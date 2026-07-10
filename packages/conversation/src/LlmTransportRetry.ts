import type { LanguageModelV3, LanguageModelV3StreamPart, LanguageModelV3StreamResult } from '@ai-sdk/provider';
import { APICallError, wrapLanguageModel } from 'ai';
import { Logger } from '@proteinjs/logger';

export type LlmTransportRetryOptions = {
  /** Total wall-clock budget for one logical call, including backoff sleeps. Default 90s. */
  budgetMs?: number;
};

export type LlmTransportRetryRunOptions = {
  abortSignal?: AbortSignal;
  /** Classify an error as a transient transport failure (retry) vs semantic (throw immediately). */
  isRetryable: (error: unknown) => boolean;
};

/**
 * Invisible, bounded retries for TRANSIENT LLM transport failures (429s, 5xx, network drops) — the
 * model and the user never see them. Semantic errors (4xx requests the provider rejected) are never
 * retried. Exhaustion throws the last transport error so the OUTER, visible layers take over
 * (FlowRunner's task-attempt retry → the blocker-ask): one retry owner per layer, no stacking — the
 * AI/OpenAI SDKs' built-in retries are disabled where this wraps them.
 *
 * Streams get the same treatment for failures that surface BEFORE any output: providers can accept
 * a stream and then deliver e.g. a `server_error` part before emitting a single delta — to the user
 * that is indistinguishable from a failed initiation, so it retries under the same budget. Once an
 * output part has flowed, a failure is not replayable and propagates immediately.
 *
 * Policy: exponential backoff with FULL jitter (base 1s, factor 2, cap 20s), a provider Retry-After
 * hint wins when longer, all bounded by a wall-clock budget (default 90s). Aborts always win
 * immediately and are never retried.
 */
export class LlmTransportRetry {
  private static readonly BASE_DELAY_MS = 1_000;
  private static readonly MAX_DELAY_MS = 20_000;
  private static readonly DEFAULT_BUDGET_MS = 90_000;

  /**
   * Stream parts that carry model OUTPUT (or, for `finish`, mark the response complete). Once one
   * has been forwarded, a failed stream cannot be transparently replayed — the consumer has already
   * seen part of THIS response. Everything else (`stream-start`, `response-metadata`, `raw`) is
   * attempt-scoped preamble, safe to discard alongside a failed attempt.
   */
  private static readonly OUTPUT_PART_TYPES: ReadonlySet<LanguageModelV3StreamPart['type']> = new Set<
    LanguageModelV3StreamPart['type']
  >([
    'text-start',
    'text-delta',
    'text-end',
    'reasoning-start',
    'reasoning-delta',
    'reasoning-end',
    'tool-input-start',
    'tool-input-delta',
    'tool-input-end',
    'tool-call',
    'tool-result',
    'tool-approval-request',
    'file',
    'source',
    'finish',
  ]);

  /**
   * Provider-declared transient error types/codes — the mid-stream analog of a retryable 429/5xx
   * status at initiation. Anything else (`invalid_request_error`, parse failures, …) is semantic
   * and surfaces, same bar as a 400 at initiation.
   */
  private static readonly TRANSIENT_PROVIDER_ERROR_TYPES: ReadonlySet<string> = new Set([
    'server_error', // OpenAI / xAI: internal 5xx surfaced mid-stream
    'rate_limit_exceeded', // OpenAI: 429 code
    'rate_limit_error', // Anthropic: 429
    'api_error', // Anthropic: 500
    'overloaded_error', // Anthropic: 529
  ]);

  private logger = new Logger({ name: this.constructor.name });
  private budgetMs: number;

  constructor(options: LlmTransportRetryOptions = {}) {
    this.budgetMs = options.budgetMs ?? LlmTransportRetry.DEFAULT_BUDGET_MS;
  }

  /**
   * Wrap a resolved model so every request retries transient failures invisibly: request INITIATION
   * (`doGenerate` / `doStream`) and, for streams, errors that surface BEFORE any output part. Once a
   * stream has emitted output a failure is not replayable and must propagate to the visible layers.
   */
  wrap(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({
      model,
      middleware: {
        specificationVersion: 'v3',
        wrapGenerate: ({ doGenerate, params }) =>
          this.run(doGenerate, { abortSignal: params.abortSignal, isRetryable: LlmTransportRetry.isSdkRetryable }),
        wrapStream: ({ doStream, params }) => this.streamWithRetry(doStream, params.abortSignal),
      },
    });
  }

  /** Retry a plain async call under the same policy — for non-SDK transports (e.g. OpenAiResponses). */
  async run<T>(fn: () => PromiseLike<T> | T, options: LlmTransportRetryRunOptions): Promise<T> {
    const startedAt = Date.now();
    for (let attempt = 0; ; attempt++) {
      try {
        return await fn();
      } catch (error: unknown) {
        if (!(await this.shouldRetryAfterBackoff(error, attempt, startedAt, options))) {
          throw error;
        }
      }
    }
  }

  /**
   * `doStream` with the retry policy covering initiation failures AND errors that surface through
   * the stream (thrown from a read, or carried by an `error` part) before any output part — both
   * share one attempt counter and wall-clock budget. Non-output preamble parts are held back until
   * the attempt proves out (first output part / completion), so a failed attempt's preamble is
   * discarded and the consumer sees exactly one attempt's parts.
   */
  private async streamWithRetry(
    doStream: () => PromiseLike<LanguageModelV3StreamResult>,
    abortSignal?: AbortSignal
  ): Promise<LanguageModelV3StreamResult> {
    const startedAt = Date.now();
    let attempt = 0;
    const options: LlmTransportRetryRunOptions = { abortSignal, isRetryable: LlmTransportRetry.isStreamRetryable };
    const shouldRetry = (error: unknown) => this.shouldRetryAfterBackoff(error, attempt++, startedAt, options);

    const initiate = async (): Promise<LanguageModelV3StreamResult> => {
      for (;;) {
        try {
          return await doStream();
        } catch (error: unknown) {
          if (!(await shouldRetry(error))) {
            throw error;
          }
        }
      }
    };

    let current = await initiate();
    let reader = current.stream.getReader();
    /** Once true, the stream is no longer replayable — every remaining part passes straight through. */
    let outputStarted = false;
    /** The current attempt's held-back preamble (`stream-start` / `response-metadata` / `raw`). */
    let preamble: LanguageModelV3StreamPart[] = [];

    const restart = async (): Promise<void> => {
      await reader.cancel().catch(() => undefined); // release the failed attempt's connection
      preamble = [];
      current = await initiate();
      reader = current.stream.getReader();
    };

    const flushPreamble = (controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>) => {
      preamble.forEach((part) => controller.enqueue(part));
      preamble = [];
    };

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      pull: async (controller) => {
        // Loop until a part is forwarded, the stream closes, or an error surfaces.
        for (;;) {
          let read: ReadableStreamReadResult<LanguageModelV3StreamPart>;
          try {
            read = await reader.read();
          } catch (error: unknown) {
            if (!outputStarted && (await shouldRetry(error))) {
              await restart();
              continue;
            }
            throw error;
          }
          if (read.done) {
            flushPreamble(controller);
            controller.close();
            return;
          }
          const part = read.value;
          if (outputStarted) {
            controller.enqueue(part);
            return;
          }
          if (part.type === 'error') {
            if (await shouldRetry(part.error)) {
              await restart();
              continue;
            }
            // Semantic or budget-exhausted: surface exactly what an unwrapped stream would.
            outputStarted = true;
            flushPreamble(controller);
            controller.enqueue(part);
            return;
          }
          if (!LlmTransportRetry.OUTPUT_PART_TYPES.has(part.type)) {
            preamble.push(part);
            continue;
          }
          // First output part — the attempt proved out; release the preamble and go passthrough.
          outputStarted = true;
          flushPreamble(controller);
          controller.enqueue(part);
          return;
        }
      },
      cancel: (reason) => reader.cancel(reason),
    });

    // request/response metadata stays from the first successful initiation (telemetry-only).
    return { ...current, stream };
  }

  /** Decide whether `error` retries, sleeping the backoff when it does. One decision point for all paths. */
  private async shouldRetryAfterBackoff(
    error: unknown,
    attempt: number,
    startedAt: number,
    options: LlmTransportRetryRunOptions
  ): Promise<boolean> {
    if (options.abortSignal?.aborted || LlmTransportRetry.isAbortError(error)) {
      return false;
    }
    if (!options.isRetryable(error)) {
      return false;
    }
    const delayMs = this.nextDelayMs(attempt, error);
    if (Date.now() - startedAt + delayMs > this.budgetMs) {
      this.logger.error({
        message: 'LLM transport retry budget exhausted; surfacing the error',
        obj: { attempt: attempt + 1, budgetMs: this.budgetMs },
        error: error as Error,
      });
      return false;
    }
    this.logger.warn({
      message: 'Transient LLM transport failure — retrying',
      obj: { attempt: attempt + 1, delayMs, error: String((error as Error)?.message ?? error) },
    });
    await LlmTransportRetry.sleepWithAbort(delayMs, options.abortSignal);
    return !options.abortSignal?.aborted;
  }

  /** The AI SDK's own classification: provider-marked transient (429/5xx/network) and nothing else. */
  private static isSdkRetryable(error: unknown): boolean {
    return APICallError.isInstance(error) && error.isRetryable === true;
  }

  /**
   * Stream-path classification: initiation classification plus its `error`-part analog. Error parts
   * carry the provider's RAW error payload, not an APICallError (there is no HTTP status mid-stream),
   * so the provider-declared error type/code is the transient signal — OpenAI nests it
   * (`{ error: { type: 'server_error' } }`), Anthropic/xAI send it flat (`{ type: 'overloaded_error' }`).
   */
  private static isStreamRetryable(error: unknown): boolean {
    if (APICallError.isInstance(error)) {
      return error.isRetryable === true;
    }
    return LlmTransportRetry.providerErrorTypes(error).some((type) =>
      LlmTransportRetry.TRANSIENT_PROVIDER_ERROR_TYPES.has(type)
    );
  }

  /** The `type`/`code` strings on a raw provider error payload — flat and nested under `error`. */
  private static providerErrorTypes(error: unknown): string[] {
    if (typeof error !== 'object' || error === null) {
      return [];
    }
    const { type, code, error: nested } = error as { type?: unknown; code?: unknown; error?: unknown };
    return [type, code, ...LlmTransportRetry.providerErrorTypes(nested)].filter(
      (value): value is string => typeof value === 'string'
    );
  }

  /** Full-jitter exponential backoff; a provider Retry-After hint wins when longer. */
  private nextDelayMs(attempt: number, error: unknown): number {
    const ceiling = Math.min(LlmTransportRetry.MAX_DELAY_MS, LlmTransportRetry.BASE_DELAY_MS * 2 ** attempt);
    const jittered = Math.ceil(Math.random() * ceiling);
    const retryAfterMs = LlmTransportRetry.retryAfterMs(error);
    return retryAfterMs !== undefined ? Math.max(jittered, retryAfterMs) : jittered;
  }

  private static retryAfterMs(error: unknown): number | undefined {
    const headers = (error as { responseHeaders?: Record<string, string> })?.responseHeaders;
    const raw = headers?.['retry-after'];
    if (!raw) {
      return undefined;
    }
    const seconds = Number(raw);
    return Number.isFinite(seconds) && seconds >= 0
      ? Math.min(seconds * 1000, LlmTransportRetry.MAX_DELAY_MS)
      : undefined;
  }

  private static isAbortError(error: unknown): boolean {
    return error instanceof Error && (error.name === 'AbortError' || /abort/i.test(error.message));
  }

  private static sleepWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
    if (!signal) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }
    if (signal.aborted) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        cleanup();
        resolve();
      }, ms);
      const onAbort = () => {
        cleanup();
        resolve();
      };
      const cleanup = () => {
        clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
      };
      signal.addEventListener('abort', onAbort, { once: true });
    });
  }
}
