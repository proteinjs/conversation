import type { LanguageModelV3, LanguageModelV3StreamPart, LanguageModelV3StreamResult } from '@ai-sdk/provider';
import { APICallError, wrapLanguageModel } from 'ai';
import { Logger } from '@proteinjs/logger';
import { TransientProviderError } from './TransientProviderError';

export type LlmTransportRetryOptions = {
  /** Total wall-clock budget for one logical call, including backoff sleeps. Default 90s. */
  budgetMs?: number;
};

/**
 * Live visibility into the retry loop, for surfaces that render the wait (e.g. the chat turn's
 * thinking-timeline provider-wait node). Purely observational — emission never alters retry
 * semantics, and the default (no listener) keeps retries invisible exactly as before.
 *
 * - `retrying` — a transient failure was absorbed and the next attempt is scheduled after
 *   `delayMs`. Emitted BEFORE the backoff sleep, so the wait can render while it happens.
 * - `recovered` — a later attempt proved out: the call resolved, or the retried stream produced
 *   output (or completed). The wait is over and the response is flowing.
 * - `gave-up` — no further attempt follows a `retrying`: the budget exhausted (the error
 *   surfaces as `TransientProviderError`), or an abort/semantic error arrived after retries had
 *   begun. Emitted only when at least one `retrying` preceded it, so consumers can treat it as
 *   the settle of the wait they began.
 */
export type LlmTransportRetryActivity =
  | {
      phase: 'retrying';
      /** 1-based count of transient failures absorbed so far (1 = first retry scheduled). */
      attempt: number;
      /** The backoff sleep before the next attempt. */
      delayMs: number;
      modelId?: string;
      /** HTTP status of the absorbed failure when known (429, 500, 529, …). */
      statusCode?: number;
      /** The transport error's message — for cause classification, never rendered raw. */
      message: string;
    }
  | { phase: 'recovered'; modelId?: string }
  | { phase: 'gave-up'; modelId?: string; statusCode?: number; message: string };

export type LlmTransportRetryWrapOptions = {
  /** See {@link LlmTransportRetryActivity}. */
  onRetryActivity?: (activity: LlmTransportRetryActivity) => void;
};

export type LlmTransportRetryRunOptions = {
  abortSignal?: AbortSignal;
  /** Classify an error as a transient transport failure (retry) vs semantic (throw immediately). */
  isRetryable: (error: unknown) => boolean;
  /** The model behind the call when known — rides the surfaced TransientProviderError so any
   *  consumer can name the provider in user-facing copy. */
  modelId?: string;
  /** See {@link LlmTransportRetryActivity}. */
  onRetryActivity?: (activity: LlmTransportRetryActivity) => void;
};

/**
 * What to do with a failed attempt's error:
 * - `retry` — transient and within budget; the backoff sleep already happened.
 * - `surface` — semantic (or abort): rethrow the ORIGINAL error untouched.
 * - `surface-transient` — TRANSIENT but the retry budget is exhausted: surface it wrapped as
 *   `TransientProviderError`, so outer layers see "the provider is down", not "the request is bad".
 */
type RetryVerdict = 'retry' | 'surface' | 'surface-transient';

/**
 * Invisible, bounded retries for TRANSIENT LLM transport failures (429s, 5xx, network drops) — the
 * model and the user never see them. Semantic errors (4xx requests the provider rejected) are never
 * retried. Exhaustion throws the last transport error WRAPPED as `TransientProviderError` — this is
 * the single choke point with ground truth on the transient/semantic distinction, so it tags the
 * error once and the OUTER, visible layers route on the type (FlowRunner's provider-wait park vs
 * task-attempt retry → blocker-ask): one retry owner per layer, no stacking — the AI/OpenAI SDKs'
 * built-in retries are disabled where this wraps them. Non-transient errors rethrow untouched.
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
  wrap(model: LanguageModelV3, options: LlmTransportRetryWrapOptions = {}): LanguageModelV3 {
    const modelId = model.modelId;
    const onRetryActivity = options.onRetryActivity;
    return wrapLanguageModel({
      model,
      middleware: {
        specificationVersion: 'v3',
        wrapGenerate: ({ doGenerate, params }) =>
          this.run(doGenerate, {
            abortSignal: params.abortSignal,
            isRetryable: LlmTransportRetry.isSdkRetryable,
            modelId,
            onRetryActivity,
          }),
        wrapStream: ({ doStream, params }) =>
          this.streamWithRetry(doStream, params.abortSignal, modelId, onRetryActivity),
      },
    });
  }

  /** Retry a plain async call under the same policy — for non-SDK transports (e.g. OpenAiResponses). */
  async run<T>(fn: () => PromiseLike<T> | T, options: LlmTransportRetryRunOptions): Promise<T> {
    const startedAt = Date.now();
    for (let attempt = 0; ; attempt++) {
      try {
        const value = await fn();
        if (attempt > 0) {
          options.onRetryActivity?.({ phase: 'recovered', modelId: options.modelId });
        }
        return value;
      } catch (error: unknown) {
        const verdict = await this.verdictAfterBackoff(error, attempt, startedAt, options);
        if (verdict !== 'retry') {
          throw LlmTransportRetry.surfaced(verdict, error, options.modelId);
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
    abortSignal?: AbortSignal,
    modelId?: string,
    onRetryActivity?: (activity: LlmTransportRetryActivity) => void
  ): Promise<LanguageModelV3StreamResult> {
    const startedAt = Date.now();
    let attempt = 0;
    const options: LlmTransportRetryRunOptions = {
      abortSignal,
      isRetryable: LlmTransportRetry.isStreamRetryable,
      modelId,
      onRetryActivity,
    };
    const verdictOf = (error: unknown) => this.verdictAfterBackoff(error, attempt++, startedAt, options);
    // The retried stream proved out (first output part, or a clean end): the wait is over.
    // Emitted at most once per logical call — post-output failures are not replayable, so a
    // stream never re-enters the retry loop after this.
    let recoveredEmitted = false;
    const emitRecoveredIfRetried = () => {
      if (attempt > 0 && !recoveredEmitted) {
        recoveredEmitted = true;
        onRetryActivity?.({ phase: 'recovered', modelId });
      }
    };

    const initiate = async (): Promise<LanguageModelV3StreamResult> => {
      for (;;) {
        try {
          return await doStream();
        } catch (error: unknown) {
          const verdict = await verdictOf(error);
          if (verdict !== 'retry') {
            throw LlmTransportRetry.surfaced(verdict, error, modelId);
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
            if (outputStarted) {
              throw error;
            }
            const verdict = await verdictOf(error);
            if (verdict === 'retry') {
              await restart();
              continue;
            }
            throw LlmTransportRetry.surfaced(verdict, error, modelId);
          }
          if (read.done) {
            emitRecoveredIfRetried();
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
            const verdict = await verdictOf(part.error);
            if (verdict === 'retry') {
              await restart();
              continue;
            }
            // Semantic or budget-exhausted: surface as a stream part exactly like an unwrapped
            // stream would — but a budget-exhausted TRANSIENT error carries the type tag.
            // The episode SETTLED as a failure (verdictAfterBackoff emitted any gave-up): the
            // stream's trailing done-read must not report a recovery.
            recoveredEmitted = true;
            outputStarted = true;
            flushPreamble(controller);
            controller.enqueue(
              verdict === 'surface-transient'
                ? { ...part, error: TransientProviderError.wrap(part.error, modelId) }
                : part
            );
            return;
          }
          if (!LlmTransportRetry.OUTPUT_PART_TYPES.has(part.type)) {
            preamble.push(part);
            continue;
          }
          // First output part — the attempt proved out; release the preamble and go passthrough.
          outputStarted = true;
          emitRecoveredIfRetried();
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

  /** Decide what to do with `error`, sleeping the backoff on `retry`. One decision point for all paths. */
  private async verdictAfterBackoff(
    error: unknown,
    attempt: number,
    startedAt: number,
    options: LlmTransportRetryRunOptions
  ): Promise<RetryVerdict> {
    // A surface verdict after retries had begun settles the observer's wait (`gave-up`); a
    // surface with no prior retry emitted nothing, so there is no wait to settle.
    const emitGaveUpIfRetried = () => {
      if (attempt > 0) {
        options.onRetryActivity?.({
          phase: 'gave-up',
          modelId: options.modelId,
          ...LlmTransportRetry.errorInfo(error),
        });
      }
    };
    if (options.abortSignal?.aborted || LlmTransportRetry.isAbortError(error)) {
      emitGaveUpIfRetried();
      return 'surface';
    }
    if (!options.isRetryable(error)) {
      emitGaveUpIfRetried();
      return 'surface';
    }
    const delayMs = this.nextDelayMs(attempt, error);
    if (Date.now() - startedAt + delayMs > this.budgetMs) {
      this.logger.error({
        message: 'LLM transport retry budget exhausted; surfacing as TransientProviderError',
        obj: { attempt: attempt + 1, budgetMs: this.budgetMs },
        error: error as Error,
      });
      emitGaveUpIfRetried();
      return 'surface-transient';
    }
    this.logger.warn({
      message: 'Transient LLM transport failure — retrying',
      obj: { attempt: attempt + 1, delayMs, error: String((error as Error)?.message ?? error) },
    });
    options.onRetryActivity?.({
      phase: 'retrying',
      attempt: attempt + 1,
      delayMs,
      modelId: options.modelId,
      ...LlmTransportRetry.errorInfo(error),
    });
    await LlmTransportRetry.sleepWithAbort(delayMs, options.abortSignal);
    if (options.abortSignal?.aborted) {
      // The abort landed during the backoff sleep — the retry we announced never runs.
      options.onRetryActivity?.({
        phase: 'gave-up',
        modelId: options.modelId,
        ...LlmTransportRetry.errorInfo(error),
      });
      return 'surface';
    }
    return 'retry';
  }

  /** The failure's status/message as the observer sees them — mirror of TransientProviderError.wrap's extraction. */
  private static errorInfo(error: unknown): { statusCode?: number; message: string } {
    const message =
      error instanceof Error ? error.message : typeof error === 'string' ? error : JSON.stringify(error ?? null);
    const statusCode = (error as { statusCode?: unknown } | null | undefined)?.statusCode;
    return { message, ...(typeof statusCode === 'number' ? { statusCode } : {}) };
  }

  /** The error a non-retry verdict throws: the original, tagged only when classified transient. */
  private static surfaced(verdict: RetryVerdict, error: unknown, modelId?: string): unknown {
    return verdict === 'surface-transient' ? TransientProviderError.wrap(error, modelId) : error;
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
