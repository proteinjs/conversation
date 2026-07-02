import type { LanguageModelV3 } from '@ai-sdk/provider';
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
 * Policy: exponential backoff with FULL jitter (base 1s, factor 2, cap 20s), a provider Retry-After
 * hint wins when longer, all bounded by a wall-clock budget (default 90s). Aborts always win
 * immediately and are never retried.
 */
export class LlmTransportRetry {
  private static readonly BASE_DELAY_MS = 1_000;
  private static readonly MAX_DELAY_MS = 20_000;
  private static readonly DEFAULT_BUDGET_MS = 90_000;

  private logger = new Logger({ name: this.constructor.name });
  private budgetMs: number;

  constructor(options: LlmTransportRetryOptions = {}) {
    this.budgetMs = options.budgetMs ?? LlmTransportRetry.DEFAULT_BUDGET_MS;
  }

  /**
   * Wrap a resolved model so every request INITIATION (`doGenerate` / `doStream`) retries transient
   * failures invisibly. Only initiation is retried — once a stream has emitted chunks a failure is
   * not replayable and must propagate to the visible layers.
   */
  wrap(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({
      model,
      middleware: {
        specificationVersion: 'v3',
        wrapGenerate: ({ doGenerate, params }) =>
          this.run(doGenerate, { abortSignal: params.abortSignal, isRetryable: LlmTransportRetry.isSdkRetryable }),
        wrapStream: ({ doStream, params }) =>
          this.run(doStream, { abortSignal: params.abortSignal, isRetryable: LlmTransportRetry.isSdkRetryable }),
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
        if (options.abortSignal?.aborted || LlmTransportRetry.isAbortError(error)) {
          throw error;
        }
        if (!options.isRetryable(error)) {
          throw error;
        }
        const delayMs = this.nextDelayMs(attempt, error);
        if (Date.now() - startedAt + delayMs > this.budgetMs) {
          this.logger.error({
            message: 'LLM transport retry budget exhausted; surfacing the error',
            obj: { attempt: attempt + 1, budgetMs: this.budgetMs },
            error: error as Error,
          });
          throw error;
        }
        this.logger.warn({
          message: 'Transient LLM transport failure — retrying',
          obj: { attempt: attempt + 1, delayMs, error: String((error as Error)?.message ?? error) },
        });
        await LlmTransportRetry.sleepWithAbort(delayMs, options.abortSignal);
        if (options.abortSignal?.aborted) {
          throw error;
        }
      }
    }
  }

  /** The AI SDK's own classification: provider-marked transient (429/5xx/network) and nothing else. */
  private static isSdkRetryable(error: unknown): boolean {
    return APICallError.isInstance(error) && error.isRetryable === true;
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
