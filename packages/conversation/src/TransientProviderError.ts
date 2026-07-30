/**
 * Cross-package instance marker (the APICallError pattern): `Symbol.for` resolves to the same
 * symbol in every copy of this module, so `isInstance` holds even when a consumer's dependency
 * tree carries a different build of `@proteinjs/conversation` than the one that threw.
 */
const MARKER = Symbol.for('@proteinjs/conversation.TransientProviderError');

/**
 * A PROVIDER-side transient failure (429/5xx/overloaded/network) that persisted past the transport
 * retry budget. Tagged at the ONE transport choke point (LlmTransportRetry — every model call runs
 * through it via Conversation.resolveModelInstance), so outer layers can distinguish "the provider
 * is down" from "this request is bad" WITHOUT re-classifying provider errors: the transport layer
 * already made that judgment with ground truth (HTTP status / provider error type) when it chose
 * to retry. Semantic errors are never wrapped — they rethrow untouched.
 *
 * Consumers (e.g. the flow runner's failure taxonomy) check `TransientProviderError.isInstance`
 * and take a wait-for-the-provider path instead of burning attempts or blaming the work itself.
 */
export class TransientProviderError extends Error {
  /** HTTP status of the underlying failure when known (429, 500, 529, …). */
  readonly statusCode?: number;
  /** The original error exactly as the transport surfaced it. */
  readonly cause: unknown;

  constructor(args: { message: string; cause: unknown; statusCode?: number }) {
    super(args.message);
    this.name = 'TransientProviderError';
    this.cause = args.cause;
    this.statusCode = args.statusCode;
    Object.defineProperty(this, MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is TransientProviderError {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[MARKER] === true;
  }

  /** Wrap the transport's original error (idempotent), preserving its message and HTTP status. */
  static wrap(error: unknown): TransientProviderError {
    if (TransientProviderError.isInstance(error)) {
      return error;
    }
    const message =
      error instanceof Error ? error.message : typeof error === 'string' ? error : JSON.stringify(error ?? null);
    const statusCode = (error as { statusCode?: unknown } | null | undefined)?.statusCode;
    return new TransientProviderError({
      message: `Model provider unavailable (transient failure persisted past the retry budget): ${message}`,
      cause: error,
      ...(typeof statusCode === 'number' ? { statusCode } : {}),
    });
  }
}
