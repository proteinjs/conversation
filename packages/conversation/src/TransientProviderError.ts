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
  /** The failing model's id when the transport knew it — lets any surface name WHO is down. */
  readonly modelId?: string;

  constructor(args: { message: string; cause: unknown; statusCode?: number; modelId?: string }) {
    super(args.message);
    this.name = 'TransientProviderError';
    this.cause = args.cause;
    this.statusCode = args.statusCode;
    this.modelId = args.modelId;
    Object.defineProperty(this, MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is TransientProviderError {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[MARKER] === true;
  }

  /** Wrap the transport's original error (idempotent), preserving its message and HTTP status. */
  static wrap(error: unknown, modelId?: string): TransientProviderError {
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
      ...(modelId ? { modelId } : {}),
    });
  }
}

/**
 * The provider's everyday name from a model id — user-facing outage copy names WHO is down
 * ("Anthropic is having trouble"), not "the model provider". Undefined for unknown families so
 * callers fall back to the generic phrase deliberately. Lives beside the typed error because the
 * error is the cross-surface carrier: every consumer that catches it needs the same mapping.
 */
export function providerDisplayName(modelId: string | undefined): string | undefined {
  const id = (modelId ?? '').toLowerCase();
  if (!id) {
    return undefined;
  }
  if (id.includes('claude')) {
    return 'Anthropic';
  }
  if (id.includes('gpt') || id.includes('openai') || /(^|[^a-z])o\d/.test(id)) {
    return 'OpenAI';
  }
  if (id.includes('gemini')) {
    return 'Google';
  }
  return undefined;
}

/**
 * Everyday-words cause for user-facing outage copy ("their systems are overloaded"); the raw
 * error text stays on records/logs. One mapping for every surface that renders the typed error.
 */
export function describeTransientCause(args: { message?: string; statusCode?: number }): string {
  const text = `${args.message ?? ''} ${args.statusCode ?? ''}`.toLowerCase();
  if (text.includes('overload') || text.includes('529')) {
    return 'their systems are overloaded';
  }
  if (text.includes('rate') && text.includes('limit')) {
    return "they're rate-limiting requests";
  }
  if (text.includes('unavailable') || text.includes('timeout') || text.includes('timed out') || text.includes('503')) {
    return "their service isn't responding";
  }
  return "they're having service trouble";
}
