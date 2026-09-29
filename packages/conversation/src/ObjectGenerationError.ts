import { NoObjectGeneratedError } from 'ai';

/**
 * Cross-package instance markers (the TransientProviderError/APICallError pattern): `Symbol.for`
 * resolves to the same symbol in every copy of this module, so `isInstance` holds even when a
 * consumer's dependency tree carries a different build of `@proteinjs/conversation`.
 */
const MARKER = Symbol.for('@proteinjs/conversation.ObjectGenerationError');
const TRUNCATED_MARKER = Symbol.for('@proteinjs/conversation.ObjectTruncatedError');
const REFUSED_MARKER = Symbol.for('@proteinjs/conversation.ObjectRefusedError');
const PARSE_MARKER = Symbol.for('@proteinjs/conversation.ObjectParseError');

/** How much of the answer's text a typed error carries, each end. */
const TEXT_EDGE_CHARS = 200;
/** How many of the schema's top-level property names name the call when the schema has no title. */
const MAX_SCHEMA_PROPERTIES = 8;

/** Who set the output cap that cut the answer: the caller's `maxTokens`, the model's own ceiling, or nobody known. */
export type ObjectCapOwner = 'caller' | 'model' | 'unknown';

/** What the library knows about the call whose answer was not the object; each part is optional. */
export type ObjectGenerationCall = {
  /** The model's id. */
  modelId?: string;
  /** The provider as the client library names it (`anthropic.messages`). */
  provider?: string;
  /** The caller's own output cap (`maxTokens`), when it set one. */
  requestedMaxTokens?: number;
  /** The provider's output ceiling for the model, when the library knows it. */
  modelMaxTokens?: number;
  /** The requested schema (a JSON schema's `title` and property names name the call). */
  schema?: unknown;
};

/** The facts every typed object error carries; read from the client library's error and the call. */
export type ObjectGenerationFacts = {
  /** The client library's unified finish reason (`length`, `content-filter`, `stop`, …). */
  finishReason: string;
  /** The provider's own word for the finish (`max_tokens`, `refusal`), when the response carried it. */
  rawStopReason?: string;
  /** The schema's `title`, when it has one. */
  schemaTitle?: string;
  /** The schema's top-level property names (a few), when it has no title. */
  schemaProperties?: string[];
  modelId?: string;
  /** The first characters of the answer's text. */
  textHead: string;
  /** The last characters of the answer's text, when it is longer than the head. */
  textTail: string;
  /** The answer's whole length in characters. */
  textLength: number;
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
};

/**
 * The library's own typed error for a structured answer that was not the object — the one thing
 * the client library raises for it (`NoObjectGeneratedError`, carrying the answer's text, the
 * finish reason and the usage) read at the one owner (`Conversation.generateObject`) and named by
 * finish reason, so every layer above classifies on a class instead of on a message:
 *
 * - {@link ObjectTruncatedError} — the answer was cut off at an output cap (finish `length`): WHOSE
 *   cap (the caller's `maxTokens` or the model's own ceiling), and whether the model RAN AWAY to
 *   its own ceiling with no reasoning — the signal that tells "the provider ran away" from "we
 *   capped it".
 * - {@link ObjectRefusedError} — the provider declined the answer (finish `content-filter`).
 * - {@link ObjectParseError} — the answer finished on its own and still did not parse.
 *
 * Every kind is deterministic on the same request: the library re-issues nothing and repairs
 * nothing through the model (a generation that ran to the cap once is the same spend the second
 * time; the transport owns the retries that heal). The client library's error stays whole as
 * `cause` (the full text, the response's headers) for whoever classifies on it; what a LOG LINE
 * prints is the stand-in `ProviderFailureLine` builds from these facts, never the text.
 */
export class ObjectGenerationError extends Error {
  readonly finishReason: string;
  readonly rawStopReason?: string;
  readonly schemaTitle?: string;
  readonly schemaProperties?: string[];
  readonly modelId?: string;
  readonly textHead: string;
  readonly textTail: string;
  readonly textLength: number;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly reasoningTokens?: number;
  /** The client library's error exactly as it raised it (the whole text and the response ride it). */
  readonly cause: unknown;

  constructor(message: string, facts: ObjectGenerationFacts, cause: unknown) {
    super(message);
    this.name = 'ObjectGenerationError';
    this.finishReason = facts.finishReason;
    this.rawStopReason = facts.rawStopReason;
    this.schemaTitle = facts.schemaTitle;
    this.schemaProperties = facts.schemaProperties;
    this.modelId = facts.modelId;
    this.textHead = facts.textHead;
    this.textTail = facts.textTail;
    this.textLength = facts.textLength;
    this.inputTokens = facts.inputTokens;
    this.outputTokens = facts.outputTokens;
    this.reasoningTokens = facts.reasoningTokens;
    this.cause = cause;
    Object.defineProperty(this, MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is ObjectGenerationError {
    return ObjectGenerationError.carries(error, MARKER);
  }

  /**
   * The typed error for the client library's `NoObjectGeneratedError`, by finish reason:
   * `length` → {@link ObjectTruncatedError}, `content-filter` → {@link ObjectRefusedError},
   * anything else → {@link ObjectParseError}. Never throws on an odd shape.
   */
  static fromSdk(error: NoObjectGeneratedError, call: ObjectGenerationCall): ObjectGenerationError {
    const facts = ObjectGenerationError.factsOf(error, call);
    if (facts.finishReason === 'length') {
      return new ObjectTruncatedError(facts, call, error);
    }
    if (facts.finishReason === 'content-filter') {
      return new ObjectRefusedError(facts, error);
    }
    return new ObjectParseError(facts, error);
  }

  /** The clause that names the call on every message: `… for schema "X" on model-id`. */
  protected static callClause(facts: ObjectGenerationFacts): string {
    const schema = facts.schemaTitle
      ? ` for schema "${facts.schemaTitle}"`
      : facts.schemaProperties?.length
        ? ` for the schema with ${facts.schemaProperties.map((name) => `"${name}"`).join(', ')}`
        : '';
    return `${schema}${facts.modelId ? ` on ${facts.modelId}` : ''}`;
  }

  /** `finish reason "length", stop reason "max_tokens"`. */
  protected static finishClause(facts: ObjectGenerationFacts): string {
    return `finish reason "${facts.finishReason}"${facts.rawStopReason ? `, stop reason "${facts.rawStopReason}"` : ''}`;
  }

  protected static carries(error: unknown, marker: symbol): boolean {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[marker] === true;
  }

  private static factsOf(error: NoObjectGeneratedError, call: ObjectGenerationCall): ObjectGenerationFacts {
    const text = typeof error.text === 'string' ? error.text : '';
    const usage = ObjectGenerationError.recordOf(error.usage);
    const outputTokens = ObjectGenerationError.countOf(usage?.outputTokens);
    const outputDetails = ObjectGenerationError.recordOf(usage?.outputTokenDetails);
    const reasoningTokens =
      ObjectGenerationError.countOf(outputDetails?.reasoningTokens) ??
      ObjectGenerationError.countOf(usage?.reasoningTokens) ??
      ObjectGenerationError.countOf(ObjectGenerationError.recordOf(usage?.outputTokens)?.reasoning);
    const schema = ObjectGenerationError.recordOf(call.schema);
    const schemaTitle = typeof schema?.title === 'string' && schema.title.trim() ? schema.title.trim() : undefined;
    const properties = ObjectGenerationError.recordOf(schema?.properties);
    const schemaProperties =
      !schemaTitle && properties ? Object.keys(properties).slice(0, MAX_SCHEMA_PROPERTIES) : undefined;
    return {
      finishReason: typeof error.finishReason === 'string' && error.finishReason ? error.finishReason : 'unknown',
      rawStopReason: ObjectGenerationError.rawStopReasonOf(error),
      ...(schemaTitle ? { schemaTitle } : {}),
      ...(schemaProperties?.length ? { schemaProperties } : {}),
      modelId: call.modelId ?? ObjectGenerationError.stringOf(ObjectGenerationError.recordOf(error.response)?.modelId),
      textHead: text.slice(0, TEXT_EDGE_CHARS),
      textTail:
        text.length > TEXT_EDGE_CHARS ? text.slice(Math.max(TEXT_EDGE_CHARS, text.length - TEXT_EDGE_CHARS)) : '',
      textLength: text.length,
      inputTokens: ObjectGenerationError.countOf(usage?.inputTokens),
      outputTokens,
      reasoningTokens,
    };
  }

  /**
   * The provider's own finish word from the response body the client library kept: Anthropic's
   * `stop_reason`, the OpenAI Responses API's `incomplete_details.reason`, a chat completion's
   * first choice's `finish_reason`. An optional-field sniff, never a schema demand.
   */
  private static rawStopReasonOf(error: NoObjectGeneratedError): string | undefined {
    const body = ObjectGenerationError.recordOf(ObjectGenerationError.recordOf(error.response)?.body);
    if (!body) {
      return undefined;
    }
    const choices = Array.isArray(body.choices) ? body.choices : [];
    return (
      ObjectGenerationError.stringOf(body.stop_reason) ??
      ObjectGenerationError.stringOf(ObjectGenerationError.recordOf(body.incomplete_details)?.reason) ??
      ObjectGenerationError.stringOf(ObjectGenerationError.recordOf(choices[0])?.finish_reason)
    );
  }

  /** A token count as the client library's usage carries it: a number, or `{ total }`. */
  private static countOf(value: unknown): number | undefined {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    const total = ObjectGenerationError.recordOf(value)?.total;
    return typeof total === 'number' && Number.isFinite(total) ? total : undefined;
  }

  private static stringOf(value: unknown): string | undefined {
    return typeof value === 'string' && value ? value : undefined;
  }

  private static recordOf(value: unknown): { [member: string]: unknown } | undefined {
    return typeof value === 'object' && value !== null ? (value as { [member: string]: unknown }) : undefined;
  }
}

/**
 * The answer was cut off at an output cap (finish `length`). `cap` is the cap that bit and
 * `capOwner` whose it was: the caller's `maxTokens` ("we capped it" — the cap is the call's to
 * raise) or the model's own ceiling. `runaway` is true when the model ran to ITS OWN ceiling with
 * no reasoning tokens at all — a generation that never converged, the provider's incident to
 * watch, not a budget the call can raise; re-issuing the request spends the ceiling again.
 */
export class ObjectTruncatedError extends ObjectGenerationError {
  readonly cap?: number;
  readonly capOwner: ObjectCapOwner;
  readonly runaway: boolean;

  constructor(facts: ObjectGenerationFacts, call: ObjectGenerationCall, cause: unknown) {
    const capOwner: ObjectCapOwner =
      call.requestedMaxTokens !== undefined ? 'caller' : call.modelMaxTokens !== undefined ? 'model' : 'unknown';
    const cap =
      capOwner === 'caller' ? call.requestedMaxTokens : capOwner === 'model' ? call.modelMaxTokens : undefined;
    const runaway =
      capOwner === 'model' &&
      facts.outputTokens !== undefined &&
      cap !== undefined &&
      facts.outputTokens >= cap &&
      (facts.reasoningTokens ?? 0) === 0;
    super(
      `The structured answer was cut off at ${cap !== undefined ? `${cap} output tokens` : 'the output limit'}` +
        `${capOwner === 'caller' ? " (the call's own cap)" : capOwner === 'model' ? " (the model's own ceiling)" : ''}` +
        `${runaway ? ' — the model ran to its ceiling with no reasoning, a runaway generation' : ''}` +
        ` (${ObjectGenerationError.finishClause(facts)})${ObjectGenerationError.callClause(facts)}.`,
      facts,
      cause
    );
    this.name = 'ObjectTruncatedError';
    this.cap = cap;
    this.capOwner = capOwner;
    this.runaway = runaway;
    Object.defineProperty(this, TRUNCATED_MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is ObjectTruncatedError {
    return ObjectGenerationError.carries(error, TRUNCATED_MARKER);
  }
}

/** The provider declined the structured answer (finish `content-filter`; Anthropic's `stop_reason: refusal`). */
export class ObjectRefusedError extends ObjectGenerationError {
  constructor(facts: ObjectGenerationFacts, cause: unknown) {
    super(
      `The model provider declined the structured answer (${ObjectGenerationError.finishClause(facts)})` +
        `${ObjectGenerationError.callClause(facts)}.`,
      facts,
      cause
    );
    this.name = 'ObjectRefusedError';
    Object.defineProperty(this, REFUSED_MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is ObjectRefusedError {
    return ObjectGenerationError.carries(error, REFUSED_MARKER);
  }
}

/** The answer finished on its own (or for a reason that is neither a cap nor a refusal) and did not parse into the schema. */
export class ObjectParseError extends ObjectGenerationError {
  constructor(facts: ObjectGenerationFacts, cause: unknown) {
    super(
      `The structured answer did not parse into the requested shape (${ObjectGenerationError.finishClause(facts)})` +
        `${ObjectGenerationError.callClause(facts)}.`,
      facts,
      cause
    );
    this.name = 'ObjectParseError';
    Object.defineProperty(this, PARSE_MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is ObjectParseError {
    return ObjectGenerationError.carries(error, PARSE_MARKER);
  }
}
