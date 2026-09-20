/**
 * What an ask for pictures came to. Four kinds, and only four, so a ledger can record each ask
 * without re-reading a vendor's error:
 *
 * - `ok` — at least one picture was made;
 * - `refused` — the vendor's moderation declined the ask (its reason is carried, in its words);
 * - `failed` — nothing was made for any other reason; `transient` says whether asking again
 *   later could work (a rate limit, a vendor outage, a dropped connection, a timeout);
 * - `stopped` — the caller's `AbortSignal` fired. No picture is surfaced, not even one the
 *   vendor had already made (those bytes are discarded inside the generator).
 *
 * EVERY kind says what is known about the spend — `sent`, `usage`, `cost` — because a vendor
 * bills for what it made whether or not the caller still wants it. `generate()` resolves with
 * one of these for every ask it accepts; an ask is never ended by a rejection that would take
 * its spend with it.
 */
export type ImageGenerationOutcome =
  | ImageGenerationOk
  | ImageGenerationRefused
  | ImageGenerationFailed
  | ImageGenerationStopped;

/** The vendor's own count of what an ask used. Every field is absent when the vendor did not report it. */
export type ImageUsage = {
  textInputTokens?: number;
  imageInputTokens?: number;
  cachedTextInputTokens?: number;
  cachedImageInputTokens?: number;
  textOutputTokens?: number;
  imageOutputTokens?: number;
  totalTokens?: number;
};

/** One picture that was made. */
export type GeneratedImage = {
  bytes: Uint8Array;
  mimeType: string;
  /** The vendor's rewrite of the prompt, where it reports one. */
  revisedPrompt?: string;
};

/** The priced split of one ask, in USD at full precision (rounding belongs to whoever displays it). */
export type ImageCostUsd = {
  textInputUsd: number;
  imageInputUsd: number;
  outputUsd: number;
  totalUsd: number;
};

type ImageGenerationOutcomeBase = {
  provider: string;
  model: string;
  /** The vendor's id for the request, for a support ticket. Never a credential. */
  vendorRequestId?: string;
  latencyMs?: number;
  /**
   * False when the ask never left this process (nothing was sent, so nothing can have been
   * billed); true once a request went out.
   */
  sent: boolean;
  /** What the vendor said the ask used — carried on ANY kind whose answer reported it. */
  usage?: ImageUsage;
  /**
   * What the ask cost. PRESENT = known — including a known zero (all four fields 0): nothing
   * was sent, or the vendor turned the ask away before making anything. ABSENT = NOT KNOWN: the
   * vendor may have billed and the amount cannot be established (no usage in its answer, no
   * rate for the model, no answer at all after the request went out). Never a guess — a record
   * of spend writes an absent cost as "not priced", never as zero.
   */
  cost?: ImageCostUsd;
};

export type ImageGenerationOk = ImageGenerationOutcomeBase & {
  kind: 'ok';
  /** The pictures actually made — may be fewer than asked for; an unreadable one is left out. */
  images: GeneratedImage[];
};

export type ImageGenerationRefused = ImageGenerationOutcomeBase & {
  kind: 'refused';
  /** The vendor's code for the refusal, e.g. `moderation_blocked`. */
  reason: string;
  /** Where the vendor's check stopped it, when it says: the ask (`input`) or the picture (`output`). */
  stage?: 'input' | 'output' | 'unknown';
  /** The vendor's categories, when it names them. */
  categories?: string[];
  /** The vendor's sentence. For records and support — not written for an end user. */
  message?: string;
};

export type ImageGenerationFailureKind =
  /** The ask itself is not valid — stopped here before the wire, or rejected by the vendor (4xx). */
  | 'invalid_request'
  /** No credential, or one the vendor does not accept. */
  | 'auth'
  /** The vendor account is out of credit or past a spend limit. Waiting does not fix it. */
  | 'billing'
  | 'rate_limited'
  /** The vendor failed (5xx). */
  | 'provider_error'
  /** The connection failed before an answer arrived. */
  | 'network'
  | 'timeout'
  /** The vendor answered 2xx with no readable picture. */
  | 'malformed_response'
  /** The adapter itself threw — a defect on this side, not an answer from the vendor. */
  | 'adapter_error';

export type ImageGenerationFailed = ImageGenerationOutcomeBase & {
  kind: 'failed';
  errorKind: ImageGenerationFailureKind;
  /** True when asking again later could work. */
  transient: boolean;
  statusCode?: number;
  /** The vendor's error code, when it gave one. */
  code?: string;
  message: string;
};

/**
 * The caller stopped the ask. There is no `images` field on purpose: whatever the vendor had
 * already made is discarded before this outcome exists and is only counted here. The spend is
 * still reported — `sent: false` with a zero cost when the stop landed before anything went
 * out; the answer's own `usage` and `cost` when it arrived anyway; `sent: true` with no cost
 * when the request was cut off mid-flight (the vendor may have finished and billed it).
 */
export type ImageGenerationStopped = ImageGenerationOutcomeBase & {
  kind: 'stopped';
  /** Pictures the vendor had made that were thrown away because of the stop. */
  discardedImages: number;
};
