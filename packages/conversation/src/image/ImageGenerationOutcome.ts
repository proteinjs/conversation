/**
 * What an ask for pictures came to. Three kinds, and only three, so a ledger can record each ask
 * as `ok`, `refused` or `failed` without re-reading a vendor's error:
 *
 * - `ok` — at least one picture was made;
 * - `refused` — the vendor's moderation declined the ask (its reason is carried, in its words);
 * - `failed` — nothing was made for any other reason; `transient` says whether asking again
 *   later could work (a rate limit, a vendor outage, a dropped connection, a timeout).
 *
 * A stop (the caller's `AbortSignal`) is none of these: `generate()` rejects.
 */
export type ImageGenerationOutcome = ImageGenerationOk | ImageGenerationRefused | ImageGenerationFailed;

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
};

export type ImageGenerationOk = ImageGenerationOutcomeBase & {
  kind: 'ok';
  /** The pictures actually made — may be fewer than asked for; an unreadable one is left out. */
  images: GeneratedImage[];
  usage?: ImageUsage;
  /** Absent = the price is NOT KNOWN (no usage from the vendor, or no rate for it). Never a guess. */
  cost?: ImageCostUsd;
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
  | 'malformed_response';

export type ImageGenerationFailed = ImageGenerationOutcomeBase & {
  kind: 'failed';
  errorKind: ImageGenerationFailureKind;
  /** True when asking again later could work. */
  transient: boolean;
  /**
   * False when the ask never left this process (nothing was sent, so nothing can have been
   * billed); true once a request went out.
   */
  sent: boolean;
  statusCode?: number;
  /** The vendor's error code, when it gave one. */
  code?: string;
  message: string;
};
