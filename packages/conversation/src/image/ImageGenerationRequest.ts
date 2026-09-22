/**
 * What a reference picture is FOR. The caller names the role; saying it to the vendor is the
 * adapter's job (a parameter where the vendor has one, words in the prompt where it does not).
 *
 * - `structure` — the layout to keep (a room's walls and windows, a page's composition);
 * - `subject` — the thing to keep recognisable (a product, a mark, a person's own photo);
 * - `style` — a look to borrow (colour, light, texture), never content to copy.
 */
export type ImageInputRole = 'structure' | 'subject' | 'style';

/** How closely the result should hold to a reference: `high` = keep it as it is, `low` = loosely. */
export type ImageInputFidelity = 'high' | 'low';

/** One reference picture sent with an ask. The bytes are the caller's; nothing here is a URL. */
export type ImageInput = {
  bytes: Uint8Array;
  /** e.g. `image/png`, `image/jpeg`, `image/webp` */
  mimeType: string;
  /** A file name for the vendor's multipart field; never shown to anyone. */
  name?: string;
  role?: ImageInputRole;
  fidelity?: ImageInputFidelity;
};

export type ImageQuality = 'low' | 'medium' | 'high' | 'xhigh' | 'max' | 'auto';
export type ImageBackground = 'transparent' | 'opaque' | 'auto';
export type ImageOutputFormat = 'png' | 'jpeg' | 'webp';

/**
 * What the vendor is asked to DO. `generate` (the default) makes pictures from the prompt and any
 * references. The two utilities take exactly one input picture and no prompt: `vectorize` traces
 * a raster into an SVG; `remove-background` returns the picture with its background cleared.
 * A vendor that does not offer a utility refuses it before anything is sent.
 */
export type ImageOperation = 'generate' | 'vectorize' | 'remove-background';

/**
 * One ask for pictures, in vendor-neutral words. `provider` + `model` come from the caller's own
 * model data (a picture model's id does not always say who serves it, so the provider is never
 * inferred from the name here).
 */
export type ImageGenerationRequest = {
  /** The adapter key: `'openai'`, … */
  provider: string;
  /** The provider's model id, exactly as the provider spells it. */
  model: string;
  /** Default `generate`. */
  operation?: ImageOperation;
  prompt: string;
  /**
   * The vendor's handle for the earlier ask this one continues (a room edited again, keeping the
   * first edit's geometry). Only a vendor that keeps such a handle uses it — `GeneratedPictures.
   * continuationId` on an `ok` outcome says whether one exists; the others ignore it.
   */
  previousInteractionId?: string;
  /** Reference pictures. Present → the ask is an edit of / from these pictures. */
  inputs?: ImageInput[];
  /** How many pictures to make. Default 1. */
  count?: number;
  /** `WIDTHxHEIGHT` or `auto`; the adapter checks the vendor's rules before anything is sent. */
  size?: string;
  quality?: ImageQuality;
  background?: ImageBackground;
  /** Default `png`. */
  outputFormat?: ImageOutputFormat;
  /**
   * The caller's stop. It is threaded to the wire; once it fires, `generate()` resolves with a
   * `stopped` outcome. No picture is ever surfaced — one the vendor had already sent is
   * discarded — but what is known about the spend (`sent`, `usage`, `cost`) still is, so the
   * caller can record it.
   */
  signal?: AbortSignal;
};
