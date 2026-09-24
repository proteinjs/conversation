import type { ImageGenerationRequest } from './ImageGenerationRequest';
import type { ImageGenerationFailed, ImageGenerationOk, ImageGenerationRefused } from './ImageGenerationOutcome';

/**
 * One vendor's way of making pictures. An adapter turns the vendor-neutral ask into that vendor's
 * request, sends it through the transport it is handed, and reads the answer back into the three
 * outcome kinds. It never prices anything (`ImageGenerator` does, from the usage the adapter
 * reports) and it never opens a connection of its own — so a test, or CI, never calls a vendor.
 *
 * An adapter REJECTS only when its context's `signal` fired before an answer arrived. Every
 * other ending — a vendor error, a dropped connection, an ask it would not send — is a result.
 */
export interface ImageProviderAdapter {
  /** The key an ask's `provider` is matched on, e.g. `'openai'`. */
  readonly provider: string;
  generate(request: ImageGenerationRequest, context: ImageAdapterContext): Promise<ImageAdapterResult>;
}

export type ImageAdapterContext = {
  transport: ImageTransport;
  /** The stop for this call: the caller's signal joined with the generator's deadline. */
  signal?: AbortSignal;
};

type Unpriced<Outcome> = Omit<Outcome, 'provider' | 'model' | 'latencyMs' | 'cost'>;

/**
 * An outcome before it is priced and stamped with the provider and model. The adapter states the
 * two vendor facts pricing needs and only it can know:
 * - on `ok`, `answeredCount` — how many pictures the vendor's answer held, readable or not (a
 *   flat price bills what the vendor made, not what could be read back);
 * - on `refused` / `failed`, `billable` — true when the vendor may have charged although no
 *   picture came back (a picture made and then withheld, a 2xx with nothing readable, a request
 *   that went out and was never answered); false when it cannot have (nothing was sent, or the
 *   vendor answered with an error before making anything).
 */
export type ImageAdapterResult =
  | (Unpriced<ImageGenerationOk> & { answeredCount: number })
  | (Unpriced<ImageGenerationRefused> & { billable: boolean })
  | (Unpriced<ImageGenerationFailed> & { billable: boolean });

/**
 * The wire, as a seam. The body is described (JSON, or named multipart parts) rather than
 * pre-encoded, so a transport double can read exactly what would have been sent.
 */
export interface ImageTransport {
  /**
   * POST the request and hand back the status and the parsed JSON body — for ANY status; a vendor
   * error is an answer, not an exception. Rejects only when no answer arrived: the connection
   * failed, or `signal` fired (then with the signal's reason).
   */
  post(request: ImageTransportRequest): Promise<ImageTransportResponse>;
}

export type ImageTransportRequest = {
  url: string;
  /** May carry a credential — a transport never logs headers. */
  headers: Record<string, string>;
  body: ImageTransportBody;
  signal?: AbortSignal;
  /** The response header the vendor names its request id in. Default `x-request-id`. */
  requestIdHeader?: string;
};

export type ImageTransportBody =
  | { kind: 'json'; json: Record<string, unknown> }
  | { kind: 'multipart'; parts: ImageTransportPart[] };

export type ImageTransportPart =
  | { name: string; value: string }
  | { name: string; bytes: Uint8Array; mimeType: string; filename: string };

export type ImageTransportResponse = {
  status: number;
  /** The parsed JSON body; `undefined` when the body was not JSON. */
  json: unknown;
  /** The vendor's request id header, when it sent one. */
  requestId?: string;
};
