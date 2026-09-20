import type { ImageGenerationRequest } from './ImageGenerationRequest';
import type {
  GeneratedImage,
  ImageGenerationFailed,
  ImageGenerationRefused,
  ImageUsage,
} from './ImageGenerationOutcome';

/**
 * One vendor's way of making pictures. An adapter turns the vendor-neutral ask into that vendor's
 * request, sends it through the transport it is handed, and reads the answer back into the three
 * outcome kinds. It never prices anything (`ImageGenerator` does, from the usage the adapter
 * reports) and it never opens a connection of its own — so a test, or CI, never calls a vendor.
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

/** An outcome before it is priced and stamped with the provider and model. */
export type ImageAdapterResult =
  | { kind: 'ok'; images: GeneratedImage[]; usage?: ImageUsage; vendorRequestId?: string }
  | Omit<ImageGenerationRefused, 'provider' | 'model' | 'latencyMs'>
  | Omit<ImageGenerationFailed, 'provider' | 'model' | 'latencyMs'>;

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
