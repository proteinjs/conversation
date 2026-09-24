import { classifyProviderBillingError } from '../ProviderBillingError';
import type { ImageGenerationRequest } from './ImageGenerationRequest';
import type { GeneratedImage } from './ImageGenerationOutcome';
import type {
  ImageAdapterContext,
  ImageAdapterResult,
  ImageProviderAdapter,
  ImageTransportPart,
  ImageTransportRequest,
  ImageTransportResponse,
} from './ImageProviderAdapter';

export type RecraftImageAdapterParams = {
  /** Defaults to the `RECRAFT_API_KEY` environment variable, read at call time. Never logged. */
  apiKey?: string;
  /** Defaults to `https://external.api.recraft.ai/v1`. */
  baseUrl?: string;
};

type AdapterFailure = Extract<ImageAdapterResult, { kind: 'failed' }>;

export const RECRAFT_SVG_MIME_TYPE = 'image/svg+xml';

/**
 * Recraft's REST API — the vector vendor. Written against
 * https://www.recraft.ai/docs/api-reference/endpoints as read on 2026-09-24, and checked on the
 * wire the same day: base `https://external.api.recraft.ai/v1`, `Authorization: Bearer`;
 * `POST /images/generations` takes any model (`prompt`, `model`, `size` as `WxH`, `n` 1–6,
 * `response_format: b64_json`) and answers `{ data: [{ b64_json, image_id }], created, credits }`
 * — a `_vector` model's `b64_json` is SVG text; `POST /images/generations/vector` is the same door
 * with anything not vector refused by the server (400 `invalid_image_type`, nothing billed), so a
 * vector model is always sent through it. A raster answer is lossless WebP unless `image_format`
 * says `png`, so a raster is always asked in the format it will be labelled with (PNG or WEBP; the
 * vendor offers no JPEG). `POST /images/vectorize` and `POST /images/removeBackground` take one
 * picture as multipart `file` (PNG, JPG, WEBP under 10 MB, at most 16 MP and 4096 px an edge, at
 * least 256 px) with `response_format: b64_json` and answer `{ image: { b64_json } }`. The vendor
 * names its request id in `x-recraft-requestid`. Past five requests in a second it answers 429
 * `rate_limit_exceeded` with no Retry-After and bills nothing.
 *
 * Prices are flat per picture (`ModelApiCost.perImageUsd` on the catalog row), so no usage object
 * is read: `answeredCount` prices what the vendor answered. `store_info_for_deep_exploration` is
 * never sent — nothing of an ask is to be kept on the vendor's side past delivery (its Developer
 * Terms). `billable`: an error answer made nothing; a 2xx with nothing readable, or a request
 * never answered, may have been billed.
 */
export class RecraftImageAdapter implements ImageProviderAdapter {
  static readonly MAX_COUNT = 6;
  /** A utility's input must be under 10 MB. */
  static readonly MAX_UTILITY_INPUT_BYTES = 10 * 1024 * 1024;
  static readonly VECTOR_MODEL = /_vector$/;
  /** The raster formats the vendor answers in (`image_format`); it offers no JPEG. */
  static readonly RASTER_FORMATS: ReadonlySet<string> = new Set(['png', 'webp']);
  static readonly REQUEST_ID_HEADER = 'x-recraft-requestid';

  readonly provider = 'recraft';
  private readonly apiKey?: string;
  private readonly baseUrl: string;

  constructor(params: RecraftImageAdapterParams = {}) {
    this.apiKey = params.apiKey;
    this.baseUrl = (params.baseUrl ?? 'https://external.api.recraft.ai/v1').replace(/\/+$/, '');
  }

  /** True when the vendor's model id says its answer is SVG path data. */
  static isVectorModel(model: string): boolean {
    return RecraftImageAdapter.VECTOR_MODEL.test(model);
  }

  async generate(request: ImageGenerationRequest, context: ImageAdapterContext): Promise<ImageAdapterResult> {
    const invalid = this.validate(request);
    if (invalid) {
      return invalid;
    }
    const apiKey = this.apiKey ?? process.env.RECRAFT_API_KEY;
    if (!apiKey) {
      return this.notSent('auth', 'No Recraft API key: pass `apiKey` or set RECRAFT_API_KEY.');
    }
    let response: ImageTransportResponse;
    try {
      response = await context.transport.post(this.buildRequest(request, apiKey, context.signal));
    } catch (error) {
      if (context.signal?.aborted) {
        throw error;
      }
      return {
        kind: 'failed',
        errorKind: 'network',
        transient: true,
        sent: true,
        billable: true,
        message: error instanceof Error ? error.message : String(error),
      };
    }
    return response.status >= 200 && response.status < 300
      ? this.readAnswer(request, response)
      : this.readError(response);
  }

  private validate(request: ImageGenerationRequest): AdapterFailure | undefined {
    const operation = request.operation ?? 'generate';
    if (operation === 'generate') {
      if (!request.prompt.trim()) {
        return this.notSent('invalid_request', 'The prompt is empty.');
      }
      const count = request.count ?? 1;
      if (!Number.isInteger(count) || count < 1 || count > RecraftImageAdapter.MAX_COUNT) {
        return this.notSent(
          'invalid_request',
          `count must be a whole number from 1 to ${RecraftImageAdapter.MAX_COUNT}; got ${count}.`
        );
      }
      if ((request.inputs?.length ?? 0) > 0) {
        return this.notSent(
          'invalid_request',
          'Recraft generation takes no reference pictures in this version (a look from references is a style, not an input).'
        );
      }
      if (request.size !== undefined && request.size !== 'auto' && !/^\d+x\d+$/.test(request.size)) {
        return this.notSent('invalid_request', `size must be WIDTHxHEIGHT or auto; got "${request.size}".`);
      }
      return RecraftImageAdapter.isVectorModel(request.model) ? undefined : this.rasterFormatInvalid(request);
    }
    const inputs = request.inputs ?? [];
    if (inputs.length !== 1) {
      return this.notSent('invalid_request', `"${operation}" takes exactly one picture; got ${inputs.length}.`);
    }
    if (inputs[0].bytes.length >= RecraftImageAdapter.MAX_UTILITY_INPUT_BYTES) {
      return this.notSent('invalid_request', 'The picture is 10 MB or more; it must be smaller.');
    }
    if (!/^image\/(png|jpeg|webp)$/.test(inputs[0].mimeType)) {
      return this.notSent('invalid_request', `"${operation}" takes a PNG, JPEG or WEBP; got ${inputs[0].mimeType}.`);
    }
    return operation === 'remove-background' ? this.rasterFormatInvalid(request) : undefined;
  }

  private rasterFormatInvalid(request: ImageGenerationRequest): AdapterFailure | undefined {
    const format = this.rasterFormat(request);
    return RecraftImageAdapter.RASTER_FORMATS.has(format)
      ? undefined
      : this.notSent('invalid_request', `Recraft answers a raster as PNG or WEBP; got ${format}.`);
  }

  /** The raster format an answer is asked in, and so labelled with. */
  private rasterFormat(request: ImageGenerationRequest): string {
    return request.outputFormat ?? 'png';
  }

  private buildRequest(
    request: ImageGenerationRequest,
    apiKey: string,
    signal: AbortSignal | undefined
  ): ImageTransportRequest {
    const headers = { Authorization: `Bearer ${apiKey}` };
    const requestIdHeader = RecraftImageAdapter.REQUEST_ID_HEADER;
    const operation = request.operation ?? 'generate';
    if (operation === 'generate') {
      const vector = RecraftImageAdapter.isVectorModel(request.model);
      const json: Record<string, unknown> = {
        prompt: request.prompt,
        model: request.model,
        n: request.count ?? 1,
        response_format: 'b64_json',
      };
      if (request.size && request.size !== 'auto') {
        json.size = request.size;
      }
      if (!vector) {
        json.image_format = this.rasterFormat(request);
      }
      const door = vector ? 'images/generations/vector' : 'images/generations';
      return { url: `${this.baseUrl}/${door}`, headers, body: { kind: 'json', json }, signal, requestIdHeader };
    }
    const input = (request.inputs ?? [])[0];
    const door = operation === 'vectorize' ? 'images/vectorize' : 'images/removeBackground';
    const format: ImageTransportPart[] =
      operation === 'remove-background' ? [{ name: 'image_format', value: this.rasterFormat(request) }] : [];
    return {
      url: `${this.baseUrl}/${door}`,
      headers,
      requestIdHeader,
      body: {
        kind: 'multipart',
        parts: [
          { name: 'response_format', value: 'b64_json' },
          ...format,
          {
            name: 'file',
            bytes: input.bytes,
            mimeType: input.mimeType,
            filename:
              input.name ?? `picture.${input.mimeType.split('/')[1] === 'jpeg' ? 'jpg' : input.mimeType.split('/')[1]}`,
          },
        ],
      },
      signal,
    };
  }

  private readAnswer(request: ImageGenerationRequest, response: ImageTransportResponse): ImageAdapterResult {
    const body = this.asRecord(response.json);
    const operation = request.operation ?? 'generate';
    const entries: unknown[] =
      operation === 'generate'
        ? Array.isArray(body?.data)
          ? (body?.data as unknown[])
          : []
        : body?.image !== undefined
          ? [body.image]
          : [];
    const svg = operation === 'vectorize' || RecraftImageAdapter.isVectorModel(request.model);
    const mimeType = svg ? RECRAFT_SVG_MIME_TYPE : `image/${this.rasterFormat(request)}`;
    const images: GeneratedImage[] = [];
    for (const entry of entries) {
      const record = this.asRecord(entry);
      const bytes = typeof record?.b64_json === 'string' ? Buffer.from(record.b64_json, 'base64') : undefined;
      if (bytes && bytes.length > 0) {
        images.push({ bytes, mimeType });
      }
    }
    const reported = response.requestId ? { vendorRequestId: response.requestId } : {};
    if (images.length === 0) {
      return {
        kind: 'failed',
        errorKind: 'malformed_response',
        transient: false,
        sent: true,
        billable: true,
        statusCode: response.status,
        message: 'The vendor answered without a readable picture.',
        ...reported,
      };
    }
    return { kind: 'ok', sent: true, images, answeredCount: entries.length, ...reported };
  }

  private readError(response: ImageTransportResponse): ImageAdapterResult {
    const body = this.asRecord(response.json);
    const code = typeof body?.code === 'string' ? body.code : undefined;
    const message = typeof body?.message === 'string' ? body.message : `HTTP ${response.status}`;
    const vendorRequestId = response.requestId ? { vendorRequestId: response.requestId } : {};
    if (/moderat|safety|nsfw|prohibited/i.test(`${code ?? ''} ${message}`) && response.status < 500) {
      return {
        kind: 'refused',
        sent: true,
        billable: false,
        reason: code ?? 'moderation',
        stage: 'unknown',
        message,
        ...vendorRequestId,
      };
    }
    const failure = (errorKind: AdapterFailure['errorKind'], transient: boolean): AdapterFailure => ({
      kind: 'failed',
      errorKind,
      transient,
      sent: true,
      billable: false,
      statusCode: response.status,
      ...(code ? { code } : {}),
      message,
      ...vendorRequestId,
    });
    if (classifyProviderBillingError({ statusCode: response.status, data: response.json })) {
      return failure('billing', false);
    }
    if (response.status === 401 || response.status === 403) {
      return failure('auth', false);
    }
    if (response.status === 429) {
      return failure('rate_limited', true);
    }
    if (response.status === 408) {
      return failure('timeout', true);
    }
    if (response.status >= 500) {
      return failure('provider_error', true);
    }
    return failure('invalid_request', false);
  }

  private notSent(errorKind: AdapterFailure['errorKind'], message: string): AdapterFailure {
    return { kind: 'failed', errorKind, transient: false, sent: false, billable: false, message };
  }

  private asRecord(value: unknown): Record<string, unknown> | undefined {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  }
}
