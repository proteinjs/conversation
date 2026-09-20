import type { ModelApiCost } from '../../src/UsageData';
import type {
  ImageTransport,
  ImageTransportRequest,
  ImageTransportResponse,
} from '../../src/image/ImageProviderAdapter';

/**
 * Recorded and documented OpenAI Images API bodies for the picture suites — no suite here calls a
 * vendor.
 *
 * RECORDED bodies are real answers captured on 2026-09-16 with `gpt-image-2.5-sunburst`, kept
 * field for field except: the picture bytes are replaced by a 1×1 PNG, and the vendor's ids
 * (`x-request-id`, `generation_id`) are replaced by made-up ones.
 *
 * DOCUMENTED bodies are written from the vendor's pages as read on 2026-09-20:
 * - https://developers.openai.com/api/docs/guides/image-generation — `moderation_blocked` with its
 *   `moderation_details` (`moderation_stage`, `categories`);
 * - https://developers.openai.com/api/reference/resources/images — the answer's shape;
 * - https://developers.openai.com/api/docs/guides/error-codes — 429 "Rate limit reached for
 *   requests", 429 `credit_balance_exhausted` (type `insufficient_quota`), 503
 *   `service_unavailable_error` / `server_is_overloaded`, 401 "Invalid Authentication".
 */

/** A 1×1 transparent PNG. */
export const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
export const TINY_PNG_BYTES = Buffer.from(TINY_PNG_BASE64, 'base64');

/** Fixture rates shaped like a by-the-token picture row (text in, picture in, pictures out). */
export const FIXTURE_TOKEN_PRICED_ROW: ModelApiCost = {
  inputUsdPer1M: 5,
  cachedInputUsdPer1M: 1.25,
  outputUsdPer1M: 0,
  imageInputUsdPer1M: 8,
  cachedImageInputUsdPer1M: 2,
  imageOutputUsdPer1M: 30,
};

/** RECORDED — `POST /images/generations`, 1024×1024, quality `high`, one picture. */
export const recordedGeneration = (): ImageTransportResponse => ({
  status: 200,
  requestId: 'req_fixture_generation',
  json: {
    created: 1789609371,
    background: 'opaque',
    data: [{ b64_json: TINY_PNG_BASE64, generation_id: '00000000-0000-4000-8000-000000000001' }],
    output_format: 'png',
    quality: 'high',
    size: '1024x1024',
    usage: {
      input_tokens: 28,
      input_tokens_details: { image_tokens: 0, text_tokens: 28 },
      output_tokens: 1756,
      output_tokens_details: { image_tokens: 1756, text_tokens: 0 },
      total_tokens: 1784,
    },
  },
});

/** RECORDED — `POST /images/edits` with 16 reference pictures, 1536×1024, quality `high`. */
export const recordedEdit = (): ImageTransportResponse => ({
  status: 200,
  requestId: 'req_fixture_edit',
  json: {
    created: 1789609867,
    background: 'opaque',
    data: [{ b64_json: TINY_PNG_BASE64, generation_id: '00000000-0000-4000-8000-000000000002' }],
    output_format: 'png',
    quality: 'high',
    size: '1536x1024',
    usage: {
      input_tokens: 11792,
      input_tokens_details: { image_tokens: 11552, text_tokens: 240 },
      output_tokens: 1372,
      output_tokens_details: { image_tokens: 1372, text_tokens: 0 },
      total_tokens: 13164,
    },
  },
});

/** RECORDED — the 2.5 model's answer to an edit that carried `input_fidelity`. */
export const recordedInputFidelityRejection = (): ImageTransportResponse => ({
  status: 400,
  requestId: 'req_fixture_input_fidelity',
  json: {
    error: {
      message: "The model 'gpt-image-2.5-sunburst' does not support the 'input_fidelity' parameter.",
      type: 'image_generation_user_error',
      param: 'input_fidelity',
      code: 'invalid_input_fidelity_model',
    },
  },
});

/** DOCUMENTED — the vendor's moderation declining an ask. */
export const documentedModerationBlocked = (): ImageTransportResponse => ({
  status: 400,
  requestId: 'req_fixture_moderation',
  json: {
    error: {
      type: 'image_generation_user_error',
      code: 'moderation_blocked',
      moderation_details: { moderation_stage: 'input', categories: ['harassment'] },
    },
  },
});

/** DOCUMENTED — the vendor's moderation withholding a picture it had already made. */
export const documentedModerationBlockedAtOutput = (): ImageTransportResponse => ({
  status: 400,
  requestId: 'req_fixture_moderation_output',
  json: {
    error: {
      type: 'image_generation_user_error',
      code: 'moderation_blocked',
      moderation_details: { moderation_stage: 'output', categories: ['violence'] },
    },
  },
});

/** DOCUMENTED — an ordinary rate limit. */
export const documentedRateLimit = (): ImageTransportResponse => ({
  status: 429,
  json: { error: { message: 'Rate limit reached for requests' } },
});

/** DOCUMENTED — an empty account, which ALSO answers 429. */
export const documentedCreditExhausted = (): ImageTransportResponse => ({
  status: 429,
  json: {
    error: { message: 'Credit balance exhausted', type: 'insufficient_quota', code: 'credit_balance_exhausted' },
  },
});

/** DOCUMENTED — the vendor overloaded. */
export const documentedOverloaded = (): ImageTransportResponse => ({
  status: 503,
  json: {
    error: {
      message: 'Model temporarily overloaded',
      type: 'service_unavailable_error',
      code: 'server_is_overloaded',
    },
  },
});

/** DOCUMENTED — a credential the vendor does not accept. */
export const documentedInvalidAuthentication = (): ImageTransportResponse => ({
  status: 401,
  json: { error: { message: 'Invalid Authentication' } },
});

/**
 * The transport double: answers every request with `answer` and keeps what it was sent. With
 * `holdUntilStopped`, it never answers — it rejects with the signal's reason when the request's
 * signal fires, the way `fetch` does.
 */
export class RecordingImageTransport implements ImageTransport {
  readonly requests: ImageTransportRequest[] = [];

  constructor(
    private readonly answer: () => ImageTransportResponse,
    private readonly options: { holdUntilStopped?: boolean } = {}
  ) {}

  async post(request: ImageTransportRequest): Promise<ImageTransportResponse> {
    this.requests.push(request);
    if (!this.options.holdUntilStopped) {
      return this.answer();
    }
    return new Promise<ImageTransportResponse>((_resolve, reject) => {
      request.signal?.addEventListener('abort', () => reject(request.signal?.reason), { once: true });
    });
  }

  /** The one request sent, or a failure naming how many there were. */
  only(): ImageTransportRequest {
    if (this.requests.length !== 1) {
      throw new Error(`expected exactly one request on the wire, saw ${this.requests.length}`);
    }
    return this.requests[0];
  }
}
