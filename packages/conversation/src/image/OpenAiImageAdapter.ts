import { classifyProviderBillingError } from '../ProviderBillingError';
import type { ImageGenerationRequest, ImageInput, ImageOutputFormat } from './ImageGenerationRequest';
import type { GeneratedImage, ImageUsage } from './ImageGenerationOutcome';
import type {
  ImageAdapterContext,
  ImageAdapterResult,
  ImageProviderAdapter,
  ImageTransportPart,
  ImageTransportRequest,
  ImageTransportResponse,
} from './ImageProviderAdapter';

export type OpenAiImageAdapterParams = {
  /** Defaults to the `OPENAI_API_KEY` environment variable, read at call time. Never logged. */
  apiKey?: string;
  /** Defaults to `https://api.openai.com/v1`. */
  baseUrl?: string;
};

type AdapterFailure = Extract<ImageAdapterResult, { kind: 'failed' }>;

/**
 * OpenAI's Images API over REST, for the GPT Image models. Written against the vendor's pages as
 * read on 2026-09-20:
 * - https://developers.openai.com/api/docs/guides/image-generation — the parameters, a transparent
 *   background only with png or webp, `moderation_blocked` and its `moderation_details`, "up to 2
 *   minutes" for a complex prompt, masking described as prompt-based guidance;
 * - https://developers.openai.com/api/reference/resources/images — `POST /images/generations`
 *   (`n` between 1 and 10), `POST /images/edits` (multipart `image[]`, "up to 16 images"), and the
 *   answer `{ created, data: [{ b64_json, revised_prompt }], output_format, usage }`;
 * - https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst — the two endpoints the
 *   model serves.
 *
 * An ask with no reference pictures goes to `/images/generations` as JSON; an ask with reference
 * pictures goes to `/images/edits` as multipart. Two things are deliberately never sent:
 * - `input_fidelity` — the 2.5 models reject the parameter outright (HTTP 400
 *   `invalid_input_fidelity_model`, recorded 2026-09-16), so what each reference is for, and how
 *   closely to hold to it, is said in words at the end of the prompt instead;
 * - `mask` — not part of this version.
 */
export class OpenAiImageAdapter implements ImageProviderAdapter {
  static readonly MAX_INPUTS = 16;
  static readonly MAX_COUNT = 10;

  readonly provider = 'openai';
  private readonly apiKey?: string;
  private readonly baseUrl: string;

  constructor(params: OpenAiImageAdapterParams = {}) {
    this.apiKey = params.apiKey;
    this.baseUrl = (params.baseUrl ?? 'https://api.openai.com/v1').replace(/\/+$/, '');
  }

  async generate(request: ImageGenerationRequest, context: ImageAdapterContext): Promise<ImageAdapterResult> {
    const invalid = this.validate(request);
    if (invalid) {
      return invalid;
    }
    const apiKey = this.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return this.notSent('auth', 'No OpenAI API key: pass `apiKey` or set OPENAI_API_KEY.');
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
        message: error instanceof Error ? error.message : String(error),
      };
    }
    return response.status >= 200 && response.status < 300
      ? this.readAnswer(response, request.outputFormat ?? 'png')
      : this.readError(response);
  }

  /** The rules this adapter can check without the vendor — a broken ask never costs a round trip. */
  private validate(request: ImageGenerationRequest): AdapterFailure | undefined {
    if (!request.prompt.trim()) {
      return this.notSent('invalid_request', 'The prompt is empty.');
    }
    const count = request.count ?? 1;
    if (!Number.isInteger(count) || count < 1 || count > OpenAiImageAdapter.MAX_COUNT) {
      return this.notSent(
        'invalid_request',
        `count must be a whole number from 1 to ${OpenAiImageAdapter.MAX_COUNT}; got ${count}.`
      );
    }
    const inputs = request.inputs?.length ?? 0;
    if (inputs > OpenAiImageAdapter.MAX_INPUTS) {
      return this.notSent(
        'invalid_request',
        `At most ${OpenAiImageAdapter.MAX_INPUTS} reference pictures can be sent; got ${inputs}.`
      );
    }
    if (request.background === 'transparent' && request.outputFormat === 'jpeg') {
      return this.notSent('invalid_request', 'A transparent background needs png or webp, not jpeg.');
    }
    return undefined;
  }

  private buildRequest(
    request: ImageGenerationRequest,
    apiKey: string,
    signal: AbortSignal | undefined
  ): ImageTransportRequest {
    const headers = { Authorization: `Bearer ${apiKey}` };
    const inputs = request.inputs ?? [];
    const fields: Record<string, string | number | undefined> = {
      model: request.model,
      prompt: this.promptWithReferences(request.prompt, inputs),
      n: request.count ?? 1,
      size: request.size,
      quality: request.quality,
      background: request.background,
      output_format: request.outputFormat ?? 'png',
    };
    if (inputs.length === 0) {
      const json: Record<string, unknown> = {};
      for (const [name, value] of Object.entries(fields)) {
        if (value !== undefined) {
          json[name] = value;
        }
      }
      return { url: `${this.baseUrl}/images/generations`, headers, body: { kind: 'json', json }, signal };
    }
    const parts: ImageTransportPart[] = [];
    for (const [name, value] of Object.entries(fields)) {
      if (value !== undefined) {
        parts.push({ name, value: String(value) });
      }
    }
    inputs.forEach((input, index) => {
      parts.push({
        name: 'image[]',
        bytes: input.bytes,
        mimeType: input.mimeType,
        filename: input.name ?? `reference-${index + 1}.${this.extensionOf(input.mimeType)}`,
      });
    });
    return { url: `${this.baseUrl}/images/edits`, headers, body: { kind: 'multipart', parts }, signal };
  }

  /**
   * What each reference is for, said in words, in the order the pictures are attached — this
   * vendor has no parameter for it. An ask whose references carry no role is sent as written.
   */
  private promptWithReferences(prompt: string, inputs: ImageInput[]): string {
    if (!inputs.some((input) => input.role)) {
      return prompt;
    }
    const lines = inputs.map((input, index) => `${index + 1}. ${this.describeReference(input)}`);
    return `${prompt}\n\nReference pictures, in the order attached:\n${lines.join('\n')}`;
  }

  private describeReference(input: ImageInput): string {
    const loosely = input.fidelity === 'low';
    switch (input.role) {
      case 'subject':
        return loosely
          ? 'The subject - keep it recognisable.'
          : 'The subject - keep it exactly as it is: the same shape, colours and details.';
      case 'structure':
        return loosely
          ? 'The layout - follow its composition loosely.'
          : 'The layout - keep its geometry and composition exactly.';
      case 'style':
        return 'A style reference - borrow only its look (colour, light, texture); do not copy what it shows.';
      default:
        return 'A reference.';
    }
  }

  private readAnswer(response: ImageTransportResponse, requestedFormat: ImageOutputFormat): ImageAdapterResult {
    const body = this.asRecord(response.json);
    const format = typeof body?.output_format === 'string' ? body.output_format : requestedFormat;
    const mimeType = `image/${format}`;
    const images: GeneratedImage[] = [];
    // Each picture stands alone: one unreadable entry never loses the others.
    const entries: unknown[] = Array.isArray(body?.data) ? (body?.data as unknown[]) : [];
    for (const entry of entries) {
      const record = this.asRecord(entry);
      const bytes = typeof record?.b64_json === 'string' ? Buffer.from(record.b64_json, 'base64') : undefined;
      if (bytes && bytes.length > 0) {
        const revisedPrompt = typeof record?.revised_prompt === 'string' ? record.revised_prompt : undefined;
        images.push({ bytes, mimeType, ...(revisedPrompt ? { revisedPrompt } : {}) });
      }
    }
    const vendorRequestId = response.requestId;
    if (images.length === 0) {
      return {
        kind: 'failed',
        errorKind: 'malformed_response',
        transient: false,
        sent: true,
        statusCode: response.status,
        message: 'The vendor answered without a readable picture.',
        ...(vendorRequestId ? { vendorRequestId } : {}),
      };
    }
    const usage = this.readUsage(body?.usage);
    return { kind: 'ok', images, ...(usage ? { usage } : {}), ...(vendorRequestId ? { vendorRequestId } : {}) };
  }

  /**
   * The vendor's `usage` object, as documented: `input_tokens` split into text and image under
   * `input_tokens_details`, and `output_tokens` (pictures; split under `output_tokens_details`
   * when present). It documents no cached split, so none is reported. A total with no split is
   * NOT turned into a split — the missing field stays missing and the ask goes unpriced.
   */
  private readUsage(value: unknown): ImageUsage | undefined {
    const usage = this.asRecord(value);
    if (!usage) {
      return undefined;
    }
    const input = this.asRecord(usage.input_tokens_details);
    const output = this.asRecord(usage.output_tokens_details);
    const read: ImageUsage = {
      textInputTokens: this.asCount(input?.text_tokens),
      imageInputTokens: this.asCount(input?.image_tokens),
      textOutputTokens: this.asCount(output?.text_tokens),
      imageOutputTokens: output ? this.asCount(output.image_tokens) : this.asCount(usage.output_tokens),
      totalTokens: this.asCount(usage.total_tokens),
    };
    const reported = Object.entries(read).filter(([, count]) => count !== undefined);
    return reported.length > 0 ? (Object.fromEntries(reported) as ImageUsage) : undefined;
  }

  private readError(response: ImageTransportResponse): ImageAdapterResult {
    const error = this.asRecord(this.asRecord(response.json)?.error);
    const code = typeof error?.code === 'string' ? error.code : undefined;
    const message = typeof error?.message === 'string' ? error.message : `HTTP ${response.status}`;
    const vendorRequestId = response.requestId ? { vendorRequestId: response.requestId } : {};

    if (code === 'moderation_blocked') {
      const details = this.asRecord(error?.moderation_details);
      const stage = details?.moderation_stage;
      const named: unknown[] = Array.isArray(details?.categories) ? (details?.categories as unknown[]) : [];
      const categories = named.filter((category): category is string => typeof category === 'string');
      return {
        kind: 'refused',
        reason: code,
        ...(stage === 'input' || stage === 'output' || stage === 'unknown' ? { stage } : {}),
        ...(categories.length > 0 ? { categories } : {}),
        ...(typeof error?.message === 'string' ? { message: error.message } : {}),
        ...vendorRequestId,
      };
    }

    const failure = (errorKind: AdapterFailure['errorKind'], transient: boolean): AdapterFailure => ({
      kind: 'failed',
      errorKind,
      transient,
      sent: true,
      statusCode: response.status,
      ...(code ? { code } : {}),
      message,
      ...vendorRequestId,
    });
    // Billing first: an empty account answers 429 and must never read as a rate limit.
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
    return { kind: 'failed', errorKind, transient: false, sent: false, message };
  }

  private extensionOf(mimeType: string): string {
    const subtype = mimeType.split('/')[1] ?? 'png';
    return subtype === 'jpeg' ? 'jpg' : subtype;
  }

  private asRecord(value: unknown): Record<string, unknown> | undefined {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  }

  private asCount(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
  }
}
