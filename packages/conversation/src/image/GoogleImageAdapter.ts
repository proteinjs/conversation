import { classifyProviderBillingError } from '../ProviderBillingError';
import type { ImageGenerationRequest, ImageInput } from './ImageGenerationRequest';
import type { GeneratedImage, ImageUsage } from './ImageGenerationOutcome';
import type {
  ImageAdapterContext,
  ImageAdapterResult,
  ImageProviderAdapter,
  ImageTransportRequest,
  ImageTransportResponse,
} from './ImageProviderAdapter';

export type GoogleImageAdapterParams = {
  /** Defaults to the `GOOGLE_GENERATIVE_AI_API_KEY` environment variable, read at call time. Never logged. */
  apiKey?: string;
  /** Defaults to `https://generativelanguage.googleapis.com/v1beta`. */
  baseUrl?: string;
};

type AdapterFailure = Extract<ImageAdapterResult, { kind: 'failed' }>;
type AdapterOk = Extract<ImageAdapterResult, { kind: 'ok' }>;

/** The vendor's output sizes, keyed by the long edge an ask names. */
const IMAGE_SIZES: ReadonlyArray<{ maxEdge: number; size: string }> = [
  { maxEdge: 512, size: '512px' },
  { maxEdge: 1024, size: '1K' },
  { maxEdge: 2048, size: '2K' },
  { maxEdge: 4096, size: '4K' },
];

/** The vendor's aspect ratios, as `[width, height]`. */
const ASPECT_RATIOS: ReadonlyArray<[number, number]> = [
  [1, 1],
  [3, 2],
  [2, 3],
  [3, 4],
  [4, 3],
  [4, 5],
  [5, 4],
  [9, 16],
  [16, 9],
  [21, 9],
];

/**
 * Google's Gemini image models through the Interactions API, over REST. Written against
 * https://ai.google.dev/gemini-api/docs/image-generation as read on 2026-09-22: `POST
 * /v1beta/interactions` with `x-goog-api-key`, a body of `{ model, input: [{ type: 'text', text }
 * | { type: 'image', mime_type, data }], response_format: { type: 'image', mime_type,
 * aspect_ratio, image_size }, previous_interaction_id }`; the picture comes back under
 * `interaction.output_image { mime_type, data }` (or in `interaction.steps[].content[]`), the
 * handle for a follow-up under `interaction.id`, and the count of what was used under `usage`
 * (`total_input_tokens`, `input_tokens_by_modality`, `total_output_tokens`,
 * `output_tokens_by_modality`, `total_thought_tokens`). Sizes `512px | 1K | 2K | 4K` (Lite: `1K`
 * only); ten aspect ratios; reference pictures ride as `image` inputs and what each is for is said
 * in words, as on OpenAI.
 *
 * Two facts measured on 2026-09-16 govern where the page and the wire disagree:
 * - `gemini-3.1-flash-image` answers HTTP 400 to `mime_type: image/png` ("Supported values:
 *   'image/jpeg'"), so JPEG is always asked for and a PNG in the product is a conversion
 *   downstream; a transparent background is refused here before the wire (the model has no alpha);
 * - the answer carries ONE picture per call, so an ask for `count` pictures is `count` calls in a
 *   row, their usage summed; the stop and the deadline cut the sequence between calls.
 *
 * `billable`: an error answer made nothing (0); a 2xx with nothing readable, or a request never
 * answered, may have been billed. A follow-up edit passes the earlier ask's `continuationId` as
 * `previousInteractionId`, and the vendor holds the earlier picture's geometry.
 */
export class GoogleImageAdapter implements ImageProviderAdapter {
  static readonly MAX_INPUTS = 14;
  static readonly MAX_COUNT = 4;
  static readonly OUTPUT_MIME_TYPE = 'image/jpeg';

  readonly provider = 'google';
  private readonly apiKey?: string;
  private readonly baseUrl: string;

  constructor(params: GoogleImageAdapterParams = {}) {
    this.apiKey = params.apiKey;
    this.baseUrl = (params.baseUrl ?? 'https://generativelanguage.googleapis.com/v1beta').replace(/\/+$/, '');
  }

  async generate(request: ImageGenerationRequest, context: ImageAdapterContext): Promise<ImageAdapterResult> {
    const invalid = this.validate(request);
    if (invalid) {
      return invalid;
    }
    const apiKey = this.apiKey ?? process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    if (!apiKey) {
      return this.notSent('auth', 'No Google API key: pass `apiKey` or set GOOGLE_GENERATIVE_AI_API_KEY.');
    }

    const count = request.count ?? 1;
    let made: AdapterOk | undefined;
    for (let index = 0; index < count; index++) {
      if (context.signal?.aborted) {
        // The stop between two pictures: what was made so far is the answer, and the caller's
        // stop wins over it in the generator.
        break;
      }
      let response: ImageTransportResponse;
      try {
        response = await context.transport.post(this.buildRequest(request, apiKey, context.signal));
      } catch (error) {
        if (context.signal?.aborted) {
          throw error;
        }
        if (made) {
          return made;
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
      const result =
        response.status >= 200 && response.status < 300 ? this.readAnswer(response) : this.readError(response);
      if (result.kind !== 'ok') {
        // A picture already made stands; the failure of a later one is not its failure.
        return made ?? result;
      }
      made = made ? this.merge(made, result) : result;
    }
    return made ?? this.notSent('invalid_request', 'Nothing was asked for.');
  }

  private validate(request: ImageGenerationRequest): AdapterFailure | undefined {
    if ((request.operation ?? 'generate') !== 'generate') {
      return this.notSent('invalid_request', `Google offers no "${request.operation}" utility.`);
    }
    if (!request.prompt.trim()) {
      return this.notSent('invalid_request', 'The prompt is empty.');
    }
    const count = request.count ?? 1;
    if (!Number.isInteger(count) || count < 1 || count > GoogleImageAdapter.MAX_COUNT) {
      return this.notSent(
        'invalid_request',
        `count must be a whole number from 1 to ${GoogleImageAdapter.MAX_COUNT}; got ${count}.`
      );
    }
    const inputs = request.inputs?.length ?? 0;
    if (inputs > GoogleImageAdapter.MAX_INPUTS) {
      return this.notSent(
        'invalid_request',
        `At most ${GoogleImageAdapter.MAX_INPUTS} reference pictures can be sent; got ${inputs}.`
      );
    }
    if (request.background === 'transparent') {
      return this.notSent('invalid_request', 'Google image models cannot make a transparent background.');
    }
    const badSize = this.sizeProblem(request.size);
    return badSize ? this.notSent('invalid_request', badSize) : undefined;
  }

  private sizeProblem(size: string | undefined): string | undefined {
    if (size === undefined || size === 'auto') {
      return undefined;
    }
    return /^(\d+)x(\d+)$/.test(size) ? undefined : `size must be WIDTHxHEIGHT or auto; got "${size}".`;
  }

  private buildRequest(
    request: ImageGenerationRequest,
    apiKey: string,
    signal: AbortSignal | undefined
  ): ImageTransportRequest {
    const inputs = request.inputs ?? [];
    const input: Record<string, unknown>[] = [
      { type: 'text', text: this.promptWithReferences(request.prompt, inputs) },
    ];
    for (const reference of inputs) {
      input.push({
        type: 'image',
        mime_type: reference.mimeType,
        data: Buffer.from(reference.bytes).toString('base64'),
      });
    }
    const responseFormat: Record<string, unknown> = { type: 'image', mime_type: GoogleImageAdapter.OUTPUT_MIME_TYPE };
    const shape = this.shapeOf(request.size);
    if (shape) {
      responseFormat.aspect_ratio = shape.aspectRatio;
      responseFormat.image_size = shape.imageSize;
    }
    const json: Record<string, unknown> = { model: request.model, input, response_format: responseFormat };
    if (request.previousInteractionId) {
      json.previous_interaction_id = request.previousInteractionId;
    }
    return {
      url: `${this.baseUrl}/interactions`,
      headers: { 'x-goog-api-key': apiKey },
      body: { kind: 'json', json },
      signal,
    };
  }

  /** The vendor's nearest aspect ratio and the size rung of the long edge, from `WIDTHxHEIGHT`. */
  private shapeOf(size: string | undefined): { aspectRatio: string; imageSize: string } | undefined {
    const match = size ? /^(\d+)x(\d+)$/.exec(size) : undefined;
    if (!match) {
      return undefined;
    }
    const [width, height] = [Number(match[1]), Number(match[2])];
    const wanted = width / height;
    let best = ASPECT_RATIOS[0];
    for (const ratio of ASPECT_RATIOS) {
      if (Math.abs(ratio[0] / ratio[1] - wanted) < Math.abs(best[0] / best[1] - wanted)) {
        best = ratio;
      }
    }
    const longEdge = Math.max(width, height);
    const rung = IMAGE_SIZES.find((entry) => longEdge <= entry.maxEdge) ?? IMAGE_SIZES[IMAGE_SIZES.length - 1];
    return { aspectRatio: `${best[0]}:${best[1]}`, imageSize: rung.size };
  }

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
          : 'The layout - keep its geometry, walls, windows and light exactly.';
      case 'style':
        return 'A style reference - borrow only its look (colour, light, texture); do not copy what it shows.';
      default:
        return 'A reference.';
    }
  }

  private readAnswer(response: ImageTransportResponse): AdapterOk | AdapterFailure {
    const body = this.asRecord(response.json);
    const interaction = this.asRecord(body?.interaction) ?? body;
    const picture = this.findPicture(interaction);
    const usage = this.readUsage(body?.usage ?? interaction?.usage);
    const reported = {
      ...(usage ? { usage } : {}),
      ...(response.requestId ? { vendorRequestId: response.requestId } : {}),
    };
    if (!picture) {
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
    const continuationId = typeof interaction?.id === 'string' ? interaction.id : undefined;
    return {
      kind: 'ok',
      sent: true,
      images: [picture],
      answeredCount: 1,
      ...(continuationId ? { continuationId } : {}),
      ...reported,
    };
  }

  /** `output_image` first; else the first image part of any step. */
  private findPicture(interaction: Record<string, unknown> | undefined): GeneratedImage | undefined {
    const direct = this.asRecord(interaction?.output_image);
    const fromPart = (part: Record<string, unknown> | undefined): GeneratedImage | undefined => {
      if (typeof part?.data !== 'string' || part.data.length === 0) {
        return undefined;
      }
      const bytes = Buffer.from(part.data, 'base64');
      const mimeType = typeof part.mime_type === 'string' ? part.mime_type : GoogleImageAdapter.OUTPUT_MIME_TYPE;
      return bytes.length > 0 ? { bytes, mimeType } : undefined;
    };
    const fromOutputImage = fromPart(direct);
    if (fromOutputImage) {
      return fromOutputImage;
    }
    const steps: unknown[] = Array.isArray(interaction?.steps) ? (interaction?.steps as unknown[]) : [];
    for (const step of steps) {
      const content: unknown[] = Array.isArray(this.asRecord(step)?.content)
        ? (this.asRecord(step)?.content as unknown[])
        : [];
      for (const part of content) {
        const record = this.asRecord(part);
        if (record?.type === 'image') {
          const found = fromPart(record);
          if (found) {
            return found;
          }
        }
      }
    }
    return undefined;
  }

  /**
   * The vendor's `usage`: totals by modality. Text output = every output token that is not a
   * picture (the model's own words and its thinking are billed as text). A total with no
   * by-modality split is NOT turned into one — the image count stays missing and the ask goes
   * unpriced.
   */
  private readUsage(value: unknown): ImageUsage | undefined {
    const usage = this.asRecord(value);
    if (!usage) {
      return undefined;
    }
    const byModality = (list: unknown, modality: string): number | undefined => {
      if (!Array.isArray(list)) {
        return undefined;
      }
      const rows = list.map((row) => this.asRecord(row)).filter((row): row is Record<string, unknown> => !!row);
      const matching = rows.filter((row) => String(row.modality ?? '').toLowerCase() === modality);
      return matching.length > 0 ? matching.reduce((sum, row) => sum + (this.asCount(row.tokens) ?? 0), 0) : undefined;
    };
    const totalInput = this.asCount(usage.total_input_tokens);
    const totalOutput = this.asCount(usage.total_output_tokens);
    const imageInput = byModality(usage.input_tokens_by_modality, 'image');
    const textInput = byModality(usage.input_tokens_by_modality, 'text');
    const imageOutput = byModality(usage.output_tokens_by_modality, 'image');
    const read: ImageUsage = {
      textInputTokens:
        textInput ?? (totalInput !== undefined ? Math.max(0, totalInput - (imageInput ?? 0)) : undefined),
      imageInputTokens: imageInput ?? (totalInput !== undefined ? 0 : undefined),
      imageOutputTokens: imageOutput,
      textOutputTokens:
        imageOutput !== undefined && totalOutput !== undefined ? Math.max(0, totalOutput - imageOutput) : undefined,
      totalTokens: totalInput !== undefined && totalOutput !== undefined ? totalInput + totalOutput : undefined,
    };
    const reported = Object.entries(read).filter(([, count]) => count !== undefined);
    return reported.length > 0 ? (Object.fromEntries(reported) as ImageUsage) : undefined;
  }

  private merge(first: AdapterOk, next: AdapterOk): AdapterOk {
    const sum = (a?: number, b?: number) => (a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0));
    const usage: ImageUsage | undefined =
      first.usage || next.usage
        ? {
            textInputTokens: sum(first.usage?.textInputTokens, next.usage?.textInputTokens),
            imageInputTokens: sum(first.usage?.imageInputTokens, next.usage?.imageInputTokens),
            textOutputTokens: sum(first.usage?.textOutputTokens, next.usage?.textOutputTokens),
            imageOutputTokens: sum(first.usage?.imageOutputTokens, next.usage?.imageOutputTokens),
            totalTokens: sum(first.usage?.totalTokens, next.usage?.totalTokens),
          }
        : undefined;
    const clean = usage
      ? (Object.fromEntries(Object.entries(usage).filter(([, count]) => count !== undefined)) as ImageUsage)
      : undefined;
    return {
      ...first,
      images: [...first.images, ...next.images],
      answeredCount: first.answeredCount + next.answeredCount,
      ...(clean ? { usage: clean } : {}),
      ...(next.continuationId ? { continuationId: next.continuationId } : {}),
    };
  }

  private readError(response: ImageTransportResponse): ImageAdapterResult {
    const error = this.asRecord(this.asRecord(response.json)?.error);
    const status = typeof error?.status === 'string' ? error.status : undefined;
    const message = typeof error?.message === 'string' ? error.message : `HTTP ${response.status}`;
    const vendorRequestId = response.requestId ? { vendorRequestId: response.requestId } : {};
    // The vendor's safety block is reported as a reason on a 400 with the words SAFETY / blocked.
    const blocked = /SAFETY|PROHIBITED_CONTENT|blocked/i.test(`${status ?? ''} ${message}`);
    if (blocked && response.status < 500) {
      return {
        kind: 'refused',
        sent: true,
        billable: false,
        reason: status ?? 'blocked',
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
      ...(status ? { code: status } : {}),
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

  private asCount(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
  }
}
