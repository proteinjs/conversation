import { Logger, LogLevel } from '@proteinjs/logger';
import type { ModelDataResolver } from '../ModelData';
import { FetchImageTransport } from './FetchImageTransport';
import { ImageCostCalculator } from './ImageCostCalculator';
import type { ImageGenerationRequest } from './ImageGenerationRequest';
import type { ImageGenerationOutcome } from './ImageGenerationOutcome';
import type { ImageAdapterResult, ImageProviderAdapter, ImageTransport } from './ImageProviderAdapter';
import { OpenAiImageAdapter } from './OpenAiImageAdapter';

export type ImageGeneratorParams = {
  /**
   * The rates pictures are priced from — the same resolver the text path takes, and required for
   * the same reason: a generator without pricing data would record every picture as free.
   */
  modelData: ModelDataResolver;
  /** One adapter per provider. Default: the OpenAI adapter. */
  adapters?: ImageProviderAdapter[];
  /** The wire. Default: the runtime's `fetch`. A test hands in a double, so CI never calls a vendor. */
  transport?: ImageTransport;
  /** The longest one ask may take before it is given up as a transient failure. Default 5 minutes. */
  timeoutMs?: number;
  logLevel?: LogLevel;
};

/**
 * Makes pictures: one vendor-neutral ask in, one outcome out — `ok`, `refused` or `failed` — priced
 * from the vendor's own usage where that is possible and left unpriced where it is not.
 *
 * The caller's `AbortSignal` is threaded to the wire. A stop REJECTS with the signal's reason, and
 * a picture that arrived after the stop is dropped, never returned.
 */
export class ImageGenerator {
  static readonly DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

  private readonly adapters: Map<string, ImageProviderAdapter>;
  private readonly transport: ImageTransport;
  private readonly costCalculator: ImageCostCalculator;
  private readonly timeoutMs: number;
  private readonly logger: Logger;

  constructor(params: ImageGeneratorParams) {
    const adapters = params.adapters ?? [new OpenAiImageAdapter()];
    this.adapters = new Map(adapters.map((adapter) => [adapter.provider, adapter]));
    this.transport = params.transport ?? new FetchImageTransport();
    this.costCalculator = new ImageCostCalculator(params.modelData);
    this.timeoutMs = params.timeoutMs ?? ImageGenerator.DEFAULT_TIMEOUT_MS;
    this.logger = new Logger({ name: 'ImageGenerator', logLevel: params.logLevel });
  }

  async generate(request: ImageGenerationRequest): Promise<ImageGenerationOutcome> {
    const adapter = this.adapters.get(request.provider);
    if (!adapter) {
      throw new Error(
        `ImageGenerator: no adapter for provider "${request.provider}". Known: ${Array.from(this.adapters.keys()).join(', ')}`
      );
    }
    this.throwIfStopped(request.signal);

    const startedAt = Date.now();
    const deadline = this.startDeadline(request.signal);
    let result: ImageAdapterResult;
    try {
      result = await adapter.generate(request, { transport: this.transport, signal: deadline.signal });
    } catch (error) {
      // A stop wins over everything. Our own deadline is a failure the caller can read; anything
      // else an adapter throws is a bug and surfaces as it is.
      this.throwIfStopped(request.signal);
      if (!deadline.timedOut()) {
        throw error;
      }
      result = {
        kind: 'failed',
        errorKind: 'timeout',
        transient: true,
        sent: true,
        message: `No answer within ${this.timeoutMs} ms.`,
      };
    } finally {
      deadline.clear();
    }
    // A transport that ignored the signal may still have answered: the stop still wins.
    this.throwIfStopped(request.signal);

    const outcome = this.finish(request, result, Date.now() - startedAt);
    this.log(outcome);
    return outcome;
  }

  /** Stamp who made it and how long it took, and price an `ok` from the usage the vendor reported. */
  private finish(
    request: ImageGenerationRequest,
    result: ImageAdapterResult,
    latencyMs: number
  ): ImageGenerationOutcome {
    const stamp = { provider: request.provider, model: request.model, latencyMs };
    if (result.kind !== 'ok') {
      return { ...result, ...stamp };
    }
    const cost = this.costCalculator.cost({
      model: request.model,
      usage: result.usage,
      imageCount: result.images.length,
    });
    return { ...result, ...stamp, ...(cost ? { cost } : {}) };
  }

  /** One line per ask: who, what came of it, how long. Never the prompt, the pictures or a header. */
  private log(outcome: ImageGenerationOutcome): void {
    const obj = {
      provider: outcome.provider,
      model: outcome.model,
      kind: outcome.kind,
      latencyMs: outcome.latencyMs,
      vendorRequestId: outcome.vendorRequestId,
      ...(outcome.kind === 'ok'
        ? { images: outcome.images.length, priced: !!outcome.cost, totalUsd: outcome.cost?.totalUsd }
        : {}),
      ...(outcome.kind === 'refused' ? { reason: outcome.reason, stage: outcome.stage } : {}),
      ...(outcome.kind === 'failed'
        ? { errorKind: outcome.errorKind, transient: outcome.transient, statusCode: outcome.statusCode }
        : {}),
    };
    if (outcome.kind === 'ok') {
      this.logger.info({ message: 'Pictures made', obj });
    } else {
      this.logger.warn({ message: `Pictures not made (${outcome.kind})`, obj });
    }
  }

  private throwIfStopped(signal: AbortSignal | undefined): void {
    if (signal?.aborted) {
      throw signal.reason ?? new Error('The ask was stopped.');
    }
  }

  /** The caller's stop joined with this generator's deadline, as the one signal the wire sees. */
  private startDeadline(callerSignal: AbortSignal | undefined): {
    signal: AbortSignal;
    timedOut: () => boolean;
    clear: () => void;
  } {
    const controller = new AbortController();
    let timedOut = false;
    const onCallerStop = () => controller.abort(callerSignal?.reason);
    callerSignal?.addEventListener('abort', onCallerStop, { once: true });
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort(new Error(`No answer within ${this.timeoutMs} ms.`));
    }, this.timeoutMs);
    return {
      signal: controller.signal,
      timedOut: () => timedOut,
      clear: () => {
        clearTimeout(timer);
        callerSignal?.removeEventListener('abort', onCallerStop);
      },
    };
  }
}
