import { Logger, LogLevel } from '@proteinjs/logger';
import type { ModelDataResolver } from '../ModelData';
import { FetchImageTransport } from './FetchImageTransport';
import { ImageCostCalculator } from './ImageCostCalculator';
import type { ImageGenerationRequest } from './ImageGenerationRequest';
import type { ImageCostUsd, ImageGenerationOutcome, ImageGenerationStopped } from './ImageGenerationOutcome';
import type { ImageAdapterResult, ImageProviderAdapter, ImageTransport } from './ImageProviderAdapter';
import { GoogleImageAdapter } from './GoogleImageAdapter';
import { OpenAiImageAdapter } from './OpenAiImageAdapter';
import { RecraftImageAdapter } from './RecraftImageAdapter';

export type ImageGeneratorParams = {
  /**
   * The rates pictures are priced from — the same resolver the text path takes, and required for
   * the same reason: a generator without pricing data would record every picture as free.
   */
  modelData: ModelDataResolver;
  /** One adapter per provider. Default: the OpenAI, Google and Recraft adapters. */
  adapters?: ImageProviderAdapter[];
  /** The wire. Default: the runtime's `fetch`. A test hands in a double, so CI never calls a vendor. */
  transport?: ImageTransport;
  /**
   * The longest one ask may take before it is given up as a transient failure. Default 4 minutes
   * — above it, keep under the runtime's own limit on a silent connection (Node's `fetch` gives
   * up at 300 seconds), or that limit fires first and the ask reads as `network`, not `timeout`.
   */
  timeoutMs?: number;
  logLevel?: LogLevel;
};

/**
 * Makes pictures: one vendor-neutral ask in, one outcome out — `ok`, `refused`, `failed` or
 * `stopped` — priced from the vendor's own usage where that is possible, a known zero where
 * nothing can have been billed, and left unpriced where the price is not known.
 *
 * `generate()` resolves for EVERY ask it accepts, so the spend of an ask always reaches the
 * caller: the caller's stop is the `stopped` outcome (a picture that arrived anyway is discarded
 * here, its usage and cost still reported), and a defect in an adapter is a `failed` outcome. It
 * rejects for one thing only — an ask for a provider with no adapter, a wiring error found before
 * anything is sent.
 */
export class ImageGenerator {
  /** The vendor documents "up to 2 minutes"; this leaves room and stays under Node fetch's 300 s. */
  static readonly DEFAULT_TIMEOUT_MS = 4 * 60 * 1000;

  private readonly adapters: Map<string, ImageProviderAdapter>;
  private readonly transport: ImageTransport;
  private readonly costCalculator: ImageCostCalculator;
  private readonly timeoutMs: number;
  private readonly logger: Logger;

  constructor(params: ImageGeneratorParams) {
    // One adapter per vendor the catalog names; each reads its own key from the environment at call time.
    const adapters = params.adapters ?? [new OpenAiImageAdapter(), new GoogleImageAdapter(), new RecraftImageAdapter()];
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
    const startedAt = Date.now();
    const wire = this.watchedTransport();
    // An ask already stopped never reaches the adapter, so never the wire.
    const result = request.signal?.aborted ? undefined : await this.ask(adapter, request, wire.transport, wire.sent);

    // The stop wins over whatever came back — a transport that ignored the signal may still have
    // answered. The answer's pictures are dropped; what it says was spent is not.
    const outcome =
      request.signal?.aborted || !result
        ? this.stopped(request, result, wire.sent(), Date.now() - startedAt)
        : this.finish(request, result, Date.now() - startedAt);
    this.log(outcome);
    return outcome;
  }

  /**
   * One adapter call under the deadline. Answers the adapter's result; a `failed` result of this
   * generator's own when the deadline fired or the adapter threw; `undefined` when the caller's
   * stop ended the call with no answer. Never rejects.
   */
  private async ask(
    adapter: ImageProviderAdapter,
    request: ImageGenerationRequest,
    transport: ImageTransport,
    sent: () => boolean
  ): Promise<ImageAdapterResult | undefined> {
    const deadline = this.startDeadline(request.signal);
    try {
      return await adapter.generate(request, { transport, signal: deadline.signal });
    } catch (error) {
      if (request.signal?.aborted) {
        return undefined;
      }
      // Once a request is out, no answer means the vendor may have made and billed the pictures.
      const failure = { kind: 'failed', transient: false, sent: sent(), billable: sent() } as const;
      if (deadline.timedOut()) {
        return { ...failure, errorKind: 'timeout', transient: true, message: `No answer within ${this.timeoutMs} ms.` };
      }
      this.logger.error({ message: 'The picture adapter threw', error });
      return {
        ...failure,
        errorKind: 'adapter_error',
        message: error instanceof Error ? error.message : String(error),
      };
    } finally {
      deadline.clear();
    }
  }

  /** Stamp who made it and how long it took, and say what it cost. */
  private finish(
    request: ImageGenerationRequest,
    result: ImageAdapterResult,
    latencyMs: number
  ): ImageGenerationOutcome {
    const stamp = { provider: request.provider, model: request.model, latencyMs };
    const cost = this.costOf(request, result);
    if (result.kind === 'ok') {
      const { answeredCount, ...ok } = result;
      return { ...ok, ...stamp, ...(cost ? { cost } : {}) };
    }
    const { billable, ...notMade } = result;
    return { ...notMade, ...stamp, ...(cost ? { cost } : {}) };
  }

  /**
   * The caller's stop as an outcome. `result` is whatever came back anyway (none, when the stop
   * cut the call off): its pictures are counted and left behind — they are never copied onto the
   * outcome, so nothing downstream can store them — while its usage and cost are carried over.
   */
  private stopped(
    request: ImageGenerationRequest,
    result: ImageAdapterResult | undefined,
    sent: boolean,
    latencyMs: number
  ): ImageGenerationStopped {
    const cost = result ? this.costOf(request, result) : sent ? undefined : this.costCalculator.nothing();
    return {
      kind: 'stopped',
      provider: request.provider,
      model: request.model,
      latencyMs,
      sent: result ? result.sent : sent,
      discardedImages: result?.kind === 'ok' ? result.images.length : 0,
      ...(result?.vendorRequestId ? { vendorRequestId: result.vendorRequestId } : {}),
      ...(result?.usage ? { usage: result.usage } : {}),
      ...(cost ? { cost } : {}),
    };
  }

  /**
   * What one adapter result cost: a known zero when nothing can have been billed; otherwise the
   * vendor's usage × the model's rates; otherwise `undefined` — not known, never a guess.
   */
  private costOf(request: ImageGenerationRequest, result: ImageAdapterResult): ImageCostUsd | undefined {
    if (!result.sent || (result.kind !== 'ok' && !result.billable)) {
      return this.costCalculator.nothing();
    }
    return this.costCalculator.cost({
      model: request.model,
      usage: result.usage,
      imageCount: result.kind === 'ok' ? result.answeredCount : undefined,
    });
  }

  /** One line per ask: who, what came of it, how long. Never the prompt, the pictures or a header. */
  private log(outcome: ImageGenerationOutcome): void {
    const obj = {
      provider: outcome.provider,
      model: outcome.model,
      kind: outcome.kind,
      latencyMs: outcome.latencyMs,
      vendorRequestId: outcome.vendorRequestId,
      sent: outcome.sent,
      priced: !!outcome.cost,
      totalUsd: outcome.cost?.totalUsd,
      ...(outcome.kind === 'ok' ? { images: outcome.images.length } : {}),
      ...(outcome.kind === 'refused' ? { reason: outcome.reason, stage: outcome.stage } : {}),
      ...(outcome.kind === 'failed'
        ? { errorKind: outcome.errorKind, transient: outcome.transient, statusCode: outcome.statusCode }
        : {}),
      ...(outcome.kind === 'stopped' ? { discardedImages: outcome.discardedImages } : {}),
    };
    if (outcome.kind === 'ok') {
      this.logger.info({ message: 'Pictures made', obj });
    } else {
      this.logger.warn({ message: `Pictures not made (${outcome.kind})`, obj });
    }
  }

  /** The wire, watched: whether a request has gone out is a fact of the wire, not an adapter's say-so. */
  private watchedTransport(): { transport: ImageTransport; sent: () => boolean } {
    let sent = false;
    return {
      transport: {
        post: (request) => {
          sent = true;
          return this.transport.post(request);
        },
      },
      sent: () => sent,
    };
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
