import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Middleware,
  LanguageModelV3StreamResult,
} from '@ai-sdk/provider';
import { wrapLanguageModel } from 'ai';
import { Logger } from '@proteinjs/logger';

/**
 * FORCED TOOL CHOICE FOLLOWS THE MODEL. A caller that forces a tool (`toolChoice: { type: 'tool' }`
 * or `'required'` — the web-search toggle's "guarantee a search this turn") is speaking to a
 * provider that may refuse forcing for the model in hand: Anthropic's always-on-thinking models
 * (Claude Fable 5.1, then Claude Opus 5.5 on 2026-09-22) answer every forced request with a 400
 * `invalid_request_error`, `tool_choice: type "tool" and "any" are not supported for this model.`
 * — and each new such model arrives after any list this library could carry.
 *
 * So the rule is not a list of ids: the PROVIDER'S OWN REFUSAL is the rule. This middleware sits
 * under the transport-retry wrapper on every resolved model and, when the provider refuses a
 * forced choice in those words, re-issues the same request with `toolChoice: auto` — the tool
 * stays attached, the model decides (the toggle softens from "guarantee" to "strongly
 * available", exactly what the id-list exception used to do by hand) — and REMEMBERS the model
 * for the process, so every later request to it is softened before it leaves (one refused
 * request per model per process, billed nothing: a 400 generates no tokens). A model newer than
 * this library that refuses forcing works on its first turn.
 *
 * Only that refusal is heard: any other 4xx surfaces untouched (the transport-retry layer's
 * semantic-error bar), and a model that accepts forcing is never softened.
 */
export class ForcedToolChoice {
  /** The provider's refusal, in its own words (Anthropic; the same clause on every model that refuses). */
  private static readonly REFUSAL_CLAUSE = /tool_choice[^.]*not supported for this model/i;

  /** Process-wide: the model ids whose provider refused a forced tool choice. */
  private static readonly refusing = new Set<string>();

  private static logger = new Logger({ name: 'ForcedToolChoice' });

  /** Wrap a resolved model so its forced tool choices follow the provider's verdict. */
  static follow(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({ model, middleware: ForcedToolChoice.middleware() });
  }

  /** Whether the provider has refused forcing for this model (in this process). */
  static refusesForcing(modelId: string): boolean {
    return ForcedToolChoice.refusing.has(modelId);
  }

  /** Whether an error is the provider's refusal of a forced tool choice. */
  static isRefusal(error: unknown): boolean {
    const message = String((error as { message?: unknown })?.message ?? error ?? '');
    return ForcedToolChoice.REFUSAL_CLAUSE.test(message);
  }

  /** Forget every refusal heard (suites only — production remembers for the process). */
  static forgetAll(): void {
    ForcedToolChoice.refusing.clear();
  }

  private static middleware(): LanguageModelV3Middleware {
    return {
      specificationVersion: 'v3',
      transformParams: async ({ params, model }) =>
        ForcedToolChoice.refusesForcing(model.modelId) ? ForcedToolChoice.softened(params) : params,
      wrapGenerate: ({ doGenerate, params, model }) =>
        ForcedToolChoice.hearing(doGenerate, params, model, (softened) => model.doGenerate(softened)),
      wrapStream: ({ doStream, params, model }) =>
        ForcedToolChoice.hearing(doStream, params, model, (softened) => model.doStream(softened)),
    };
  }

  /**
   * Run the request; when the provider refuses the forced choice, remember the model and run it
   * again softened. A request that was not forcing anything, or a refusal already heard (the
   * transform softened it before dispatch), never reaches the second call.
   */
  private static async hearing<T extends LanguageModelV3GenerateResult | LanguageModelV3StreamResult>(
    run: () => PromiseLike<T>,
    params: LanguageModelV3CallOptions,
    model: LanguageModelV3,
    rerun: (softened: LanguageModelV3CallOptions) => PromiseLike<T>
  ): Promise<T> {
    try {
      return await run();
    } catch (error: unknown) {
      if (!ForcedToolChoice.isForcing(params) || !ForcedToolChoice.isRefusal(error)) {
        throw error;
      }
      ForcedToolChoice.refusing.add(model.modelId);
      ForcedToolChoice.logger.warn({
        message: 'The provider refused a forced tool choice for this model — re-issuing with auto and remembering it',
        obj: { modelId: model.modelId, toolChoice: params.toolChoice, clause: String((error as Error)?.message) },
      });
      return await rerun(ForcedToolChoice.softened(params));
    }
  }

  private static isForcing(params: LanguageModelV3CallOptions): boolean {
    const type = params.toolChoice?.type;
    return type === 'tool' || type === 'required';
  }

  /** The same request with the forcing dropped — the tools stay, the model decides. */
  private static softened(params: LanguageModelV3CallOptions): LanguageModelV3CallOptions {
    return ForcedToolChoice.isForcing(params) ? { ...params, toolChoice: { type: 'auto' } } : params;
  }
}
