import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Middleware,
  LanguageModelV3StreamResult,
} from '@ai-sdk/provider';
import { getModelCapabilities } from '@ai-sdk/anthropic/internal';
import { wrapLanguageModel } from 'ai';
import { Logger } from '@proteinjs/logger';

/**
 * FORCED TOOL CHOICE FOLLOWS THE MODEL. A caller that forces a tool (`toolChoice: { type: 'tool' }`
 * or `'required'` — the web-search toggle's "guarantee a search this turn") is speaking to a
 * provider that may refuse forcing for the model in hand: Anthropic's always-on-thinking models
 * (Claude Fable 5.1, then Claude Opus 5.5 on 2026-09-22, Claude Sonnet 5.5 after it) answer every
 * forced request with a 400 `invalid_request_error`, `tool_choice: type "tool" and "any" are not
 * supported for this model.` — and so does any model while budget thinking is on. Each new such
 * model arrives after any list this library could carry.
 *
 * So the rule is not a house list of ids. This middleware sits under the transport-retry wrapper on
 * every resolved model and softens a forced choice to `auto` — THE TOOLS STAY, every one of them,
 * only the forcing is dropped; the model decides, and the toggle softens from "guarantee" to
 * "strongly available" — on either of two words:
 *
 * - THE VENDOR'S OWN VERDICT, before any request. `@ai-sdk/anthropic` keeps a per-model table
 *   (`getModelCapabilities`, exported from `@ai-sdk/anthropic/internal`) that marks the ids
 *   rejecting forced tool use (Sonnet 5.5, Opus 5.5, Fable 5.1 at ^3.0.138), and its own
 *   `prepareTools` would soften such a request itself (3.0.120/3.0.125 for the table's ids;
 *   3.0.131 under budget thinking) — with a DIFFERENT outcome: a named choice becomes `auto` with
 *   THAT TOOL ALONE on the request, every client tool dropped from the step and, with the first
 *   tool, the house's cache breakpoint on the roster. A turn whose second step carries the whole
 *   roster again then never reads the prompt cache its first step wrote (measured 2026-11-04 over a
 *   scripted transport: Sonnet 5.5 step 2 read 0 tokens; Sonnet 5 read the roster). So this
 *   middleware asks the vendor's table FIRST ({@link refusesForcing}) and softens the request with
 *   the whole roster before the adapter sees it: the adapter's narrowing is never reached, no
 *   warning rides the stream, and the roster — the cache prefix — is the same on every step.
 * - THE PROVIDER'S REFUSAL, heard once. An id the table does not know yet (a model newer than the
 *   adapter) is forwarded as forced; when the provider refuses it in those words, the request is
 *   re-issued softened and the model is REMEMBERED for the process, so every later request to it
 *   is softened before it leaves (one refused request per model per process, billed nothing: a 400
 *   generates no tokens). A model newer than this library that refuses forcing works on its first
 *   turn.
 *
 * Only that refusal is heard: any other 4xx surfaces untouched (the transport-retry layer's
 * semantic-error bar), and a model that accepts forcing is never softened.
 */
export class ForcedToolChoice {
  /** The provider's refusal, in its own words (Anthropic; the same clause on every model that refuses). */
  private static readonly REFUSAL_CLAUSE = /tool_choice[^.]*not supported for this model/i;

  /** Process-wide: the model ids whose provider refused a forced tool choice (ids the vendor's table did not know). */
  private static readonly heard = new Set<string>();

  private static logger = new Logger({ name: 'ForcedToolChoice' });

  /** Wrap a resolved model so its forced tool choices follow the provider's verdict. */
  static follow(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({ model, middleware: ForcedToolChoice.middleware() });
  }

  /**
   * Whether a forced tool choice is refused for this model — and so is softened before it leaves:
   * the vendor's table says so for the id, the request's thinking shape says so (budget thinking,
   * when `params` are given), or the provider's refusal was heard in this process.
   */
  static refusesForcing(modelId: string, params?: LanguageModelV3CallOptions): boolean {
    return ForcedToolChoice.heard.has(modelId) || ForcedToolChoice.vendorRefusesForcing(modelId, params);
  }

  /** Whether the provider's refusal was heard for this model in this process (an id the vendor's table did not know). */
  static heardRefusal(modelId: string): boolean {
    return ForcedToolChoice.heard.has(modelId);
  }

  /** Whether an error is the provider's refusal of a forced tool choice. */
  static isRefusal(error: unknown): boolean {
    const message = String((error as { message?: unknown })?.message ?? error ?? '');
    return ForcedToolChoice.REFUSAL_CLAUSE.test(message);
  }

  /** Forget every refusal heard (suites only — production remembers for the process). */
  static forgetAll(): void {
    ForcedToolChoice.heard.clear();
  }

  private static middleware(): LanguageModelV3Middleware {
    return {
      specificationVersion: 'v3',
      transformParams: async ({ params, model }) =>
        ForcedToolChoice.isForcing(params) && ForcedToolChoice.refusesForcing(model.modelId, params)
          ? ForcedToolChoice.softened(params)
          : params,
      wrapGenerate: ({ doGenerate, params, model }) =>
        ForcedToolChoice.hearing(doGenerate, params, model, (softened) => model.doGenerate(softened)),
      wrapStream: ({ doStream, params, model }) =>
        ForcedToolChoice.hearing(doStream, params, model, (softened) => model.doStream(softened)),
    };
  }

  /**
   * Run the request; when the provider refuses the forced choice, remember the model and run it
   * again softened. A request that was not forcing anything, or one softened before dispatch (the
   * vendor's verdict, a refusal already heard), never reaches the second call.
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
      ForcedToolChoice.heard.add(model.modelId);
      ForcedToolChoice.logger.warn({
        message: 'The provider refused a forced tool choice for this model — re-issuing with auto and remembering it',
        obj: { modelId: model.modelId, toolChoice: params.toolChoice, clause: String((error as Error)?.message) },
      });
      return await rerun(ForcedToolChoice.softened(params));
    }
  }

  /**
   * The vendor's own verdict on a request, before it is sent — the two clauses `@ai-sdk/anthropic`
   * softens by itself (`rejectsForcedToolUse || thinkingType === 'enabled'`, read after its
   * substitutions): its table marks the id, or the request carries budget thinking on a model that
   * takes a budget (on a model that rejects budgets the adapter swaps it for adaptive thinking,
   * under which forcing is accepted). An id the table does not know is forwarded as sent.
   */
  private static vendorRefusesForcing(modelId: string, params?: LanguageModelV3CallOptions): boolean {
    const capabilities = getModelCapabilities(modelId);
    if (capabilities.rejectsForcedToolUse) {
      return true;
    }
    const thinking = (params?.providerOptions?.anthropic as { thinking?: { type?: unknown } } | undefined)?.thinking;
    return thinking?.type === 'enabled' && !capabilities.rejectsBudgetThinking;
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
