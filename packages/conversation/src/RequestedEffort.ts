import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Middleware,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from '@ai-sdk/provider';
import { wrapLanguageModel } from 'ai';
import { Logger } from '@proteinjs/logger';

/**
 * THE REQUESTED EFFORT FOLLOWS THE MODEL. A caller asks for a reasoning effort the way it asks
 * every model, and the provider may refuse that value for the model in hand: GPT-6 Astra lists no
 * `none` (OpenAI, 2026-09-22: `Unsupported value: 'none' is not supported with the 'gpt-6-astra'
 * model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.`, HTTP 400), xAI
 * refuses `none` on grok-4.5 (2026-09-29: `This model does not support \`reasoning_effort\`
 * value \`none\`.`), and each new such model arrives after any list this library could carry.
 *
 * So the rule is not a list of ids: the PROVIDER'S OWN REFUSAL is the rule. This middleware sits
 * under the transport-retry wrapper on every resolved model and, when the provider refuses the
 * request for its effort value, re-issues it ONCE with the effort OMITTED — the request the
 * caller would have made with no effort at all, so the provider applies its own documented
 * default for the model (the ruling of 2026-09-29: never a level of this library's choosing;
 * "leave the current functionality of omitting effort so the api can default") — and REMEMBERS
 * the omission for the model for the process, so every later request to it at that value leaves
 * without the effort (one refused request per model per process, billed nothing: a 400
 * generates no tokens). The omission is surfaced once, as a warning on the re-issued call (the
 * SDK's warning channel — it rides `stream-start` and the generate result's `warnings`, and the
 * conversation forwards it on its own stream as the `step-start` part), never as an error to the
 * person. A consumer that keeps a catalog of what each model accepts reads that warning as the
 * fact that its entry is wrong: this is the transport's LAST guard, not the product's rule.
 *
 * Only that refusal is heard: any other 4xx surfaces untouched (the transport-retry layer's
 * semantic-error bar), a model that accepts the value is never touched (its request leaves
 * byte-identical), and a request that carries no effort has nothing to hear.
 */
export class RequestedEffort {
  /** Where each provider carries the effort in `providerOptions` (the field this library writes). */
  private static readonly EFFORT_FIELDS: Readonly<Record<string, readonly string[]>> = {
    openai: ['reasoningEffort'],
    xai: ['reasoningEffort'],
    anthropic: ['effort'],
    google: ['thinkingConfig', 'thinkingLevel'],
  };

  /** The parameter names the providers refuse the effort under (OpenAI's `param`; the field path in a message). */
  private static readonly EFFORT_PARAM =
    /(reasoning[._]?effort|output_config\.effort|\beffort\b|thinking[._]?level|thinking[._]?config)/i;

  /** The words of a refused value, in every provider's grammar. */
  private static readonly REFUSED =
    /not supported|unsupported|invalid|not (?:a )?valid|must be one of|does not support/i;

  /** Process-wide: model id → the effort values the provider refused (each now omitted before dispatch). */
  private static readonly omissions = new Map<string, Set<string>>();

  private static logger = new Logger({ name: 'RequestedEffort' });

  /** Wrap a resolved model so its requested effort follows the provider's verdict. */
  static follow(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({ model, middleware: RequestedEffort.middleware() });
  }

  /** Whether `effort` is omitted for this model (in this process) because the provider refused it. */
  static omits(modelId: string, effort: string): boolean {
    return RequestedEffort.omissions.get(modelId)?.has(effort) ?? false;
  }

  /**
   * Whether an error is the provider's refusal of the request for its effort value: a request
   * refusal (a 400 when the status is known) that names the effort parameter — OpenAI's `param`,
   * or the field path in the message — or that quotes the value this request sent as unsupported.
   */
  static isRefusal(error: unknown, sent: string): boolean {
    const status = (error as { statusCode?: unknown })?.statusCode;
    if (typeof status === 'number' && status !== 400) {
      return false;
    }
    // The provider's clause, or the SDK's own (`parseProviderOptions`: "invalid xai provider options",
    // the refused value named only in its cause) — a level the installed provider package cannot
    // carry is refused before any request, and it is the same refusal to this rule.
    const text = RequestedEffort.words(error);
    if (!RequestedEffort.REFUSED.test(text)) {
      return false;
    }
    const param = RequestedEffort.namedParam(error);
    if (param !== undefined) {
      return RequestedEffort.EFFORT_PARAM.test(param);
    }
    return RequestedEffort.EFFORT_PARAM.test(text) || new RegExp(`['"\`]${sent}['"\`]`).test(text);
  }

  /** Forget every omission heard (suites only — production remembers for the process). */
  static forgetAll(): void {
    RequestedEffort.omissions.clear();
  }

  private static middleware(): LanguageModelV3Middleware {
    return {
      specificationVersion: 'v3',
      transformParams: async ({ params, model }) => RequestedEffort.remembered(params, model.modelId),
      wrapGenerate: ({ doGenerate, params, model }) =>
        RequestedEffort.hearing(
          doGenerate,
          params,
          model,
          (substituted) => model.doGenerate(substituted),
          (result, warning) => ({
            ...result,
            warnings: [...(result.warnings ?? []), warning],
          })
        ),
      wrapStream: ({ doStream, params, model }) =>
        RequestedEffort.hearing(
          doStream,
          params,
          model,
          (substituted) => model.doStream(substituted),
          (result, warning) => ({
            ...result,
            stream: result.stream.pipeThrough(RequestedEffort.warned(warning)),
          })
        ),
    };
  }

  /**
   * Run the request; when the provider refuses it for its effort value, run it again ONCE with
   * the effort omitted, and — once that request is accepted — remember the omission for the model
   * and surface it on the result as a warning. A request with no effort, or an omission already
   * heard (the transform applied it before dispatch), never reaches the second call; a refusal of
   * the effort-less request itself surfaces.
   */
  private static async hearing<T extends LanguageModelV3GenerateResult | LanguageModelV3StreamResult>(
    run: () => PromiseLike<T>,
    params: LanguageModelV3CallOptions,
    model: LanguageModelV3,
    rerun: (substituted: LanguageModelV3CallOptions) => PromiseLike<T>,
    warned: (result: T, warning: SharedV3Warning) => T
  ): Promise<T> {
    try {
      return await run();
    } catch (error: unknown) {
      const sent = RequestedEffort.sentEffort(params);
      if (!sent || !RequestedEffort.isRefusal(error, sent.value)) {
        throw error;
      }
      const clause = RequestedEffort.words(error);
      RequestedEffort.logger.warn({
        message:
          'The provider refused the requested reasoning effort for this model — re-issuing once with the effort omitted (the provider’s own default) and remembering it',
        obj: { modelId: model.modelId, requested: sent.value, clause },
      });
      const result = await rerun(RequestedEffort.withoutEffort(params, sent));
      RequestedEffort.remember(model.modelId, sent.value);
      return warned(result, {
        type: 'compatibility',
        feature: 'reasoningEffort',
        details: `${model.modelId} does not accept reasoning effort '${sent.value}' — re-issued with the effort omitted (the provider's own default applies) and remembered for this process. The provider: ${clause}`,
      });
    }
  }

  /** The request with a remembered omission applied before it leaves — else the request as it came. */
  private static remembered(params: LanguageModelV3CallOptions, modelId: string): LanguageModelV3CallOptions {
    const sent = RequestedEffort.sentEffort(params);
    return sent && RequestedEffort.omits(modelId, sent.value) ? RequestedEffort.withoutEffort(params, sent) : params;
  }

  private static remember(modelId: string, refused: string): void {
    const forModel = RequestedEffort.omissions.get(modelId) ?? new Set<string>();
    forModel.add(refused);
    RequestedEffort.omissions.set(modelId, forModel);
  }

  /** The effort this request carries — the provider key it rides under, its field path, its value. */
  private static sentEffort(
    params: LanguageModelV3CallOptions
  ): { provider: string; path: readonly string[]; value: string } | undefined {
    for (const [provider, path] of Object.entries(RequestedEffort.EFFORT_FIELDS)) {
      const value = path.reduce<unknown>(
        (node, key) => (node as Record<string, unknown> | undefined)?.[key],
        params.providerOptions?.[provider]
      );
      if (typeof value === 'string') {
        return { provider, path, value };
      }
    }
    return undefined;
  }

  /** The same request with the effort field removed — nothing else touched. */
  private static withoutEffort(
    params: LanguageModelV3CallOptions,
    sent: { provider: string; path: readonly string[] }
  ): LanguageModelV3CallOptions {
    const options = { ...(params.providerOptions?.[sent.provider] ?? {}) } as Record<string, unknown>;
    let node = options;
    for (const key of sent.path.slice(0, -1)) {
      node[key] = { ...((node[key] as Record<string, unknown> | undefined) ?? {}) };
      node = node[key] as Record<string, unknown>;
    }
    delete node[sent.path[sent.path.length - 1]];
    return { ...params, providerOptions: { ...params.providerOptions, [sent.provider]: options } as never };
  }

  /** OpenAI names the refused parameter in the error body (`error.param`); the SDK keeps the body on `data`. */
  /** An error's words with its cause's — the SDK names the refused option only in the cause. */
  private static words(error: unknown): string {
    const message = String((error as { message?: unknown })?.message ?? error ?? '');
    const cause = (error as { cause?: unknown })?.cause;
    if (cause === undefined || cause === null) {
      return message;
    }
    return `${message} — ${String((cause as { message?: unknown })?.message ?? cause)}`;
  }

  private static namedParam(error: unknown): string | undefined {
    const param = (error as { data?: { error?: { param?: unknown } } })?.data?.error?.param;
    return typeof param === 'string' ? param : undefined;
  }

  /** The stream with the warning added to its `stream-start` — the part the SDK reads a call's warnings from. */
  private static warned(
    warning: SharedV3Warning
  ): TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart> {
    return new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
      transform(part, controller) {
        controller.enqueue(
          part.type === 'stream-start' ? { ...part, warnings: [...(part.warnings ?? []), warning] } : part
        );
      },
    });
  }
}
