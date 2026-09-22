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
 * every model — the bounded utterance (the acknowledgment line a turn speaks first) asks for
 * `none` — and the provider may refuse that value for the model in hand: GPT-6 Astra lists no
 * `none` (OpenAI, 2026-09-22: `Unsupported value: 'none' is not supported with the 'gpt-6-astra'
 * model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.`, HTTP 400), and each
 * new such model arrives after any list this library could carry. Before this rule every Astra
 * turn ran without its acknowledgment line and paid a refused request each time.
 *
 * So the rule is not a list of ids: the PROVIDER'S OWN REFUSAL is the rule. This middleware sits
 * under the transport-retry wrapper on every resolved model and, when the provider refuses the
 * request for its effort value, re-issues it ONCE at the nearest level the model accepts — the
 * model's own ladder when the clause lists it (OpenAI and Google name the supported values in
 * the refusal), else the provider's lowest level — and REMEMBERS the substitution for the model
 * for the process, so every later request to it is substituted before it leaves (one refused
 * request per model per process, billed nothing: a 400 generates no tokens). The substitution
 * is surfaced once, as a warning on the re-issued call (the SDK's warning channel — it rides
 * `stream-start` and the generate result's `warnings`), never as an error to the person.
 *
 * Only that refusal is heard: any other 4xx surfaces untouched (the transport-retry layer's
 * semantic-error bar), a model that accepts the value is never touched (its request leaves
 * byte-identical), and a request that carries no effort has nothing to hear.
 */
export class RequestedEffort {
  /**
   * The effort ladder this library speaks, lowest first — the words every provider uses
   * (OpenAI `reasoning.effort`, Anthropic `output_config.effort`, Google `thinking_level`,
   * xAI `reasoning_effort`), so "nearest" is one distance for all of them.
   */
  static readonly LADDER: readonly string[] = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

  /** The provider's lowest reasoning level — the floor when a refusal lists no ladder. */
  static readonly FLOOR = 'low';

  /** Where each provider carries the effort in `providerOptions` (the field this library writes). */
  private static readonly EFFORT_FIELDS: Readonly<Record<string, readonly string[]>> = {
    openai: ['reasoningEffort'],
    xai: ['reasoningEffort'],
    anthropic: ['effort'],
    google: ['thinkingConfig', 'thinkingLevel'],
  };

  /** The parameter names the providers refuse the effort under (OpenAI's `param`; the field path in a message). */
  private static readonly EFFORT_PARAM = /(reasoning[._]effort|output_config\.effort|\beffort\b|thinking[._]?level|thinking[._]?config)/i;

  /** The words of a refused value, in every provider's grammar. */
  private static readonly REFUSED = /not supported|unsupported|invalid|not (?:a )?valid|must be one of/i;

  /** The ladder a refusal lists (OpenAI `Supported values are: 'low', …`; Google `Supported values: …`). */
  private static readonly LISTED = /supported values(?: are)?\s*:?\s*([^.]*)/i;

  /** Process-wide: model id → (refused effort → the level the model accepted in its place). */
  private static readonly substitutions = new Map<string, Map<string, string>>();

  private static logger = new Logger({ name: 'RequestedEffort' });

  /** Wrap a resolved model so its requested effort follows the provider's verdict. */
  static follow(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({ model, middleware: RequestedEffort.middleware() });
  }

  /** The level the provider accepted in place of `effort` for this model (in this process), if any. */
  static substituteFor(modelId: string, effort: string): string | undefined {
    return RequestedEffort.substitutions.get(modelId)?.get(effort);
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
    const message = String((error as { message?: unknown })?.message ?? error ?? '');
    if (!RequestedEffort.REFUSED.test(message)) {
      return false;
    }
    const param = RequestedEffort.namedParam(error);
    if (param !== undefined) {
      return RequestedEffort.EFFORT_PARAM.test(param);
    }
    return RequestedEffort.EFFORT_PARAM.test(message) || new RegExp(`['"\`]${sent}['"\`]`).test(message);
  }

  /**
   * The nearest level to `refused` among `accepted` (the ladder the clause lists, else the
   * provider's floor) — a tie goes to the lower level; never the refused value itself.
   */
  static nearest(refused: string, accepted: readonly string[]): string | undefined {
    const ladder = RequestedEffort.LADDER;
    const from = ladder.indexOf(refused);
    const candidates = accepted.filter((level) => level !== refused && ladder.includes(level));
    if (candidates.length === 0) {
      return undefined;
    }
    if (from < 0) {
      return candidates[0];
    }
    return candidates
      .map((level) => ({ level, distance: Math.abs(ladder.indexOf(level) - from), rank: ladder.indexOf(level) }))
      .sort((a, b) => a.distance - b.distance || a.rank - b.rank)[0].level;
  }

  /** The levels a refusal lists as accepted, in the ladder's words; empty when it lists none. */
  static listedLadder(message: string): string[] {
    const listed = RequestedEffort.LISTED.exec(message)?.[1] ?? '';
    return Array.from(listed.matchAll(/['"`]([a-z]+)['"`]/gi), (match) => match[1].toLowerCase()).filter((level) =>
      RequestedEffort.LADDER.includes(level)
    );
  }

  /** Forget every substitution heard (suites only — production remembers for the process). */
  static forgetAll(): void {
    RequestedEffort.substitutions.clear();
  }

  private static middleware(): LanguageModelV3Middleware {
    return {
      specificationVersion: 'v3',
      transformParams: async ({ params, model }) => RequestedEffort.remembered(params, model.modelId),
      wrapGenerate: ({ doGenerate, params, model }) =>
        RequestedEffort.hearing(doGenerate, params, model, (substituted) => model.doGenerate(substituted), (result, warning) => ({
          ...result,
          warnings: [...(result.warnings ?? []), warning],
        })),
      wrapStream: ({ doStream, params, model }) =>
        RequestedEffort.hearing(doStream, params, model, (substituted) => model.doStream(substituted), (result, warning) => ({
          ...result,
          stream: result.stream.pipeThrough(RequestedEffort.warned(warning)),
        })),
    };
  }

  /**
   * Run the request; when the provider refuses it for its effort value, run it again ONCE at the
   * nearest level the model accepts, and — once that request is accepted — remember the
   * substitution for the model and surface it on the result as a warning. A request with no
   * effort, or a substitution already heard (the transform applied it before dispatch), never
   * reaches the second call; a refusal of the substitute itself surfaces.
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
      const clause = String((error as Error)?.message ?? error);
      const listed = RequestedEffort.listedLadder(clause);
      const substitute = RequestedEffort.nearest(sent.value, listed.length > 0 ? listed : [RequestedEffort.FLOOR]);
      if (!substitute) {
        throw error;
      }
      RequestedEffort.logger.warn({
        message:
          'The provider refused the requested reasoning effort for this model — re-issuing once at the nearest level it accepts and remembering it',
        obj: { modelId: model.modelId, requested: sent.value, substitute, listed, clause },
      });
      const result = await rerun(RequestedEffort.substituted(params, sent, substitute));
      RequestedEffort.remember(model.modelId, sent.value, substitute);
      return warned(result, {
        type: 'compatibility',
        feature: 'reasoningEffort',
        details: `${model.modelId} does not accept reasoning effort '${sent.value}' — re-issued at '${substitute}', the nearest level it accepts, and remembered for this process. The provider: ${clause}`,
      });
    }
  }

  /** The request with a remembered substitution applied before it leaves — else the request as it came. */
  private static remembered(params: LanguageModelV3CallOptions, modelId: string): LanguageModelV3CallOptions {
    const sent = RequestedEffort.sentEffort(params);
    const substitute = sent && RequestedEffort.substituteFor(modelId, sent.value);
    return substitute ? RequestedEffort.substituted(params, sent, substitute) : params;
  }

  private static remember(modelId: string, refused: string, substitute: string): void {
    const forModel = RequestedEffort.substitutions.get(modelId) ?? new Map<string, string>();
    forModel.set(refused, substitute);
    RequestedEffort.substitutions.set(modelId, forModel);
  }

  /** The effort this request carries — the provider key it rides under, its field path, its value. */
  private static sentEffort(params: LanguageModelV3CallOptions): { provider: string; path: readonly string[]; value: string } | undefined {
    for (const [provider, path] of Object.entries(RequestedEffort.EFFORT_FIELDS)) {
      const value = path.reduce<unknown>((node, key) => (node as Record<string, unknown> | undefined)?.[key], params.providerOptions?.[provider]);
      if (typeof value === 'string') {
        return { provider, path, value };
      }
    }
    return undefined;
  }

  /** The same request with the effort replaced — nothing else touched. */
  private static substituted(
    params: LanguageModelV3CallOptions,
    sent: { provider: string; path: readonly string[] },
    substitute: string
  ): LanguageModelV3CallOptions {
    const options = { ...(params.providerOptions?.[sent.provider] ?? {}) } as Record<string, unknown>;
    let node = options;
    for (const key of sent.path.slice(0, -1)) {
      node[key] = { ...((node[key] as Record<string, unknown> | undefined) ?? {}) };
      node = node[key] as Record<string, unknown>;
    }
    node[sent.path[sent.path.length - 1]] = substitute;
    return { ...params, providerOptions: { ...params.providerOptions, [sent.provider]: options } as never };
  }

  /** OpenAI names the refused parameter in the error body (`error.param`); the SDK keeps the body on `data`. */
  private static namedParam(error: unknown): string | undefined {
    const param = (error as { data?: { error?: { param?: unknown } } })?.data?.error?.param;
    return typeof param === 'string' ? param : undefined;
  }

  /** The stream with the warning added to its `stream-start` — the part the SDK reads a call's warnings from. */
  private static warned(warning: SharedV3Warning): TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart> {
    return new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
      transform(part, controller) {
        controller.enqueue(
          part.type === 'stream-start' ? { ...part, warnings: [...(part.warnings ?? []), warning] } : part
        );
      },
    });
  }
}
