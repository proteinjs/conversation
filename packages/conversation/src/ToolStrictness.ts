import type { ToolSet } from 'ai';

/**
 * What a function tool says about STRICT mode to the provider it is handed to — the one owner of
 * that statement, read by every path that puts a function tool in a provider's tool list
 * (`Conversation`'s AI SDK tools, the tools a skill hands in already built through
 * `ConversationSkill.getProviderDefinedTools`, `OpenAiResponses`' polling-mode tools).
 *
 * `provider` is always the provider that RECEIVES the call — `routedProvider(model)`, the same
 * routing decision `resolveModel` builds the model from — never a guess from the model's name: a
 * name no pattern recognizes is routed to OpenAI, and must be told what OpenAI is told.
 *
 * Why it must be said: OpenAI's Responses API reads a function tool that states nothing as
 * STRICT. It rewrites the tool's schema so every property is required, and the model then fills
 * every OPTIONAL property of the tool with something — an empty string, `false`, the first enum
 * value, an invented object. A tool that reads "this optional object is present" as the model's
 * decision (a reminder's `repeat`, a question's `credential`) then acts on a value the model never
 * chose. A schema the rewrite cannot express (an open object anywhere inside it) silently stays
 * non-strict, so which tools were force-filled was an accident of their schemas.
 *
 * Tool schemas in this library are ordinary JSON Schema: an optional property means optional on
 * every provider. So the statement is explicit — a tool is strict only when its own definition
 * declares it (`definition.strict: true`, for a tool whose schema was written for strict mode),
 * and non-strict otherwise. Providers that do not default to strict are told nothing: their wire
 * is unchanged.
 */
export class ToolStrictness {
  /** Providers whose API treats a function tool that states nothing as strict. */
  private static readonly DEFAULTS_TO_STRICT: ReadonlySet<string> = new Set(['openai']);

  /**
   * The `strict` statement to spread onto the provider's tool entry (empty = say nothing).
   * `tool` is whatever declares the tool — a `Function`'s definition or an AI SDK tool.
   */
  static statementFor(provider: string, tool: { strict?: boolean | null }): { strict?: boolean } {
    if (!ToolStrictness.DEFAULTS_TO_STRICT.has(provider)) {
      return {};
    }
    return { strict: tool.strict === true };
  }

  /**
   * The same statement on tools that arrive ALREADY BUILT as AI SDK tools (a skill's
   * `getProviderDefinedTools`), which never pass through the library's own tool builder. Every
   * tool the provider reads as a FUNCTION (an AI SDK tool whose `type` is absent, `'function'` or
   * `'dynamic'`) carries it; a provider's native tool (`type: 'provider'` — a text editor, a web
   * search) has no function schema to be strict about and is handed on untouched. The skill's own
   * tool objects are never mutated.
   */
  static statedOn(provider: string, tools: ToolSet): ToolSet {
    const stated: ToolSet = {};
    for (const [name, tool] of Object.entries(tools)) {
      stated[name] = tool.type === 'provider' ? tool : { ...tool, ...ToolStrictness.statementFor(provider, tool) };
    }
    return stated;
  }
}
