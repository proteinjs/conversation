import type { Function } from './Function';

/**
 * What a function tool says about STRICT mode to the provider it is handed to — the one owner of
 * that statement, read by every path that builds a provider's tool list (`Conversation`'s AI SDK
 * tools, `OpenAiResponses`' polling-mode tools).
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

  /** The `strict` statement to spread onto the provider's tool entry (empty = say nothing). */
  static statementFor(provider: string, definition: Function['definition']): { strict?: boolean } {
    if (!ToolStrictness.DEFAULTS_TO_STRICT.has(provider)) {
      return {};
    }
    return { strict: definition.strict === true };
  }
}
