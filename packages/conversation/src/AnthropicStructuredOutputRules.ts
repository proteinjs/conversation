import { getModelCapabilities } from '@ai-sdk/anthropic/internal';

/** The structured output mode the house names on an Anthropic request (`providerOptions.anthropic.structuredOutputMode`). */
export type AnthropicStructuredOutputMode = 'outputFormat' | 'jsonTool';

/** What the house measures on a response schema — counted the way the adapter and the grammar compiler count. */
export type StructuredOutputSchemaStats = {
  /**
   * Properties at EVERY depth not listed in their owning object's `required` — through `$ref` (per use:
   * the grammar expands a definition wherever it is referenced), `items`, `anyOf` / `oneOf` / `allOf`,
   * `properties` and the rest of the subschema keywords.
   */
  optionalProperties: number;
  /** Schema nodes typed as a union — `anyOf` / `oneOf`, or a `type` array (`["string", "null"]`). */
  unionTypedParameters: number;
  /**
   * An object the grammar cannot close: `additionalProperties` true or a schema, `patternProperties`, or an
   * object type with no `properties` and `additionalProperties` not false (a map, a free-form record).
   */
  hasOpenObject: boolean;
  /** A local `$ref` that reaches itself — a recursive schema, which the grammar compiler does not take. */
  circular: boolean;
};

/**
 * THE STRUCTURED OUTPUT MODE IS THE HOUSE'S DECISION, NOT THE ADAPTER'S GUESS. An Anthropic
 * `generateObject` rides one of two roads: `output_config.format` (the grammar path — the API compiles
 * the JSON schema into a grammar and every token is constrained to it) or a `json` tool (the adapter's
 * fallback — the schema rides as a tool's `input_schema`, nothing is compiled, and the answer is the
 * tool call's input). `@ai-sdk/anthropic` chooses between them in its default mode, `auto`, by its OWN
 * conservatism: from 3.0.13x on (`supportsNativeStructuredOutputSchema`,
 * src/supports-native-structured-output-schema.ts — `MAX_OPTIONAL_PROPERTIES = 10`, read at 3.0.138 and
 * 3.0.139) it demotes any schema with more than TEN optional properties, or an open object, to the json
 * tool — silently, no warning on the stream. The API's DOCUMENTED limit is 24 ({@link DOCUMENTED}), and the
 * house budgets its schemas against the documented figures (a consumer's readiness checkpoint is reshaped
 * to 20 optionals on purpose): under `auto`, every house schema carrying 11–24 optionals left the grammar
 * path for the json tool the day the floor rose to ^3.0.138 — a vendor-floor behavior change nothing in
 * our code asked for.
 *
 * So the mode is SAID, from the documented limits, on every object request — never 'auto', the house never
 * guesses ({@link mode}):
 * - `'outputFormat'` when the schema is natively representable within the published limits: no open object
 *   (the adapter's own definition, mirrored in {@link measure}), no recursive reference, at most 24
 *   optional properties at every depth counted the way the adapter counts, at most 16 union-typed
 *   parameters;
 * - `'jsonTool'` when it is not — said, so the adapter never re-decides it;
 * - nothing at all (the adapter's road) for a model the adapter's own table marks as lacking native
 *   structured output (`getModelCapabilities`, exported from `@ai-sdk/anthropic/internal` — Sonnet 4.0,
 *   Opus 4.0, the third generation and earlier): forcing `'outputFormat'` there would send
 *   `output_config.format` to a model that refuses it (HTTP 400). The table is the adapter's and is read
 *   from it, never copied.
 *
 * THE DOCUMENTED LIMITS ({@link DOCUMENTATION}; "Schema complexity limits", fetched 2026-11-06), per request
 * across every strict schema — the object road sends one JSON output schema and no strict tools, so the
 * schema's own count is the request's:
 * - "Optional parameters | 24 | Total optional parameters across all strict tool schemas and JSON output
 *   schemas. Each parameter not listed in `required` counts toward this limit."
 * - "Parameters with union types | 16 | Total parameters that use `anyOf` or type arrays (for example,
 *   `"type": ["string", "null"]`) across all strict schemas."
 * - "Strict tools per request | 20" — none ride the object road.
 *
 * One owner: this class is the only place the mode is decided; `Conversation.buildProviderOptions` reads it
 * on the generateObject road and nowhere else (the text and tool roads carry no response schema).
 */
export class AnthropicStructuredOutputRules {
  /** The API's published per-request ceilings for the grammar path ({@link DOCUMENTATION}). */
  static readonly DOCUMENTED: Readonly<{ optionalProperties: number; unionTypedParameters: number }> = {
    optionalProperties: 24,
    unionTypedParameters: 16,
  };

  /** Where the figures come from, and when they were read. */
  static readonly DOCUMENTATION: Readonly<{ url: string; section: string; fetched: string }> = {
    url: 'https://platform.claude.com/docs/en/build-with-claude/structured-outputs',
    section: 'Schema complexity limits',
    fetched: '2026-11-06',
  };

  /** The keywords whose value is one subschema (or a list of them), walked the way the adapter walks them. */
  private static readonly SUBSCHEMA_KEYWORDS = [
    'additionalProperties',
    'contains',
    'else',
    'if',
    'items',
    'not',
    'then',
  ];
  /** The keywords whose value is a list of subschemas. */
  private static readonly SUBSCHEMA_LIST_KEYWORDS = ['allOf', 'anyOf', 'oneOf', 'prefixItems'];
  /** The keywords whose value is a map of subschemas. */
  private static readonly SUBSCHEMA_MAP_KEYWORDS = ['dependentSchemas', 'patternProperties', 'properties'];

  /**
   * The mode the house says for this model and response schema: `'outputFormat'` when the schema is
   * {@link representable}, `'jsonTool'` when it is not — or undefined (nothing said; the adapter's own road)
   * for a model the adapter's table marks as lacking native structured output.
   */
  static mode(modelId: string, schema: unknown): AnthropicStructuredOutputMode | undefined {
    if (!getModelCapabilities(AnthropicStructuredOutputRules.bare(modelId)).supportsStructuredOutput) {
      return undefined;
    }
    return AnthropicStructuredOutputRules.representable(schema) ? 'outputFormat' : 'jsonTool';
  }

  /** Whether the grammar path takes this schema within the documented limits: closed objects, no recursion, the counts under the ceilings. */
  static representable(schema: unknown): boolean {
    const stats = AnthropicStructuredOutputRules.measure(schema);
    return (
      !stats.circular &&
      !stats.hasOpenObject &&
      stats.optionalProperties <= AnthropicStructuredOutputRules.DOCUMENTED.optionalProperties &&
      stats.unionTypedParameters <= AnthropicStructuredOutputRules.DOCUMENTED.unionTypedParameters
    );
  }

  /** The schema's counts, the way the adapter counts them (every subschema keyword, `$ref` per use, memoized per definition). */
  static measure(schema: unknown): StructuredOutputSchemaStats {
    return AnthropicStructuredOutputRules.stats(schema, schema, new Map(), new Set());
  }

  // ─── helpers ──────────────────────────────────────────────────────────────

  private static stats(
    node: unknown,
    root: unknown,
    memo: Map<string, StructuredOutputSchemaStats>,
    resolving: Set<string>
  ): StructuredOutputSchemaStats {
    const stats: StructuredOutputSchemaStats = {
      optionalProperties: 0,
      unionTypedParameters: 0,
      hasOpenObject: false,
      circular: false,
    };
    if (!AnthropicStructuredOutputRules.isRecord(node)) {
      return stats;
    }
    const add = (child: StructuredOutputSchemaStats) => {
      stats.optionalProperties += child.optionalProperties;
      stats.unionTypedParameters += child.unionTypedParameters;
      stats.hasOpenObject = stats.hasOpenObject || child.hasOpenObject;
      stats.circular = stats.circular || child.circular;
    };
    const walk = (child: unknown) => add(AnthropicStructuredOutputRules.stats(child, root, memo, resolving));

    if (typeof node.$ref === 'string' && node.$ref.startsWith('#')) {
      if (resolving.has(node.$ref)) {
        stats.circular = true;
      } else {
        let resolved = memo.get(node.$ref);
        if (resolved === undefined) {
          resolving.add(node.$ref);
          resolved = AnthropicStructuredOutputRules.stats(
            AnthropicStructuredOutputRules.resolvePointer(root, node.$ref),
            root,
            memo,
            resolving
          );
          resolving.delete(node.$ref);
          memo.set(node.$ref, resolved);
        }
        add(resolved);
      }
    }
    if (AnthropicStructuredOutputRules.isOpenObject(node)) {
      stats.hasOpenObject = true;
    }
    if (Array.isArray(node.type) || Array.isArray(node.anyOf) || Array.isArray(node.oneOf)) {
      stats.unionTypedParameters += 1;
    }
    if (AnthropicStructuredOutputRules.isRecord(node.properties)) {
      const required = new Set(Array.isArray(node.required) ? node.required.map(String) : []);
      stats.optionalProperties += Object.keys(node.properties).filter((name) => !required.has(name)).length;
    }
    for (const keyword of AnthropicStructuredOutputRules.SUBSCHEMA_KEYWORDS) {
      const value = node[keyword];
      if (Array.isArray(value)) {
        value.forEach(walk);
      } else {
        walk(value);
      }
    }
    for (const keyword of AnthropicStructuredOutputRules.SUBSCHEMA_LIST_KEYWORDS) {
      const value = node[keyword];
      if (Array.isArray(value)) {
        value.forEach(walk);
      }
    }
    for (const keyword of AnthropicStructuredOutputRules.SUBSCHEMA_MAP_KEYWORDS) {
      const value = node[keyword];
      if (AnthropicStructuredOutputRules.isRecord(value)) {
        Object.values(value).forEach(walk);
      }
    }
    return stats;
  }

  /** The adapter's definition of an object the grammar cannot close. */
  private static isOpenObject(node: Record<string, unknown>): boolean {
    const { additionalProperties, patternProperties, properties, type } = node;
    if (additionalProperties === true || AnthropicStructuredOutputRules.isRecord(additionalProperties)) {
      return true;
    }
    if (AnthropicStructuredOutputRules.isRecord(patternProperties) && Object.keys(patternProperties).length > 0) {
      return true;
    }
    const isObjectType = type === 'object' || (Array.isArray(type) && type.includes('object'));
    return isObjectType && properties == null && additionalProperties !== false;
  }

  /** A local JSON pointer (`#/$defs/name`, `#/definitions/name`) resolved against the root; undefined when it names nothing. */
  private static resolvePointer(root: unknown, pointer: string): unknown {
    if (pointer === '#') {
      return root;
    }
    if (!pointer.startsWith('#/')) {
      return undefined;
    }
    let decoded: string;
    try {
      decoded = decodeURIComponent(pointer.slice(2));
    } catch {
      return undefined;
    }
    let current: unknown = root;
    for (const rawSegment of decoded.split('/')) {
      const segment = rawSegment.replace(/~1/g, '/').replace(/~0/g, '~');
      if (Array.isArray(current)) {
        current = current[Number(segment)];
      } else if (AnthropicStructuredOutputRules.isRecord(current)) {
        current = current[segment];
      } else {
        return undefined;
      }
    }
    return current;
  }

  /** The id after any `provider:` prefix — what the adapter's table is asked about. */
  private static bare(modelId: string): string {
    const raw = String(modelId ?? '').trim();
    const colon = raw.indexOf(':');
    return colon > 0 ? raw.slice(colon + 1) : raw;
  }

  private static isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }
}
