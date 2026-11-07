import { getModelCapabilities } from '@ai-sdk/anthropic/internal';
import { Logger } from '@proteinjs/logger';

/** The structured output mode the house names on an Anthropic request (`providerOptions.anthropic.structuredOutputMode`). */
export type AnthropicStructuredOutputMode = 'outputFormat' | 'jsonTool';

/**
 * What the house measures on a structured-output schema — ONE walk ({@link AnthropicStructuredOutputRules.measure}),
 * counted the way the adapter and the grammar compiler count, and read by every door that budgets a schema
 * (flow-server's StructuredOutputSchemaBudget reads it; it walks nothing of its own).
 */
export type StructuredOutputSchemaStats = {
  /**
   * Properties at EVERY depth not listed in their owning object's `required` — through `$ref` (per use:
   * the grammar expands a definition wherever it is referenced), `items`, `anyOf` / `oneOf` / `allOf`,
   * `prefixItems`, `properties` and the rest of the subschema keywords. The documented limit's unit.
   */
  optionalProperties: number;
  /** Schema nodes typed as a union — `anyOf` / `oneOf`, or a `type` array (`["string", "null"]`). */
  unionTypedParameters: number;
  /**
   * An object the grammar cannot close: `additionalProperties` true or a schema, `patternProperties`, or an
   * object type with no `properties` and `additionalProperties` not false (a map, a free-form record).
   */
  hasOpenObject: boolean;
  /**
   * Where each open object sits, as a JSON pointer from the root (`#/properties/byName`); an object inside a
   * definition is named once, at the definition (`#/$defs/item/properties/tags`).
   */
  openObjects: string[];
  /** A local `$ref` that reaches itself — a recursive schema, which the grammar compiler does not take. */
  circular: boolean;
  /**
   * The root level's properties not listed in their `required` — the root object's own, and those of the
   * subschemas describing the root alongside it (a union's alternatives, `allOf`, a `$ref`). Information; the
   * limit's unit is every depth.
   */
  topLevelOptionalProperties: number;
  /**
   * The longest chain of objects that each carry an optional property — through properties, array items and
   * references; the root is level 1 and a fully required object breaks the chain. Information here (the doc:
   * "deeply nested objects with optional fields compound the complexity", no published figure); a consumer
   * budgets it.
   */
  optionalBearingNesting: number;
  /** The deepest object schema (root = 1), through properties and array items. */
  objectNesting: number;
  /** Property declarations at every depth (a definition's per use). */
  properties: number;
  /** Enum values at every depth (a definition's per use). */
  enumValues: number;
};

/** The figures a verdict is read against: the documented ones by default, or a consumer's tighter budget. */
export type StructuredOutputCeilings = { optionalProperties: number; unionTypedParameters: number };

/** One limit a schema crosses, named. */
export type StructuredOutputLimitCrossed = {
  limit: 'optionalProperties' | 'unionTypedParameters' | 'openObject' | 'circularReference';
  /** The line that names it: the figure, the ceiling and the API's own number — or where the object sits. */
  text: string;
};

/** The rule read over a schema: its measure, and every limit it crosses. */
export type StructuredOutputVerdict = {
  stats: StructuredOutputSchemaStats;
  /** Every limit crossed, named; empty when the grammar path takes the schema. */
  crossed: StructuredOutputLimitCrossed[];
  representable: boolean;
};

/** One subtree's counts, before the root's own figures are read off it. */
type SchemaTally = {
  optionalProperties: number;
  /** The optional properties at this node's own level: its own, and its alongside subschemas'. */
  levelOptionalProperties: number;
  unionTypedParameters: number;
  openObjects: string[];
  circular: boolean;
  properties: number;
  enumValues: number;
  /** The subtree's object depth, its own node at 1 (0 when it holds no object). */
  objectHeight: number;
  /** The run of optional-bearing objects that starts at this node's level — what an ancestor's chain continues. */
  chainTop: number;
  /** The longest run anywhere in the subtree. */
  chainBest: number;
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
 * on the generateObject road and nowhere else (the text and tool roads carry no response schema). It is
 * also the house's ONE counter of the table: {@link measure} is the only walk of a structured-output schema
 * and {@link verdict} the only reading of it against the limits — flow-server's StructuredOutputSchemaBudget
 * reads both (at its own tighter figures) instead of carrying a second walk, which disagreed with this one
 * on `$defs`/`$ref` and `allOf` schemas (row 3254's probe: 26 here, 10 there).
 *
 * A schema the grammar path does not take is SAID, once, at the owner ({@link mode}): one log line naming
 * the schema (its `title`, else its top-level property names), the model and every limit crossed — once per
 * schema identity per process, the identity being that name and those limits (a process-wide set of keys,
 * bounded by the house's distinct schemas; a schema whose verdict changes is said again).
 */
export class AnthropicStructuredOutputRules {
  /** The API's published per-request ceilings for the grammar path ({@link DOCUMENTATION}). */
  static readonly DOCUMENTED: Readonly<
    StructuredOutputCeilings & {
      strictToolsPerRequest: number;
    }
  > = {
    optionalProperties: 24,
    unionTypedParameters: 16,
    strictToolsPerRequest: 20,
  };

  /** Where the figures come from, and when they were read. */
  static readonly DOCUMENTATION: Readonly<{ url: string; section: string; fetched: string }> = {
    url: 'https://platform.claude.com/docs/en/build-with-claude/structured-outputs',
    section: 'Schema complexity limits',
    fetched: '2026-11-06',
  };

  /**
   * The keywords walked, the way the adapter walks them (every subschema keyword; `$defs` only through a
   * `$ref`), grouped by where the subschema sits. MEMBERS: the object's own members, one level down — an
   * optional-bearing chain continues through them.
   */
  private static readonly MEMBER_MAP_KEYWORDS = ['patternProperties', 'properties'];
  private static readonly MEMBER_KEYWORDS = ['additionalProperties'];
  /** ITEMS: an array's elements, one level down — the chain passes through the array to them. */
  private static readonly ITEM_KEYWORDS = ['contains', 'items'];
  private static readonly ITEM_LIST_KEYWORDS = ['prefixItems'];
  /** ALONGSIDE: subschemas that describe the same value, at the same level. */
  private static readonly ALONGSIDE_KEYWORDS = ['else', 'if', 'not', 'then'];
  private static readonly ALONGSIDE_LIST_KEYWORDS = ['allOf', 'anyOf', 'oneOf'];
  private static readonly ALONGSIDE_MAP_KEYWORDS = ['dependentSchemas'];

  /** Where a schema off the grammar path is said (the house logger). */
  private static logger = new Logger({ name: 'AnthropicStructuredOutputRules' });
  /** The demotions already said in this process — keyed by the schema's name and the limits it crosses. */
  private static said = new Set<string>();

  /**
   * The mode the house says for this model and response schema: `'outputFormat'` when the schema is
   * {@link representable}, `'jsonTool'` when it is not — or undefined (nothing said; the adapter's own road)
   * for a model the adapter's table marks as lacking native structured output.
   */
  static mode(modelId: string, schema: unknown): AnthropicStructuredOutputMode | undefined {
    if (!getModelCapabilities(AnthropicStructuredOutputRules.bare(modelId)).supportsStructuredOutput) {
      return undefined;
    }
    const verdict = AnthropicStructuredOutputRules.verdict(schema);
    if (verdict.representable) {
      return 'outputFormat';
    }
    AnthropicStructuredOutputRules.sayOffTheGrammarPath(modelId, schema, verdict);
    return 'jsonTool';
  }

  /** Whether the grammar path takes this schema within the documented limits: closed objects, no recursion, the counts under the ceilings. */
  static representable(schema: unknown): boolean {
    return AnthropicStructuredOutputRules.verdict(schema).representable;
  }

  /**
   * THE RULE: the schema's measure read against the ceilings — the documented ones unless a consumer hands
   * its own tighter budget — with every limit it crosses named. An open object and a recursive reference
   * cross whatever the figures: the grammar path cannot take either.
   */
  static verdict(
    schema: unknown,
    ceilings: StructuredOutputCeilings = AnthropicStructuredOutputRules.DOCUMENTED
  ): StructuredOutputVerdict {
    const stats = AnthropicStructuredOutputRules.measure(schema);
    const documented = AnthropicStructuredOutputRules.DOCUMENTED;
    const crossed: StructuredOutputLimitCrossed[] = [];
    if (stats.optionalProperties > ceilings.optionalProperties) {
      crossed.push({
        limit: 'optionalProperties',
        text: `optional parameters at any depth ${stats.optionalProperties} > ${ceilings.optionalProperties} (API limit ${documented.optionalProperties} per request across every strict schema)`,
      });
    }
    if (stats.unionTypedParameters > ceilings.unionTypedParameters) {
      crossed.push({
        limit: 'unionTypedParameters',
        text: `union-typed parameters ${stats.unionTypedParameters} > ${ceilings.unionTypedParameters} (API limit ${documented.unionTypedParameters})`,
      });
    }
    if (stats.openObjects.length > 0) {
      crossed.push({
        limit: 'openObject',
        text: `an open object the grammar cannot close at ${stats.openObjects.join(', ')}`,
      });
    }
    if (stats.circular) {
      crossed.push({ limit: 'circularReference', text: 'a recursive reference (a $ref that reaches itself)' });
    }
    return { stats, crossed, representable: crossed.length === 0 };
  }

  /** THE MEASURE: the schema's counts, the way the adapter counts them (every subschema keyword, `$ref` per use, memoized per definition). */
  static measure(schema: unknown): StructuredOutputSchemaStats {
    const tally = AnthropicStructuredOutputRules.tally(schema, '#', schema, new Map(), new Set());
    return {
      optionalProperties: tally.optionalProperties,
      unionTypedParameters: tally.unionTypedParameters,
      hasOpenObject: tally.openObjects.length > 0,
      openObjects: tally.openObjects,
      circular: tally.circular,
      topLevelOptionalProperties: tally.levelOptionalProperties,
      optionalBearingNesting: tally.chainBest,
      objectNesting: tally.objectHeight,
      properties: tally.properties,
      enumValues: tally.enumValues,
    };
  }

  // ─── helpers ──────────────────────────────────────────────────────────────

  /** One line, once per schema identity per process: the schema left the grammar path, and why. */
  private static sayOffTheGrammarPath(modelId: string, schema: unknown, verdict: StructuredOutputVerdict): void {
    const name = AnthropicStructuredOutputRules.nameOf(schema);
    const crossed = verdict.crossed.map((limit) => limit.text);
    const key = JSON.stringify([name, crossed]);
    if (AnthropicStructuredOutputRules.said.has(key)) {
      return;
    }
    AnthropicStructuredOutputRules.said.add(key);
    AnthropicStructuredOutputRules.logger.info({
      message: `Structured output for ${name} rides the json tool, not the grammar path (output_config.format): ${crossed.join('; ')}`,
      obj: { model: modelId, schema: name, crossed },
    });
  }

  private static tally(
    node: unknown,
    pointer: string,
    root: unknown,
    memo: Map<string, SchemaTally>,
    resolving: Set<string>
  ): SchemaTally {
    const tally: SchemaTally = {
      optionalProperties: 0,
      levelOptionalProperties: 0,
      unionTypedParameters: 0,
      openObjects: [],
      circular: false,
      properties: 0,
      enumValues: 0,
      objectHeight: 0,
      chainTop: 0,
      chainBest: 0,
    };
    if (!AnthropicStructuredOutputRules.isRecord(node)) {
      return tally;
    }
    const Rules = AnthropicStructuredOutputRules;
    const walk = (child: unknown, at: string) => Rules.tally(child, at, root, memo, resolving);
    const members: SchemaTally[] = [];
    const items: SchemaTally[] = [];
    const alongside: SchemaTally[] = [];

    if (typeof node.$ref === 'string' && node.$ref.startsWith('#')) {
      if (resolving.has(node.$ref)) {
        tally.circular = true;
      } else {
        let resolved = memo.get(node.$ref);
        if (resolved === undefined) {
          resolving.add(node.$ref);
          resolved = walk(Rules.resolvePointer(root, node.$ref), node.$ref);
          resolving.delete(node.$ref);
          memo.set(node.$ref, resolved);
        }
        alongside.push(resolved);
      }
    }
    if (Rules.isOpenObject(node)) {
      tally.openObjects.push(pointer);
    }
    if (Array.isArray(node.type) || Array.isArray(node.anyOf) || Array.isArray(node.oneOf)) {
      tally.unionTypedParameters += 1;
    }
    if (Array.isArray(node.enum)) {
      tally.enumValues += node.enum.length;
    }
    const optional = Rules.optionalOf(node);
    tally.optionalProperties += optional;
    tally.properties += Rules.isRecord(node.properties) ? Object.keys(node.properties).length : 0;

    for (const keyword of Rules.MEMBER_MAP_KEYWORDS) {
      Rules.eachOfMap(node[keyword], (child, name) => members.push(walk(child, `${pointer}/${keyword}/${name}`)));
    }
    for (const keyword of Rules.MEMBER_KEYWORDS) {
      members.push(walk(node[keyword], `${pointer}/${keyword}`));
    }
    for (const keyword of Rules.ITEM_KEYWORDS) {
      Rules.eachOf(node[keyword], `${pointer}/${keyword}`, (child, at) => items.push(walk(child, at)));
    }
    for (const keyword of Rules.ITEM_LIST_KEYWORDS) {
      Rules.eachOfList(node[keyword], `${pointer}/${keyword}`, (child, at) => items.push(walk(child, at)));
    }
    for (const keyword of Rules.ALONGSIDE_KEYWORDS) {
      alongside.push(walk(node[keyword], `${pointer}/${keyword}`));
    }
    for (const keyword of Rules.ALONGSIDE_LIST_KEYWORDS) {
      Rules.eachOfList(node[keyword], `${pointer}/${keyword}`, (child, at) => alongside.push(walk(child, at)));
    }
    for (const keyword of Rules.ALONGSIDE_MAP_KEYWORDS) {
      Rules.eachOfMap(node[keyword], (child, name) => alongside.push(walk(child, `${pointer}/${keyword}/${name}`)));
    }

    const below = members.concat(items);
    const children = below.concat(alongside);
    for (const child of children) {
      tally.optionalProperties += child.optionalProperties;
      tally.unionTypedParameters += child.unionTypedParameters;
      tally.circular = tally.circular || child.circular;
      tally.properties += child.properties;
      tally.enumValues += child.enumValues;
      for (const at of child.openObjects) {
        if (tally.openObjects.indexOf(at) < 0) {
          tally.openObjects.push(at);
        }
      }
    }
    tally.levelOptionalProperties = optional + alongside.reduce((sum, child) => sum + child.levelOptionalProperties, 0);
    const isObject = Rules.isObjectNode(node);
    // Depth: an object is 1; a member or an item one level below its parent; an alongside subschema at its level.
    tally.objectHeight = Math.max(
      isObject ? 1 : 0,
      ...below.map((child) => (child.objectHeight > 0 ? child.objectHeight + 1 : 0)),
      ...alongside.map((child) => child.objectHeight)
    );
    // The chain: an object carrying an optional property extends it through its members; a fully required
    // object breaks it; an array's items and an alongside subschema continue the chain the node was reached on.
    const own = isObject && optional > 0 ? 1 + Math.max(0, ...members.map((child) => child.chainTop)) : 0;
    tally.chainTop = Math.max(own, ...items.concat(alongside).map((child) => child.chainTop));
    tally.chainBest = Math.max(tally.chainTop, ...children.map((child) => child.chainBest));
    return tally;
  }

  /** The schema's name on a line: its `title`, else its top-level property names. */
  private static nameOf(schema: unknown): string {
    const record = AnthropicStructuredOutputRules.isRecord(schema) ? schema : {};
    if (typeof record.title === 'string' && record.title.trim()) {
      return `schema "${record.title.trim()}"`;
    }
    const names = AnthropicStructuredOutputRules.isRecord(record.properties) ? Object.keys(record.properties) : [];
    return names.length
      ? `the schema with ${names.map((name) => `"${name}"`).join(', ')}`
      : 'a schema with no top-level properties';
  }

  /** An object node's own properties not listed in its `required`. */
  private static optionalOf(node: unknown): number {
    if (!AnthropicStructuredOutputRules.isRecord(node) || !AnthropicStructuredOutputRules.isRecord(node.properties)) {
      return 0;
    }
    const required = new Set(Array.isArray(node.required) ? node.required.map(String) : []);
    return Object.keys(node.properties).filter((name) => !required.has(name)).length;
  }

  /** An object schema: typed `object` (alone or in a type array), or carrying `properties`. */
  private static isObjectNode(node: Record<string, unknown>): boolean {
    const { properties, type } = node;
    return (
      type === 'object' ||
      (Array.isArray(type) && type.includes('object')) ||
      AnthropicStructuredOutputRules.isRecord(properties)
    );
  }

  /** One subschema or a list of them (`items` as a tuple), each with its pointer. */
  private static eachOf(value: unknown, at: string, visit: (child: unknown, at: string) => void): void {
    if (Array.isArray(value)) {
      AnthropicStructuredOutputRules.eachOfList(value, at, visit);
    } else {
      visit(value, at);
    }
  }

  private static eachOfList(value: unknown, at: string, visit: (child: unknown, at: string) => void): void {
    if (Array.isArray(value)) {
      value.forEach((child, index) => visit(child, `${at}/${index}`));
    }
  }

  /** A map of subschemas, each with its name escaped as a JSON pointer segment. */
  private static eachOfMap(value: unknown, visit: (child: unknown, name: string) => void): void {
    if (AnthropicStructuredOutputRules.isRecord(value)) {
      const map = value;
      Object.keys(map).forEach((name) => visit(map[name], name.replace(/~/g, '~0').replace(/\//g, '~1')));
    }
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
