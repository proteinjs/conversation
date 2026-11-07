import { Logger, type Log } from '@proteinjs/logger';
import { AnthropicStructuredOutputRules } from '../../src/AnthropicStructuredOutputRules';

/**
 * THE STRUCTURED OUTPUT MODE IS THE HOUSE'S DECISION — the rule's counting and its edges, pure. The
 * ceilings are the API's documented ones (24 optional parameters at every depth, 16 union-typed
 * parameters; "Schema complexity limits", fetched 2026-11-06), never the adapter's own 10; an open object or
 * a recursive reference leaves the grammar path; a model the adapter's table marks as lacking native
 * structured output gets no verdict at all (the adapter's road). The wire — what the request body carries
 * on each road — is pinned in conversation.buildProviderOptions.test.ts.
 *
 * ONE COUNTER: the measure here is the house's only walk of a structured-output schema — flow-server's
 * StructuredOutputSchemaBudget reads it (and the verdict, at its own tighter figures) instead of walking
 * the schema itself. The two counters disagreed on a $defs/$ref schema (26 here, 10 there) and on an
 * allOf one (27, 0) — row 3254's probe, whose shapes are pinned below.
 */

/** An object with `n` optional string properties (and one required, so the root is never empty). */
const optionals = (n: number, prefix = 'f'): Record<string, unknown> => ({
  type: 'object',
  properties: Object.fromEntries([
    ['id', { type: 'string' }],
    ...Array.from({ length: n }, (_, i) => [`${prefix}${i}`, { type: 'string' }]),
  ]),
  required: ['id'],
  additionalProperties: false,
});

/** An object with `n` REQUIRED nullable properties — union-typed, none optional. */
const unions = (n: number): Record<string, unknown> => {
  const names = Array.from({ length: n }, (_, i) => `u${i}`);
  return {
    type: 'object',
    properties: Object.fromEntries(names.map((name) => [name, { type: ['string', 'null'] }])),
    required: names,
    additionalProperties: false,
  };
};

describe('AnthropicStructuredOutputRules — the documented ceilings, not the adapter’s 10', () => {
  test('24 optional properties at the root are representable; the 25th is not', () => {
    expect(AnthropicStructuredOutputRules.representable(optionals(24))).toBe(true);
    expect(AnthropicStructuredOutputRules.representable(optionals(25))).toBe(false);
  });

  test('optionals count at every depth — through items, nested objects and anyOf — toward the same 24', () => {
    // 10 at the root + 8 in an array's items + 6 inside a nullable object = 24; one more inside tips it.
    const deep = (extra: number) => ({
      type: 'object',
      properties: {
        ...(optionals(10).properties as Record<string, unknown>),
        rows: { type: 'array', items: optionals(8, 'r') },
        maybe: { anyOf: [optionals(6 + extra, 'm'), { type: 'null' }] },
      },
      required: ['id', 'rows', 'maybe'],
      additionalProperties: false,
    });
    expect(AnthropicStructuredOutputRules.measure(deep(0)).optionalProperties).toBe(24);
    expect(AnthropicStructuredOutputRules.representable(deep(0))).toBe(true);
    expect(AnthropicStructuredOutputRules.representable(deep(1))).toBe(false);
  });

  test('16 union-typed parameters are representable; the 17th is not — and none of them is optional', () => {
    expect(AnthropicStructuredOutputRules.measure(unions(16))).toMatchObject({
      unionTypedParameters: 16,
      optionalProperties: 0,
    });
    expect(AnthropicStructuredOutputRules.representable(unions(16))).toBe(true);
    expect(AnthropicStructuredOutputRules.representable(unions(17))).toBe(false);
  });

  test('a definition is counted per use — the grammar expands it wherever it is referenced', () => {
    const schema = {
      type: 'object',
      properties: { a: { $ref: '#/$defs/item' }, b: { $ref: '#/$defs/item' } },
      required: ['a', 'b'],
      additionalProperties: false,
      $defs: { item: optionals(3) },
    };
    expect(AnthropicStructuredOutputRules.measure(schema).optionalProperties).toBe(6);
  });

  test('a recursive reference leaves the grammar path', () => {
    const schema = {
      type: 'object',
      properties: { root: { $ref: '#/$defs/node' } },
      required: ['root'],
      additionalProperties: false,
      $defs: {
        node: {
          type: 'object',
          properties: { child: { $ref: '#/$defs/node' } },
          required: ['child'],
          additionalProperties: false,
        },
      },
    };
    expect(AnthropicStructuredOutputRules.measure(schema).circular).toBe(true);
    expect(AnthropicStructuredOutputRules.representable(schema)).toBe(false);
  });
});

describe('AnthropicStructuredOutputRules — an open object (the adapter’s definition) leaves the grammar path', () => {
  const withField = (field: Record<string, unknown>) => ({
    type: 'object',
    properties: { id: { type: 'string' }, field },
    required: ['id', 'field'],
    additionalProperties: false,
  });

  test.each([
    ['a map (additionalProperties as a schema)', { type: 'object', additionalProperties: { type: 'string' } }],
    [
      'additionalProperties: true',
      { type: 'object', properties: { x: { type: 'string' } }, additionalProperties: true },
    ],
    ['patternProperties', { type: 'object', patternProperties: { '^x': { type: 'string' } } }],
    ['an object type with no properties', { type: 'object' }],
  ] as Array<[string, Record<string, unknown>]>)('%s is open', (_name, field) => {
    expect(AnthropicStructuredOutputRules.measure(withField(field)).hasOpenObject).toBe(true);
    expect(AnthropicStructuredOutputRules.representable(withField(field))).toBe(false);
  });

  test.each([
    [
      'an object closed with additionalProperties: false and no properties',
      { type: 'object', additionalProperties: false },
    ],
    [
      'an object with properties and additionalProperties unsaid',
      { type: 'object', properties: { x: { type: 'string' } } },
    ],
  ] as Array<[string, Record<string, unknown>]>)('%s is closed', (_name, field) => {
    expect(AnthropicStructuredOutputRules.measure(withField(field)).hasOpenObject).toBe(false);
    expect(AnthropicStructuredOutputRules.representable(withField(field))).toBe(true);
  });
});

describe('AnthropicStructuredOutputRules.mode — said from the limits, deferred to the adapter for the models its table excludes', () => {
  test.each([
    ['claude-opus-5-5', optionals(20), 'outputFormat'],
    ['claude-fable-5-1', optionals(24), 'outputFormat'],
    ['anthropic:claude-sonnet-5-5', optionals(5), 'outputFormat'],
    ['claude-haiku-4-5', optionals(20), 'outputFormat'],
    ['claude-opus-5-5', optionals(25), 'jsonTool'],
    ['claude-opus-5-5', { type: 'object', additionalProperties: { type: 'number' } }, 'jsonTool'],
    // The adapter's own table marks these as lacking native structured output: nothing is said — its road.
    ['claude-sonnet-4-20250514', optionals(5), undefined],
    ['claude-opus-4-20250514', optionals(5), undefined],
    ['claude-3-haiku-20240307', optionals(5), undefined],
  ] as Array<[string, Record<string, unknown>, string | undefined]>)(
    'mode(%s, …) → %s',
    (modelId, schema, expected) => {
      expect(AnthropicStructuredOutputRules.mode(modelId, schema)).toBe(expected);
    }
  );
});

describe('AnthropicStructuredOutputRules.verdict — the one measure, the limit crossed named (row 3254’s probe)', () => {
  /** The probe's $defs/$ref shape: 10 optionals at the root + a definition of 8 optionals used twice. */
  const REFERENCED = {
    type: 'object',
    properties: {
      ...(optionals(10, 'r').properties as Record<string, unknown>),
      a: { $ref: '#/$defs/item' },
      b: { $ref: '#/$defs/item' },
    },
    required: ['id', 'a', 'b'],
    additionalProperties: false,
    $defs: { item: optionals(8, 'd') },
  };

  test('the $defs/$ref schema measures 26 (the definition per use) — past the documented 24, the limit named', () => {
    const verdict = AnthropicStructuredOutputRules.verdict(REFERENCED);
    expect(verdict.stats.optionalProperties).toBe(26);
    expect(verdict.representable).toBe(false);
    expect(verdict.crossed).toEqual([
      {
        limit: 'optionalProperties',
        text: expect.stringMatching(/^optional parameters at any depth 26 > 24 \(API limit 24 /),
      },
    ]);
  });

  test('optionals carried by allOf count — 15 + 12 = 27 — and a consumer’s tighter figures name its own line', () => {
    const allOf = {
      type: 'object',
      allOf: [optionals(15, 'x'), optionals(12, 'y')],
      properties: { id: { type: 'string' } },
      required: ['id'],
      additionalProperties: false,
    };
    expect(AnthropicStructuredOutputRules.verdict(allOf).crossed.map((c) => c.text)).toEqual([
      expect.stringMatching(/^optional parameters at any depth 27 > 24 /),
    ]);
    expect(
      AnthropicStructuredOutputRules.verdict(unions(9), { optionalProperties: 20, unionTypedParameters: 8 }).crossed
    ).toEqual([{ limit: 'unionTypedParameters', text: 'union-typed parameters 9 > 8 (API limit 16)' }]);
  });

  test('an open object (a z.record’s map) is named where it sits; a recursive reference is named too', () => {
    const map = {
      type: 'object',
      properties: { title: { type: 'string' }, byName: { type: 'object', additionalProperties: { type: 'string' } } },
      required: ['title', 'byName'],
      additionalProperties: false,
    };
    const verdict = AnthropicStructuredOutputRules.verdict(map);
    expect(verdict.stats.openObjects).toEqual(['#/properties/byName']);
    expect(verdict.crossed).toEqual([{ limit: 'openObject', text: expect.stringContaining('#/properties/byName') }]);
    const recursive = {
      type: 'object',
      properties: { root: { $ref: '#/$defs/node' } },
      required: ['root'],
      additionalProperties: false,
      $defs: { node: { type: 'object', properties: { child: { $ref: '#/$defs/node' } }, required: ['child'] } },
    };
    expect(AnthropicStructuredOutputRules.verdict(recursive).crossed.map((c) => c.limit)).toEqual([
      'circularReference',
    ]);
  });

  test('the structure a consumer budgets — optional-bearing objects in a chain, object depth — counted through items and references', () => {
    // root (optional) → questions[] (optional) → options[] (optional): three in a chain, objects five deep;
    // the same option object reached through a reference continues the chain at its use site.
    const option = {
      type: 'object',
      properties: { label: { type: 'string' }, note: { type: 'string' } },
      required: ['label'],
    };
    const shape = (options: unknown) => ({
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['go', 'stop'] },
        note: { type: 'string' },
        questions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              text: { type: 'string' },
              options: { type: 'array', items: options },
              hint: { type: 'string' },
            },
            required: ['text'],
          },
        },
      },
      required: ['action'],
      $defs: { option },
    });
    for (const schema of [shape(option), shape({ $ref: '#/$defs/option' })]) {
      expect(AnthropicStructuredOutputRules.measure(schema)).toMatchObject({
        topLevelOptionalProperties: 2,
        optionalProperties: 5,
        optionalBearingNesting: 3,
        objectNesting: 5,
        properties: 8,
        enumValues: 2,
      });
    }
    // A fully required object breaks the chain: the options alone are then one deep.
    const required = shape(option) as { properties: { questions: { items: { required: string[] } } } };
    required.properties.questions.items.required = ['text', 'options', 'hint'];
    expect(AnthropicStructuredOutputRules.measure(required).optionalBearingNesting).toBe(1);
    // The root level reads through the subschemas alongside it: a root union's alternatives are its top level.
    expect(
      AnthropicStructuredOutputRules.measure({ anyOf: [optionals(3, 'a'), optionals(2, 'b')] })
        .topLevelOptionalProperties
    ).toBe(5);
  });
});

describe('AnthropicStructuredOutputRules.mode — a schema off the grammar path is said ONCE, at the owner', () => {
  /** The owner's logger, reached by a typed cast (a private static) and pointed at a capturing writer. */
  type RulesInternals = { logger: Logger; said: Set<string> };
  const internals = AnthropicStructuredOutputRules as unknown as RulesInternals;
  let lines: Log[] = [];
  let original: Logger;

  beforeEach(() => {
    lines = [];
    original = internals.logger;
    internals.logger = new Logger({
      name: 'AnthropicStructuredOutputRules',
      logWriter: { write: (log: Log) => void lines.push(log) } as never,
    });
  });

  afterEach(() => {
    internals.logger = original;
  });

  test('jsonTool writes one line naming the schema and the limit crossed; the same schema again writes none', () => {
    const readiness = { ...optionals(25), title: 'PlanReadiness' };
    expect(AnthropicStructuredOutputRules.mode('claude-opus-5-5', readiness)).toBe('jsonTool');
    expect(AnthropicStructuredOutputRules.mode('claude-opus-5-5', { ...optionals(25), title: 'PlanReadiness' })).toBe(
      'jsonTool'
    );
    expect(lines).toHaveLength(1);
    expect(lines[0].message).toContain('"PlanReadiness"');
    expect(lines[0].message).toContain('optional parameters at any depth 25 > 24');
    expect(lines[0].obj).toMatchObject({ model: 'claude-opus-5-5', crossed: [expect.stringContaining('25 > 24')] });
  });

  test('outputFormat writes nothing; another schema off the path writes its own line', () => {
    expect(AnthropicStructuredOutputRules.mode('claude-opus-5-5', optionals(24))).toBe('outputFormat');
    expect(lines).toEqual([]);
    const map = {
      type: 'object',
      properties: { title: { type: 'string' }, byName: { type: 'object', additionalProperties: { type: 'string' } } },
      required: ['title', 'byName'],
      additionalProperties: false,
    };
    expect(AnthropicStructuredOutputRules.mode('claude-fable-5-1', map)).toBe('jsonTool');
    expect(lines).toHaveLength(1);
    expect(lines[0].message).toContain('"title", "byName"');
    expect(lines[0].message).toContain('#/properties/byName');
  });
});
