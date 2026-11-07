import { AnthropicStructuredOutputRules } from '../../src/AnthropicStructuredOutputRules';

/**
 * THE STRUCTURED OUTPUT MODE IS THE HOUSE'S DECISION — the rule's counting and its edges, pure. The
 * ceilings are the API's documented ones (24 optional parameters at every depth, 16 union-typed
 * parameters; "Schema complexity limits", fetched 2026-11-06), never the adapter's own 10; an open object or
 * a recursive reference leaves the grammar path; a model the adapter's table marks as lacking native
 * structured output gets no verdict at all (the adapter's road). The wire — what the request body carries
 * on each road — is pinned in conversation.buildProviderOptions.test.ts.
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
