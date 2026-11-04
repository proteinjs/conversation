import { AnthropicModelRules } from '../../src/AnthropicModelRules';

/**
 * THE PER-MODEL RULES FOR ANTHROPIC, read from the vendor's versioning grammar — never a list of
 * ids. The off switch per family is the vendor's word (the model documentation and
 * `@ai-sdk/anthropic`'s own model catalog, read 2026-11-04): `disabled` where the model takes it,
 * `between_tools` on Sonnet from 5.5 on, and none at all on Fable, Mythos and Opus from 5.5 on —
 * thinking is always on there, and the least those models take is effort 'low'.
 */
describe('AnthropicModelRules', () => {
  test.each([
    ['claude-sonnet-5-5', 'sonnet'],
    ['claude-opus-4-8', 'opus'],
    ['claude-haiku-4-5-20251001', 'haiku'],
    ['claude-fable-5-1', 'fable'],
    ['claude-mythos-5-1', 'mythos'],
    ['anthropic:claude-sonnet-5', 'sonnet'],
    ['claude-3-5-haiku-20241022', 'haiku'],
    ['claude-3-opus-20240229', 'opus'],
    ['claude-2.1', undefined],
    ['gpt-5.5', undefined],
  ] as Array<[string, string | undefined]>)('tier(%s) → %s', (id, expected) => {
    expect(AnthropicModelRules.tier(id)).toBe(expected);
  });

  test.each([
    ['claude-sonnet-5-5', 5.5],
    ['claude-sonnet-5', 5],
    ['claude-opus-4-8', 4.8],
    ['claude-opus-4-1-20250805', 4.1],
    ['claude-haiku-4-5-20251001', 4.5],
    ['claude-sonnet-4-20250514', 4],
    ['claude-fable-5-1', 5.1],
    ['claude-3-5-haiku-20241022', 3.5],
    ['claude-3-7-sonnet-20250219', 3.7],
    ['claude-3-opus-20240229', 3],
    ['claude-2.1', 2.1],
    ['anthropic:claude-opus-5-5', 5.5],
    ['gpt-5.5', undefined],
  ] as Array<[string, number | undefined]>)('generation(%s) → %s', (id, expected) => {
    expect(AnthropicModelRules.generation(id)).toBe(expected);
  });

  test.each([
    ['claude-sonnet-5', { type: 'disabled' }],
    ['claude-opus-5', { type: 'disabled' }],
    ['claude-haiku-5-5', { type: 'disabled' }],
    ['claude-haiku-4-5', { type: 'disabled' }],
    ['claude-opus-4-8', { type: 'disabled' }],
    ['claude-sonnet-4-6', { type: 'disabled' }],
    ['claude-3-5-haiku-20241022', { type: 'disabled' }],
    ['claude-sonnet-5-5', { type: 'between_tools' }],
    ['claude-sonnet-6', { type: 'between_tools' }],
    ['claude-opus-5-5', undefined],
    ['claude-opus-6', undefined],
    ['claude-fable-5-1', undefined],
    ['claude-fable-5', undefined],
    ['claude-mythos-5-1', undefined],
    ['anthropic:claude-opus-5-5', undefined],
  ] as Array<[string, { type: string } | undefined]>)('thinkingOff(%s) → %s', (id, expected) => {
    expect(AnthropicModelRules.thinkingOff(id)).toEqual(expected);
  });
});
