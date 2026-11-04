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

  // THE FAMILY DECIDES 'auto' AND THE LEVELS TOO (one owner; the builder reads it for every branch): adaptive
  // thinking with the reasoning text shown, the level as `effort`, on every family — except Haiku through the
  // fourth generation, which takes budget thinking only (no adaptive: 'auto' a 10k budget, a level its budget).
  // Haiku from the fifth generation on is adaptive like the rest (5.5 refuses a budget — the vendor's table).
  test.each([
    ['claude-haiku-4-5', 'auto', true],
    ['claude-haiku-4-5-20251001', 'low', true],
    ['claude-3-5-haiku-20241022', 'high', true],
    ['claude-haiku-5-5', 'auto', false],
    ['claude-haiku-5', 'low', false],
    ['claude-sonnet-4-6', 'auto', false],
    ['claude-opus-5-5', 'max', false],
  ] as Array<[string, string, boolean]>)('budgetThinkingOnly(%s) at %s → %s', (id, _effort, expected) => {
    expect(AnthropicModelRules.budgetThinkingOnly(id)).toBe(expected);
  });

  const adaptive = { type: 'adaptive', display: 'summarized' };
  test.each([
    ['claude-haiku-4-5', 'auto', { thinking: { type: 'enabled', budgetTokens: 10000 } }],
    ['claude-haiku-4-5', 'low', { thinking: { type: 'enabled', budgetTokens: 5000 } }],
    ['claude-haiku-4-5', 'medium', { thinking: { type: 'enabled', budgetTokens: 10000 } }],
    ['claude-3-5-haiku-20241022', 'high', { thinking: { type: 'enabled', budgetTokens: 50000 } }],
    ['claude-haiku-4-5', 'max', { thinking: { type: 'enabled', budgetTokens: 10000 } }],
    ['claude-haiku-5-5', 'auto', { thinking: adaptive }],
    ['claude-haiku-5-5', 'low', { thinking: adaptive, effort: 'low' }],
    ['claude-haiku-5-5', 'high', { thinking: adaptive, effort: 'high' }],
    ['claude-sonnet-5-5', 'auto', { thinking: adaptive }],
    ['claude-sonnet-5-5', 'max', { thinking: adaptive, effort: 'max' }],
    ['claude-opus-4-8', 'medium', { thinking: adaptive, effort: 'medium' }],
    ['claude-opus-5-5', 'low', { thinking: adaptive, effort: 'low' }],
    ['claude-fable-5-1', 'xhigh', { thinking: adaptive, effort: 'xhigh' }],
    ['anthropic:claude-haiku-4-5', 'low', { thinking: { type: 'enabled', budgetTokens: 5000 } }],
  ] as Array<[string, 'auto' | 'low' | 'medium' | 'high' | 'xhigh' | 'max', Record<string, unknown>]>)(
    'thinkingOn(%s, %s) → %j',
    (id, effort, expected) => {
      expect(AnthropicModelRules.thinkingOn(id, effort)).toEqual(expected);
    }
  );

  // THE LEAST A CALLER MAY NAME — the fact a consumer's catalog reads from this one owner (its least-thinking level
  // and utterance effort for an Anthropic row): 'none' where the family has an off switch, 'low' where thinking is
  // always on.
  test.each([
    ['claude-sonnet-5', 'none'],
    ['claude-sonnet-5-5', 'none'],
    ['claude-haiku-5-5', 'none'],
    ['claude-haiku-4-5', 'none'],
    ['claude-opus-4-8', 'none'],
    ['claude-opus-5', 'none'],
    ['claude-opus-5-5', 'low'],
    ['claude-fable-5-1', 'low'],
    ['claude-mythos-5-1', 'low'],
    ['anthropic:claude-opus-5-5', 'low'],
  ] as Array<[string, 'none' | 'low']>)('leastEffort(%s) → %s', (id, expected) => {
    expect(AnthropicModelRules.leastEffort(id)).toBe(expected);
  });
});
