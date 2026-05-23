import { Conversation, ReasoningEffort } from '../../src/Conversation';

/**
 * Unit tests for the provider-options builder.
 *
 * No API calls — this exercises the pure mapping from
 * `(provider, reasoningEffort, modelString)` to the providerOptions object
 * passed to the AI SDK.
 *
 * The Anthropic branch is what these tests primarily protect: Opus 4.7
 * changed `thinking.display`'s default to `'omitted'`, so adaptive thinking
 * now requires `display: 'summarized'` on the request to get reasoning
 * text back on the stream. A regression here silently hides reasoning in
 * the UI without throwing.
 */

type AnthropicProviderOptions = {
  thinking?: { type: 'adaptive' | 'enabled' | 'disabled'; display?: 'summarized' | 'omitted'; budgetTokens?: number };
  effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
};

const buildAnthropic = (effort: ReasoningEffort | undefined, modelString: string): AnthropicProviderOptions => {
  const conv = new Conversation({ name: 'test-providerOptions' });
  // buildProviderOptions is private — testing it directly via cast keeps the
  // production API surface narrow.
  const opts = (conv as any).buildProviderOptions('anthropic', { reasoningEffort: effort }, modelString);
  return opts.anthropic as AnthropicProviderOptions;
};

describe('Conversation.buildProviderOptions (anthropic)', () => {
  describe('adaptive thinking (Opus 4.7 / Sonnet 4.6)', () => {
    test.each([
      ['claude-opus-4-7', 'auto'],
      ['claude-sonnet-4-6', 'auto'],
    ] as Array<[string, ReasoningEffort]>)(
      'sets thinking: { type: adaptive, display: summarized } for model=%s effort=%s',
      (model, effort) => {
        const anthropic = buildAnthropic(effort, model);
        expect(anthropic.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
        expect(anthropic.effort).toBeUndefined();
      }
    );

    test.each([['low'], ['medium'], ['high'], ['xhigh'], ['max']] as Array<[ReasoningEffort]>)(
      'sets thinking: { type: adaptive, display: summarized } and effort: %s on Opus 4.7',
      (effort) => {
        const anthropic = buildAnthropic(effort, 'claude-opus-4-7');
        expect(anthropic.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
        expect(anthropic.effort).toBe(effort);
      }
    );

    test('omits thinking entirely when effort is "none"', () => {
      const anthropic = buildAnthropic('none', 'claude-opus-4-7');
      expect(anthropic.thinking).toBeUndefined();
      expect(anthropic.effort).toBeUndefined();
    });
  });

  describe('extended thinking (Haiku 4.5)', () => {
    test('auto effort → enabled with default budget, no display field', () => {
      const anthropic = buildAnthropic('auto', 'claude-haiku-4-5');
      expect(anthropic.thinking).toEqual({ type: 'enabled', budgetTokens: 10000 });
      expect(anthropic.effort).toBeUndefined();
    });

    test.each([
      ['low', 5000],
      ['medium', 10000],
      ['high', 50000],
    ] as Array<[ReasoningEffort, number]>)('effort=%s maps to budgetTokens=%i, no display field', (effort, budget) => {
      const anthropic = buildAnthropic(effort, 'claude-haiku-4-5');
      expect(anthropic.thinking).toEqual({ type: 'enabled', budgetTokens: budget });
      expect(anthropic.effort).toBeUndefined();
    });

    test('omits thinking entirely when effort is "none"', () => {
      const anthropic = buildAnthropic('none', 'claude-haiku-4-5');
      expect(anthropic.thinking).toBeUndefined();
    });
  });
});
