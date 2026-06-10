import { Conversation, ReasoningEffort } from '../../src/Conversation';

/**
 * Unit tests for the provider-options builder.
 *
 * No API calls — this exercises the pure mapping from
 * `(provider, reasoningEffort, modelString)` to the providerOptions object
 * passed to the AI SDK.
 *
 * These tests primarily protect each provider's "show reasoning text in
 * the UI" path, which is provider-specific and easy to break silently:
 *
 * - Anthropic: adaptive thinking defaults `thinking.display` to 'omitted',
 *   so it requires `display: 'summarized'` on the request to get reasoning
 *   text back on the stream.
 * - OpenAI: the Responses API only emits `reasoning-delta` chunks when
 *   `reasoningSummary` is set (default is no summary).
 *
 * A regression in either silently hides reasoning in the UI without
 * throwing.
 */

type AnthropicProviderOptions = {
  thinking?: { type: 'adaptive' | 'enabled' | 'disabled'; display?: 'summarized' | 'omitted'; budgetTokens?: number };
  effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
};

type OpenAIProviderOptions = {
  reasoningEffort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh';
  reasoningSummary?: 'auto' | 'concise' | 'detailed' | 'none';
  serviceTier?: string;
};

type GoogleProviderOptions = {
  thinkingConfig?: { includeThoughts?: boolean; thinkingLevel?: 'minimal' | 'low' | 'medium' | 'high' };
};

const conv = new Conversation({ name: 'test-providerOptions' });

const buildAnthropic = (effort: ReasoningEffort | undefined, modelString: string): AnthropicProviderOptions => {
  // buildProviderOptions is private — testing it directly via cast keeps the
  // production API surface narrow.
  const opts = (conv as any).buildProviderOptions('anthropic', { reasoningEffort: effort }, modelString);
  return opts.anthropic as AnthropicProviderOptions;
};

const buildOpenAI = (effort: ReasoningEffort | undefined, modelString: string): OpenAIProviderOptions => {
  const opts = (conv as any).buildProviderOptions('openai', { reasoningEffort: effort }, modelString);
  return opts.openai as OpenAIProviderOptions;
};

const buildGoogle = (effort: ReasoningEffort | undefined, modelString: string): GoogleProviderOptions => {
  const opts = (conv as any).buildProviderOptions('google', { reasoningEffort: effort }, modelString);
  return opts.google as GoogleProviderOptions;
};

describe('Conversation.buildProviderOptions (anthropic)', () => {
  describe('adaptive thinking (Opus / Sonnet)', () => {
    test.each([
      ['claude-opus-4-8', 'auto'],
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
      'sets thinking: { type: adaptive, display: summarized } and effort: %s on Opus',
      (effort) => {
        const anthropic = buildAnthropic(effort, 'claude-opus-4-8');
        expect(anthropic.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
        expect(anthropic.effort).toBe(effort);
      }
    );

    test('omits thinking entirely when effort is "none"', () => {
      const anthropic = buildAnthropic('none', 'claude-opus-4-8');
      expect(anthropic.thinking).toBeUndefined();
      expect(anthropic.effort).toBeUndefined();
    });
  });

  describe('extended thinking (Haiku)', () => {
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

describe('Conversation.buildProviderOptions (openai)', () => {
  test.each([
    ['gpt-5.5', 'auto'],
    ['gpt-5.5', 'none'],
    ['gpt-5.5', 'low'],
    ['gpt-5.5', 'medium'],
    ['gpt-5.5', 'high'],
    ['gpt-5.5', 'xhigh'],
    ['gpt-5.4-mini', 'auto'],
  ] as Array<[string, ReasoningEffort]>)(
    'always sets reasoningSummary: auto for model=%s effort=%s',
    (model, effort) => {
      const openai = buildOpenAI(effort, model);
      expect(openai.reasoningSummary).toBe('auto');
    }
  );

  test.each([
    ['none', 'none'],
    ['low', 'low'],
    ['medium', 'medium'],
    ['high', 'high'],
    ['xhigh', 'xhigh'],
    ['max', 'xhigh'], // 'max' maps to 'xhigh' (OpenAI's highest)
  ] as Array<[ReasoningEffort, string]>)('forwards reasoningEffort %s → %s', (input, expected) => {
    const openai = buildOpenAI(input, 'gpt-5.5');
    expect(openai.reasoningEffort).toBe(expected);
  });

  test('omits reasoningEffort on "auto" but keeps reasoningSummary', () => {
    const openai = buildOpenAI('auto', 'gpt-5.5');
    expect(openai.reasoningEffort).toBeUndefined();
    expect(openai.reasoningSummary).toBe('auto');
  });
});

describe('Conversation.buildProviderOptions (google)', () => {
  test.each([['gemini-3.1-pro-preview'], ['gemini-3.5-flash']])(
    'auto effort sets thinkingConfig.includeThoughts: true (model=%s)',
    (model) => {
      const google = buildGoogle('auto', model);
      expect(google.thinkingConfig).toEqual({ includeThoughts: true });
    }
  );

  test.each([
    ['low', 'low'],
    ['medium', 'medium'],
    ['high', 'high'],
    ['xhigh', 'high'], // Gemini caps at 'high'
    ['max', 'high'],
  ] as Array<[ReasoningEffort, string]>)(
    'effort %s → thinkingLevel %s, always includes thoughts',
    (input, expected) => {
      const google = buildGoogle(input, 'gemini-3.5-flash');
      expect(google.thinkingConfig?.includeThoughts).toBe(true);
      expect(google.thinkingConfig?.thinkingLevel).toBe(expected);
    }
  );

  test('omits thinkingConfig entirely when effort is "none"', () => {
    const google = buildGoogle('none', 'gemini-3.5-flash');
    expect(google.thinkingConfig).toBeUndefined();
  });
});

type XaiProviderOptions = {
  reasoningEffort?: 'low' | 'high';
};

const buildXai = (
  effort: ReasoningEffort | undefined,
  modelString: string,
  webSearch?: boolean
): XaiProviderOptions => {
  const opts = (conv as any).buildProviderOptions('xai', { reasoningEffort: effort, webSearch }, modelString);
  return opts.xai as XaiProviderOptions;
};

describe('Conversation.buildProviderOptions (xai)', () => {
  describe('reasoningEffort gate', () => {
    test('Fast models accept low/high mapped values', () => {
      expect(buildXai('low', 'grok-4-1-fast-reasoning').reasoningEffort).toBe('low');
      expect(buildXai('high', 'grok-4-1-fast-reasoning').reasoningEffort).toBe('high');
      expect(buildXai('medium', 'grok-4-1-fast-reasoning').reasoningEffort).toBe('high');
    });

    test('Flagship models do not accept reasoningEffort (model decides internally)', () => {
      expect(buildXai('high', 'grok-4.3').reasoningEffort).toBeUndefined();
      expect(buildXai('low', 'grok-4').reasoningEffort).toBeUndefined();
    });

    test('auto omits reasoningEffort everywhere', () => {
      expect(buildXai('auto', 'grok-4-1-fast-reasoning').reasoningEffort).toBeUndefined();
      expect(buildXai('auto', 'grok-4.3').reasoningEffort).toBeUndefined();
    });
  });

  describe('search wiring lives elsewhere now', () => {
    // xAI deprecated the Chat Completions `search_parameters` API and the
    // new Agent Tools API uses the `webSearch` tool factory via the
    // Responses endpoint. So buildProviderOptions doesn't set any search
    // option for xAI anymore — see getWebSearchTools for the tool wiring.
    test('does not set any search-related field on xai providerOptions', () => {
      expect(buildXai('auto', 'grok-4.3', true)).not.toHaveProperty('searchParameters');
      expect(buildXai('auto', 'grok-4-1-fast-reasoning', false)).not.toHaveProperty('searchParameters');
    });
  });
});
