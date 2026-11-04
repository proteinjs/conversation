import { createAnthropic } from '@ai-sdk/anthropic';
import { Conversation, ReasoningEffort } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';
import { PromptCacheScriptedTransport, type WireBody } from './PromptCacheScriptedTransport';

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
  thinking?: {
    type: 'adaptive' | 'enabled' | 'disabled' | 'between_tools';
    display?: 'summarized' | 'omitted';
    budgetTokens?: number;
  };
  effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
};

type OpenAIProviderOptions = {
  reasoningEffort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';
  reasoningSummary?: 'auto' | 'concise' | 'detailed' | 'none';
  serviceTier?: string;
  forceReasoning?: boolean;
  store?: boolean;
  include?: string[];
};

type GoogleProviderOptions = {
  thinkingConfig?: { includeThoughts?: boolean; thinkingLevel?: 'minimal' | 'low' | 'medium' | 'high' };
};

const conv = new Conversation({ modelData: fixtureModelData, name: 'test-providerOptions' });

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

    // 'none' is the EXPLICIT off, in the vendor's word for the family (AnthropicModelRules.thinkingOff).
    // RED at 6.20.6: nothing was set for 'none' — on the fifth generation a request with no thinking
    // block runs ADAPTIVE thinking (Sonnet 5 reasoned on 19 of 50 short questions sent that way,
    // 80–427 reasoning tokens a row, 2026-11-04), so the branch sent the opposite of off.
    test.each([['claude-sonnet-5'], ['claude-opus-5'], ['claude-opus-4-8'], ['claude-sonnet-4-6']])(
      'effort "none" sends thinking: { type: disabled } and no effort on %s',
      (model) => {
        const anthropic = buildAnthropic('none', model);
        expect(anthropic.thinking).toEqual({ type: 'disabled' });
        expect(anthropic.effort).toBeUndefined();
      }
    );

    test('effort "none" on Sonnet 5.5 is between_tools — the family\'s lowest thinking setting (disabled is refused)', () => {
      const anthropic = buildAnthropic('none', 'claude-sonnet-5-5');
      expect(anthropic.thinking).toEqual({ type: 'between_tools' });
      expect(anthropic.effort).toBeUndefined();
    });

    // Thinking is always on there — no off switch (disabled and between_tools alike are refused, HTTP
    // 400): 'none' is not a level those models offer, refused by name as on Gemini; the least is 'low'.
    test.each([['claude-opus-5-5'], ['claude-fable-5-1'], ['claude-mythos-5-1']])(
      'refuses "none" by name on %s — thinking is always on there, no off switch',
      (model) => {
        expect(() => buildAnthropic('none', model)).toThrow(/no off switch/);
        expect(() => buildAnthropic('none', model)).toThrow(model);
      }
    );
  });

  // Haiku through the fourth generation takes budget thinking only — no adaptive (AnthropicModelRules.budgetThinkingOnly).
  describe('extended thinking (Haiku through 4.x)', () => {
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

    // Haiku from the fifth generation on is adaptive like Opus and Sonnet — 5.5 REFUSES a budget (the vendor's
    // table; the adapter would swap a budget for bare adaptive with a warning and drop the display). RED at
    // 9afaf30c: `/haiku/` sent every Haiku a budget — one owner of the family now (AnthropicModelRules).
    test('Haiku 5.5 at auto is adaptive with the reasoning text shown — never a budget', () => {
      const anthropic = buildAnthropic('auto', 'claude-haiku-5-5');
      expect(anthropic.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
      expect(anthropic.effort).toBeUndefined();
    });

    test.each([['low'], ['high']] as Array<[ReasoningEffort]>)(
      'Haiku 5.5 at %s is adaptive with the level as effort — never a budget',
      (effort) => {
        const anthropic = buildAnthropic(effort, 'claude-haiku-5-5');
        expect(anthropic.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
        expect(anthropic.effort).toBe(effort);
      }
    );

    // Haiku takes `disabled` in every generation (5.5: at effort high or below — none is sent here).
    test.each([['claude-haiku-4-5'], ['claude-haiku-5-5']])(
      'effort "none" sends thinking: { type: disabled } and no effort on %s',
      (model) => {
        const anthropic = buildAnthropic('none', model);
        expect(anthropic.thinking).toEqual({ type: 'disabled' });
        expect(anthropic.effort).toBeUndefined();
      }
    );
  });
});

/**
 * THE ADAPTER AT THE FLOOR, ON THE WIRE. The built options are only the request when the installed
 * `@ai-sdk/anthropic` carries them unchanged: its options schema refuses a literal it does not know
 * before any request (`between_tools` arrived in 3.0.125 — the old lock, 3.0.107, threw
 * AI_InvalidArgumentError and sent nothing), and its per-model substitutions rewrite a shape a model
 * rejects with a warning (a budget on Haiku 5.5 becomes bare adaptive, the display dropped; `disabled`
 * on Sonnet 5.5 becomes `between_tools`). So every family's shape is read on the request body the
 * real provider sends over a scripted transport, with the stream's warnings — ZERO: the house sends
 * each model what it takes, and nothing is rewritten on the way out. No network, no key.
 */
type ThinkingWire = WireBody & {
  thinking?: Record<string, unknown>;
  output_config?: { effort?: string };
};

/** One request through the real adapter: the body it sent, and every warning the step's start carried (the SDK's own log of the same warnings is captured off the console). */
async function wireRequest(
  modelId: string,
  reasoningEffort: ReasoningEffort
): Promise<{ body: ThinkingWire; warnings: unknown[] }> {
  const g = globalThis as { AI_SDK_LOG_WARNINGS?: unknown };
  const prior = g.AI_SDK_LOG_WARNINGS;
  g.AI_SDK_LOG_WARNINGS = () => undefined;
  try {
    const transport = new PromptCacheScriptedTransport();
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: `wire-${modelId}-${reasoningEffort}`,
      logLevel: 'error',
      limits: { enforceLimits: false },
    });
    const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })(modelId);
    const result = await conversation.generateStream({ messages: ['hi'], model: model as never, reasoningEffort });
    const stepWarnings: unknown[] = [];
    for await (const part of result.fullStream as AsyncIterable<{ type: string; warnings?: unknown[] }>) {
      if (part.type === 'step-start') {
        stepWarnings.push(...(part.warnings ?? []));
      }
    }
    await result.usage;
    expect(transport.requests).toHaveLength(1);
    return { body: transport.requests[0] as ThinkingWire, warnings: stepWarnings };
  } finally {
    if (prior === undefined) {
      delete g.AI_SDK_LOG_WARNINGS;
    } else {
      g.AI_SDK_LOG_WARNINGS = prior;
    }
  }
}

describe('the Anthropic adapter at the floor carries the built options to the wire — nothing refused, nothing rewritten', () => {
  const adaptiveSummarized = { type: 'adaptive', display: 'summarized' };
  test.each([
    // effort 'none' — the family's explicit off, as sent
    ['claude-sonnet-5', 'none', { type: 'disabled' }, undefined],
    ['claude-opus-4-8', 'none', { type: 'disabled' }, undefined],
    ['claude-haiku-5-5', 'none', { type: 'disabled' }, undefined],
    ['claude-sonnet-5-5', 'none', { type: 'between_tools' }, undefined],
    // 'auto' and the levels — adaptive with the text shown, the level under output_config; a budget on Haiku 4.x
    ['claude-haiku-5-5', 'low', adaptiveSummarized, 'low'],
    ['claude-haiku-4-5', 'low', { type: 'enabled', budget_tokens: 5000 }, undefined],
    ['claude-opus-5-5', 'low', adaptiveSummarized, 'low'],
    ['claude-sonnet-5-5', 'auto', adaptiveSummarized, undefined],
  ] as Array<[string, ReasoningEffort, Record<string, unknown>, string | undefined]>)(
    '%s at %s: the request carries thinking %j, effort %s, and the adapter warns of nothing',
    async (modelId, effort, thinking, outputEffort) => {
      const { body, warnings } = await wireRequest(modelId, effort);
      expect(body.model).toBe(modelId);
      expect(body.thinking).toEqual(thinking);
      expect(body.output_config?.effort).toBe(outputEffort);
      expect(warnings).toEqual([]);
    },
    30_000
  );
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
    ['max', 'xhigh'], // 'max' is GPT-5's top level, xhigh (a level of its own from GPT-6 — openAiModelRules.test)
  ] as Array<[ReasoningEffort, string]>)('forwards reasoningEffort %s → %s', (input, expected) => {
    const openai = buildOpenAI(input, 'gpt-5.5');
    expect(openai.reasoningEffort).toBe(expected);
  });

  test('omits reasoningEffort on "auto" but keeps reasoningSummary', () => {
    const openai = buildOpenAI('auto', 'gpt-5.5');
    expect(openai.reasoningEffort).toBeUndefined();
    expect(openai.reasoningSummary).toBe('auto');
  });

  // Every OpenAI request is stateless: `store: false` (the create reference's `store` "Defaults
  // to true when omitted"), and on a reasoning model the encrypted reasoning is asked for by name
  // so a tool loop can replay it (the reasoning guide, "Preserve reasoning without stored
  // responses"). A non-reasoning model has no reasoning items to keep, so it asks for nothing.
  test.each([
    ['gpt-5.5', 'auto'],
    ['gpt-5.5', 'high'],
    ['gpt-6-sol', 'max'],
    ['gpt-5-chat-latest', 'auto'],
    ['gpt-4o', undefined],
  ] as Array<[string, ReasoningEffort | undefined]>)('says store: false for model=%s effort=%s', (model, effort) => {
    const openai = buildOpenAI(effort, model);
    expect(openai.store).toBe(false);
  });

  test.each([['gpt-5.5'], ['gpt-6-sol'], ['o3-mini']])(
    'asks for the encrypted reasoning on the reasoning model %s',
    (model) => {
      const openai = buildOpenAI('auto', model);
      expect(openai.include).toEqual(['reasoning.encrypted_content']);
    }
  );

  test.each([['gpt-5-chat-latest'], ['gpt-4o']])(
    'asks for no encrypted reasoning on the non-reasoning model %s',
    (model) => {
      const openai = buildOpenAI('auto', model);
      expect(openai.include).toBeUndefined();
    }
  );
});

describe('Conversation.buildProviderOptions (google)', () => {
  test.each([['gemini-3.1-pro-preview'], ['gemini-3.5-flash']])(
    'auto effort sets thinkingConfig.includeThoughts: true (model=%s)',
    (model) => {
      const google = buildGoogle('auto', model);
      expect(google.thinkingConfig).toEqual({ includeThoughts: true });
    }
  );

  // A caller that names NO level (a chat at auto over a catalog row that lists no auto thinking
  // level: the provider's default stands, never a level of ours) is the auto shape on the wire:
  // summaries on, no level sent. Before the fix the whole thinkingConfig went with the level, so
  // Google was never asked for thought summaries and a model that thinks at its default showed
  // nothing on the timeline.
  test.each([['gemini-3.1-pro-preview'], ['gemini-3.5-flash']])(
    'an undefined effort is the auto shape: includeThoughts: true and no thinkingLevel (model=%s)',
    (model) => {
      const google = buildGoogle(undefined, model);
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

  // Gemini 3 and later have NO off switch (Google's thinking page lists no Off for any of them): a request
  // without thinkingConfig is the model's own dynamic default, the opposite of 'none'. 'none' is a caller's
  // error, refused by name — the least a caller can ask for is the model's lowest listed level, and the
  // owner of that answer is the caller's catalog. RED at d2fa1b4: 'none' omitted thinkingConfig entirely
  // ("the one off switch"), and the acknowledgment on Gemini 4 Argon ran at the model's default.
  test.each([['gemini-4-argon'], ['gemini-3.1-pro-preview'], ['gemini-3.5-flash']])(
    'refuses "none" by name — no off switch on Gemini (model=%s)',
    (model) => {
      expect(() => buildGoogle('none', model)).toThrow(/no off switch/);
      expect(() => buildGoogle('none', model)).toThrow(model);
    }
  );
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
  describe('reasoningEffort — sent as asked for every xAI model (the models API states each model’s levels; grok-4.5 accepted low…xhigh live, 2026-09-29)', () => {
    test('every level rides as its own word on the flagship models — the /fast/ gate that held them back is gone', () => {
      expect(buildXai('low', 'grok-4.5').reasoningEffort).toBe('low');
      expect(buildXai('medium', 'grok-4.5').reasoningEffort).toBe('medium');
      expect(buildXai('high', 'grok-4.5').reasoningEffort).toBe('high');
      expect(buildXai('xhigh', 'grok-4.5').reasoningEffort).toBe('xhigh');
      expect(buildXai('none', 'grok-4.3').reasoningEffort).toBe('none');
      expect(buildXai('high', 'grok-4-1-fast-reasoning').reasoningEffort).toBe('high');
    });

    test('max — a level xAI does not have — is sent as its top, xhigh', () => {
      expect(buildXai('max', 'grok-4.5').reasoningEffort).toBe('xhigh');
    });

    test('auto omits reasoningEffort everywhere (the model’s own default)', () => {
      expect(buildXai('auto', 'grok-4-1-fast-reasoning').reasoningEffort).toBeUndefined();
      expect(buildXai('auto', 'grok-4.3').reasoningEffort).toBeUndefined();
      expect(buildXai(undefined, 'grok-4.5').reasoningEffort).toBeUndefined();
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
