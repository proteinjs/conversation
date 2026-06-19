import { Conversation } from '../../src/Conversation';

/**
 * Provider-agnostic USAGE contract: every model we ship must report sane token
 * accounting AND resolve to a non-zero price. We loop one representative model
 * per provider (plus both Anthropic tiers, since the flow uses Opus and the
 * chat uses Sonnet, and both live xAI ids, since those were the models silently
 * billing $0). Each case is gated on its provider's API key, so the suite
 * degrades gracefully where a key is absent — same pattern as the reasoning
 * suite.
 *
 * This is the live counterpart to the pure-unit guards:
 *  - conversation.usageCost.test.ts pins the cost MATH (no network);
 *  - chat-common's modelCatalogPricing.test.ts pins catalog↔pricing coverage;
 *  - this test proves the REAL provider responses populate the token fields and
 *    flow all the way through to a priced UsageData.
 */

type UsageCase = { provider: string; model: string; keyEnv: string };

const USAGE_MODELS: UsageCase[] = [
  { provider: 'anthropic', model: 'claude-opus-4-8', keyEnv: 'ANTHROPIC_API_KEY' },
  { provider: 'anthropic', model: 'claude-sonnet-4-6', keyEnv: 'ANTHROPIC_API_KEY' },
  { provider: 'openai', model: 'gpt-5.5', keyEnv: 'OPENAI_API_KEY' },
  { provider: 'google', model: 'gemini-3.5-flash', keyEnv: 'GOOGLE_GENERATIVE_AI_API_KEY' },
  { provider: 'xai', model: 'grok-4.3', keyEnv: 'XAI_API_KEY' },
  { provider: 'xai', model: 'grok-4-1-fast-reasoning', keyEnv: 'XAI_API_KEY' },
];

const TIMEOUT = 120_000;

// Short, deterministic prompt — we exercise token accounting, not output length.
const PROMPT = 'Reply with exactly the word: hello';

describe('Conversation.generateResponse — every shipped model reports token usage and a non-zero price', () => {
  for (const { provider, model, keyEnv } of USAGE_MODELS) {
    const testIfKey = process.env[keyEnv] ? test : test.skip;

    testIfKey(
      `${provider}/${model}: populates token counts and resolves a price`,
      async () => {
        const conversation = new Conversation({ name: `test-usage-${model}` });

        const result = await conversation.generateResponse({ messages: [PROMPT], model });

        const usage = result.usage;
        const tu = usage.totalTokenUsage;
        const cost = usage.totalCostUsd;

        // Diagnostic — shows the full per-model breakdown when the suite runs.
        // eslint-disable-next-line no-console
        console.log(
          `[usage] ${provider}/${model}: in=${tu.inputTokens} cached=${tu.cachedInputTokens} ` +
            `reasoning=${tu.reasoningTokens} out=${tu.outputTokens} total=${tu.totalTokens} ` +
            `model=${usage.model} totalUsd=${cost.totalUsd}`
        );

        // ── Token accounting contract ──
        expect(tu.inputTokens).toBeGreaterThan(0);
        expect(tu.outputTokens).toBeGreaterThan(0);
        // totalTokens accounts for at least input + output (providers may add more).
        expect(tu.totalTokens).toBeGreaterThanOrEqual(tu.inputTokens + tu.outputTokens);
        // cached is a SUBSET of input (the UI renders "X% of input cached" off this).
        expect(tu.cachedInputTokens).toBeGreaterThanOrEqual(0);
        expect(tu.cachedInputTokens).toBeLessThanOrEqual(tu.inputTokens);
        // reasoning is a subset of output.
        expect(tu.reasoningTokens).toBeGreaterThanOrEqual(0);
        expect(tu.reasoningTokens).toBeLessThanOrEqual(tu.outputTokens);

        // ── Provenance + price contract ──
        expect(usage.model).toBeTruthy();
        // The model is priced — a real billable call must not record as $0/free.
        expect(cost.totalUsd).toBeGreaterThan(0);
      },
      TIMEOUT
    );
  }
});
