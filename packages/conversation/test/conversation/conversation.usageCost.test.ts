import { calculateUsageCostUsd, aggregateUsageData } from '../../src/UsageData';
import type { TokenUsage, UsageData } from '../../src/UsageData';

/**
 * Pure-unit guards for the cost math in `calculateUsageCostUsd` — no API keys,
 * no network, runs in CI. These pin the invariants that the per-step rounding
 * and the cached-token accounting must satisfy, independent of any provider.
 */

const tokens = (over: Partial<TokenUsage>): TokenUsage => ({
  inputTokens: 0,
  cachedInputTokens: 0,
  cacheWriteTokens: 0,
  reasoningTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  ...over,
});

describe('calculateUsageCostUsd', () => {
  // claude-opus-4-8: input 5.0, cachedInput 0.5, output 25.0 per 1M.
  const MODEL = 'claude-opus-4-8';

  it('does NOT double-count cached input in the total (regression: totalUsd was input+cached+output)', () => {
    // The audit's verified example: 1M input of which 800k cached, 200k output.
    // nonCached 200k*5 + cached 800k*0.5 = 1.0 + 0.4 = 1.4 input; 200k*25 = 5.0 output.
    // Correct total = 6.40. The pre-fix bug added cachedInputUsd (0.40) again → 6.80.
    const cost = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 800_000, outputTokens: 200_000, totalTokens: 1_200_000 })
    );

    expect(cost.totalUsd).toBeCloseTo(6.4, 5);
    // total is exactly input + output — cachedInputUsd is already inside inputUsd.
    expect(cost.totalUsd).toBeCloseTo(cost.inputUsd + cost.outputUsd, 5);
    // and explicitly NOT the double-counted figure.
    expect(cost.totalUsd).not.toBeCloseTo(cost.inputUsd + cost.cachedInputUsd + cost.outputUsd, 5);
  });

  it('prices cache-WRITE tokens at a premium above fresh input (Anthropic cache creation)', () => {
    // claude-opus-4-8: input 5.0, cacheWrite 6.25 per 1M.
    const fresh = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 1_000_000, outputTokens: 0 }));
    const allWrite = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cacheWriteTokens: 1_000_000, outputTokens: 0 })
    );

    // 1M cache-write @ 6.25 = $6.25 vs 1M fresh @ 5.0 = $5.00.
    expect(allWrite.totalUsd).toBeCloseTo(6.25, 5);
    expect(allWrite.totalUsd).toBeGreaterThan(fresh.totalUsd);
  });

  it('carries sub-cent precision through (no per-step rounding to $0)', () => {
    // A tiny request whose true cost is well under a cent must NOT record as 0.
    const cost = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 100, outputTokens: 100, totalTokens: 200 }));
    expect(cost.totalUsd).toBeGreaterThan(0);
    expect(cost.totalUsd).toBeLessThan(0.01);
  });

  it('prices cached input below fresh input (cache discount is applied, not the full rate)', () => {
    const allFresh = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 1_000_000, outputTokens: 200_000 }));
    const allCached = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 1_000_000, outputTokens: 200_000 })
    );

    expect(allCached.totalUsd).toBeLessThan(allFresh.totalUsd);
  });

  it('excludes reasoning from the total (reasoning tokens are already inside output_tokens)', () => {
    const withReasoning = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 100_000, outputTokens: 50_000, reasoningTokens: 30_000, totalTokens: 150_000 })
    );

    // reasoningUsd is itemized but must not inflate the total beyond input + output.
    expect(withReasoning.totalUsd).toBeCloseTo(withReasoning.inputUsd + withReasoning.outputUsd, 5);
  });

  it('returns all-zero cost for a model with no pricing row (the $0 fallback is explicit)', () => {
    const cost = calculateUsageCostUsd(
      'definitely-not-a-real-model-xyz',
      tokens({ inputTokens: 1_000_000, outputTokens: 1_000_000, totalTokens: 2_000_000 })
    );

    expect(cost.totalUsd).toBe(0);
  });
});

const usageData = (model: string, outputTokens: number, totalUsd: number): UsageData =>
  ({
    model,
    initialRequestTokenUsage: tokens({}),
    initialRequestCostUsd: { inputUsd: 0, cachedInputUsd: 0, reasoningUsd: 0, outputUsd: 0, totalUsd: 0 },
    totalTokenUsage: tokens({ outputTokens, totalTokens: outputTokens }),
    totalCostUsd: { inputUsd: 0, cachedInputUsd: 0, reasoningUsd: 0, outputUsd: totalUsd, totalUsd },
    totalRequestsToAssistant: 1,
    callsPerTool: {},
    totalToolCalls: 0,
  }) as UsageData;

describe('aggregateUsageData', () => {
  it('labels the aggregate with the dominant (most-output) model, not whichever ran first', () => {
    // A tiny utility/title call on a cheap model ran FIRST; the real work ran on Opus.
    const agg = aggregateUsageData([usageData('gpt-5-nano', 1_000, 0.001), usageData('claude-opus-4-8', 50_000, 2.5)]);

    expect(agg?.model).toBe('claude-opus-4-8');
    // tokens still sum across both models
    expect(agg?.totalTokenUsage.outputTokens).toBe(51_000);
  });

  it('falls back to the first model when nothing has output yet', () => {
    const agg = aggregateUsageData([usageData('gpt-5-nano', 0, 0), usageData('claude-opus-4-8', 0, 0)]);
    expect(agg?.model).toBe('gpt-5-nano');
  });
});
