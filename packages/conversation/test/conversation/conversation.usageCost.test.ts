import { calculateUsageCostUsd, aggregateUsageData } from '../../src/UsageData';
import type { TokenUsage, UsageData } from '../../src/UsageData';
import type { UsagePricingTier } from '../../src/ModelData';
import { modelDataFromRows } from './fixtureModelData';

/**
 * Pure-unit guards for the cost math in `calculateUsageCostUsd` — no API keys,
 * no network, runs in CI. These pin the invariants that the per-step rounding
 * and the cached-token accounting must satisfy, independent of any provider.
 *
 * All rows here are FIXTURES: pricing data is injected through the
 * `ModelDataResolver` seam (its one real implementation lives in
 * `@n3xah/chat-common`); this suite owns the MATH and the seam mechanics only.
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
  // Fixture row with full cache economics: input 5.0, cachedInput 0.5,
  // cacheWrite 6.25, output 25.0 per 1M (the shape of a standard Anthropic row).
  const MODEL = 'fixture-model';
  const modelData = modelDataFromRows({
    standard: {
      'fixture-model': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, cacheWriteUsdPer1M: 6.25, outputUsdPer1M: 25.0 },
      // Two rows differing ONLY in the cache-read rate — one at a deep 0.025x
      // tier, one at the conventional 0.1x (the fable-5-1 vs fable-5 shape).
      'fixture-cache-deep-discount': {
        inputUsdPer1M: 10.0,
        cachedInputUsdPer1M: 0.25,
        cacheWriteUsdPer1M: 12.5,
        outputUsdPer1M: 50.0,
      },
      'fixture-cache-standard-discount': {
        inputUsdPer1M: 10.0,
        cachedInputUsdPer1M: 1.0,
        cacheWriteUsdPer1M: 12.5,
        outputUsdPer1M: 50.0,
      },
      // No cachedInputUsdPer1M at all (the pro-tier shape): cache reads must
      // bill at the FULL input rate, not free and not at a phantom discount.
      'fixture-no-cache-discount': { inputUsdPer1M: 30.0, outputUsdPer1M: 180.0 },
    },
    flex: {
      'fixture-model': { inputUsdPer1M: 1.0, cachedInputUsdPer1M: 0.1, outputUsdPer1M: 4.0 },
    },
  });

  it('does NOT double-count cached input in the total (regression: totalUsd was input+cached+output)', () => {
    // The audit's verified example: 1M input of which 800k cached, 200k output.
    // nonCached 200k*5 + cached 800k*0.5 = 1.0 + 0.4 = 1.4 input; 200k*25 = 5.0 output.
    // Correct total = 6.40. The pre-fix bug added cachedInputUsd (0.40) again → 6.80.
    const cost = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 800_000, outputTokens: 200_000, totalTokens: 1_200_000 }),
      { modelData }
    );

    expect(cost.totalUsd).toBeCloseTo(6.4, 5);
    // total is exactly input + output — cachedInputUsd is already inside inputUsd.
    expect(cost.totalUsd).toBeCloseTo(cost.inputUsd + cost.outputUsd, 5);
    // and explicitly NOT the double-counted figure.
    expect(cost.totalUsd).not.toBeCloseTo(cost.inputUsd + cost.cachedInputUsd + cost.outputUsd, 5);
  });

  it('prices cache-WRITE tokens at a premium above fresh input (Anthropic cache creation)', () => {
    // fixture-model: input 5.0, cacheWrite 6.25 per 1M.
    const fresh = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 1_000_000, outputTokens: 0 }), { modelData });
    const allWrite = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cacheWriteTokens: 1_000_000, outputTokens: 0 }),
      { modelData }
    );

    // 1M cache-write @ 6.25 = $6.25 vs 1M fresh @ 5.0 = $5.00.
    expect(allWrite.totalUsd).toBeCloseTo(6.25, 5);
    expect(allWrite.totalUsd).toBeGreaterThan(fresh.totalUsd);
  });

  it('carries sub-cent precision through (no per-step rounding to $0)', () => {
    // A tiny request whose true cost is well under a cent must NOT record as 0.
    const cost = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 100, outputTokens: 100, totalTokens: 200 }), {
      modelData,
    });
    expect(cost.totalUsd).toBeGreaterThan(0);
    expect(cost.totalUsd).toBeLessThan(0.01);
  });

  it('prices cached input below fresh input (cache discount is applied, not the full rate)', () => {
    const allFresh = calculateUsageCostUsd(MODEL, tokens({ inputTokens: 1_000_000, outputTokens: 200_000 }), {
      modelData,
    });
    const allCached = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 1_000_000, outputTokens: 200_000 }),
      { modelData }
    );

    expect(allCached.totalUsd).toBeLessThan(allFresh.totalUsd);
  });

  it("prices cache reads at the ROW's own rate (a deep 0.025x tier is honored, not a copied 0.1x)", () => {
    // The fable-5-1 regression shape: same base rates, different cache-read
    // tier — 1M all-cached input = $0.25 on the deep-discount row vs $1.00 on
    // the standard-discount row. An unpriced id would read $0; a copy-pasted
    // standard rate would read $1.00 on both.
    const deep = calculateUsageCostUsd(
      'fixture-cache-deep-discount',
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 1_000_000, totalTokens: 1_000_000 }),
      { modelData }
    );
    expect(deep.totalUsd).toBeCloseTo(0.25, 5);
    const standard = calculateUsageCostUsd(
      'fixture-cache-standard-discount',
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 1_000_000, totalTokens: 1_000_000 }),
      { modelData }
    );
    expect(standard.totalUsd).toBeCloseTo(1.0, 5);
  });

  it('bills cache reads at the FULL input rate when a row has no cachedInputUsdPer1M (pro-tier shape)', () => {
    const cost = calculateUsageCostUsd(
      'fixture-no-cache-discount',
      tokens({ inputTokens: 1_000_000, cachedInputTokens: 1_000_000, totalTokens: 1_000_000 }),
      { modelData }
    );
    // 1M all-cached input @ the full 30.0 input rate — no discount exists on this row.
    expect(cost.totalUsd).toBeCloseTo(30.0, 5);
  });

  it('excludes reasoning from the total (reasoning tokens are already inside output_tokens)', () => {
    const withReasoning = calculateUsageCostUsd(
      MODEL,
      tokens({ inputTokens: 100_000, outputTokens: 50_000, reasoningTokens: 30_000, totalTokens: 150_000 }),
      { modelData }
    );

    // reasoningUsd is itemized but must not inflate the total beyond input + output.
    expect(withReasoning.totalUsd).toBeCloseTo(withReasoning.inputUsd + withReasoning.outputUsd, 5);
  });

  it('returns all-zero cost for a model with no pricing row (the $0 fallback is explicit)', () => {
    const cost = calculateUsageCostUsd(
      'definitely-not-a-real-model-xyz',
      tokens({ inputTokens: 1_000_000, outputTokens: 1_000_000, totalTokens: 2_000_000 }),
      { modelData }
    );

    expect(cost.totalUsd).toBe(0);
  });

  it('hands the resolver the NORMALIZED model id (provider prefixes stripped)', () => {
    const seen: string[] = [];
    const spying = {
      pricing: (modelId: string, tier: UsagePricingTier) => {
        seen.push(modelId);
        return modelData.pricing(modelId, tier);
      },
    };
    const prefixed = calculateUsageCostUsd(
      'openai:fixture-model',
      tokens({ inputTokens: 1_000_000, outputTokens: 0, totalTokens: 1_000_000 }),
      { modelData: spying }
    );
    const slashed = calculateUsageCostUsd(
      'openai/fixture-model',
      tokens({ inputTokens: 1_000_000, outputTokens: 0, totalTokens: 1_000_000 }),
      { modelData: spying }
    );

    expect(seen).toEqual(['fixture-model', 'fixture-model']);
    // and the normalized lookup actually priced the request.
    expect(prefixed.totalUsd).toBeCloseTo(5.0, 5);
    expect(slashed.totalUsd).toBeCloseTo(5.0, 5);
  });

  it('hands the resolver the NORMALIZED service tier (known tiers pass through, junk falls to standard)', () => {
    const seen: UsagePricingTier[] = [];
    const spying = {
      pricing: (modelId: string, tier: UsagePricingTier) => {
        seen.push(tier);
        return modelData.pricing(modelId, tier);
      },
    };
    const probe = tokens({ inputTokens: 1_000_000, outputTokens: 0, totalTokens: 1_000_000 });

    const flex = calculateUsageCostUsd(MODEL, probe, { modelData: spying, serviceTier: 'flex' });
    calculateUsageCostUsd(MODEL, probe, { modelData: spying, serviceTier: 'auto' });
    calculateUsageCostUsd(MODEL, probe, { modelData: spying });

    expect(seen).toEqual(['flex', 'standard', 'standard']);
    // the flex request billed the flex row (input 1.0), not the standard row (5.0).
    expect(flex.totalUsd).toBeCloseTo(1.0, 5);
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
    // A tiny utility/title call on a cheap model ran FIRST; the real work ran on the big model.
    const agg = aggregateUsageData([
      usageData('fixture-nano-model', 1_000, 0.001),
      usageData('fixture-model', 50_000, 2.5),
    ]);

    expect(agg?.model).toBe('fixture-model');
    // tokens still sum across both models
    expect(agg?.totalTokenUsage.outputTokens).toBe(51_000);
  });

  it('falls back to the first model when nothing has output yet', () => {
    const agg = aggregateUsageData([usageData('fixture-nano-model', 0, 0), usageData('fixture-model', 0, 0)]);
    expect(agg?.model).toBe('fixture-nano-model');
  });
});
