import type { ModelApiCost } from '../../src/UsageData';
import type { ModelDataResolver, UsagePricingTier } from '../../src/ModelData';

/**
 * Fixture pricing for the test estate. Pricing DATA is owned by the platform
 * embedding this library (n3xa: chat-common's models module) — no suite here
 * may depend on real rates or real catalog contents; mechanism tests pass
 * fixture rows and pin the MATH/lookup seam only.
 */

/** A standard-shaped cost row (full cache economics: read discount + write premium). */
export const FIXTURE_STANDARD_ROW: ModelApiCost = {
  inputUsdPer1M: 5.0,
  cachedInputUsdPer1M: 0.5,
  cacheWriteUsdPer1M: 6.25,
  outputUsdPer1M: 25.0,
};

/**
 * Prices EVERY model id at {@link FIXTURE_STANDARD_ROW} on every tier — the
 * default resolver for suites that just need "a priced model" (live transport
 * suites asserting usage flows through to a non-zero cost, tool-loop suites,
 * etc.). Suites pinning pricing-lookup semantics use {@link modelDataFromRows}.
 */
export const fixtureModelData: ModelDataResolver = {
  pricing: () => FIXTURE_STANDARD_ROW,
};

export type FixtureModelRows = Partial<Record<UsagePricingTier, Record<string, ModelApiCost>>>;

/** A resolver over explicit per-tier fixture rows — absent id/tier = unpriced. */
export const modelDataFromRows = (rows: FixtureModelRows): ModelDataResolver => ({
  pricing: (modelId: string, tier: UsagePricingTier) => rows[tier]?.[modelId],
});
