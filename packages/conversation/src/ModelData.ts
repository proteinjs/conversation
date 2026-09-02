import type { ModelApiCost } from './UsageData';

/**
 * Service tier a request bills at. Providers price the same model differently
 * per tier (OpenAI: standard/batch/flex/priority); `standard` is the default
 * when a request carries no tier.
 */
export type UsagePricingTier = 'standard' | 'batch' | 'flex' | 'priority';

/**
 * The model-data seam: WHAT models cost is operating data owned by the
 * platform embedding this library (one authoring home), not by the transport.
 * The transport keeps the cost MATH (`calculateUsageCostUsd`, the accumulator,
 * tier/id normalization) and RECEIVES the data through this resolver —
 * required at every construction seam (`ConversationParams.modelData`), so a
 * call site without pricing data is a compile error, never a request that
 * silently bills $0 against an absent table.
 *
 * Step-1 shape of the model-catalog-as-data design: pricing only. Step 2
 * widens this same interface with capability facts (forced-tool-choice
 * support, hard input caps) so the transport's per-model gates read data
 * instead of hardcoded id lists.
 */
export interface ModelDataResolver {
  /**
   * Resolve the cost row for a model at a service tier, or `undefined` when
   * the model has no priced row — the caller then records the explicit
   * all-zero cost (consumers see `priced: false`, never a fabricated price).
   *
   * `modelId` arrives normalized by the transport (provider prefixes like
   * `openai:`/`openai/` stripped), but implementations own the lookup
   * semantics against their dataset (e.g. `-latest` suffix fallback) and must
   * tolerate raw ids, since they also serve non-transport callers.
   */
  pricing(modelId: string, tier: UsagePricingTier): ModelApiCost | undefined;
}
