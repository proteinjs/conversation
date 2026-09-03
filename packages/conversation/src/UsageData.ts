import { TiktokenModel } from 'tiktoken';
import type { ModelDataResolver, UsagePricingTier } from './ModelData';

export type TokenUsage = {
  inputTokens: number;
  cachedInputTokens: number;
  /**
   * Tokens written to the provider's prompt cache this request (e.g. Anthropic
   * `cache_creation_input_tokens`). A subset of `inputTokens`, but priced at a
   * PREMIUM (cache writes cost more than fresh input), unlike `cachedInputTokens`
   * (cache reads) which are cheaper. 0 for providers/requests that don't write a cache.
   */
  cacheWriteTokens: number;
  reasoningTokens: number;
  outputTokens: number;
  totalTokens: number;
};

export type ModelApiCost = {
  /** USD per 1M input tokens */
  inputUsdPer1M: number;
  /** USD per 1M cached input tokens (cache reads; if supported) */
  cachedInputUsdPer1M?: number;
  /** USD per 1M cache-write tokens (cache creation; if supported, typically > input rate) */
  cacheWriteUsdPer1M?: number;
  /** USD per 1M output tokens */
  outputUsdPer1M: number;
};

export type UsageCostUsd = {
  inputUsd: number;
  cachedInputUsd: number;
  reasoningUsd: number;
  outputUsd: number;
  totalUsd: number;
};

/**
 * ONE loop step's usage — one billed provider request inside a tool loop (the initial request,
 * then every tool-call continuation). The SDK reports each step's usage on the step itself
 * (`StepResult.usage`); `UsageData.steps` keeps that list instead of only its sum, so a
 * downstream ledger can tell the FIRST request (the cross-turn prompt-cache read) from the
 * later steps (within-turn re-reads of the same prefix) — `totalTokenUsage` cannot: it is the
 * sum. `toolCalls` = how many tools the model called in that step.
 */
export type StepUsage = TokenUsage & { toolCalls: number };

/**
 * Usage data accumulated throughout the lifecycle of a single call to
 * `OpenAi.generateResponse` or `OpenAi.generateStreamingResponse`.
 */
export type UsageData = {
  /** The model used by the assistant */
  model: TiktokenModel;
  /** The token usage of the initial request sent to the assistant */
  initialRequestTokenUsage: TokenUsage;
  /** The USD cost of the initial request */
  initialRequestCostUsd: UsageCostUsd;
  /** The total token usage of all requests sent to the assistant (ie. initial request + all subsequent tool call requests) */
  totalTokenUsage: TokenUsage;
  /** The total USD cost of all requests sent to the assistant */
  totalCostUsd: UsageCostUsd;
  /** The number of requests sent to the assistant */
  totalRequestsToAssistant: number;
  /** The number of times each tool was called by the assistant */
  callsPerTool: { [toolName: string]: number };
  /** The total number of tool calls made by the assistant */
  totalToolCalls: number;
  /**
   * Per-step usage in loop order (see `StepUsage`), present when the provider reported usage per
   * step — Σ over `steps` reconciles to `totalTokenUsage`. Absent on paths that only know the
   * sum (the in-flight partial reports, the single-shot object path's pre-step shape).
   */
  steps?: StepUsage[];
};

type UsageDataAccumulatorParams = {
  model: TiktokenModel;
  /**
   * Pricing data for the models this accumulator bills — see
   * {@link ModelDataResolver}. Required: an accumulator without pricing data
   * would silently record $0 for every request.
   */
  modelData: ModelDataResolver;
};

export class UsageDataAccumulator {
  private processedInitialRequest = false;
  private modelData: ModelDataResolver;
  public usageData: UsageData;

  constructor({ model, modelData }: UsageDataAccumulatorParams) {
    this.modelData = modelData;
    this.usageData = {
      model,
      initialRequestTokenUsage: {
        inputTokens: 0,
        reasoningTokens: 0,
        cachedInputTokens: 0,
        cacheWriteTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
      },
      initialRequestCostUsd: {
        inputUsd: 0,
        cachedInputUsd: 0,
        reasoningUsd: 0,
        outputUsd: 0,
        totalUsd: 0,
      },
      totalTokenUsage: {
        inputTokens: 0,
        cachedInputTokens: 0,
        cacheWriteTokens: 0,
        reasoningTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
      },
      totalCostUsd: {
        inputUsd: 0,
        cachedInputUsd: 0,
        reasoningUsd: 0,
        outputUsd: 0,
        totalUsd: 0,
      },
      totalRequestsToAssistant: 0,
      callsPerTool: {},
      totalToolCalls: 0,
    };
  }

  addTokenUsage(tokenUsage: TokenUsage, opts?: { serviceTier?: string }) {
    this.usageData.totalRequestsToAssistant++;

    const cost = calculateUsageCostUsd(this.usageData.model, tokenUsage, {
      modelData: this.modelData,
      serviceTier: opts?.serviceTier,
    });

    if (!this.processedInitialRequest) {
      this.usageData.initialRequestTokenUsage = tokenUsage;
      this.usageData.initialRequestCostUsd = cost;
      this.processedInitialRequest = true;
    }

    if (cost) {
      if (!this.usageData.totalCostUsd) {
        this.usageData.totalCostUsd = { ...cost };
      } else {
        this.usageData.totalCostUsd = {
          inputUsd: this.usageData.totalCostUsd.inputUsd + cost.inputUsd,
          cachedInputUsd: this.usageData.totalCostUsd.cachedInputUsd + cost.cachedInputUsd,
          reasoningUsd: this.usageData.totalCostUsd.reasoningUsd + cost.reasoningUsd,
          outputUsd: this.usageData.totalCostUsd.outputUsd + cost.outputUsd,
          totalUsd: this.usageData.totalCostUsd.totalUsd + cost.totalUsd,
        };
      }
      // NB: no per-step rounding here. Rounding each request to cents drops
      // sub-cent costs (a real money-loss bug when summing many small requests);
      // we carry full precision and round only at the display/ledger boundary.
    }

    this.usageData.totalTokenUsage = {
      inputTokens: this.usageData.totalTokenUsage.inputTokens + tokenUsage.inputTokens,
      cachedInputTokens: this.usageData.totalTokenUsage.cachedInputTokens + tokenUsage.cachedInputTokens,
      cacheWriteTokens: this.usageData.totalTokenUsage.cacheWriteTokens + tokenUsage.cacheWriteTokens,
      reasoningTokens: this.usageData.totalTokenUsage.reasoningTokens + tokenUsage.reasoningTokens,
      outputTokens: this.usageData.totalTokenUsage.outputTokens + tokenUsage.outputTokens,
      totalTokens: this.usageData.totalTokenUsage.totalTokens + tokenUsage.totalTokens,
    };
  }

  recordToolCall(toolName: string) {
    if (!this.usageData.callsPerTool[toolName]) {
      this.usageData.callsPerTool[toolName] = 0;
    }

    this.usageData.callsPerTool[toolName]++;
    this.usageData.totalToolCalls++;
  }
}

/**
 * Aggregate multiple UsageData objects into a single UsageData.
 */
export function aggregateUsageData(list: UsageData[]): UsageData | undefined {
  if (!Array.isArray(list) || list.length === 0) {
    return undefined;
  }

  const first = list[0];

  const out: UsageData = {
    // The representative model is the one that did the most work, NOT whichever
    // ran first — otherwise a run whose real work is on (say) Opus but that also
    // made one tiny utility call (a cheap title/routing model) gets mislabeled by
    // the incidental call. See pickRepresentativeModel.
    model: pickRepresentativeModel(list),
    initialRequestTokenUsage: { ...first.initialRequestTokenUsage },
    totalTokenUsage: { ...first.totalTokenUsage },
    totalRequestsToAssistant: first.totalRequestsToAssistant,
    callsPerTool: { ...first.callsPerTool },
    totalToolCalls: first.totalToolCalls,
    initialRequestCostUsd: { ...first.initialRequestCostUsd },
    totalCostUsd: { ...first.totalCostUsd },
  };

  for (const u of list.slice(1)) {
    out.totalTokenUsage.inputTokens += u.totalTokenUsage.inputTokens;
    out.totalTokenUsage.cachedInputTokens += u.totalTokenUsage.cachedInputTokens;
    out.totalTokenUsage.cacheWriteTokens += u.totalTokenUsage.cacheWriteTokens;
    out.totalTokenUsage.reasoningTokens += u.totalTokenUsage.reasoningTokens;
    out.totalTokenUsage.outputTokens += u.totalTokenUsage.outputTokens;
    out.totalTokenUsage.totalTokens += u.totalTokenUsage.totalTokens;

    out.totalRequestsToAssistant += u.totalRequestsToAssistant;
    out.totalToolCalls += u.totalToolCalls;

    for (const [k, v] of Object.entries(u.callsPerTool)) {
      out.callsPerTool[k] = (out.callsPerTool[k] ?? 0) + v;
    }

    out.totalCostUsd.inputUsd += u.totalCostUsd.inputUsd;
    out.totalCostUsd.cachedInputUsd += u.totalCostUsd.cachedInputUsd;
    out.totalCostUsd.reasoningUsd += u.totalCostUsd.reasoningUsd;
    out.totalCostUsd.outputUsd += u.totalCostUsd.outputUsd;
    out.totalCostUsd.totalUsd += u.totalCostUsd.totalUsd;
    // Full precision retained; rounding happens only at display/ledger.
  }

  // Per-step lists concatenate in call order — the aggregate's steps are every call's steps.
  const steps = list.flatMap((u) => u.steps ?? []);
  if (steps.length > 0) {
    out.steps = steps;
  }

  return out;
}

/**
 * The model that best represents a multi-model aggregate: the one that produced
 * the most output tokens (i.e. did the most work), tie-broken by spend. Entries
 * with no model are ignored; falls back to the first entry's model when nothing
 * has output/cost yet.
 */
function pickRepresentativeModel(list: UsageData[]): UsageData['model'] {
  let best: UsageData | undefined;
  for (const u of list) {
    if (!u?.model) {
      continue;
    }
    if (!best) {
      best = u;
      continue;
    }
    const moreOutput = u.totalTokenUsage.outputTokens > best.totalTokenUsage.outputTokens;
    const sameOutput = u.totalTokenUsage.outputTokens === best.totalTokenUsage.outputTokens;
    const moreCost = (u.totalCostUsd?.totalUsd ?? 0) > (best.totalCostUsd?.totalUsd ?? 0);
    if (moreOutput || (sameOutput && moreCost)) {
      best = u;
    }
  }
  return best?.model ?? list[0].model;
}

const TOKENS_PER_1M = 1_000_000;

const normalizeModelIdForPricing = (model: string): string => {
  const raw = String(model ?? '').trim();
  if (!raw) {
    return '';
  }

  // handle e.g. "openai:gpt-4o" or "openai/gpt-4o"
  const afterColon = raw.includes(':') ? raw.split(':').pop() ?? raw : raw;
  const afterSlash = afterColon.includes('/') ? afterColon.split('/').pop() ?? afterColon : afterColon;
  return afterSlash;
};

const normalizeServiceTierForPricing = (serviceTier?: string): UsagePricingTier => {
  const v = String(serviceTier ?? '')
    .trim()
    .toLowerCase();
  if (v === 'priority') {
    return 'priority';
  }
  if (v === 'flex') {
    return 'flex';
  }
  if (v === 'batch') {
    return 'batch';
  }
  return 'standard';
};

/**
 * Compute the USD cost of one request's token usage from the pricing row the
 * given {@link ModelDataResolver} resolves for `model` (pricing DATA lives
 * with the resolver's owner; this function owns only the math). Returns the
 * explicit all-zero cost when the resolver has no row for the model — callers
 * distinguish that "unpriced" case via the resolver (e.g. a `priced` flag on
 * the ledger), never by a fabricated price.
 */
export const calculateUsageCostUsd = (
  model: string,
  tokenUsage: TokenUsage,
  opts: { modelData: ModelDataResolver; serviceTier?: string }
): UsageCostUsd => {
  const tier = normalizeServiceTierForPricing(opts.serviceTier);
  const normalizedModelId = normalizeModelIdForPricing(model);
  const pricing = normalizedModelId ? opts.modelData.pricing(normalizedModelId, tier) : undefined;
  if (!pricing) {
    return {
      inputUsd: 0,
      cachedInputUsd: 0,
      reasoningUsd: 0,
      outputUsd: 0,
      totalUsd: 0,
    };
  }

  const input = Number.isFinite(tokenUsage.inputTokens) ? Number(tokenUsage.inputTokens) : 0;
  const cachedInput = Number.isFinite(tokenUsage.cachedInputTokens) ? Number(tokenUsage.cachedInputTokens) : 0;
  const cacheWrite = Number.isFinite(tokenUsage.cacheWriteTokens) ? Number(tokenUsage.cacheWriteTokens) : 0;
  const reasoning = Number.isFinite(tokenUsage.reasoningTokens) ? Number(tokenUsage.reasoningTokens) : 0;
  const output = Number.isFinite(tokenUsage.outputTokens) ? Number(tokenUsage.outputTokens) : 0;

  const inputTokens = Math.max(0, input);
  const cachedInputTokens = Math.max(0, cachedInput);
  const cacheWriteTokens = Math.max(0, cacheWrite);
  // Both cache reads and cache writes are carved out of inputTokens; whatever
  // remains is fresh input priced at the full rate.
  const nonCachedInputTokens = Math.max(0, inputTokens - cachedInputTokens - cacheWriteTokens);
  const reasoningTokens = Math.max(0, reasoning);
  const outputTokens = Math.max(0, output);

  const cachedRate =
    typeof pricing.cachedInputUsdPer1M === 'number' ? pricing.cachedInputUsdPer1M : pricing.inputUsdPer1M;
  // Cache writes cost a PREMIUM over fresh input (e.g. Anthropic 5-min cache = 1.25x).
  // Fall back to the full input rate where a model has no distinct write rate.
  const cacheWriteRate =
    typeof pricing.cacheWriteUsdPer1M === 'number' ? pricing.cacheWriteUsdPer1M : pricing.inputUsdPer1M;

  const inputUsd =
    (nonCachedInputTokens * pricing.inputUsdPer1M +
      cachedInputTokens * cachedRate +
      cacheWriteTokens * cacheWriteRate) /
    TOKENS_PER_1M;
  const cachedInputUsd = (cachedInputTokens * cachedRate) / TOKENS_PER_1M;
  const reasoningUsd = (reasoningTokens * pricing.outputUsdPer1M) / TOKENS_PER_1M;
  const outputUsd = (outputTokens * pricing.outputUsdPer1M) / TOKENS_PER_1M;
  // total = input + output ONLY. `inputUsd` already folds in cached + cache-write;
  // `reasoningUsd` is already inside `outputUsd` (providers count reasoning within
  // output_tokens). Adding either would double-count. Full precision is retained
  // here — rounding to cents happens at the display/ledger boundary, never per
  // request (per-step rounding silently zeroed sub-cent calls).
  const totalUsd = inputUsd + outputUsd;

  return {
    inputUsd,
    cachedInputUsd,
    reasoningUsd,
    outputUsd,
    totalUsd,
  };
};
