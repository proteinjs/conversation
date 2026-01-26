import { TiktokenModel } from 'tiktoken';

export type TokenUsage = {
  inputTokens: number;
  cachedInputTokens: number;
  reasoningTokens: number;
  outputTokens: number;
  totalTokens: number;
};

export type ModelApiCost = {
  /** USD per 1M input tokens */
  inputUsdPer1M: number;
  /** USD per 1M cached input tokens (if supported) */
  cachedInputUsdPer1M?: number;
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
};

type UsageDataAccumulatorParams = {
  model: TiktokenModel;
};

export class UsageDataAccumulator {
  private processedInitialRequest = false;
  public usageData: UsageData;

  constructor({ model }: UsageDataAccumulatorParams) {
    this.usageData = {
      model,
      initialRequestTokenUsage: {
        inputTokens: 0,
        reasoningTokens: 0,
        cachedInputTokens: 0,
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

    const cost = calculateUsageCostUsd(this.usageData.model, tokenUsage, { serviceTier: opts?.serviceTier });

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

      this.usageData.totalCostUsd = roundUsageCostUsdToCents(this.usageData.totalCostUsd);
    }

    this.usageData.totalTokenUsage = {
      inputTokens: this.usageData.totalTokenUsage.inputTokens + tokenUsage.inputTokens,
      cachedInputTokens: this.usageData.totalTokenUsage.cachedInputTokens + tokenUsage.cachedInputTokens,
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
    model: first.model,
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

    out.totalCostUsd = roundUsageCostUsdToCents(out.totalCostUsd);
  }

  return out;
}

/**
 * Standard tier costs (USD per 1M tokens) based on the pricing on OpenAI's website.
 */
export const MODEL_API_COST_USD_PER_1M_TOKENS_STANDARD: Record<string, ModelApiCost> = {
  'gpt-5.2': { inputUsdPer1M: 1.75, cachedInputUsdPer1M: 0.175, outputUsdPer1M: 14.0 },
  'gpt-5.1': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-5': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-5-mini': { inputUsdPer1M: 0.25, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 2.0 },
  'gpt-5-nano': { inputUsdPer1M: 0.05, cachedInputUsdPer1M: 0.005, outputUsdPer1M: 0.4 },

  'gpt-5.2-chat-latest': { inputUsdPer1M: 1.75, cachedInputUsdPer1M: 0.175, outputUsdPer1M: 14.0 },
  'gpt-5.1-chat-latest': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-5-chat-latest': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },

  'gpt-5.2-codex': { inputUsdPer1M: 1.75, cachedInputUsdPer1M: 0.175, outputUsdPer1M: 14.0 },
  'gpt-5.1-codex-max': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-5.1-codex': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-5-codex': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },

  'gpt-5.2-pro': { inputUsdPer1M: 21.0, outputUsdPer1M: 168.0 },
  'gpt-5-pro': { inputUsdPer1M: 15.0, outputUsdPer1M: 120.0 },

  'gpt-4.1': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 8.0 },
  'gpt-4.1-mini': { inputUsdPer1M: 0.4, cachedInputUsdPer1M: 0.1, outputUsdPer1M: 1.6 },
  'gpt-4.1-nano': { inputUsdPer1M: 0.1, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 0.4 },

  'gpt-4o': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 1.25, outputUsdPer1M: 10.0 },
  'gpt-4o-2024-05-13': { inputUsdPer1M: 5.0, outputUsdPer1M: 15.0 },
  'gpt-4o-mini': { inputUsdPer1M: 0.15, cachedInputUsdPer1M: 0.075, outputUsdPer1M: 0.6 },

  'gpt-realtime': { inputUsdPer1M: 4.0, cachedInputUsdPer1M: 0.4, outputUsdPer1M: 16.0 },
  'gpt-realtime-mini': { inputUsdPer1M: 0.6, cachedInputUsdPer1M: 0.06, outputUsdPer1M: 2.4 },

  'gpt-4o-realtime-preview': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 2.5, outputUsdPer1M: 20.0 },
  'gpt-4o-mini-realtime-preview': { inputUsdPer1M: 0.6, cachedInputUsdPer1M: 0.3, outputUsdPer1M: 2.4 },

  'gpt-audio': { inputUsdPer1M: 2.5, outputUsdPer1M: 10.0 },
  'gpt-audio-mini': { inputUsdPer1M: 0.6, outputUsdPer1M: 2.4 },
  'gpt-4o-audio-preview': { inputUsdPer1M: 2.5, outputUsdPer1M: 10.0 },
  'gpt-4o-mini-audio-preview': { inputUsdPer1M: 0.15, outputUsdPer1M: 0.6 },

  o1: { inputUsdPer1M: 15.0, cachedInputUsdPer1M: 7.5, outputUsdPer1M: 60.0 },
  'o1-pro': { inputUsdPer1M: 150.0, outputUsdPer1M: 600.0 },
  'o1-mini': { inputUsdPer1M: 1.1, cachedInputUsdPer1M: 0.55, outputUsdPer1M: 4.4 },

  o3: { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 8.0 },
  'o3-pro': { inputUsdPer1M: 20.0, outputUsdPer1M: 80.0 },
  'o3-mini': { inputUsdPer1M: 1.1, cachedInputUsdPer1M: 0.55, outputUsdPer1M: 4.4 },
  'o3-deep-research': { inputUsdPer1M: 10.0, cachedInputUsdPer1M: 2.5, outputUsdPer1M: 40.0 },

  'o4-mini': { inputUsdPer1M: 1.1, cachedInputUsdPer1M: 0.275, outputUsdPer1M: 4.4 },
  'o4-mini-deep-research': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 8.0 },

  'gpt-5-search-api': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gpt-4o-mini-search-preview': { inputUsdPer1M: 0.15, outputUsdPer1M: 0.6 },
  'gpt-4o-search-preview': { inputUsdPer1M: 2.5, outputUsdPer1M: 10.0 },
  'computer-use-preview': { inputUsdPer1M: 3.0, outputUsdPer1M: 12.0 },

  'gpt-5.1-codex-mini': { inputUsdPer1M: 0.25, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 2.0 },
  'codex-mini-latest': { inputUsdPer1M: 1.5, cachedInputUsdPer1M: 0.375, outputUsdPer1M: 6.0 },
};

export const MODEL_API_COST_USD_PER_1M_TOKENS_BATCH: Record<string, ModelApiCost> = {
  'gpt-5.2': { inputUsdPer1M: 0.875, cachedInputUsdPer1M: 0.0875, outputUsdPer1M: 7.0 },
  'gpt-5.1': { inputUsdPer1M: 0.625, cachedInputUsdPer1M: 0.0625, outputUsdPer1M: 5.0 },
  'gpt-5': { inputUsdPer1M: 0.625, cachedInputUsdPer1M: 0.0625, outputUsdPer1M: 5.0 },
  'gpt-5-mini': { inputUsdPer1M: 0.125, cachedInputUsdPer1M: 0.0125, outputUsdPer1M: 1.0 },
  'gpt-5-nano': { inputUsdPer1M: 0.025, cachedInputUsdPer1M: 0.0025, outputUsdPer1M: 0.2 },

  'gpt-5.2-pro': { inputUsdPer1M: 10.5, outputUsdPer1M: 84.0 },
  'gpt-5-pro': { inputUsdPer1M: 7.5, outputUsdPer1M: 60.0 },

  'gpt-4.1': { inputUsdPer1M: 1.0, outputUsdPer1M: 4.0 },
  'gpt-4.1-mini': { inputUsdPer1M: 0.2, outputUsdPer1M: 0.8 },
  'gpt-4.1-nano': { inputUsdPer1M: 0.05, outputUsdPer1M: 0.2 },

  'gpt-4o': { inputUsdPer1M: 1.25, outputUsdPer1M: 5.0 },
  'gpt-4o-2024-05-13': { inputUsdPer1M: 2.5, outputUsdPer1M: 7.5 },
  'gpt-4o-mini': { inputUsdPer1M: 0.075, outputUsdPer1M: 0.3 },

  o1: { inputUsdPer1M: 7.5, outputUsdPer1M: 30.0 },
  'o1-pro': { inputUsdPer1M: 75.0, outputUsdPer1M: 300.0 },

  'o3-pro': { inputUsdPer1M: 10.0, outputUsdPer1M: 40.0 },
  o3: { inputUsdPer1M: 1.0, outputUsdPer1M: 4.0 },
  'o3-deep-research': { inputUsdPer1M: 5.0, outputUsdPer1M: 20.0 },

  'o4-mini': { inputUsdPer1M: 0.55, outputUsdPer1M: 2.2 },
  'o4-mini-deep-research': { inputUsdPer1M: 1.0, outputUsdPer1M: 4.0 },
  'o3-mini': { inputUsdPer1M: 0.55, outputUsdPer1M: 2.2 },
  'o1-mini': { inputUsdPer1M: 0.55, outputUsdPer1M: 2.2 },

  'computer-use-preview': { inputUsdPer1M: 1.5, outputUsdPer1M: 6.0 },
};

export const MODEL_API_COST_USD_PER_1M_TOKENS_FLEX: Record<string, ModelApiCost> = {
  'gpt-5.2': { inputUsdPer1M: 0.875, cachedInputUsdPer1M: 0.0875, outputUsdPer1M: 7.0 },
  'gpt-5.1': { inputUsdPer1M: 0.625, cachedInputUsdPer1M: 0.0625, outputUsdPer1M: 5.0 },
  'gpt-5': { inputUsdPer1M: 0.625, cachedInputUsdPer1M: 0.0625, outputUsdPer1M: 5.0 },
  'gpt-5-mini': { inputUsdPer1M: 0.125, cachedInputUsdPer1M: 0.0125, outputUsdPer1M: 1.0 },
  'gpt-5-nano': { inputUsdPer1M: 0.025, cachedInputUsdPer1M: 0.0025, outputUsdPer1M: 0.2 },

  o3: { inputUsdPer1M: 1.0, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 4.0 },
  'o4-mini': { inputUsdPer1M: 0.55, cachedInputUsdPer1M: 0.138, outputUsdPer1M: 2.2 },
};

export const MODEL_API_COST_USD_PER_1M_TOKENS_PRIORITY: Record<string, ModelApiCost> = {
  'gpt-5.2': { inputUsdPer1M: 3.5, cachedInputUsdPer1M: 0.35, outputUsdPer1M: 28.0 },
  'gpt-5.1': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 20.0 },
  'gpt-5': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 20.0 },
  'gpt-5-mini': { inputUsdPer1M: 0.45, cachedInputUsdPer1M: 0.045, outputUsdPer1M: 3.6 },

  'gpt-5.2-codex': { inputUsdPer1M: 3.5, cachedInputUsdPer1M: 0.35, outputUsdPer1M: 28.0 },
  'gpt-5.1-codex-max': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 20.0 },
  'gpt-5.1-codex': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 20.0 },
  'gpt-5-codex': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 20.0 },

  'gpt-4.1': { inputUsdPer1M: 3.5, cachedInputUsdPer1M: 0.875, outputUsdPer1M: 14.0 },
  'gpt-4.1-mini': { inputUsdPer1M: 0.7, cachedInputUsdPer1M: 0.175, outputUsdPer1M: 2.8 },
  'gpt-4.1-nano': { inputUsdPer1M: 0.2, cachedInputUsdPer1M: 0.05, outputUsdPer1M: 0.8 },

  'gpt-4o': { inputUsdPer1M: 4.25, cachedInputUsdPer1M: 2.125, outputUsdPer1M: 17.0 },
  'gpt-4o-2024-05-13': { inputUsdPer1M: 8.75, outputUsdPer1M: 26.25 },
  'gpt-4o-mini': { inputUsdPer1M: 0.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 1.0 },

  o3: { inputUsdPer1M: 3.5, cachedInputUsdPer1M: 0.875, outputUsdPer1M: 14.0 },
  'o4-mini': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 8.0 },
};

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

type UsagePricingTier = 'standard' | 'batch' | 'flex' | 'priority';

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

const resolveModelApiCost = (model: string, tier?: UsagePricingTier): ModelApiCost | undefined => {
  const m = normalizeModelIdForPricing(model);
  if (!m) {
    return undefined;
  }

  const t: UsagePricingTier = tier ?? 'standard';
  const table =
    t === 'priority'
      ? MODEL_API_COST_USD_PER_1M_TOKENS_PRIORITY
      : t === 'flex'
        ? MODEL_API_COST_USD_PER_1M_TOKENS_FLEX
        : t === 'batch'
          ? MODEL_API_COST_USD_PER_1M_TOKENS_BATCH
          : MODEL_API_COST_USD_PER_1M_TOKENS_STANDARD;

  const direct = table[m];
  if (direct) {
    return direct;
  }

  // common suffix normalization (only if the base exists)
  if (m.endsWith('-latest')) {
    const base = m.slice(0, -'-latest'.length);
    const baseCost = table[base];
    if (baseCost) {
      return baseCost;
    }
  }

  return undefined;
};

export const calculateUsageCostUsd = (
  model: string,
  tokenUsage: TokenUsage,
  opts?: { serviceTier?: string }
): UsageCostUsd => {
  const tier = normalizeServiceTierForPricing(opts?.serviceTier);
  const pricing = resolveModelApiCost(model, tier);
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
  const reasoning = Number.isFinite(tokenUsage.reasoningTokens) ? Number(tokenUsage.reasoningTokens) : 0;
  const output = Number.isFinite(tokenUsage.outputTokens) ? Number(tokenUsage.outputTokens) : 0;

  const inputTokens = Math.max(0, input);
  const cachedInputTokens = Math.max(0, cachedInput);
  const nonCachedInputTokens = Math.max(0, inputTokens - cachedInputTokens);
  const reasoningTokens = Math.max(0, reasoning);
  const outputTokens = Math.max(0, output);

  const cachedRate =
    typeof pricing.cachedInputUsdPer1M === 'number' ? pricing.cachedInputUsdPer1M : pricing.inputUsdPer1M;

  const inputUsd = (nonCachedInputTokens * pricing.inputUsdPer1M + cachedInputTokens * cachedRate) / TOKENS_PER_1M;
  const cachedInputUsd = (cachedInputTokens * cachedRate) / TOKENS_PER_1M;
  const reasoningUsd = (reasoningTokens * pricing.outputUsdPer1M) / TOKENS_PER_1M;
  const outputUsd = (outputTokens * pricing.outputUsdPer1M) / TOKENS_PER_1M;
  const totalUsd = inputUsd + outputUsd;

  return roundUsageCostUsdToCents({
    inputUsd,
    cachedInputUsd,
    reasoningUsd,
    outputUsd,
    totalUsd,
  });
};

function roundToHundredths(value: number): number {
  return Math.round(value * 100) / 100;
}

function roundUsageCostUsdToCents(cost: UsageCostUsd): UsageCostUsd {
  const inputUsd = roundToHundredths(cost.inputUsd);
  const cachedInputUsd = roundToHundredths(cost.cachedInputUsd);
  const reasoningUsd = roundToHundredths(cost.reasoningUsd);
  const outputUsd = roundToHundredths(cost.outputUsd);
  const totalUsd = roundToHundredths(inputUsd + cachedInputUsd + outputUsd);

  return {
    inputUsd,
    cachedInputUsd,
    reasoningUsd,
    outputUsd,
    totalUsd,
  };
}
