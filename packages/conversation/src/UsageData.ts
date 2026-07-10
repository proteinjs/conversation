import { TiktokenModel } from 'tiktoken';

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

/**
 * Standard tier costs (USD per 1M tokens) based on the pricing on OpenAI's website.
 */
export const MODEL_API_COST_USD_PER_1M_TOKENS_STANDARD: Record<string, ModelApiCost> = {
  // Short-context pricing from developers.openai.com/api/docs/pricing.
  // Long-context rates (>~272k input) are ~2x these; tracked here as the
  // default because chats below the threshold hit this tier.
  'gpt-5.5': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 30.0 },
  'gpt-5.4': { inputUsdPer1M: 2.5, cachedInputUsdPer1M: 0.25, outputUsdPer1M: 15.0 },
  'gpt-5.4-mini': { inputUsdPer1M: 0.4, cachedInputUsdPer1M: 0.1, outputUsdPer1M: 1.6 },
  'gpt-5.4-nano': { inputUsdPer1M: 0.1, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 0.4 },
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

  // ── Anthropic Claude models ──
  // cacheWriteUsdPer1M = 1.25x input — the 5-minute ephemeral cache write rate
  // (N3XA writes default ephemeral caches via applyAnthropicPromptCaching, no
  // explicit ttl). cachedInputUsdPer1M is the cache-READ rate (0.1x input).
  'claude-fable-5': { inputUsdPer1M: 10.0, cachedInputUsdPer1M: 1.0, cacheWriteUsdPer1M: 12.5, outputUsdPer1M: 50.0 },
  'claude-opus-4-8': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, cacheWriteUsdPer1M: 6.25, outputUsdPer1M: 25.0 },
  'claude-opus-4-7': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, cacheWriteUsdPer1M: 6.25, outputUsdPer1M: 25.0 },
  'claude-opus-4-6': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, cacheWriteUsdPer1M: 6.25, outputUsdPer1M: 25.0 },
  'claude-opus-4.5': { inputUsdPer1M: 5.0, cachedInputUsdPer1M: 0.5, cacheWriteUsdPer1M: 6.25, outputUsdPer1M: 25.0 },
  'claude-opus-4-20250514': {
    inputUsdPer1M: 15.0,
    cachedInputUsdPer1M: 1.5,
    cacheWriteUsdPer1M: 18.75,
    outputUsdPer1M: 75.0,
  },
  'claude-opus-4.1': { inputUsdPer1M: 15.0, cachedInputUsdPer1M: 1.5, cacheWriteUsdPer1M: 18.75, outputUsdPer1M: 75.0 },
  'claude-sonnet-4-6': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.3, cacheWriteUsdPer1M: 3.75, outputUsdPer1M: 15.0 },
  'claude-sonnet-4.6': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.3, cacheWriteUsdPer1M: 3.75, outputUsdPer1M: 15.0 },
  'claude-sonnet-4-5': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.3, cacheWriteUsdPer1M: 3.75, outputUsdPer1M: 15.0 },
  'claude-sonnet-4.5': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.3, cacheWriteUsdPer1M: 3.75, outputUsdPer1M: 15.0 },
  'claude-sonnet-4-20250514': {
    inputUsdPer1M: 3.0,
    cachedInputUsdPer1M: 0.3,
    cacheWriteUsdPer1M: 3.75,
    outputUsdPer1M: 15.0,
  },
  'claude-haiku-4-5': { inputUsdPer1M: 1.0, cachedInputUsdPer1M: 0.1, cacheWriteUsdPer1M: 1.25, outputUsdPer1M: 5.0 },
  'claude-haiku-4.5': { inputUsdPer1M: 1.0, cachedInputUsdPer1M: 0.1, cacheWriteUsdPer1M: 1.25, outputUsdPer1M: 5.0 },
  'claude-3-haiku-20240307': {
    inputUsdPer1M: 0.25,
    cachedInputUsdPer1M: 0.03,
    cacheWriteUsdPer1M: 0.3125,
    outputUsdPer1M: 1.25,
  },

  // ── Google Gemini models ──
  // cachedInputUsdPer1M reflects Google's documented implicit-cache discount
  // (~75% off input, i.e. 0.25x). Without it, cached tokens fall back to the
  // full input rate in calculateUsageCostUsd and overcharge ~4x.
  'gemini-3.1-pro-preview': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 12.0 },
  'gemini-3-pro-preview': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 12.0 },
  'gemini-3-flash-preview': { inputUsdPer1M: 0.5, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 3.0 },
  'gemini-3-flash': { inputUsdPer1M: 0.5, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 3.0 },
  // Live catalog id (ChatSettings modelCatalog) — mirrors gemini-3-flash.
  'gemini-3.5-flash': { inputUsdPer1M: 0.5, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 3.0 },
  'gemini-2.5-pro': { inputUsdPer1M: 1.25, cachedInputUsdPer1M: 0.125, outputUsdPer1M: 10.0 },
  'gemini-2.5-flash': { inputUsdPer1M: 0.3, cachedInputUsdPer1M: 0.03, outputUsdPer1M: 2.5 },
  'gemini-2.0-flash': { inputUsdPer1M: 0.1, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 0.4 },
  'gemini-2.0-flash-lite': { inputUsdPer1M: 0.1, cachedInputUsdPer1M: 0.025, outputUsdPer1M: 0.4 },

  // ── xAI Grok models ──
  // cachedInputUsdPer1M reflects xAI's published cache-read rate (0.25x input).
  // Live catalog ids: grok-4.3 (flagship, mirrors grok-4.20) and
  // grok-4-1-fast-reasoning (mirrors grok-4.1-fast) — both were absent and
  // therefore billed $0 on real calls.
  // grok-4.5 (released 2026-07-08): $2/$6 base, cache reads at a 75% discount
  // (0.25x input). xAI doubles both rates past 200k input tokens; ModelApiCost
  // has no context-tiered rates, so we record the base tier.
  'grok-4.5': { inputUsdPer1M: 2.0, cachedInputUsdPer1M: 0.5, outputUsdPer1M: 6.0 },
  'grok-4.3': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.75, outputUsdPer1M: 15.0 },
  'grok-4.20': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.75, outputUsdPer1M: 15.0 },
  'grok-4': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.75, outputUsdPer1M: 15.0 },
  'grok-4-fast': { inputUsdPer1M: 0.2, cachedInputUsdPer1M: 0.05, outputUsdPer1M: 0.5 },
  'grok-4.1-fast': { inputUsdPer1M: 0.2, cachedInputUsdPer1M: 0.05, outputUsdPer1M: 0.5 },
  'grok-4-1-fast-reasoning': { inputUsdPer1M: 0.2, cachedInputUsdPer1M: 0.05, outputUsdPer1M: 0.5 },
  'grok-3': { inputUsdPer1M: 3.0, cachedInputUsdPer1M: 0.75, outputUsdPer1M: 15.0 },
  'grok-3-mini': { inputUsdPer1M: 0.3, cachedInputUsdPer1M: 0.075, outputUsdPer1M: 0.5 },
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

/**
 * True when `model` resolves to a pricing row — i.e. its recorded cost is real,
 * not the all-zero fallback `calculateUsageCostUsd` returns for unknown models.
 * Lets consumers (e.g. the usage ledger / UI) distinguish "cost unavailable"
 * from a genuinely zero-cost request.
 */
export const isModelPriced = (model: string): boolean => !!resolveModelApiCost(model);

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
