import { Conversation } from '../../src/Conversation';
import { aggregateUsageData, type UsageData } from '../../src/UsageData';
import { fixtureModelData } from './fixtureModelData';

/**
 * Per-step usage (the `UsageData.steps` seam): a tool loop's steps are individually billed
 * provider requests, and the SDK stamps each step's usage on the step. The mapping must keep
 * that list — in loop order, every token class per step, the step's tool-call count — beside
 * the summed totals it already produced, so a ledger can split the FIRST request (the
 * cross-turn cache read) from the later re-reads. Pure unit: no network, the private mapper
 * reached through a typed cast on an instance (the house idiom).
 */
type ConversationInternals = {
  mapSdkUsage(
    sdkUsage: unknown,
    modelString: string,
    steps?: Array<{ toolCalls?: Array<{ toolName?: string }>; usage?: unknown }>
  ): UsageData;
};

const conversation = () =>
  new Conversation({ modelData: fixtureModelData, name: 'per-step-usage' }) as unknown as ConversationInternals;

const sdkUsage = (over: {
  inputTokens: number;
  outputTokens: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
}) => ({
  inputTokens: over.inputTokens,
  outputTokens: over.outputTokens,
  totalTokens: over.inputTokens + over.outputTokens,
  inputTokenDetails: { cacheReadTokens: over.cacheRead ?? 0, cacheWriteTokens: over.cacheWrite ?? 0 },
  outputTokenDetails: { reasoningTokens: over.reasoning ?? 0 },
});

describe('UsageData.steps — per-step usage rides beside the summed totals', () => {
  test('a two-step loop lists both steps in order with every token class and the tool-call count', () => {
    const step1 = sdkUsage({
      inputTokens: 40_000,
      outputTokens: 300,
      cacheRead: 9_000,
      cacheWrite: 31_000,
      reasoning: 120,
    });
    const step2 = sdkUsage({
      inputTokens: 41_000,
      outputTokens: 900,
      cacheRead: 40_000,
      cacheWrite: 1_000,
      reasoning: 200,
    });
    const summed = sdkUsage({
      inputTokens: 81_000,
      outputTokens: 1_200,
      cacheRead: 49_000,
      cacheWrite: 32_000,
      reasoning: 320,
    });

    const usage = conversation().mapSdkUsage(summed, 'claude-opus-4-8', [
      { toolCalls: [{ toolName: 'editThoughts' }], usage: step1 },
      { toolCalls: [], usage: step2 },
    ]);

    // The summed totals are untouched by the seam.
    expect(usage.totalTokenUsage.inputTokens).toBe(81_000);
    expect(usage.totalTokenUsage.cachedInputTokens).toBe(49_000);
    expect(usage.totalRequestsToAssistant).toBe(2);
    expect(usage.callsPerTool).toEqual({ editThoughts: 1 });

    expect(usage.steps).toEqual([
      {
        inputTokens: 40_000,
        cachedInputTokens: 9_000,
        cacheWriteTokens: 31_000,
        reasoningTokens: 120,
        outputTokens: 300,
        totalTokens: 40_300,
        toolCalls: 1,
      },
      {
        inputTokens: 41_000,
        cachedInputTokens: 40_000,
        cacheWriteTokens: 1_000,
        reasoningTokens: 200,
        outputTokens: 900,
        totalTokens: 41_900,
        toolCalls: 0,
      },
    ]);
    // Σ steps reconciles to the summed totals — the list is a partition of the sum, not a copy.
    const stepInput = usage.steps!.reduce((sum, step) => sum + step.inputTokens, 0);
    expect(stepInput).toBe(usage.totalTokenUsage.inputTokens);
  });

  test('steps without usage (the partial-usage placeholders) produce NO list — never fabricated zero rows', () => {
    const summed = sdkUsage({ inputTokens: 1_000, outputTokens: 50 });
    const usage = conversation().mapSdkUsage(summed, 'claude-opus-4-8', [{}, {}, {}]);
    expect(usage.totalRequestsToAssistant).toBe(3);
    expect('steps' in usage).toBe(false);
  });

  test('aggregateUsageData concatenates the calls’ step lists in call order', () => {
    const conv = conversation();
    const first = conv.mapSdkUsage(sdkUsage({ inputTokens: 100, outputTokens: 10 }), 'claude-opus-4-8', [
      { usage: sdkUsage({ inputTokens: 100, outputTokens: 10 }) },
    ]);
    const second = conv.mapSdkUsage(sdkUsage({ inputTokens: 700, outputTokens: 30 }), 'claude-opus-4-8', [
      { usage: sdkUsage({ inputTokens: 300, outputTokens: 10 }) },
      { usage: sdkUsage({ inputTokens: 400, outputTokens: 20 }) },
    ]);
    const aggregate = aggregateUsageData([first, second])!;
    expect(aggregate.steps?.map((step) => step.inputTokens)).toEqual([100, 300, 400]);
    expect(aggregate.totalTokenUsage.inputTokens).toBe(800);
  });
});
