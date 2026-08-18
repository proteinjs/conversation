import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, totalTokensReach } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';

/**
 * `GenerateStreamParams.maxTotalTokens` — the loop-seam token ceiling (a budget guardrail for
 * unattended callers, e.g. scheduled routine ticks): usage is judged at every step boundary via
 * the same `stopWhen` seam as `maxToolCalls`, and once the completed steps' cumulative total
 * reaches the ceiling the loop stops scheduling further steps. No network: a MockLanguageModelV3
 * scripts a would-be three-step tool loop with known per-step usage; the ceiling must cut it.
 */

const TIMEOUT = 30_000;

/** 500 in + 100 out = 600 total tokens per scripted step. */
const STEP_TOKENS = 600;
const usage = {
  inputTokens: { total: 500, noCache: 500, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 100, text: 100, reasoning: 0 },
};

const toolCallStep = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'doWork', input: '{}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

function buildSkill(fn: Function): ConversationSkill {
  return {
    getId: () => 'max-total-tokens-test-skill',
    getName: () => 'MaxTotalTokensTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [fn],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

const workTool: Function = {
  definition: {
    name: 'doWork',
    description: 'Does one unit of work.',
    parameters: { type: 'object', properties: {} },
  },
  call: async () => ({ ok: true }),
};

/** A model that would run `toolSteps` tool steps then a closing text step, counting calls. */
function buildScriptedModel(toolSteps: number): { model: MockLanguageModelV3; calls: () => number } {
  let call = 0;
  const model = new MockLanguageModelV3({
    doStream: async () => {
      call++;
      return { stream: call <= toolSteps ? toolCallStep(`tc-${call}`) : textStep('done') };
    },
  });
  return { model, calls: () => call };
}

function buildConversation(): Conversation {
  return new Conversation({
    name: 'max-total-tokens-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [buildSkill(workTool)],
  });
}

describe('Conversation.generateStream — maxTotalTokens loop ceiling', () => {
  test(
    'without a ceiling the scripted loop runs all steps',
    async () => {
      const { model, calls } = buildScriptedModel(2);
      const result = await buildConversation().generateResponse({
        messages: ['do the work'],
        model: model as never,
      });
      expect(result.text).toBe('done');
      expect(calls()).toBe(3);
    },
    TIMEOUT
  );

  test(
    'the ceiling stops the loop at the step boundary where cumulative usage reaches it',
    async () => {
      const { model, calls } = buildScriptedModel(2);
      // Step 1 ends at 600 < 1000 → step 2 runs; step 2 ends at 1200 ≥ 1000 → the loop wraps up:
      // the closing text step is never scheduled.
      await buildConversation().generateResponse({
        messages: ['do the work'],
        model: model as never,
        maxTotalTokens: STEP_TOKENS + 400,
      });
      expect(calls()).toBe(2);
    },
    TIMEOUT
  );

  test('totalTokensReach sums step usage and trips exactly at the budget', () => {
    const steps = [{ usage: { totalTokens: 600 } }, { usage: { totalTokens: 600 } }] as never;
    expect(totalTokensReach(1201)({ steps })).toBe(false);
    expect(totalTokensReach(1200)({ steps })).toBe(true);
    expect(totalTokensReach(100)({ steps })).toBe(true);
    // Falls back to input+output when a provider reports no total.
    const partial = [{ usage: { inputTokens: 500, outputTokens: 100 } }] as never;
    expect(totalTokensReach(600)({ steps: partial })).toBe(true);
    expect(totalTokensReach(601)({ steps: partial })).toBe(false);
    // No steps yet → never trips.
    expect(totalTokensReach(1)({ steps: [] as never })).toBe(false);
  });
});
