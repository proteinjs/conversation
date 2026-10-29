import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

/**
 * A skill's provider-defined tools are asked for WITH THE MODEL the call rides (prod 2026-10-28, v1.32.0:
 * the dev skill's BrowserSkill handed Anthropic's `computer_20251124` to every Anthropic model, and
 * claude-sonnet-5-5 refuses that tool type — HTTP 400 `invalid_request_error`, "'claude-sonnet-5-5' does
 * not support tool types: computer_20251124" — on the bounded utterance's call and the step's alike, so
 * every turn in a workspace-bound chat on that model ended before a word). A provider's built-in tool
 * types are accepted PER MODEL, so the hook that hands them is told the model id (the catalog's id, after
 * any `provider:` prefix) and hands a tool only where the provider takes it.
 */

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

type HookCall = { provider: string; modelId: string | undefined };

const recordingSkill = (calls: HookCall[]): ConversationSkill => ({
  getId: () => 'provider-tools-carry-the-model-skill',
  getName: () => 'ProviderToolsCarryTheModelSkill',
  getSystemMessages: () => [],
  getFunctions: () => [],
  getMessageModerators: () => [] as MessageModerator[],
  getProviderDefinedTools: (provider: string, modelId?: string) => {
    calls.push({ provider, modelId });
    return {};
  },
});

const runTurn = async (modelId: string): Promise<HookCall[]> => {
  const calls: HookCall[] = [];
  const model = new MockLanguageModelV3({
    modelId,
    doStream: async () => ({ stream: textStep('done') }),
  });
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'provider-tools-carry-the-model-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [recordingSkill(calls)],
  });
  const result = await conversation.generateStream({ messages: ['hello'], model: model as never });
  for await (const part of result.fullStream) {
    void part;
  }
  return calls;
};

type ModelIdInternals = { constructor: { bareModelId(modelString: string): string } };

describe('provider-defined tools carry the model they are asked for', () => {
  test('the hook is told the provider AND the model id of the call', async () => {
    const calls = await runTurn('claude-sonnet-5-5');
    expect(calls.length).toBeGreaterThan(0);
    for (const call of calls) {
      expect(call.provider).toBe('anthropic');
      expect(call.modelId).toBe('claude-sonnet-5-5');
    }
  });

  test('another provider is named with its own model id', async () => {
    const calls = await runTurn('gpt-5.6-sol');
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.every((call) => call.provider === 'openai' && call.modelId === 'gpt-5.6-sol')).toBe(true);
  });

  test("the model id handed to the hook is the catalog's id — any `provider:` prefix stripped", () => {
    const conversation = new Conversation({ modelData: fixtureModelData, name: 'bare-model-id', logLevel: 'error' });
    const { constructor } = conversation as unknown as ModelIdInternals;
    expect(constructor.bareModelId('anthropic:claude-sonnet-5-5')).toBe('claude-sonnet-5-5');
    expect(constructor.bareModelId('claude-sonnet-5-5')).toBe('claude-sonnet-5-5');
  });
});
