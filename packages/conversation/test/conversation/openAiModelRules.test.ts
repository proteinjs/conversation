import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { OpenAiModelRules } from '../../src/OpenAiModelRules';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE PER-MODEL RULES FOR OPENAI, read from the vendor's versioning grammar. Recorded live
 * 2026-09-22: `@ai-sdk/openai` 3.0.65 read `gpt-6-sol` as a non-reasoning model (its list is a
 * `gpt-5` prefix), warned `reasoningEffort is not supported for non-reasoning models`, and sent
 * the request with neither the effort nor the reasoning summary — the call "passed" with the
 * effort never applied. The rules below do not trail a release, and are stated to the SDK on
 * every call (`forceReasoning`).
 */

describe('OpenAiModelRules', () => {
  test.each([
    ['gpt-6-sol', 6],
    ['gpt-6-astra', 6],
    ['gpt-5.6-terra', 5],
    ['gpt-5.5-pro', 5],
    ['gpt-4o', 4],
    ['openai:gpt-6-luna', 6],
    ['o3', undefined],
    ['claude-opus-5-5', undefined],
  ] as Array<[string, number | undefined]>)('generation(%s) → %s', (id, expected) => {
    expect(OpenAiModelRules.generation(id)).toBe(expected);
  });

  test.each([
    ['gpt-6-sol', true],
    ['gpt-6-luna', true],
    ['gpt-7-anything', true],
    ['gpt-5.6-sol', true],
    ['gpt-5.5-pro', true],
    ['gpt-5-chat-latest', false],
    ['gpt-4o', false],
    ['o1', true],
    ['o3-pro', true],
    ['o4-mini', true],
    ['openai:gpt-6-astra', true],
  ] as Array<[string, boolean]>)('reasons(%s) → %s', (id, expected) => {
    expect(OpenAiModelRules.reasons(id)).toBe(expected);
  });

  test.each([
    ['gpt-6-sol', 'max', 'max'],
    ['gpt-6-astra', 'max', 'max'],
    ['gpt-5.6-sol', 'max', 'xhigh'],
    ['gpt-5.5-pro', 'max', 'xhigh'],
    ['o3', 'max', 'xhigh'],
    ['gpt-6-sol', 'xhigh', 'xhigh'],
    ['gpt-6-sol', 'none', 'none'],
    ['gpt-6-sol', 'low', 'low'],
    ['gpt-6-sol', 'auto', undefined],
    ['gpt-6-sol', undefined, undefined],
  ] as Array<[string, any, string | undefined]>)('reasoningEffort(%s, %s) → %s', (id, effort, expected) => {
    expect(OpenAiModelRules.reasoningEffort(id, effort)).toBe(expected);
  });
});

describe('Conversation.buildProviderOptions (openai) states the rules to the SDK', () => {
  const conv = new Conversation({ modelData: fixtureModelData, name: 'test-openai-rules' });
  const build = (effort: any, model: string) =>
    (conv as any).buildProviderOptions('openai', { reasoningEffort: effort }, model).openai;

  // Every request is stateless (OpenAiResponseRetention): `store: false`, and on a reasoning
  // model the encrypted reasoning asked for by name — the two fields ride beside the rules.
  const stateless = { store: false, include: ['reasoning.encrypted_content'] };

  test('a GPT-6 model reasons — forceReasoning true, and max is max', () => {
    expect(build('max', 'gpt-6-sol')).toEqual({
      reasoningEffort: 'max',
      reasoningSummary: 'auto',
      forceReasoning: true,
      ...stateless,
    });
    expect(build('low', 'gpt-6-luna')).toEqual({
      reasoningEffort: 'low',
      reasoningSummary: 'auto',
      forceReasoning: true,
      ...stateless,
    });
    expect(build('auto', 'gpt-6-astra')).toEqual({ reasoningSummary: 'auto', forceReasoning: true, ...stateless });
  });

  test('a GPT-5 model reasons — max is its top level, xhigh', () => {
    expect(build('max', 'gpt-5.6-sol')).toEqual({
      reasoningEffort: 'xhigh',
      reasoningSummary: 'auto',
      forceReasoning: true,
      ...stateless,
    });
  });

  test('a non-reasoning model is said so', () => {
    expect(build(undefined, 'gpt-4o')).toEqual({ reasoningSummary: 'auto', forceReasoning: false, store: false });
  });
});

describe('the rules reach the request', () => {
  const usage = {
    inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 5, text: 5, reasoning: 0 },
  };
  const answer = () =>
    convertArrayToReadableStream([
      { type: 'stream-start' as const, warnings: [] },
      { type: 'text-start' as const, id: 't1' },
      { type: 'text-delta' as const, id: 't1', delta: '4' },
      { type: 'text-end' as const, id: 't1' },
      { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
    ]);

  test('a GPT-6 call carries forceReasoning and the effort in its provider options', async () => {
    const model = new MockLanguageModelV3({
      provider: 'openai.responses',
      modelId: 'gpt-6-sol',
      doStream: async () => ({ stream: answer() }),
    });
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'test-openai-rules-request',
      logLevel: 'error',
      limits: { enforceLimits: false },
    });
    const result = await conversation.generateResponse({
      messages: ['2+2?'],
      model: model as never,
      reasoningEffort: 'low',
    });
    expect(result.text).toBe('4');
    expect(model.doStreamCalls).toHaveLength(1);
    expect(model.doStreamCalls[0].providerOptions?.openai).toEqual({
      reasoningEffort: 'low',
      reasoningSummary: 'auto',
      forceReasoning: true,
      store: false,
      include: ['reasoning.encrypted_content'],
    });
  });
});
