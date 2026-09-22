import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import type { LanguageModelV3CallOptions } from '@ai-sdk/provider';
import { Conversation, type StreamPart } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * ONE GENERATION PATH FOR OPENAI — no network, no keys. A MockLanguageModelV3 stands in as the
 * OpenAI Responses model, so every `generateStream` round runs the REAL wiring (Conversation →
 * transport retry → the model) and the mock records what the request carried.
 *
 * Until 2026-09-22 `generateStream` routed OpenAI calls at high · xhigh · max effort, and every
 * model whose id carried "pro", to a separate polling transport that answered in one piece,
 * asked for no reasoning summary, attached no web-search tool and flattened image parts to
 * text — measured live on GPT-6 Sol and GPT-6 Astra: 80 reasoning pieces at medium, zero at high.
 * The heuristic is gone: every effort and every id streams through the same path. Background mode
 * survives only as the explicit `backgroundMode: true` opt-in.
 */
const TIMEOUT = 30_000;
const usage = {
  inputTokens: { total: 4_500, noCache: 68, cacheRead: 4_432, cacheWrite: 0 },
  outputTokens: { total: 110, text: 70, reasoning: 40 },
};
/** A reasoned answer the way the Responses stream delivers one: a summary in pieces, then the text in pieces. */
const reasonedAnswer = () =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'reasoning-start' as const, id: 'r1' },
    { type: 'reasoning-delta' as const, id: 'r1', delta: 'First leg 2 h 45 min, ' },
    { type: 'reasoning-delta' as const, id: 'r1', delta: 'then the stop, then 2 h.' },
    { type: 'reasoning-end' as const, id: 'r1' },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: '14:50. ' },
    { type: 'text-delta' as const, id: 't1', delta: 'The trip takes 5 h 10 min.' },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'one-generation-path-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

/** An OpenAI-shaped model that records every call. */
const recordingModel = (modelId: string, calls: LanguageModelV3CallOptions[]) =>
  new MockLanguageModelV3({
    provider: 'openai.responses',
    modelId,
    doStream: async (call) => {
      calls.push(call);
      return { stream: reasonedAnswer() };
    },
  });

const drain = async (stream: AsyncIterable<StreamPart>): Promise<StreamPart[]> => {
  const parts: StreamPart[] = [];
  for await (const part of stream) {
    parts.push(part);
  }
  return parts;
};

describe('one generation path for OpenAI', () => {
  test(
    'a high-effort call streams through the same path as any other: the reasoning summary in pieces, the text in pieces, the effort and the summary on the request',
    async () => {
      const calls: LanguageModelV3CallOptions[] = [];
      const model = recordingModel('gpt-6-sol', calls);
      const result = await newConversation().generateStream({
        messages: ['A train leaves at 9:40…'],
        model,
        reasoningEffort: 'high',
      });
      const parts = await drain(result.fullStream);
      expect(calls).toHaveLength(1);
      expect(parts.filter((p) => p.type === 'reasoning-delta')).toHaveLength(2);
      expect(parts.filter((p) => p.type === 'text-delta')).toHaveLength(2);
      expect(await result.reasoning).toBe('First leg 2 h 45 min, then the stop, then 2 h.');
      expect(await result.text).toBe('14:50. The trip takes 5 h 10 min.');
      const openai = (calls[0].providerOptions as { openai?: Record<string, unknown> } | undefined)?.openai ?? {};
      expect(openai.reasoningEffort).toBe('high');
      expect(openai.reasoningSummary).toBe('auto');
      expect(openai.forceReasoning).toBe(true);
      // The usage is the stream's, complete: the cached read and the reasoning tokens.
      const totals = (await result.usage).totalTokenUsage;
      expect(totals.reasoningTokens).toBe(40);
      expect(totals.cachedInputTokens).toBe(4_432);
      expect(totals.inputTokens).toBe(4_500);
    },
    TIMEOUT
  );

  test(
    'xhigh and max, and an id that carries "pro", take the same path — the mock answers every one',
    async () => {
      for (const [modelId, effort] of [
        ['gpt-6-sol', 'xhigh'],
        ['gpt-6-sol', 'max'],
        ['gpt-5.5-pro', 'auto'],
        ['gpt-6-astra', 'max'],
      ] as const) {
        const calls: LanguageModelV3CallOptions[] = [];
        const result = await newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model: recordingModel(modelId, calls),
          reasoningEffort: effort,
        });
        const parts = await drain(result.fullStream);
        expect({ modelId, effort, calls: calls.length }).toEqual({ modelId, effort, calls: 1 });
        expect(parts.some((p) => p.type === 'reasoning-delta')).toBe(true);
      }
    },
    TIMEOUT
  );

  test(
    'the web-search tool rides a high-effort request exactly as it rides a medium one',
    async () => {
      const at = async (effort: 'medium' | 'high') => {
        const calls: LanguageModelV3CallOptions[] = [];
        const result = await newConversation().generateStream({
          messages: ['What is a news headline from today?'],
          model: recordingModel('gpt-6-sol', calls),
          reasoningEffort: effort,
          webSearch: true,
        });
        await drain(result.fullStream);
        return (calls[0].tools ?? []).map((t) => t.name);
      };
      const medium = await at('medium');
      expect(medium).toContain('web_search');
      expect(await at('high')).toEqual(medium);
    },
    TIMEOUT
  );

  test(
    'an image part in the turn reaches the model intact at max effort (the polling path flattened it to text)',
    async () => {
      const calls: LanguageModelV3CallOptions[] = [];
      const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).toString('base64');
      const result = await newConversation().generateStream({
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'What color is this?' },
              { type: 'image_url', image_url: { url: `data:image/png;base64,${png}` } },
            ],
          },
        ],
        model: recordingModel('gpt-6-astra', calls),
        reasoningEffort: 'max',
      });
      await drain(result.fullStream);
      const user = calls[0].prompt.find((m) => m.role === 'user');
      const kinds = Array.isArray(user?.content) ? user!.content.map((p) => p.type) : ['<string>'];
      expect(kinds).toEqual(['text', 'file']);
    },
    TIMEOUT
  );

  test(
    'background mode is the explicit opt-in only — a caller that asks for it still leaves the stream',
    async () => {
      const calls: LanguageModelV3CallOptions[] = [];
      const model = recordingModel('gpt-6-astra', calls);
      // The polling transport needs a real client; with no caller-supplied key it cannot be built
      // — the point pinned here is only that the mock (the streaming path) was NOT consulted.
      await expect(
        newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model,
          reasoningEffort: 'low',
          backgroundMode: true,
        })
      ).rejects.toBeDefined();
      expect(calls).toHaveLength(0);
    },
    TIMEOUT
  );
});
