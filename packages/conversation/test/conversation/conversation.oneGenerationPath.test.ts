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
    'the polling transport is gone: a caller that still passes `backgroundMode` streams like everyone else',
    async () => {
      const calls: LanguageModelV3CallOptions[] = [];
      const model = recordingModel('gpt-6-astra', calls);
      const result = await newConversation().generateStream({
        messages: ['A train leaves at 9:40…'],
        model,
        reasoningEffort: 'low',
        backgroundMode: true,
        maxBackgroundWaitMs: 60_000,
      });
      await drain(result.fullStream);
      expect(calls).toHaveLength(1);
      expect(await result.text).toBe('14:50. The trip takes 5 h 10 min.');
    },
    TIMEOUT
  );

  test(
    'a structured call on a "pro" id at high effort goes through the model too (no polling for generateObject either)',
    async () => {
      const calls: LanguageModelV3CallOptions[] = [];
      const model = new MockLanguageModelV3({
        provider: 'openai.responses',
        modelId: 'gpt-5.5-pro',
        doGenerate: async (call) => {
          calls.push(call);
          return {
            content: [{ type: 'text', text: '{"arrival":"14:50"}' }],
            finishReason: { unified: 'stop', raw: 'stop' },
            usage,
            warnings: [],
          };
        },
      });
      const result = await newConversation().generateObject<{ arrival: string }>({
        messages: ['When does the train arrive?'],
        model,
        reasoningEffort: 'high',
        schema: { type: 'object', properties: { arrival: { type: 'string' } }, required: ['arrival'] },
      });
      expect(calls).toHaveLength(1);
      expect(result.object).toEqual({ arrival: '14:50' });
    },
    TIMEOUT
  );

  test(
    'a long silent think is not a dead connection: the provider’s keepalives (raw chunks) keep the liveness guard fed, and never reach the consumer',
    async () => {
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      const idle = process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
      process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = '250';
      try {
        // Nine keepalives 100 ms apart (900 ms with no mapped part — three idle windows), then the answer.
        const heartbeats = (withKeepalives: boolean) =>
          new ReadableStream({
            async start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] });
              for (let i = 0; i < 9; i++) {
                await sleep(100);
                if (withKeepalives) {
                  controller.enqueue({ type: 'raw', rawValue: { type: 'keepalive' } });
                }
              }
              controller.enqueue({ type: 'text-start', id: 't1' });
              controller.enqueue({ type: 'text-delta', id: 't1', delta: '14:50.' });
              controller.enqueue({ type: 'text-end', id: 't1' });
              controller.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage });
              controller.close();
            },
          });
        const streaming = new MockLanguageModelV3({
          provider: 'openai.responses',
          modelId: 'gpt-5.5-pro',
          doStream: async (call) => ({ stream: heartbeats(call.includeRawChunks === true) }),
        });
        const result = await newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model: streaming,
          reasoningEffort: 'xhigh',
        });
        const parts = await drain(result.fullStream);
        expect(parts.some((p) => (p as { type: string }).type === 'raw')).toBe(false);
        expect(await result.text).toBe('14:50.');
        expect(await result.failure).toBeUndefined();
      } finally {
        if (idle === undefined) {
          delete process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
        } else {
          process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = idle;
        }
      }
    },
    TIMEOUT
  );

  test(
    'the liveness guard itself swallows the heartbeat: what it hands the round loop carries no raw part (a raw part there would release a held step-finish and clear the boundary)',
    async () => {
      type GuardInternals = {
        guardStreamLiveness(
          stream: AsyncIterable<unknown>,
          controller: AbortController,
          modelString: string
        ): AsyncIterable<{ type: string }>;
      };
      const guard = newConversation() as unknown as GuardInternals;
      const fed = (async function* () {
        yield { type: 'stream-start', warnings: [] };
        yield { type: 'raw', rawValue: { type: 'keepalive' } };
        yield { type: 'raw', rawValue: { type: 'keepalive' } };
        yield { type: 'text-delta', id: 't1', delta: '14:50.' };
        yield { type: 'finish-step', finishReason: 'stop' };
      })();
      const out: string[] = [];
      for await (const part of guard.guardStreamLiveness(fed, new AbortController(), 'gpt-5.5-pro')) {
        out.push(part.type);
      }
      expect(out).toEqual(['stream-start', 'text-delta', 'finish-step']);
    },
    TIMEOUT
  );
});
