import { MockLanguageModelV3 } from 'ai/test';
import type { LanguageModelV3CallOptions, LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Conversation, type StreamPart } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE PROVIDER'S HEARTBEATS RIDE EVERY STREAM — no network, no keys. Since 2026-09-22 every
 * `generateStream` round asks the provider for its raw chunks (`includeRawChunks`) so a silent
 * think's keepalives reset the liveness guard's idle timer. That request goes to EVERY provider,
 * and a raw chunk now crosses the transport-retry wrapper untouched. This suite pins what must
 * stay true for every provider with a scripted transport that behaves like the real adapters do:
 * a raw chunk is enqueued for every wire event, ahead of the part it maps to, whenever the call
 * asked for raw chunks.
 *
 *  (a) an Anthropic stream and a Google stream read by the consumer are the same part sequence
 *      whether the adapter emits raw chunks or not (byte-identical after JSON);
 *  (b) an OpenAI stream whose only traffic for many idle windows is keepalives is NOT aborted;
 *  (c) a stream that truly stalls — no bytes at all — IS aborted at the guard;
 *  (d) a retry after a pre-output failure replays correctly with keepalives ahead of the preamble
 *      — the consumer sees exactly one attempt: one `stream-start`, never the failed attempt's
 *      response metadata — and the SDK still sees `stream-start` FIRST, so the provider's call
 *      warnings land on the step (they reach the SDK's warning log);
 *  (e) usage on the streaming Responses path for a pro-class model is complete: input, output,
 *      reasoning and cached-input tokens, and the cost priced from them.
 */
const TIMEOUT = 60_000;

const usage = {
  inputTokens: { total: 10_000, noCache: 2_000, cacheRead: 8_000, cacheWrite: 0 },
  outputTokens: { total: 3_000, text: 500, reasoning: 2_500 },
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The wire events of a reasoned answer, as the adapters map them: one mapped part per event. */
const reasonedParts = (): LanguageModelV3StreamPart[] => [
  { type: 'response-metadata', id: 'resp-1', modelId: 'm', timestamp: new Date(0) },
  { type: 'reasoning-start', id: 'r1' },
  { type: 'reasoning-delta', id: 'r1', delta: 'First leg 2 h 45 min, ' },
  { type: 'reasoning-delta', id: 'r1', delta: 'then the stop, then 2 h.' },
  { type: 'reasoning-end', id: 'r1' },
  { type: 'text-start', id: 't1' },
  { type: 'text-delta', id: 't1', delta: '14:50. ' },
  { type: 'text-delta', id: 't1', delta: 'The trip takes 5 h 10 min.' },
  { type: 'text-end', id: 't1' },
  { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage },
];

/**
 * A stream the way a real adapter builds one: `stream-start` first (the transform's `start`),
 * then per wire event a raw chunk (only when the call asked for them) followed by the mapped part.
 * `keepalivesBefore[i]` = raw-only events (no mapped part) to emit before part i, `gapMs` apart.
 */
const adapterStream = (
  parts: LanguageModelV3StreamPart[],
  opts: { includeRawChunks: boolean; keepalivesBefore?: Record<number, number>; gapMs?: number; warnings?: unknown[] }
) =>
  new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller) {
      controller.enqueue({ type: 'stream-start', warnings: (opts.warnings ?? []) as never });
      for (let i = 0; i < parts.length; i++) {
        const keepalives = opts.keepalivesBefore?.[i] ?? 0;
        for (let k = 0; k < keepalives; k++) {
          await sleep(opts.gapMs ?? 0);
          if (opts.includeRawChunks) {
            controller.enqueue({ type: 'raw', rawValue: { type: 'keepalive' } });
          }
        }
        if (opts.includeRawChunks) {
          controller.enqueue({ type: 'raw', rawValue: { type: `event-${parts[i].type}` } });
        }
        controller.enqueue(parts[i]);
      }
      controller.close();
    },
  });

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'heartbeats-on-every-stream',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

const drain = async (stream: AsyncIterable<StreamPart>): Promise<StreamPart[]> => {
  const parts: StreamPart[] = [];
  for await (const part of stream) {
    parts.push(part);
  }
  return parts;
};

/** The consumer's whole read of one stream, as JSON: what the timeline and the ledger would see. */
const consumerRead = async (model: MockLanguageModelV3, modelId: string) => {
  const result = await newConversation().generateStream({
    messages: ['A train leaves at 9:40…'],
    model,
    reasoningEffort: 'high',
  });
  const parts = await drain(result.fullStream);
  return JSON.stringify(
    {
      parts,
      text: await result.text,
      reasoning: await result.reasoning,
      sources: await result.sources,
      usage: (await result.usage).totalTokenUsage,
      failure: await result.failure,
      modelId,
    },
    null,
    1
  );
};

const withIdleWindow = async (ms: number, run: () => Promise<void>) => {
  const prior = process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
  process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = String(ms);
  try {
    await run();
  } finally {
    if (prior === undefined) {
      delete process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
    } else {
      process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = prior;
    }
  }
};

/** The @ai-sdk/openai Responses transport's mid-stream error part (no HTTP status; retryable). */
const openAiServerErrorPart = (): LanguageModelV3StreamPart => ({
  type: 'error',
  error: {
    type: 'error',
    sequence_number: 0,
    error: { type: 'server_error', code: 'server_error', message: 'The server had an error.', param: null },
  },
});

describe('the provider’s heartbeats ride every stream', () => {
  test(
    '(a) Anthropic and Google: the consumer reads the same parts whether the adapter emits raw chunks or not',
    async () => {
      const dumpDir = process.env.HEARTBEAT_CHECK_DUMP_DIR;
      for (const [provider, modelId] of [
        ['anthropic.messages', 'claude-opus-5-5'],
        ['google.generative-ai', 'gemini-3-pro'],
      ] as const) {
        const calls: LanguageModelV3CallOptions[] = [];
        // The adapter as it really behaves: raw chunks only when asked for them.
        const honoring = new MockLanguageModelV3({
          provider,
          modelId,
          doStream: async (call) => {
            calls.push(call);
            return { stream: adapterStream(reasonedParts(), { includeRawChunks: call.includeRawChunks === true }) };
          },
        });
        // The same adapter never emitting a raw chunk (the pre-2026-09-22 request).
        const silent = new MockLanguageModelV3({
          provider,
          modelId,
          doStream: async () => ({ stream: adapterStream(reasonedParts(), { includeRawChunks: false }) }),
        });
        const withRaw = await consumerRead(honoring, modelId);
        const withoutRaw = await consumerRead(silent, modelId);
        expect(calls[0].includeRawChunks).toBe(true);
        expect(withRaw).toBe(withoutRaw);
        expect(withRaw).not.toContain('"raw"');
        if (dumpDir) {
          mkdirSync(dumpDir, { recursive: true });
          writeFileSync(join(dumpDir, `${provider}.json`), withRaw);
        }
      }
    },
    TIMEOUT
  );

  test(
    '(b) an OpenAI stream whose only traffic for forty idle windows is keepalives is not aborted',
    () =>
      withIdleWindow(300, async () => {
        const model = new MockLanguageModelV3({
          provider: 'openai.responses',
          modelId: 'gpt-5.5-pro',
          doStream: async (call) => ({
            // 120 keepalives 100 ms apart before the first reasoning piece: 12 s with no mapped part.
            stream: adapterStream(reasonedParts(), {
              includeRawChunks: call.includeRawChunks === true,
              keepalivesBefore: { 1: 120 },
              gapMs: 100,
            }),
          }),
        });
        const startedAt = Date.now();
        const result = await newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model,
          reasoningEffort: 'xhigh',
        });
        const parts = await drain(result.fullStream);
        expect(Date.now() - startedAt).toBeGreaterThan(11_000);
        expect(await result.failure).toBeUndefined();
        expect(await result.text).toBe('14:50. The trip takes 5 h 10 min.');
        expect(parts.filter((p) => p.type === 'reasoning-delta')).toHaveLength(2);
        expect(parts.some((p) => (p as { type: string }).type === 'raw')).toBe(false);
      }),
    TIMEOUT
  );

  test(
    '(c) a stream that truly stalls — no bytes at all — is aborted at the guard',
    () =>
      withIdleWindow(300, async () => {
        let aborted = false;
        const model = new MockLanguageModelV3({
          provider: 'openai.responses',
          modelId: 'gpt-5.5-pro',
          doStream: async (call) => {
            call.abortSignal?.addEventListener('abort', () => {
              aborted = true;
            });
            return {
              stream: new ReadableStream<LanguageModelV3StreamPart>({
                start(controller) {
                  controller.enqueue({ type: 'stream-start', warnings: [] });
                  // then nothing, ever
                },
              }),
            };
          },
        });
        const result = await newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model,
          reasoningEffort: 'xhigh',
        });
        // The streaming egress throws the round's failure from `fullStream` (the library's contract).
        await expect(drain(result.fullStream)).rejects.toThrow(
          /Model stream stalled: no parts from gpt-5\.5-pro for 0s \(silent connection loss\)/
        );
        expect(aborted).toBe(true);
      }),
    TIMEOUT
  );

  /** The pre-output failure then the answer, both attempts heartbeating, `stream-start` carrying a warning. */
  const warning = { type: 'unsupported', feature: 'a setting the model ignores', details: 'dropped' };
  const retriedModel = (attempts: { n: number }) =>
    new MockLanguageModelV3({
      provider: 'openai.responses',
      modelId: 'gpt-5.5-pro',
      doStream: async (call) => {
        attempts.n++;
        const raw = call.includeRawChunks === true;
        if (attempts.n === 1) {
          // stream-start · raw(created) · response-metadata(failed) · 3 keepalives · error
          return {
            stream: adapterStream([{ type: 'response-metadata', id: 'failed-attempt' }, openAiServerErrorPart()], {
              includeRawChunks: raw,
              keepalivesBefore: { 1: 3 },
              warnings: [warning],
            }),
          };
        }
        return {
          stream: adapterStream(reasonedParts(), {
            includeRawChunks: raw,
            keepalivesBefore: { 1: 2 },
            warnings: [warning],
          }),
        };
      },
    });

  test(
    '(d1) a retry after a pre-output failure replays with keepalives ahead of the preamble: the SDK is handed one attempt, and stream-start first',
    async () => {
      const attempts = { n: 0 };
      const model = retriedModel(attempts);
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [], includeRawChunks: true });
      const reader = stream.getReader();
      const seen: LanguageModelV3StreamPart[] = [];
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        seen.push(read.value);
      }
      expect(attempts.n).toBe(2);
      const types = seen.map((p) => p.type);
      expect(types.filter((t) => t === 'stream-start')).toHaveLength(1);
      expect(seen.filter((p) => p.type === 'response-metadata').map((p) => (p as { id?: string }).id)).toEqual([
        'resp-1',
      ]);
      // The mapped parts, in order, once; the failed attempt contributed nothing but heartbeats.
      expect(types.filter((t) => t !== 'raw' && t !== 'stream-start')).toEqual(reasonedParts().map((p) => p.type));
      // The provider's call warnings ride stream-start, and the SDK reads them only when
      // stream-start is the FIRST part it sees (ai 6.0.14 `streamText`: any earlier part opens
      // the step with `warnings: []`). A heartbeat must not overtake it.
      expect(types[0]).toBe('stream-start');
    },
    TIMEOUT
  );

  test(
    '(d2) through the real wiring the retried turn answers once, and the provider’s call warning reaches the SDK’s warning log',
    async () => {
      const attempts = { n: 0 };
      const model = retriedModel(attempts);
      const logged: Array<{ warnings: unknown[] }> = [];
      const g = globalThis as { AI_SDK_LOG_WARNINGS?: unknown };
      const priorLogger = g.AI_SDK_LOG_WARNINGS;
      g.AI_SDK_LOG_WARNINGS = (entry: { warnings: unknown[] }) => logged.push(entry);
      try {
        const retries: string[] = [];
        const result = await newConversation().generateStream({
          messages: ['A train leaves at 9:40…'],
          model,
          reasoningEffort: 'xhigh',
          onTransportRetry: (activity) => retries.push(activity.phase),
        });
        const parts = await drain(result.fullStream);
        expect(attempts.n).toBe(2);
        expect(retries).toEqual(['retrying', 'recovered']);
        expect(await result.failure).toBeUndefined();
        expect(await result.text).toBe('14:50. The trip takes 5 h 10 min.');
        expect(parts.filter((p) => p.type === 'text-delta')).toHaveLength(2);
        expect(parts.some((p) => (p as { type: string }).type === 'raw')).toBe(false);
        expect(logged.flatMap((e) => e.warnings)).toEqual([warning]);
      } finally {
        if (priorLogger === undefined) {
          delete g.AI_SDK_LOG_WARNINGS;
        } else {
          g.AI_SDK_LOG_WARNINGS = priorLogger;
        }
      }
    },
    TIMEOUT
  );

  test(
    '(e) usage on the streaming Responses path for a pro-class model is complete: input, output, reasoning, cached, and the cost',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'openai.responses',
        modelId: 'gpt-5.5-pro',
        doStream: async (call) => ({
          stream: adapterStream(reasonedParts(), {
            includeRawChunks: call.includeRawChunks === true,
            keepalivesBefore: { 1: 2 },
          }),
        }),
      });
      const reported: unknown[] = [];
      const result = await newConversation().generateStream({
        messages: ['A train leaves at 9:40…'],
        model,
        reasoningEffort: 'max',
        onUsageData: async (u) => {
          reported.push(u);
        },
      });
      await drain(result.fullStream);
      const u = await result.usage;
      expect(u.totalTokenUsage.inputTokens).toBe(10_000);
      expect(u.totalTokenUsage.outputTokens).toBe(3_000);
      expect(u.totalTokenUsage.reasoningTokens).toBe(2_500);
      expect(u.totalTokenUsage.cachedInputTokens).toBe(8_000);
      expect(u.totalRequestsToAssistant).toBe(1);
      // Priced from the fixture's rates ($5 / $0.50 cached / $25 per 1M) by the library's contract:
      // `inputUsd` folds the 8,000 cache reads (at the cached rate) in with the 2,000 fresh tokens;
      // `cachedInputUsd` is that cached portion alone; `reasoningUsd` (the 2,500) sits INSIDE
      // `outputUsd` (all 3,000 output tokens at the output rate); total = input + output.
      expect(u.totalCostUsd.inputUsd).toBeCloseTo((2_000 * 5.0 + 8_000 * 0.5) / 1_000_000, 9);
      expect(u.totalCostUsd.cachedInputUsd).toBeCloseTo((8_000 * 0.5) / 1_000_000, 9);
      expect(u.totalCostUsd.reasoningUsd).toBeCloseTo((2_500 * 25.0) / 1_000_000, 9);
      expect(u.totalCostUsd.outputUsd).toBeCloseTo((3_000 * 25.0) / 1_000_000, 9);
      expect(u.totalCostUsd.totalUsd).toBeCloseTo((2_000 * 5.0 + 8_000 * 0.5 + 3_000 * 25.0) / 1_000_000, 9);
      expect(reported.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );
});
