import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, GenerateStreamParams } from '../../src/Conversation';
import { LlmTransportRetry, LlmTransportRetryActivity } from '../../src/LlmTransportRetry';
import { TransientProviderError } from '../../src/TransientProviderError';
import { fixtureModelData } from './fixtureModelData';

/**
 * Transport-retry ACTIVITY tests: the observer hook that lets visible surfaces (the chat turn's
 * provider wait node) render the retries the transport layer absorbs. Same harness as
 * conversation.transportRetry.test.ts — MockLanguageModelV3 as the transport, exercising the real
 * wiring: Conversation.generateStream({ onTransportRetry }) → LlmTransportRetry.wrap → model.
 *
 * The retry SEMANTICS (what retries, budgets, typed exhaustion) are pinned by
 * conversation.transportRetry.test.ts; these tests pin only the observability contract:
 * retrying → recovered on a healed outage, retrying → gave-up on an exhausted one, and silence
 * when nothing was retried.
 */

const TIMEOUT = 30_000;

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'transport-retry-activity-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

/** Anthropic's 529 capacity overload as the AI SDK surfaces it at request initiation. */
const overloadedError = () =>
  new APICallError({
    message: 'Overloaded',
    url: 'https://api.anthropic.test',
    requestBodyValues: {},
    statusCode: 529,
    responseHeaders: {},
    responseBody: '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
  });

/** Anthropic's overload as a MID-STREAM error part (flat payload — no HTTP status mid-stream). */
const overloadedErrorPart = () => ({
  type: 'error' as const,
  error: { type: 'overloaded_error', message: 'Overloaded' },
});

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const goodStream = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    {
      type: 'finish' as const,
      finishReason: { unified: 'stop' as const, raw: 'stop' },
      usage,
    },
  ]);

const collectText = async (result: { fullStream: AsyncIterable<{ type: string; textDelta?: string }> }) => {
  let text = '';
  for await (const part of result.fullStream) {
    if (part.type === 'text-delta') {
      text += part.textDelta;
    }
  }
  return text;
};

const streamWithActivity = async (model: MockLanguageModelV3, extras: Partial<GenerateStreamParams> = {}) => {
  const activities: LlmTransportRetryActivity[] = [];
  const result = await newConversation().generateStream({
    messages: ['hi'],
    model: model as never,
    onTransportRetry: (activity) => activities.push(activity),
    ...extras,
  });
  return { activities, result };
};

describe('generateStream onTransportRetry — healed outage reports retrying → recovered', () => {
  test(
    'a 529 at stream initiation: retrying (with status + attempt), then recovered when output flows',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            throw overloadedError();
          }
          return { stream: goodStream('hello world') };
        },
      });

      const { activities, result } = await streamWithActivity(model);
      const text = await collectText(result);

      expect(calls).toBe(2);
      expect(text).toBe('hello world');
      expect(activities.map((a) => a.phase)).toEqual(['retrying', 'recovered']);
      const retrying = activities[0] as Extract<LlmTransportRetryActivity, { phase: 'retrying' }>;
      expect(retrying.attempt).toBe(1);
      expect(retrying.statusCode).toBe(529);
      expect(retrying.message).toContain('Overloaded');
      expect(retrying.delayMs).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'an overloaded_error part MID-STREAM before any output: retried, and the recovery is reported',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                overloadedErrorPart(),
              ]),
            };
          }
          return { stream: goodStream('hello world') };
        },
      });

      const { activities, result } = await streamWithActivity(model);
      const text = await collectText(result);

      expect(calls).toBe(2);
      expect(text).toBe('hello world');
      expect(activities.map((a) => a.phase)).toEqual(['retrying', 'recovered']);
    },
    TIMEOUT
  );

  test(
    'a clean stream reports NOTHING — no listener noise when no retry happened',
    async () => {
      const model = new MockLanguageModelV3({
        doStream: async () => ({ stream: goodStream('all good') }),
      });

      const { activities, result } = await streamWithActivity(model);
      const text = await collectText(result);

      expect(text).toBe('all good');
      expect(activities).toEqual([]);
    },
    TIMEOUT
  );
});

describe('generateStream onTransportRetry — exhausted outage reports retrying → gave-up', () => {
  // Deterministic tiny backoff so the budget admits at least one retry and the test stays fast:
  // full-jitter delay = ceil(random * ceiling), so random=0.001 gives 1-2ms delays.
  let randomSpy: jest.SpyInstance<number, []>;
  beforeEach(() => {
    randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.001);
  });
  afterEach(() => {
    randomSpy.mockRestore();
  });

  test(
    'an overload that outlives the budget: retrying at least once, then gave-up, then the typed error surfaces',
    async () => {
      const model = new MockLanguageModelV3({
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            overloadedErrorPart(),
          ]),
        }),
      });

      const activities: LlmTransportRetryActivity[] = [];
      // The retry layer itself (Conversation's budget isn't injectable per call — and shouldn't
      // be): a small budget that fits the 1-2ms mocked delays a few times, then exhausts.
      const wrapped = new LlmTransportRetry({ budgetMs: 30 }).wrap(model as never, {
        onRetryActivity: (activity) => activities.push(activity),
      });

      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();
      const parts: Array<{ type: string; error?: unknown }> = [];
      for (;;) {
        const result = await reader.read();
        if (result.done) {
          break;
        }
        parts.push(result.value as { type: string; error?: unknown });
      }

      // The budget admitted at least one retry, and the last word is gave-up.
      expect(activities.length).toBeGreaterThanOrEqual(2);
      expect(activities.slice(0, -1).every((a) => a.phase === 'retrying')).toBe(true);
      expect(activities[activities.length - 1].phase).toBe('gave-up');
      // The surfaced part carries the typed error — the visible layers' routing contract.
      const errorPart = parts.find((p) => p.type === 'error');
      expect(errorPart).toBeTruthy();
      expect(TransientProviderError.isInstance(errorPart!.error)).toBe(true);
    },
    TIMEOUT
  );

  test(
    'run(): recovery after retries reports retrying → recovered (the doGenerate/OpenAiResponses path)',
    async () => {
      const activities: LlmTransportRetryActivity[] = [];
      let calls = 0;
      const value = await new LlmTransportRetry().run(
        async () => {
          calls++;
          if (calls === 1) {
            throw overloadedError();
          }
          return 'ok';
        },
        {
          isRetryable: () => true,
          modelId: 'claude-opus-4-6',
          onRetryActivity: (activity) => activities.push(activity),
        }
      );

      expect(value).toBe('ok');
      expect(activities.map((a) => a.phase)).toEqual(['retrying', 'recovered']);
      expect(activities[0]).toMatchObject({ modelId: 'claude-opus-4-6', statusCode: 529 });
    },
    TIMEOUT
  );
});
