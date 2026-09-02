import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { LlmTransportRetry, LlmTransportRetryActivity } from '../../src/LlmTransportRetry';
import { fixtureModelData } from './fixtureModelData';

/**
 * STOP reaches INTO the transport retry loop (prod defect 2026-09-02: a turn stuck in the
 * provider-overload retry loop would not stop). The one-act-stop contract at this layer:
 * an abort issued MID-RETRY-BACKOFF cancels the pending retry (the announced attempt never
 * runs), settles the wrapped stream promptly, and reports the settle to the activity observer
 * AS AN ABORT — `gave-up` with `aborted: true` — so wait-rendering surfaces settle with stop
 * words instead of a bogus provider-outage verdict ("didn't recover" for a turn the USER
 * stopped). Covered legs: stream initiation (thrown 529), mid-stream error part, the tool-loop
 * continuation call (a later step's initiation), and the run()/doGenerate path.
 *
 * The exhaustion counter-pin keeps the discriminator honest: a budget-exhausted gave-up must
 * NOT carry the aborted tag.
 */

const TIMEOUT = 30_000;

const newConversation = (skills: ConversationSkill[] = []) =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'transport-retry-abort-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills,
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

const toolCallStep = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'doWork', input: '{}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const workTool: Function = {
  definition: {
    name: 'doWork',
    description: 'Does one unit of work.',
    parameters: { type: 'object', properties: {} },
  },
  call: async () => ({ ok: true }),
};

const workSkill: ConversationSkill = {
  getId: () => 'transport-retry-abort-test-skill',
  getName: () => 'TransportRetryAbortTestSkill',
  getSystemMessages: () => [],
  getFunctions: () => [workTool],
  getMessageModerators: () => [] as MessageModerator[],
};

/** Drain a fullStream; resolves the collected parts on settle, or 'timeout' after `boundMs`. */
const drainWithBound = async (
  fullStream: AsyncIterable<{ type: string }>,
  boundMs: number
): Promise<Array<{ type: string }> | 'timeout'> => {
  const drain = (async () => {
    const parts: Array<{ type: string }> = [];
    for await (const part of fullStream) {
      parts.push(part);
    }
    return parts;
  })();
  return await Promise.race([
    drain,
    new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), boundMs)),
  ]);
};

describe('stop mid-retry-backoff — the abort cancels the pending retry and settles the stream', () => {
  // Deterministic backoff: full-jitter delay = ceil(random * ceiling); 0.9 gives a ~900ms first
  // backoff — wide enough that the abort below provably lands MID-SLEEP.
  let randomSpy: jest.SpyInstance<number, []>;
  beforeEach(() => {
    randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.9);
  });
  afterEach(() => {
    randomSpy.mockRestore();
  });

  /** Abort ~100ms after the first `retrying` report — i.e. mid-backoff-sleep. */
  const abortOnFirstRetry = (controller: AbortController, activities: LlmTransportRetryActivity[]) => {
    return (activity: LlmTransportRetryActivity) => {
      const firstRetry = activities.filter((a) => a.phase === 'retrying').length === 0;
      activities.push(activity);
      if (activity.phase === 'retrying' && firstRetry) {
        setTimeout(() => controller.abort(), 100);
      }
    };
  };

  test(
    'STREAMING leg, thrown 529 at initiation: no attempt runs after the abort; gave-up is abort-tagged',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          throw overloadedError();
        },
      });

      const controller = new AbortController();
      const activities: LlmTransportRetryActivity[] = [];
      const result = await newConversation().generateStream({
        messages: ['hi'],
        model: model as never,
        abortSignal: controller.signal,
        onTransportRetry: abortOnFirstRetry(controller, activities),
      });

      // The announced backoff was ~900ms; settling by 5s proves the abort cut it, with margin.
      const settled = await drainWithBound(result.fullStream, 5_000);
      expect(settled).not.toBe('timeout');
      // The retry announced before the abort never ran — the pending timer was cancelled.
      expect(calls).toBe(1);
      const last = activities[activities.length - 1];
      expect(last.phase).toBe('gave-up');
      expect((last as Extract<LlmTransportRetryActivity, { phase: 'gave-up' }>).aborted).toBe(true);
    },
    TIMEOUT
  );

  test(
    'STREAMING leg, mid-stream overloaded_error part: no attempt runs after the abort; gave-up is abort-tagged',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              overloadedErrorPart(),
            ]),
          };
        },
      });

      const controller = new AbortController();
      const activities: LlmTransportRetryActivity[] = [];
      const result = await newConversation().generateStream({
        messages: ['hi'],
        model: model as never,
        abortSignal: controller.signal,
        onTransportRetry: abortOnFirstRetry(controller, activities),
      });

      const settled = await drainWithBound(result.fullStream, 5_000);
      expect(settled).not.toBe('timeout');
      expect(calls).toBe(1);
      const last = activities[activities.length - 1];
      expect(last.phase).toBe('gave-up');
      expect((last as Extract<LlmTransportRetryActivity, { phase: 'gave-up' }>).aborted).toBe(true);
    },
    TIMEOUT
  );

  test(
    'TOOL-CALL leg: a later step 529s; the stop mid-backoff settles the loop the same way',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            return { stream: toolCallStep('tc-1') };
          }
          // Step 2 (the continuation after the tool result) hits the overload storm.
          throw overloadedError();
        },
      });

      const controller = new AbortController();
      const activities: LlmTransportRetryActivity[] = [];
      const result = await newConversation([workSkill]).generateStream({
        messages: ['do the work'],
        model: model as never,
        abortSignal: controller.signal,
        onTransportRetry: abortOnFirstRetry(controller, activities),
      });

      const settled = await drainWithBound(result.fullStream, 5_000);
      expect(settled).not.toBe('timeout');
      // Step 1 (tool round) + step 2's single failed initiation — the announced retry never ran.
      expect(calls).toBe(2);
      const last = activities[activities.length - 1];
      expect(last.phase).toBe('gave-up');
      expect((last as Extract<LlmTransportRetryActivity, { phase: 'gave-up' }>).aborted).toBe(true);
    },
    TIMEOUT
  );

  test(
    'run() leg (doGenerate/OpenAiResponses path): the abort mid-backoff surfaces with an abort-tagged gave-up',
    async () => {
      const controller = new AbortController();
      const activities: LlmTransportRetryActivity[] = [];
      let calls = 0;
      let caught: unknown;
      try {
        await new LlmTransportRetry().run(
          async () => {
            calls++;
            throw overloadedError();
          },
          {
            abortSignal: controller.signal,
            isRetryable: () => true,
            modelId: 'claude-opus-4-6',
            onRetryActivity: abortOnFirstRetry(controller, activities),
          }
        );
      } catch (error) {
        caught = error;
      }

      expect(calls).toBe(1);
      expect(APICallError.isInstance(caught)).toBe(true); // the original error, untouched
      const last = activities[activities.length - 1];
      expect(last.phase).toBe('gave-up');
      expect((last as Extract<LlmTransportRetryActivity, { phase: 'gave-up' }>).aborted).toBe(true);
    },
    TIMEOUT
  );

  test(
    'COUNTER-PIN: budget exhaustion (no abort) reports gave-up WITHOUT the aborted tag',
    async () => {
      randomSpy.mockReturnValue(0.001); // 1-2ms delays so the tiny budget admits a retry
      const activities: LlmTransportRetryActivity[] = [];
      let caught: unknown;
      try {
        await new LlmTransportRetry({ budgetMs: 30 }).run(
          async () => {
            throw overloadedError();
          },
          {
            isRetryable: () => true,
            modelId: 'claude-opus-4-6',
            onRetryActivity: (activity) => activities.push(activity),
          }
        );
      } catch (error) {
        caught = error;
      }

      expect(caught).toBeTruthy();
      const last = activities[activities.length - 1];
      expect(last.phase).toBe('gave-up');
      expect((last as Extract<LlmTransportRetryActivity, { phase: 'gave-up' }>).aborted).toBeUndefined();
    },
    TIMEOUT
  );
});
