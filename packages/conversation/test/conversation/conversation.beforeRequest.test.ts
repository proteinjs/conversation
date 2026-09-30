import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type OutgoingRequest } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

/**
 * `beforeRequest` — the PER-REQUEST seat: every request a call sends passes through it with the
 * final outgoing messages, including every step of a tool loop, not only the first. A consumer's
 * ceiling check that ran before the first send alone missed the loop's later steps, where tool
 * results accumulate until the provider refused the request itself ("prompt is too long",
 * 2026-09-29). A throw from the hook refuses THAT request before the transport is asked: the
 * call ends on the hook's own error object (identity kept), so a typed refusal is classified
 * by the consumer as its own. No network: a scripted model runs the loop.
 */

const TIMEOUT = 30_000;

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

const objectResult = (json: string) => ({
  content: [{ type: 'text' as const, text: json }],
  finishReason: { unified: 'stop' as const, raw: 'stop' },
  usage,
  warnings: [],
});

function buildSkill(fn: Function): ConversationSkill {
  return {
    getId: () => 'before-request-test-skill',
    getName: () => 'BeforeRequestTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [fn],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

/** Every call returns the same 3,000-character result — a loop grows by that much each step. */
const RESULT = 'result line: the quick brown fox jumps over the lazy dog.\n'.repeat(50);
const workTool: Function = {
  definition: { name: 'doWork', description: 'Does one unit of work.', parameters: { type: 'object', properties: {} } },
  call: async () => RESULT,
};

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
    modelData: fixtureModelData,
    name: 'before-request-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [buildSkill(workTool)],
  });
}

/** The consumer's own typed refusal (a plain Error carrying a name — the package compiles to ES5, where a subclass of Error loses its prototype). */
const refusal = (chars: number): Error =>
  Object.assign(new Error(`refused: ${chars} characters is over the window`), { name: 'RefusedError' });

/** A consumer's ceiling: the request's text, counted in characters, may not exceed `limit`; the error it threw is kept for the identity check. */
function ceiling(limit: number, seen: OutgoingRequest[], thrown: Error[] = []): (request: OutgoingRequest) => void {
  return (request) => {
    seen.push(request);
    const chars = JSON.stringify(request.messages).length;
    if (chars > limit) {
      const error = refusal(chars);
      thrown.push(error);
      throw error;
    }
  };
}

describe('Conversation — beforeRequest runs before EVERY request of a call', () => {
  test(
    'the hook sees every step of a tool loop, in order, with the step number and the final messages',
    async () => {
      const { model, calls } = buildScriptedModel(2);
      const seen: OutgoingRequest[] = [];
      await buildConversation().generateResponse({
        messages: ['do the work'],
        model: model as never,
        beforeRequest: ceiling(Number.MAX_SAFE_INTEGER, seen),
      });
      expect(calls()).toBe(3);
      expect(seen.map((r) => r.stepNumber)).toEqual([0, 1, 2]);
      // The request grows as results accumulate — the hook measures what is about to be sent.
      const sizes = seen.map((r) => JSON.stringify(r.messages).length);
      expect(sizes[1]).toBeGreaterThan(sizes[0] + RESULT.length);
      expect(sizes[2]).toBeGreaterThan(sizes[1] + RESULT.length);
      // The provider's own count of the previous request rides along once a step has finished.
      expect(seen[0].previousRequestInputTokens).toBeUndefined();
      expect(seen[1].previousRequestInputTokens).toBe(500);
      expect(seen[0].modelString).toContain('mock');
    },
    TIMEOUT
  );

  test(
    'the third step’s request is refused BEFORE the transport is asked — the call ends on the hook’s own error object',
    async () => {
      const { model, calls } = buildScriptedModel(3);
      const seen: OutgoingRequest[] = [];
      const thrown: Error[] = [];
      // Steps 0 and 1 fit; step 2 carries two results and does not.
      const limit = 200 + RESULT.length * 2 - 1;
      const failure = await buildConversation()
        .generateResponse({
          messages: ['do the work'],
          model: model as never,
          beforeRequest: ceiling(limit, seen, thrown),
        })
        .then(
          () => undefined,
          (error: unknown) => error
        );
      // The very object the hook threw — not a re-worded copy — so a typed error stays its owner's.
      expect(thrown).toHaveLength(1);
      expect(failure).toBe(thrown[0]);
      expect((failure as Error).name).toBe('RefusedError');
      // Two requests went out; the third was refused at the seat — never sent.
      expect(calls()).toBe(2);
      expect(seen).toHaveLength(3);
    },
    TIMEOUT
  );

  test(
    'the object tool loop runs its steps through the same seat',
    async () => {
      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          call++;
          return {
            stream:
              call <= 2
                ? toolCallStep(`tc-${call}`)
                : convertArrayToReadableStream([
                    { type: 'stream-start' as const, warnings: [] },
                    {
                      type: 'tool-call' as const,
                      toolCallId: 'submit',
                      toolName: 'submit_result',
                      input: '{"answer":"ok"}',
                    },
                    {
                      type: 'finish' as const,
                      finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' },
                      usage,
                    },
                  ]),
          };
        },
      });
      const seen: OutgoingRequest[] = [];
      const thrown: Error[] = [];
      const failure = await buildConversation()
        .generateObject<{ answer: string }>({
          messages: ['investigate, then answer'],
          model: model as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
          maxToolCalls: 5,
          beforeRequest: ceiling(200 + RESULT.length * 2 - 1, seen, thrown),
        })
        .then(
          () => undefined,
          (error: unknown) => error
        );
      expect(failure).toBe(thrown[0]);
      expect(call).toBe(2);
      expect(seen.map((r) => r.stepNumber)).toEqual([0, 1, 2]);
    },
    TIMEOUT
  );

  test(
    'the single-shot object call passes through it too — refused with nothing dispatched',
    async () => {
      let call = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          call++;
          return objectResult('{"answer":"ok"}');
        },
      });
      const seen: OutgoingRequest[] = [];
      const thrown: Error[] = [];
      const failure = await buildConversation()
        .generateObject<{ answer: string }>({
          messages: ['answer'],
          model: model as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
          beforeRequest: ceiling(1, seen, thrown),
        })
        .then(
          () => undefined,
          (error: unknown) => error
        );
      expect(failure).toBe(thrown[0]);
      expect(call).toBe(0);
      expect(seen).toHaveLength(1);
      expect(seen[0].stepNumber).toBe(0);
    },
    TIMEOUT
  );
});
