import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type RefusedStep, type StreamPart } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

/**
 * `GenerateStreamParams.refusalLadder` — a model step the provider REFUSES (the SDK's
 * `content-filter` finish on a step that produced nothing) is re-run on the next rung of the
 * ladder the caller hands in, over the same transcript, and the response stays ONE response. No
 * network: one MockLanguageModelV3 per rung scripts what each model does with the step, and the
 * prompts each rung received (the mocks' `doStreamCalls`) prove what was re-sent.
 *
 *  - the refused step is the one after a tool ran: the next rung gets the transcript WITH the
 *    tool call and its result — the tool runs exactly once, the model step alone re-runs;
 *  - the consumer sees no text and no `step-finish` from the refused attempt: one answer, one
 *    `step-finish`, plus a `model-rerun` part recording the switch for the surfaces that show it;
 *  - the ladder is asked with the refused step's facts (the model, everything that declined so
 *    far, the provider's own stop reason and stop details);
 *  - every rung refusing surfaces the refusal exactly as it surfaces without a ladder — the
 *    `content-filter` step-finish — naming every model that declined;
 *  - a ladder naming a model that already declined this step has nothing below (never re-sent);
 *  - a refusal AFTER text was shown keeps the text and is never re-run;
 *  - no ladder = today's behavior, byte for byte.
 */

const TIMEOUT = 30_000;

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

/** A step that calls the tool. */
const toolCallStep = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'doWork', input: '{}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

/** A step the provider refused — nothing produced; the shape the Anthropic adapter yields. */
const refusedStep = () =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    {
      type: 'finish' as const,
      finishReason: { unified: 'content-filter' as const, raw: 'refusal' },
      usage,
      providerMetadata: { anthropic: { stopDetails: { type: 'refusal', category: 'policy', explanation: 'no' } } },
    },
  ]);

/** A step the provider's filter ended AFTER some text had streamed. */
const refusedAfterTextStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'content-filter' as const, raw: 'refusal' }, usage },
  ]);

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

type Step = 'tool' | 'refuse' | 'refuse-after-text' | { text: string };

/** A model that plays `steps` in order, one per request it receives. */
function scriptedModel(modelId: string, steps: Step[]): MockLanguageModelV3 {
  let call = 0;
  return new MockLanguageModelV3({
    modelId,
    doStream: async () => {
      const step = steps[Math.min(call, steps.length - 1)];
      call++;
      if (step === 'tool') {
        return { stream: toolCallStep(`tc-${modelId}-${call}`) };
      }
      if (step === 'refuse') {
        return { stream: refusedStep() };
      }
      if (step === 'refuse-after-text') {
        return { stream: refusedAfterTextStep('partial answer') };
      }
      return { stream: textStep(step.text) };
    },
  });
}

function buildSkill(fn: Function): ConversationSkill {
  return {
    getId: () => 'refusal-ladder-test-skill',
    getName: () => 'RefusalLadderTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [fn],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

function buildConversation(): { conversation: Conversation; toolCalls: () => number } {
  let toolCalls = 0;
  const workTool: Function = {
    definition: {
      name: 'doWork',
      description: 'Does one unit of work.',
      parameters: { type: 'object', properties: {} },
    },
    call: async () => {
      toolCalls++;
      return { ok: true, receipt: 'WORK-RECEIPT-7' };
    },
  };
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'refusal-ladder-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [buildSkill(workTool)],
  });
  return { conversation, toolCalls: () => toolCalls };
}

/** Drive the full stream to its end, collecting every part and the text. */
async function drain(result: {
  fullStream: AsyncIterable<StreamPart>;
}): Promise<{ parts: StreamPart[]; text: string }> {
  const parts: StreamPart[] = [];
  let text = '';
  for await (const part of result.fullStream) {
    parts.push(part);
    if (part.type === 'text-delta') {
      text += part.textDelta;
    }
  }
  return { parts, text };
}

const stepFinishes = (parts: StreamPart[]) => parts.filter((p) => p.type === 'step-finish');
const reruns = (parts: StreamPart[]) => parts.filter((p) => p.type === 'model-rerun');

/** The LanguageModelV3 prompt a mock received on its n-th request, flattened for assertions. */
function promptOf(model: MockLanguageModelV3, index: number): Array<{ role: string; text: string }> {
  const prompt = model.doStreamCalls[index].prompt as Array<{ role: string; content: unknown }>;
  return prompt.map((m) => ({ role: m.role, text: JSON.stringify(m.content) }));
}

describe('Conversation.generateStream — the refusal ladder', () => {
  test(
    'rung 1 refuses the step after the tool ran; rung 2 answers it — the tool runs once, one response, the switch recorded',
    async () => {
      const first = scriptedModel('claude-first', ['tool', 'refuse']);
      const second = scriptedModel('claude-second', [{ text: 'done' }]);
      const asked: RefusedStep[] = [];
      const { conversation, toolCalls } = buildConversation();

      const result = await conversation.generateStream({
        messages: ['do the work'],
        model: first as never,
        refusalLadder: (refused) => {
          asked.push(refused);
          return second as never;
        },
      });
      const { parts, text } = await drain(result);

      // The answer came from the second rung; the consumer saw ONE response.
      expect(text).toBe('done');
      expect(stepFinishes(parts)).toEqual([
        { type: 'step-finish', finishReason: 'tool-calls' },
        { type: 'step-finish', finishReason: 'stop' },
      ]);
      expect(reruns(parts)).toEqual([
        {
          type: 'model-rerun',
          reason: 'refusal',
          from: 'claude-first',
          to: 'claude-second',
          declined: ['claude-first'],
        },
      ]);
      // The switch part sits at the step boundary: after the tool settled, before the answer.
      const rerunAt = parts.findIndex((p) => p.type === 'model-rerun');
      const firstTextAt = parts.findIndex((p) => p.type === 'text-delta');
      expect(parts.findIndex((p) => p.type === 'tool-settled')).toBeLessThan(rerunAt);
      expect(rerunAt).toBeLessThan(firstTextAt);

      // The first rung ran the tool step and the refused step; the second ran the step ONCE.
      expect(first.doStreamCalls).toHaveLength(2);
      expect(second.doStreamCalls).toHaveLength(1);
      // The tool ran exactly once — the model step re-ran, never the tool.
      expect(toolCalls()).toBe(1);
      // The second rung was sent the transcript the refused step was sent: the tool call and
      // its result are in it, so nothing was redone and nothing was lost.
      const secondPrompt = promptOf(second, 0);
      expect(secondPrompt.some((m) => m.role === 'assistant' && m.text.includes('doWork'))).toBe(true);
      expect(secondPrompt.some((m) => m.role === 'tool' && m.text.includes('WORK-RECEIPT-7'))).toBe(true);
      expect(secondPrompt).toEqual(promptOf(first, 1));

      // The ladder was asked once, with the refused step's facts.
      expect(asked).toEqual([
        {
          modelId: 'claude-first',
          declined: ['claude-first'],
          provider: 'anthropic',
          finishReason: 'content-filter',
          rawFinishReason: 'refusal',
          category: 'policy',
          explanation: 'no',
        },
      ]);
      // Usage covers both rungs' requests.
      const usageData = await result.usage;
      expect(usageData.totalRequestsToAssistant).toBe(3);
    },
    TIMEOUT
  );

  test(
    'every rung refuses: the refusal surfaces once, as the content-filter step-finish naming every model that declined',
    async () => {
      const first = scriptedModel('claude-first', ['refuse']);
      const second = scriptedModel('claude-second', ['refuse']);
      const asked: RefusedStep[] = [];
      const { conversation } = buildConversation();

      const result = await conversation.generateStream({
        messages: ['do the work'],
        model: first as never,
        refusalLadder: (refused) => {
          asked.push(refused);
          return refused.declined.length === 1 ? (second as never) : undefined;
        },
      });
      const { parts, text } = await drain(result);

      expect(text).toBe('');
      expect(reruns(parts)).toEqual([
        {
          type: 'model-rerun',
          reason: 'refusal',
          from: 'claude-first',
          to: 'claude-second',
          declined: ['claude-first'],
        },
      ]);
      expect(stepFinishes(parts)).toEqual([
        { type: 'step-finish', finishReason: 'content-filter', declined: ['claude-first', 'claude-second'] },
      ]);
      expect(first.doStreamCalls).toHaveLength(1);
      expect(second.doStreamCalls).toHaveLength(1);
      expect(asked.map((r) => [r.modelId, [...r.declined]])).toEqual([
        ['claude-first', ['claude-first']],
        ['claude-second', ['claude-first', 'claude-second']],
      ]);
    },
    TIMEOUT
  );

  test(
    'a ladder naming a model that already declined this step has nothing below — the refusal surfaces, nothing is re-sent',
    async () => {
      const first = scriptedModel('claude-first', ['refuse']);
      const { conversation } = buildConversation();

      const result = await conversation.generateStream({
        messages: ['do the work'],
        model: first as never,
        refusalLadder: () => first as never,
      });
      const { parts, text } = await drain(result);

      expect(text).toBe('');
      expect(reruns(parts)).toEqual([]);
      expect(stepFinishes(parts)).toEqual([{ type: 'step-finish', finishReason: 'content-filter' }]);
      expect(first.doStreamCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'a refusal after text was shown keeps the text and is never re-run — the ladder is not asked',
    async () => {
      const first = scriptedModel('claude-first', ['refuse-after-text']);
      const ladder = jest.fn(() => scriptedModel('claude-second', [{ text: 'never' }]) as never);
      const { conversation } = buildConversation();

      const result = await conversation.generateStream({
        messages: ['do the work'],
        model: first as never,
        refusalLadder: ladder,
      });
      const { parts, text } = await drain(result);

      expect(text).toBe('partial answer');
      expect(ladder).not.toHaveBeenCalled();
      expect(reruns(parts)).toEqual([]);
      expect(stepFinishes(parts)).toEqual([{ type: 'step-finish', finishReason: 'content-filter' }]);
    },
    TIMEOUT
  );

  test(
    'without a ladder a refusal surfaces as it always has: one content-filter step-finish, no record',
    async () => {
      const first = scriptedModel('claude-first', ['tool', 'refuse']);
      const { conversation, toolCalls } = buildConversation();

      const result = await conversation.generateStream({ messages: ['do the work'], model: first as never });
      const { parts, text } = await drain(result);

      expect(text).toBe('');
      expect(toolCalls()).toBe(1);
      expect(reruns(parts)).toEqual([]);
      expect(stepFinishes(parts)).toEqual([
        { type: 'step-finish', finishReason: 'tool-calls' },
        { type: 'step-finish', finishReason: 'content-filter' },
      ]);
      expect(first.doStreamCalls).toHaveLength(2);
    },
    TIMEOUT
  );
});
