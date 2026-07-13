import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';

/**
 * `GenerateStreamParams.drainInjectedContext` — mid-call user context spliced at step boundaries.
 * No network: a MockLanguageModelV3 scripts a multi-step tool loop, and each step's OUTGOING
 * prompt (captured from `doStream`) proves the splice semantics:
 *  - a note arriving mid-step is spliced as a `role:'user'` message at the NEXT step, anchored
 *    after everything the loop had produced when it drained;
 *  - the note is spliced exactly once per call — later steps re-project it at the SAME anchor
 *    (present exactly once, before the messages appended after the drain);
 *  - a call with no injected context sends byte-identical prompts to one without the hook.
 */

const TIMEOUT = 30_000;

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

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

/** Flatten a LanguageModelV3 prompt message's content to text for assertions. */
const messageText = (msg: { content: unknown }): string => {
  if (typeof msg.content === 'string') {
    return msg.content;
  }
  if (Array.isArray(msg.content)) {
    return msg.content
      .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? part.text ?? '' : ''))
      .join('');
  }
  return '';
};

const userMessagesContaining = (prompt: Array<{ role: string; content: unknown }>, needle: string) =>
  prompt.filter((m) => m.role === 'user' && messageText(m).includes(needle));

function buildSkill(fn: Function): ConversationSkill {
  return {
    getId: () => 'injected-context-test-skill',
    getName: () => 'InjectedContextTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [fn],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

describe('Conversation.generateStream — drainInjectedContext step splice', () => {
  test(
    'a note arriving mid-step is spliced at the next step exactly once and re-projected at its anchor',
    async () => {
      const NOTE = 'MID-CALL NOTE: also cover the audit log.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          // Two tool steps, then a text step — the note (pushed during step 1's tool run) must be
          // in steps 2 and 3's prompts, once each.
          return { stream: call <= 2 ? toolCallStep(`tc-${call}`) : textStep('done') };
        },
      });

      const workTool: Function = {
        definition: {
          name: 'doWork',
          description: 'Does one unit of work.',
          parameters: { type: 'object', properties: {} },
        },
        call: async () => {
          if (call === 1) {
            // The user's note arrives WHILE step 1's tool executes.
            inbox.push(NOTE);
          }
          return { ok: true };
        },
      };

      const conversation = new Conversation({
        name: 'injected-context-splice-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [buildSkill(workTool)],
      });

      const drains: number[] = [];
      const result = await conversation.generateResponse({
        messages: ['do the work'],
        model: model as never,
        drainInjectedContext: () => {
          drains.push(call);
          return inbox.splice(0, inbox.length);
        },
      });

      expect(result.text).toBe('done');
      expect(capturedPrompts).toHaveLength(3);
      // The hook ran at every step boundary (the drain is per-step, not per-call).
      expect(drains.length).toBe(3);

      // Step 1 (before the note existed): no splice.
      expect(userMessagesContaining(capturedPrompts[0], NOTE)).toHaveLength(0);

      // Step 2: the note is spliced exactly once, AFTER everything step 1 produced (the anchor is
      // the drain-time message count — the note is the last message of step 2's prompt).
      const step2Notes = userMessagesContaining(capturedPrompts[1], NOTE);
      expect(step2Notes).toHaveLength(1);
      expect(messageText(capturedPrompts[1][capturedPrompts[1].length - 1] as never)).toContain(NOTE);

      // Step 3: still exactly once (no re-splice), at the SAME anchor — i.e. BEFORE the messages
      // step 2 appended (the note is no longer last; step 2's assistant/tool messages follow it).
      const step3Notes = userMessagesContaining(capturedPrompts[2], NOTE);
      expect(step3Notes).toHaveLength(1);
      const step3 = capturedPrompts[2];
      const noteIndex = step3.findIndex((m) => m.role === 'user' && messageText(m).includes(NOTE));
      expect(noteIndex).toBeGreaterThan(-1);
      expect(noteIndex).toBeLessThan(step3.length - 1);
      // The prefix up to and including the note is byte-identical to step 2's prompt (prompt-cache
      // stability across the splice).
      expect(JSON.stringify(step3.slice(0, capturedPrompts[1].length))).toBe(JSON.stringify(capturedPrompts[1]));
    },
    TIMEOUT
  );

  test(
    'an empty drain leaves the outgoing prompts untouched',
    async () => {
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];
      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          return { stream: call === 1 ? toolCallStep('tc-1') : textStep('done') };
        },
      });

      const workTool: Function = {
        definition: { name: 'doWork', description: 'noop', parameters: { type: 'object', properties: {} } },
        call: async () => ({ ok: true }),
      };

      const conversation = new Conversation({
        name: 'injected-context-empty-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [buildSkill(workTool)],
      });

      const result = await conversation.generateResponse({
        messages: ['do the work'],
        model: model as never,
        drainInjectedContext: () => [],
      });

      expect(result.text).toBe('done');
      expect(capturedPrompts).toHaveLength(2);
      for (const prompt of capturedPrompts) {
        expect(prompt.filter((m) => m.role === 'user').map((m) => messageText(m as never))).toEqual(['do the work']);
      }
    },
    TIMEOUT
  );
});
