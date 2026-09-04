import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

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
        modelData: fixtureModelData,
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
        modelData: fixtureModelData,
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

/**
 * The server-tool boundary (R7 finding 9, 2026-09-04 — the founder's follow-up mid-research turn
 * killed the turn with Anthropic's 400 "`web_search` tool use with id `srvtoolu_…` was found without
 * a corresponding `web_search_tool_result` block").
 *
 * When the model batches a SERVER tool (Anthropic web_search) with a CLIENT tool in one parallel
 * group, the API ends the message at the client tool (`stop_reason: tool_use`) WITHOUT running the
 * server tool: the assistant message carries the `server_tool_use` with no result. The API runs it at
 * the start of the next request — but only if that request's follow-up user message carries nothing
 * but the client `tool_result` blocks. Any block after them (the spliced mid-turn note) tells the API
 * the assistant turn is over, and the unresolved server call fails the whole request (the server-tools
 * contract, "Mixing server tools and client tools in one turn"). So at a boundary where the last
 * assistant message is still OPEN on a server tool, the drain is not consumed: the notes wait in the
 * caller's inbox for the next boundary (the one after the server tool's result lands) or the exit.
 */
describe('Conversation.generateStream — mid-turn notes wait while the assistant turn is open on a server tool', () => {
  const serverToolStep = (results: { serverId: string; clientId: string }, settled?: { serverId: string }) =>
    convertArrayToReadableStream([
      { type: 'stream-start' as const, warnings: [] },
      ...(settled
        ? [
            {
              type: 'tool-result' as const,
              toolCallId: settled.serverId,
              toolName: 'web_search',
              result: [{ url: 'https://example.com', title: 'Example' }],
            },
          ]
        : []),
      ...(results.serverId
        ? [
            {
              type: 'tool-call' as const,
              toolCallId: results.serverId,
              toolName: 'web_search',
              input: '{"query":"frontier models"}',
              providerExecuted: true,
              dynamic: true,
            },
          ]
        : []),
      { type: 'tool-call' as const, toolCallId: results.clientId, toolName: 'doWork', input: '{}' },
      { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
    ]);

  const lastRole = (prompt: Array<{ role: string; content: unknown }>) => prompt[prompt.length - 1]?.role;

  test(
    'a note that arrives while a server tool is still open is NOT spliced into that continuation; it is spliced once at the next boundary, after the result landed',
    async () => {
      const NOTE = 'MID-CALL NOTE: also cover the robotics advances.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          if (call === 1) {
            // Step 1: web_search (server) batched with doWork (client) — the API stops at the client
            // tool; the server call has no result yet.
            return { stream: serverToolStep({ serverId: 'srv-1', clientId: 'tc-1' }) };
          }
          if (call === 2) {
            // Step 2: the API ran the deferred search first (its result opens the message), then the
            // model called doWork again — a clean client-tool boundary follows.
            return { stream: serverToolStep({ serverId: '', clientId: 'tc-2' }, { serverId: 'srv-1' }) };
          }
          return { stream: textStep('done') };
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
            // The user's note arrives WHILE step 1's client tool executes — before step 2's boundary.
            inbox.push(NOTE);
          }
          return { ok: true };
        },
      };

      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'injected-context-open-server-tool-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [buildSkill(workTool)],
      });

      const drainedAtCall: number[] = [];
      const result = await conversation.generateResponse({
        messages: ['research the frontier models'],
        model: model as never,
        drainInjectedContext: () => {
          drainedAtCall.push(call);
          return inbox.splice(0, inbox.length);
        },
      });

      expect(result.text).toBe('done');
      expect(capturedPrompts).toHaveLength(3);

      // Step 2's request: the follow-up carries ONLY the client tool result — no note anywhere, and
      // the transcript ends on the tool message (what the API needs to run the deferred search).
      expect(userMessagesContaining(capturedPrompts[1], NOTE)).toHaveLength(0);
      expect(lastRole(capturedPrompts[1])).toBe('tool');
      // The drain was NOT consumed at that boundary (boundaries 1 and 3 drained; 2 was held).
      expect(drainedAtCall).toEqual([0, 2]);

      // Step 3's request: the note rides exactly once, anchored at the boundary AFTER the search's
      // result landed — the last message of the prompt, after step 2's assistant + tool messages.
      const step3 = capturedPrompts[2];
      expect(userMessagesContaining(step3, NOTE)).toHaveLength(1);
      expect(lastRole(step3)).toBe('user');
      expect(messageText(step3[step3.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );

  test(
    'exit absorption leaves the notes to settle when the final response is paused on a server tool (pause_turn) — a spliced note would end the turn',
    async () => {
      const NOTE = 'MID-CALL NOTE: also cover the robotics advances.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          // The note lands during the round; the API paused the server-side loop mid-search.
          inbox.push(NOTE);
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              {
                type: 'tool-call' as const,
                toolCallId: 'srv-9',
                toolName: 'web_search',
                input: '{"query":"paused"}',
                providerExecuted: true,
                dynamic: true,
              },
              { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'pause_turn' }, usage },
            ]),
          };
        },
      });

      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'injected-context-paused-exit-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [],
      });

      // Exit absorption is a live-stream contract (the buffered generateResponse never holds a
      // stream open) — consume the fullStream the way the chat turn does.
      const result = await conversation.generateStream({
        messages: ['research the frontier models'],
        model: model as never,
        absorbExitNotes: true,
        drainInjectedContext: () => inbox.splice(0, inbox.length),
      });
      const parts: string[] = [];
      for await (const part of result.fullStream as AsyncIterable<{ type: string }>) {
        parts.push(part.type);
      }

      // ONE request: no continuation round was started on the paused response, and the note is
      // still in the caller's inbox (the settle path re-raises it — never silently cleared).
      expect(capturedPrompts).toHaveLength(1);
      expect(inbox).toEqual([NOTE]);
      expect(parts).toContain('step-finish');
    },
    TIMEOUT
  );
});

describe('Conversation.generateStream — a server tool whose result already landed does not hold the drain', () => {
  test(
    'web_search resolved in the same message, then a client tool: the note rides the very next boundary (no hold)',
    async () => {
      const NOTE = 'MID-CALL NOTE: also cover the robotics advances.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          if (call === 1) {
            // The founder's shape for searches 1–2: the API ran the search inside the message (call +
            // result, paired by id) and the model then called a client tool — a clean boundary.
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                {
                  type: 'tool-call' as const,
                  toolCallId: 'srv-1',
                  toolName: 'web_search',
                  input: '{"query":"frontier models"}',
                  providerExecuted: true,
                  dynamic: true,
                },
                {
                  type: 'tool-result' as const,
                  toolCallId: 'srv-1',
                  toolName: 'web_search',
                  result: [{ url: 'https://example.com', title: 'Example' }],
                },
                { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'doWork', input: '{}' },
                { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
              ]),
            };
          }
          return { stream: textStep('done') };
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
            inbox.push(NOTE);
          }
          return { ok: true };
        },
      };

      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'injected-context-settled-server-tool-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [buildSkill(workTool)],
      });

      const result = await conversation.generateResponse({
        messages: ['research the frontier models'],
        model: model as never,
        drainInjectedContext: () => inbox.splice(0, inbox.length),
      });

      expect(result.text).toBe('done');
      expect(capturedPrompts).toHaveLength(2);
      // Nothing is open on the server side, so the boundary is a real one: the note is spliced now,
      // last in step 2's prompt — exactly as at any client-tool boundary.
      const step2 = capturedPrompts[1];
      expect(userMessagesContaining(step2, NOTE)).toHaveLength(1);
      expect(messageText(step2[step2.length - 1] as never)).toContain(NOTE);
      expect(inbox).toEqual([]);
    },
    TIMEOUT
  );
});
