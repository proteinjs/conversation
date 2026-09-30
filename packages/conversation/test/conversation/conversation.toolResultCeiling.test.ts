import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { encoding_for_model } from 'tiktoken';
import { jsonSchema } from 'ai';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { ToolResultOverflow, type ToolResultRecord } from '../../src/ToolResultOverflow';
import { fixtureModelData } from './fixtureModelData';

/**
 * `toolResultCeiling` — an oversized tool result never enters the conversation WHOLE: over the
 * per-result ceiling it is set aside (the whole kept — in the conversation's registry for the
 * model's read door, and in the consumer's store for its durable record) and the transcript gets
 * its head and a pointer the model opens through `read_tool_result` (a range of lines, or a
 * search). The law covers every tool the loop executes in-process, function tools and
 * provider-defined tools alike; a result evicted by the per-turn budget gets the same pointer
 * instead of a dead placeholder. No network: a scripted model runs the loop and the test reads
 * what each step's request carried.
 */

const TIMEOUT = 30_000;

const usage = {
  inputTokens: { total: 500, noCache: 500, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 100, text: 100, reasoning: 0 },
};

type Step = { type: 'tool'; toolName: string; input: string } | { type: 'text'; text: string };

const stepStream = (step: Step, id: string) =>
  step.type === 'tool'
    ? convertArrayToReadableStream([
        { type: 'stream-start' as const, warnings: [] },
        { type: 'tool-call' as const, toolCallId: id, toolName: step.toolName, input: step.input },
        { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
      ])
    : convertArrayToReadableStream([
        { type: 'stream-start' as const, warnings: [] },
        { type: 'text-start' as const, id: 't1' },
        { type: 'text-delta' as const, id: 't1', delta: step.text },
        { type: 'text-end' as const, id: 't1' },
        { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
      ]);

/** A model that plays the scripted steps and records every request's prompt as the provider saw it. */
function scriptedModel(steps: Step[]): { model: MockLanguageModelV3; prompts: () => unknown[][] } {
  const prompts: unknown[][] = [];
  let call = 0;
  const model = new MockLanguageModelV3({
    doStream: async ({ prompt }) => {
      prompts.push(prompt as unknown[]);
      const step = steps[Math.min(call, steps.length - 1)];
      call++;
      return { stream: stepStream(step, `tc-${call}`) };
    },
  });
  return { model, prompts: () => prompts };
}

/** The text of every tool-result part in a provider prompt, in order. */
function toolResultTexts(prompt: unknown[]): string[] {
  const texts: string[] = [];
  for (const message of prompt as Array<{ role: string; content: unknown }>) {
    if (message.role !== 'tool' || !Array.isArray(message.content)) {
      continue;
    }
    for (const part of message.content as Array<{ type: string; output?: { type: string; value: unknown } }>) {
      if (part.type !== 'tool-result' || !part.output) {
        continue;
      }
      const { type, value } = part.output;
      if (type === 'text' || type === 'error-text') {
        texts.push(String(value));
      } else if (type === 'json') {
        texts.push(JSON.stringify(value));
      } else if (type === 'content' && Array.isArray(value)) {
        texts.push(
          (value as Array<{ type: string; text?: string }>)
            .filter((p) => p.type === 'text')
            .map((p) => p.text ?? '')
            .join('\n')
        );
      }
    }
  }
  return texts;
}

/** The tool names the model was offered on a request. */
function offeredTools(model: MockLanguageModelV3, index: number): string[] {
  const call = model.doStreamCalls[index];
  return (call.tools ?? []).map((t) => t.name);
}

const LISTING_LINES = 3_000;
/** ≈ 40 characters a line: a listing the model asked for that is far larger than the ceiling. */
const LISTING = Array.from({ length: LISTING_LINES }, (_, i) => `plans/mocks/area-${i % 17}/frame-${i}.png`).join('\n');
const SMALL = 'src/a.ts\nsrc/b.ts';

function buildSkill(functions: Function[]): ConversationSkill {
  return {
    getId: () => 'tool-result-ceiling-test-skill',
    getName: () => 'ToolResultCeilingTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => functions,
    getMessageModerators: () => [] as MessageModerator[],
  };
}

const globTool: Function = {
  definition: {
    name: 'Glob',
    description: 'Find files.',
    parameters: { type: 'object', properties: { pattern: { type: 'string' } }, required: ['pattern'] },
  },
  call: async ({ pattern }: { pattern: string }) => (pattern === '**/plans/**' ? LISTING : SMALL),
};

const encoder = encoding_for_model('gpt-4o');
const countTokens = (text: string) => encoder.encode_ordinary(text).length;

function buildConversation(functions: Function[], toolResultTokenBudget?: number): Conversation {
  return new Conversation({
    modelData: fixtureModelData,
    name: 'tool-result-ceiling-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [buildSkill(functions)],
    ...(toolResultTokenBudget ? { toolResultTokenBudget } : {}),
  });
}

describe('Conversation — toolResultCeiling: an oversized tool result never enters the conversation whole', () => {
  test(
    'over the ceiling: the transcript carries the head + pointer, the store keeps the whole, a small result rides untouched',
    async () => {
      const kept: ToolResultRecord[] = [];
      const { model, prompts } = scriptedModel([
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"src/*.ts"}' },
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"**/plans/**"}' },
        { type: 'text', text: 'done' },
      ]);
      const result = await buildConversation([globTool]).generateResponse({
        messages: ['list the plans'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: 1_000, store: { keep: (record) => void kept.push(record) } },
      });
      expect(result.text).toBe('done');
      // The third request carries both results: the small one whole, the listing as head + pointer.
      const carried = toolResultTexts(prompts()[2]);
      expect(carried).toHaveLength(2);
      expect(carried[0]).toBe(SMALL);
      expect(carried[1]).not.toContain(`frame-${LISTING_LINES - 1}.png`);
      expect(carried[1]).toContain('Tool result set aside:');
      expect(carried[1]).toContain('tool result "tc-2"');
      expect(carried[1]).toContain('read_tool_result');
      expect(carried[1]).toContain('plans/mocks/area-0/frame-0.png');
      expect(carried[1].length).toBeLessThan(LISTING.length / 10);
      // The whole was kept — nothing lost.
      expect(kept).toHaveLength(1);
      expect(kept[0]).toMatchObject({ id: 'tc-2', toolName: 'Glob', reason: 'over-ceiling', lines: LISTING_LINES });
      expect(kept[0].text).toBe(LISTING);
      expect(kept[0].input).toEqual({ pattern: '**/plans/**' });
      expect(kept[0].tokens).toBeGreaterThan(1_000);
      // The read door was offered on every request of the call.
      expect(offeredTools(model, 0)).toContain(ToolResultOverflow.READ_TOOL_NAME);
    },
    TIMEOUT
  );

  test(
    'the model opens the whole through read_tool_result: a range of numbered lines, and a search',
    async () => {
      const { model, prompts } = scriptedModel([
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"**/plans/**"}' },
        { type: 'tool', toolName: ToolResultOverflow.READ_TOOL_NAME, input: '{"id":"tc-1","from":2001,"lines":3}' },
        { type: 'tool', toolName: ToolResultOverflow.READ_TOOL_NAME, input: '{"id":"tc-1","find":"frame-2999"}' },
        { type: 'tool', toolName: ToolResultOverflow.READ_TOOL_NAME, input: '{"id":"nope"}' },
        { type: 'text', text: 'done' },
      ]);
      await buildConversation([globTool]).generateResponse({
        messages: ['list the plans'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: 1_000 },
      });
      const carried = toolResultTexts(prompts()[4]);
      expect(carried).toHaveLength(4);
      expect(carried[1]).toBe(
        [
          `Lines 2,001–2,003 of ${LISTING_LINES.toLocaleString('en-US')} (tool result "tc-1", Glob).`,
          '2001: plans/mocks/area-11/frame-2000.png',
          '2002: plans/mocks/area-12/frame-2001.png',
          '2003: plans/mocks/area-13/frame-2002.png',
        ].join('\n')
      );
      expect(carried[2]).toBe(
        [
          `Matches for "frame-2999" in tool result "tc-1" (Glob, 3,000 lines): 1 of 1 shown.`,
          '3000: plans/mocks/area-7/frame-2999.png',
        ].join('\n')
      );
      expect(carried[3]).toBe('No set-aside tool result has id "nope".');
    },
    TIMEOUT
  );

  test(
    'a read is itself bounded by the ceiling — cut from the end, the cut named',
    async () => {
      const { model, prompts } = scriptedModel([
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"**/plans/**"}' },
        { type: 'tool', toolName: ToolResultOverflow.READ_TOOL_NAME, input: '{"id":"tc-1","from":1,"lines":1000}' },
        { type: 'text', text: 'done' },
      ]);
      await buildConversation([globTool]).generateResponse({
        messages: ['list the plans'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: 1_000 },
      });
      const read = toolResultTexts(prompts()[2])[1];
      expect(read).toContain('[cut at the 1,000-token limit after');
      expect(read.split('\n').length).toBeLessThan(200);
    },
    TIMEOUT
  );

  test(
    'a provider-defined tool with its own execute is bounded the same way (the law is every tool, not one kind)',
    async () => {
      const kept: ToolResultRecord[] = [];
      const viewer: ConversationSkill = {
        ...buildSkill([]),
        getProviderDefinedTools: () => ({
          view_file: {
            type: 'function',
            description: 'View a file.',
            inputSchema: jsonSchema({ type: 'object', properties: { path: { type: 'string' } } }),
            execute: async () => LISTING,
          } as never,
        }),
      };
      const { model, prompts } = scriptedModel([
        { type: 'tool', toolName: 'view_file', input: '{"path":"big.txt"}' },
        { type: 'text', text: 'done' },
      ]);
      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'tool-result-ceiling-provider-tool-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [viewer],
      });
      await conversation.generateResponse({
        messages: ['view it'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: 1_000, store: { keep: (record) => void kept.push(record) } },
      });
      const carried = toolResultTexts(prompts()[1]);
      expect(carried).toHaveLength(1);
      expect(carried[0]).toContain('Tool result set aside:');
      expect(carried[0]).not.toContain(`frame-${LISTING_LINES - 1}.png`);
      expect(kept).toHaveLength(1);
      expect(kept[0]).toMatchObject({ id: 'tc-1', toolName: 'view_file', reason: 'over-ceiling' });
      expect(kept[0].text).toBe(LISTING);
    },
    TIMEOUT
  );

  test(
    'a result the per-turn budget evicts gets the same pointer — re-openable, never a dead placeholder',
    async () => {
      const kept: ToolResultRecord[] = [];
      const { model, prompts } = scriptedModel([
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"a"}' },
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"b"}' },
        { type: 'tool', toolName: 'Glob', input: '{"pattern":"c"}' },
        { type: 'tool', toolName: ToolResultOverflow.READ_TOOL_NAME, input: '{"id":"tc-1","from":1,"lines":2}' },
        { type: 'text', text: 'done' },
      ]);
      const mid = Array.from({ length: 60 }, (_, i) => `mid/file-${i}.ts`).join('\n');
      const midTool: Function = { ...globTool, call: async () => mid };
      // Budget: two results fit, three do not — the oldest is evicted from the third step's request,
      // and one eviction reaches the hysteresis floor (75%), so the second result stays live.
      const each = countTokens(mid);
      const budget = Math.floor(each * 2.8);
      const conversation = buildConversation([midTool], budget);
      await conversation.generateResponse({
        messages: ['go'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: 100_000, store: { keep: (record) => void kept.push(record) } },
      });
      const third = toolResultTexts(prompts()[3]);
      expect(third[0]).toContain('Tool result set aside to fit the context budget');
      expect(third[0]).toContain('tool result "tc-1"');
      expect(third[1]).toBe(mid);
      expect(third[2]).toBe(mid);
      // The evicted whole was kept, once, and the model read it back through the door.
      await new Promise((resolve) => setImmediate(resolve));
      expect(kept.filter((r) => r.id === 'tc-1')).toHaveLength(1);
      expect(kept[0]).toMatchObject({ reason: 'over-budget', toolName: 'Glob' });
      expect(kept[0].text).toBe(mid);
      const read = toolResultTexts(prompts()[4])[3];
      expect(read).toBe(
        ['Lines 1–2 of 60 (tool result "tc-1", Glob).', '1: mid/file-0.ts', '2: mid/file-1.ts'].join('\n')
      );
    },
    TIMEOUT
  );

  test('without toolResultCeiling nothing changes: the read tool is not offered', async () => {
    const { model } = scriptedModel([{ type: 'text', text: 'done' }]);
    await buildConversation([globTool]).generateResponse({ messages: ['hi'], model: model as never });
    expect(offeredTools(model, 0)).toEqual(['Glob']);
  });
});
