import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { encoding_for_model } from 'tiktoken';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import type { ToolInvocationResult } from '../../src/OpenAi';
import { calculateUsageCostUsd, type StepToolUsage, type TokenUsage, type UsageData } from '../../src/UsageData';
import { fixtureModelData, modelDataFromRows } from './fixtureModelData';

/**
 * Per-step TOOL usage (`StepUsage.tools` + `StepUsage.costUsd`): a loop step's usage names the
 * tools it called, in call order, with what each call put into the conversation on each side —
 * its arguments and its result, counted with the package's shared encoder — whether the call's
 * execution succeeded, and the step's own price. Two laws carry the weight: the sizes are joined
 * to the SDK's step by `toolCallId` (never by position), and `resultTokens` is the result AS THE
 * MODEL RECEIVED IT — after the per-result ceiling, a set-aside result counts its head + pointer,
 * never the kept whole the execute wrapper captured. Pure units reach the private mapper through
 * a typed cast on an instance (the house idiom); the ceiling and the two consumer roads run a
 * scripted model through the real loop. No network.
 */

const TIMEOUT = 30_000;

type SdkStep = {
  toolCalls?: Array<{ toolCallId?: string; toolName?: string; input?: unknown }>;
  toolResults?: Array<{ toolCallId?: string; output?: unknown }>;
  content?: Array<{ type: string; toolCallId?: string; error?: unknown }>;
  usage?: unknown;
};

type ConversationInternals = {
  mapSdkUsage(
    sdkUsage: unknown,
    modelString: string,
    steps?: SdkStep[],
    capturedInvocations?: ToolInvocationResult[]
  ): UsageData;
};

const MODEL = 'model-a';

const internals = (modelData = fixtureModelData) =>
  new Conversation({ modelData, name: 'step-tool-usage' }) as unknown as ConversationInternals;

const encoder = encoding_for_model('gpt-4o');
/** The package's own counter: the shared o200k encoder over the text. */
const countTokens = (text: string) => encoder.encode_ordinary(text).length;

const sdkUsage = (over: {
  inputTokens: number;
  outputTokens: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
}) => ({
  inputTokens: over.inputTokens,
  outputTokens: over.outputTokens,
  totalTokens: over.inputTokens + over.outputTokens,
  inputTokenDetails: { cacheReadTokens: over.cacheRead ?? 0, cacheWriteTokens: over.cacheWrite ?? 0 },
  outputTokenDetails: { reasoningTokens: over.reasoning ?? 0 },
});

const tokenUsageOf = (u: ReturnType<typeof sdkUsage>): TokenUsage => ({
  inputTokens: u.inputTokens,
  cachedInputTokens: u.inputTokenDetails.cacheReadTokens,
  cacheWriteTokens: u.inputTokenDetails.cacheWriteTokens,
  reasoningTokens: u.outputTokenDetails.reasoningTokens,
  outputTokens: u.outputTokens,
  totalTokens: u.totalTokens,
});

const invocation = (over: Partial<ToolInvocationResult> & { id: string; name: string }): ToolInvocationResult => ({
  startedAt: new Date(0),
  finishedAt: new Date(0),
  input: {},
  ok: true,
  ...over,
});

// ── A two-step fixture: the first step called two tools; its results arrive REVERSED ──
const SEARCH_INPUT = { query: 'alpha' };
const SEARCH_OUTPUT = 'alpha: three hits — a.txt, b.txt, c.txt';
const OPEN_INPUT = { id: 7 };
const OPEN_OUTPUT = { title: 'Seven', body: 'A longer body than the search result carries, by some margin.' };
const STEP_1 = sdkUsage({
  inputTokens: 40_000,
  outputTokens: 300,
  cacheRead: 9_000,
  cacheWrite: 31_000,
  reasoning: 120,
});
const STEP_2 = sdkUsage({
  inputTokens: 41_000,
  outputTokens: 900,
  cacheRead: 40_000,
  cacheWrite: 1_000,
  reasoning: 200,
});
const SUMMED = sdkUsage({
  inputTokens: 81_000,
  outputTokens: 1_200,
  cacheRead: 49_000,
  cacheWrite: 32_000,
  reasoning: 320,
});

const twoSteps = (): SdkStep[] => [
  {
    toolCalls: [
      { toolCallId: 'call-a', toolName: 'search', input: SEARCH_INPUT },
      { toolCallId: 'call-b', toolName: 'open', input: OPEN_INPUT },
    ],
    // Reversed on purpose: a join by position would hand each call the other's result.
    toolResults: [
      { toolCallId: 'call-b', output: OPEN_OUTPUT },
      { toolCallId: 'call-a', output: SEARCH_OUTPUT },
    ],
    usage: STEP_1,
  },
  { toolCalls: [], toolResults: [], usage: STEP_2 },
];

describe('StepUsage.tools — a step names its tool calls with their sizes', () => {
  test('L1: one entry per call in call order — the SDK names and ids, the counted argument JSON and result text; the counts and callsPerTool unchanged', () => {
    const usage = internals().mapSdkUsage(SUMMED, MODEL, twoSteps());

    const expected: StepToolUsage[] = [
      {
        toolCallId: 'call-a',
        name: 'search',
        ok: true,
        argTokens: countTokens(JSON.stringify(SEARCH_INPUT)),
        resultTokens: countTokens(SEARCH_OUTPUT),
      },
      {
        toolCallId: 'call-b',
        name: 'open',
        ok: true,
        argTokens: countTokens(JSON.stringify(OPEN_INPUT)),
        resultTokens: countTokens(JSON.stringify(OPEN_OUTPUT)),
      },
    ];
    expect(usage.steps?.[0].tools).toEqual(expected);
    expect(usage.steps?.[1].tools).toEqual([]);
    // The two results differ in size — the join is by id, and the fixture's reversal would show.
    expect(expected[0].resultTokens).not.toBe(expected[1].resultTokens);

    // Everything the mapper produced before is untouched.
    expect(usage.steps?.map((step) => step.toolCalls)).toEqual([2, 0]);
    expect(usage.callsPerTool).toEqual({ search: 1, open: 1 });
    expect(usage.totalToolCalls).toBe(2);
    expect(usage.totalRequestsToAssistant).toBe(2);
    expect(usage.totalTokenUsage).toEqual(tokenUsageOf(SUMMED));
    expect(usage.initialRequestTokenUsage).toEqual(tokenUsageOf(SUMMED));
    expect(usage.steps?.[0]).toMatchObject(tokenUsageOf(STEP_1));
    expect(usage.steps?.[1]).toMatchObject(tokenUsageOf(STEP_2));
    expect(Object.keys(usage).sort()).toEqual(
      [
        'model',
        'initialRequestTokenUsage',
        'initialRequestCostUsd',
        'totalTokenUsage',
        'totalCostUsd',
        'totalRequestsToAssistant',
        'callsPerTool',
        'totalToolCalls',
        'steps',
      ].sort()
    );
  });

  test('L3: costUsd per step prices the step’s OWN usage; Σ steps reconciles to the total; 0 per step for an unpriced model', () => {
    const priced = internals().mapSdkUsage(SUMMED, MODEL, twoSteps());
    const perStep = [STEP_1, STEP_2].map(
      (step) => calculateUsageCostUsd(MODEL, tokenUsageOf(step), { modelData: fixtureModelData }).totalUsd
    );
    expect(perStep[0]).toBeGreaterThan(0);
    expect(perStep[0]).not.toBeCloseTo(perStep[1], 12);
    expect(priced.steps?.map((step) => step.costUsd)).toEqual(perStep);
    const summed = priced.steps!.reduce((sum, step) => sum + step.costUsd, 0);
    expect(Math.abs(summed - priced.totalCostUsd.totalUsd)).toBeLessThanOrEqual(1e-9 * priced.steps!.length);

    const unpriced = internals(modelDataFromRows({})).mapSdkUsage(SUMMED, MODEL, twoSteps());
    expect(unpriced.steps?.map((step) => step.costUsd)).toEqual([0, 0]);
    expect(unpriced.totalCostUsd.totalUsd).toBe(0);
  });

  test('L4: a step without usage (the partial-usage placeholders) lists no step — and so no tools — while its calls still count', () => {
    const usage = internals().mapSdkUsage(SUMMED, MODEL, [
      { toolCalls: [{ toolCallId: 'call-a', toolName: 'search', input: SEARCH_INPUT }] },
      {},
    ]);
    expect(usage.totalRequestsToAssistant).toBe(2);
    expect(usage.callsPerTool).toEqual({ search: 1 });
    expect('steps' in usage).toBe(false);
  });

  test('L6: ok is false for a call whose captured execution failed (its error text is what the model read); true for a call with no capture', () => {
    const failure = new Error('the index is locked');
    const captured = [
      invocation({ id: 'call-a', name: 'search', ok: false, error: { message: failure.message } }),
      invocation({ id: 'call-b', name: 'open', ok: true, data: OPEN_OUTPUT }),
    ];
    const usage = internals().mapSdkUsage(
      SUMMED,
      MODEL,
      [
        {
          toolCalls: [
            { toolCallId: 'call-a', toolName: 'search', input: SEARCH_INPUT },
            { toolCallId: 'call-b', toolName: 'open', input: OPEN_INPUT },
            { toolCallId: 'call-c', toolName: 'web_search', input: { query: 'beta' } },
          ],
          // The failed call has no result part — the SDK records its error, and the model receives the error text.
          toolResults: [
            { toolCallId: 'call-b', output: OPEN_OUTPUT },
            { toolCallId: 'call-c', output: { results: ['one', 'two'] } },
          ],
          content: [{ type: 'tool-error', toolCallId: 'call-a', error: failure }],
          usage: STEP_1,
        },
      ],
      captured
    );
    const tools = usage.steps?.[0].tools ?? [];
    expect(tools.map((t) => [t.toolCallId, t.ok])).toEqual([
      ['call-a', false],
      ['call-b', true],
      ['call-c', true],
    ]);
    expect(tools[0].resultTokens).toBe(countTokens(failure.message));
    expect(tools[1].resultTokens).toBe(countTokens(JSON.stringify(OPEN_OUTPUT)));
    expect(tools[2].resultTokens).toBe(countTokens(JSON.stringify({ results: ['one', 'two'] })));
  });
});

// ── The real loop: a scripted model, the ceiling, both consumer roads ──

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
      texts.push(type === 'json' ? JSON.stringify(value) : String(value));
    }
  }
  return texts;
}

const LISTING_LINES = 3_000;
/** A listing far larger than any ceiling the suite sets. */
const LISTING = Array.from({ length: LISTING_LINES }, (_, i) => `docs/area-${i % 17}/page-${i}.md`).join('\n');
const SMALL = 'src/a.ts\nsrc/b.ts';

const globTool: Function = {
  definition: {
    name: 'Glob',
    description: 'Find files.',
    parameters: { type: 'object', properties: { pattern: { type: 'string' } }, required: ['pattern'] },
  },
  call: async ({ pattern }: { pattern: string }) => (pattern === '**/docs/**' ? LISTING : SMALL),
};

function buildConversation(functions: Function[]): Conversation {
  const skill: ConversationSkill = {
    getId: () => 'step-tool-usage-test-skill',
    getName: () => 'StepToolUsageTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => functions,
    getMessageModerators: () => [] as MessageModerator[],
  };
  return new Conversation({
    modelData: fixtureModelData,
    name: 'step-tool-usage-loop',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [skill],
  });
}

const SCRIPT: Step[] = [
  { type: 'tool', toolName: 'Glob', input: '{"pattern":"src/*.ts"}' },
  { type: 'tool', toolName: 'Glob', input: '{"pattern":"**/docs/**"}' },
  { type: 'text', text: 'done' },
];

describe('StepUsage.tools — through the loop', () => {
  test(
    'L2: over the per-result ceiling, resultTokens counts the head + pointer the model received — never the kept whole',
    async () => {
      const wholeTokens = countTokens(LISTING);
      const ceiling = Math.floor(wholeTokens / 10);
      expect(wholeTokens).toBeGreaterThanOrEqual(10 * ceiling);

      const { model, prompts } = scriptedModel(SCRIPT);
      const result = await buildConversation([globTool]).generateResponse({
        messages: ['list the docs'],
        model: model as never,
        toolResultCeiling: { tokensPerResult: ceiling },
      });
      expect(result.text).toBe('done');

      // What the model received on the next request: the small result whole, the listing as head + pointer.
      const carried = toolResultTexts(prompts()[2]);
      expect(carried[0]).toBe(SMALL);
      expect(carried[1]).toContain('Tool result set aside:');
      expect(countTokens(carried[1])).toBeLessThanOrEqual(ceiling);

      const steps = result.usage.steps!;
      expect(steps).toHaveLength(3);
      expect(steps[0].tools).toEqual([
        {
          toolCallId: 'tc-1',
          name: 'Glob',
          ok: true,
          argTokens: countTokens('{"pattern":"src/*.ts"}'),
          resultTokens: countTokens(SMALL),
        },
      ]);
      expect(steps[1].tools).toEqual([
        {
          toolCallId: 'tc-2',
          name: 'Glob',
          ok: true,
          argTokens: countTokens('{"pattern":"**/docs/**"}'),
          resultTokens: countTokens(carried[1]),
        },
      ]);
      expect(steps[1].tools[0].resultTokens).toBeLessThan(wholeTokens / 5);
      expect(steps[2].tools).toEqual([]);

      // The execute wrapper's capture still holds the WHOLE (its job) — the measure did not read it.
      const capturedWhole = result.toolInvocations.find((r) => r.id === 'tc-2');
      expect(capturedWhole?.data).toBe(LISTING);
      expect(countTokens(LISTING)).toBe(wholeTokens);
    },
    TIMEOUT
  );

  test(
    'L5: the streaming road and the buffered road produce the same tools lists for the same steps',
    async () => {
      const streamed = await buildConversation([globTool]).generateStream({
        messages: ['list the docs'],
        model: scriptedModel(SCRIPT).model as never,
      });
      for await (const part of streamed.fullStream) {
        void part;
      }
      const streamedUsage = await streamed.usage;

      const buffered = await buildConversation([globTool]).generateResponse({
        messages: ['list the docs'],
        model: scriptedModel(SCRIPT).model as never,
      });

      const streamedTools = streamedUsage.steps!.map((step) => step.tools);
      const bufferedTools = buffered.usage.steps!.map((step) => step.tools);
      expect(streamedTools).toHaveLength(3);
      expect(streamedTools[0]).toHaveLength(1);
      expect(streamedTools[1][0].resultTokens).toBe(countTokens(LISTING));
      expect(streamedTools).toEqual(bufferedTools);
      expect(streamedUsage.steps!.map((step) => step.costUsd)).toEqual(
        buffered.usage.steps!.map((step) => step.costUsd)
      );
      expect(streamedUsage.steps![0].costUsd).toBeGreaterThan(0);
    },
    TIMEOUT
  );
});
