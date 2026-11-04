import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';

/**
 * The generateObject tool loop (`generateObjectViaToolLoop`) — request-shape pins.
 *
 * Two production regressions live here, both measured on a dev-skill implement leg
 * (2026-08-26, $45.57 / 105 min):
 *
 * 1. CACHING — the loop dispatched steps with NO per-step projection, so Anthropic
 *    requests carried zero cache breakpoints: every step re-sent the entire
 *    accumulated context at full input price (review gates measured 0% cache reads —
 *    57% of the leg's cost). The loop must apply the SAME `prepareStep` projection as
 *    the streaming writer loop (`projectOutgoingStepMessages`), so every step's
 *    outgoing request carries `providerOptions.anthropic.cacheControl` marks.
 *
 * 2. CALL SHAPE — the loop used non-streaming `generateText`: a large step held the
 *    whole response server-side for minutes before the first byte and died on undici's
 *    Headers Timeout (observed: 5-min stall → retry budget burned). The loop must
 *    dispatch via the STREAMING model path (`doStream`), never `doGenerate`.
 *
 * No network: a MockLanguageModelV3 scripts an investigate-then-submit loop and
 * captures each step's outgoing prompt at the model seam. The mock answers BOTH
 * `doGenerate` and `doStream`, so these tests pin behavior (marks present, streaming
 * used) rather than merely which SDK entrypoint got mocked.
 */

const TIMEOUT = 30_000;

const usage = {
  inputTokens: { total: 500, noCache: 500, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 100, text: 100, reasoning: 0 },
};

type CapturedPrompt = Array<{
  role: string;
  content: unknown;
  providerOptions?: Record<string, Record<string, unknown>>;
}>;

const cacheControlOf = (msg: CapturedPrompt[number]) =>
  (msg.providerOptions?.anthropic as { cacheControl?: { type: string } } | undefined)?.cacheControl;
const markedCount = (prompt: CapturedPrompt) => prompt.filter((m) => cacheControlOf(m)).length;

const investigateTool: Function = {
  definition: {
    name: 'investigate',
    description: 'Reads one thing.',
    parameters: { type: 'object', properties: {} },
  },
  call: async () => ({ finding: 'the answer is 42' }),
};

const answerSchema = {
  type: 'object',
  properties: { answer: { type: 'string' } },
  required: ['answer'],
};

/** Stream step: one investigate tool call. */
const streamToolCallStep = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'investigate', input: '{}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

/** Stream step: the closing submit_result call. */
const streamSubmitStep = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'submit_result', input: '{"answer":"42"}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

/**
 * A scripted model (investigate on step 1, submit_result on step 2) that answers both
 * call shapes and records, per shape, the prompts it was dispatched with.
 */
function buildScriptedModel(modelId: string) {
  const streamPrompts: CapturedPrompt[] = [];
  const generatePrompts: CapturedPrompt[] = [];
  let streamCalls = 0;
  let generateCalls = 0;
  const model = new MockLanguageModelV3({
    modelId,
    doStream: async ({ prompt }) => {
      streamPrompts.push(prompt as unknown as CapturedPrompt);
      streamCalls++;
      return { stream: streamCalls === 1 ? streamToolCallStep('tc-1') : streamSubmitStep('tc-2') };
    },
    doGenerate: async ({ prompt }) => {
      generatePrompts.push(prompt as unknown as CapturedPrompt);
      generateCalls++;
      return {
        content: [
          generateCalls === 1
            ? { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'investigate', input: '{}' }
            : { type: 'tool-call' as const, toolCallId: 'tc-2', toolName: 'submit_result', input: '{"answer":"42"}' },
        ],
        finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' },
        usage,
        warnings: [],
      };
    },
  });
  return {
    model,
    streamPrompts,
    generatePrompts,
    calls: () => ({ stream: streamCalls, generate: generateCalls }),
  };
}

function runLoop(model: MockLanguageModelV3) {
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'generate-object-tool-loop-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });
  return conversation.generateObject<{ answer: string }>({
    messages: ['Investigate, then answer.'],
    model: model as never,
    schema: answerSchema,
    maxToolCalls: 5,
    tools: [investigateTool],
  });
}

/**
 * THE OUTPUT CEILING IS THE VENDOR'S. When the caller sets no `maxTokens`, each step of the loop asks for the
 * model's real maximum — the per-model number `@ai-sdk/anthropic` keeps in its own table (`getModelCapabilities`),
 * never a house map of ids: the map this replaced (2026-11-04) held Haiku 5.5 at 64k, half its maximum, and asked
 * the third generation for 64k it never had (an over-ask is a hard 400). The same number names the cap on a
 * length failure (`ObjectGenerationError`).
 */
describe("Conversation.generateObject tool loop — the output ceiling is the vendor's per-model maximum", () => {
  test.each([
    ['claude-haiku-5-5', 128_000],
    ['claude-sonnet-5-5', 128_000],
    ['claude-opus-4-5', 64_000],
    ['claude-haiku-4-5-20251001', 64_000],
    ['claude-3-haiku-20240307', 4_096],
  ] as Array<[string, number]>)(
    '%s: every step asks for %s output tokens',
    async (modelId, ceiling) => {
      const { model } = buildScriptedModel(modelId);
      const result = await runLoop(model);

      expect(result.object).toEqual({ answer: '42' });
      expect(model.doStreamCalls).toHaveLength(2);
      expect(model.doStreamCalls.map((call) => call.maxOutputTokens)).toEqual([ceiling, ceiling]);
    },
    TIMEOUT
  );
});

describe('Conversation.generateObject tool loop — request shape', () => {
  test(
    'dispatches every step through the streaming path, never doGenerate',
    async () => {
      const { model, calls } = buildScriptedModel('claude-fable-5');
      const result = await runLoop(model);

      expect(result.object).toEqual({ answer: '42' });
      expect(calls().stream).toBe(2);
      expect(calls().generate).toBe(0);
    },
    TIMEOUT
  );

  test(
    'every Anthropic step carries cache breakpoints: last system + last two non-system messages',
    async () => {
      const { model, streamPrompts, generatePrompts } = buildScriptedModel('claude-fable-5');
      const result = await runLoop(model);
      expect(result.object).toEqual({ answer: '42' });

      // Both call shapes captured — assert on whichever the loop actually used, so this
      // test fails on missing MARKS (the caching bug), not on the call-shape change.
      const prompts = streamPrompts.length > 0 ? streamPrompts : generatePrompts;
      expect(prompts.length).toBe(2);

      for (const prompt of prompts) {
        // Exactly the writer-loop policy: last system message + rolling last two
        // non-system messages (Anthropic caps breakpoints at 4 total; shorter
        // prompts mark what exists).
        const nonSystemCount = prompt.filter((m) => m.role !== 'system').length;
        expect(markedCount(prompt)).toBe(1 + Math.min(2, nonSystemCount));
        const lastSystemIndex = prompt.map((m) => m.role).lastIndexOf('system');
        expect(cacheControlOf(prompt[lastSystemIndex])).toEqual({ type: 'ephemeral' });
        const nonSystemIndexes = prompt
          .map((m, i) => (m.role === 'system' ? -1 : i))
          .filter((i) => i >= 0)
          .slice(-2);
        for (const i of nonSystemIndexes) {
          expect(cacheControlOf(prompt[i])).toEqual({ type: 'ephemeral' });
        }
      }

      // Step 2 (post tool-result) must carry the full rolling set — the breakpoints are
      // what convert each step's re-sent context into cache reads.
      const step2 = prompts[1];
      expect(step2.length).toBeGreaterThan(prompts[0].length);
      expect(markedCount(step2)).toBe(3);
    },
    TIMEOUT
  );

  test(
    'non-Anthropic steps carry no Anthropic cache marks',
    async () => {
      const { model, streamPrompts, generatePrompts } = buildScriptedModel('gpt-5');
      const result = await runLoop(model);
      expect(result.object).toEqual({ answer: '42' });

      const prompts = streamPrompts.length > 0 ? streamPrompts : generatePrompts;
      expect(prompts.length).toBe(2);
      for (const prompt of prompts) {
        expect(markedCount(prompt)).toBe(0);
      }
    },
    TIMEOUT
  );
});
