import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance, type DrainedInput } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE BOUNDED UTTERANCE (plans/FREE_AGENT.md §M.3 part 2c — the load-bearing guarantee of the
 * 10-second bar): before the mind takes an input into a step, the loop asks it for ONE LINE in a
 * separate no-tools, no-thinking call over the same transcript, streams that line as its own step
 * (a `step-finish` flagged `utterance`), and runs the step with the line riding as the agent's own
 * prior message + the continue framing. On every door: turn start (the idle path), the step
 * boundary (`prepareStep`), the thinking-phase restart, the mid-text cut, the exit absorption.
 *
 * RED before part 2c: no separate call — the first words about an input were the next step's own
 * text, after that step's thinking.
 */

const TIMEOUT = 30_000;
const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

type Prompt = Array<{ role: string; content: unknown }>;
type CallOptions = {
  prompt: Prompt;
  abortSignal?: AbortSignal;
  tools?: unknown[];
  maxOutputTokens?: number;
  providerOptions?: Record<string, Record<string, unknown>>;
};
type Part = {
  type: string;
  textDelta?: string;
  finishReason?: string;
  utterance?: true;
  toolName?: string;
  text?: string;
};

/** A provider stream that stalls `ms` before its first part — a model "thinking" with nothing on the wire. */
const stalledStream = (ms: number, parts: unknown[]) => {
  let started = false;
  const queue = [...parts];
  return new ReadableStream<any>({
    async pull(controller) {
      if (!started) {
        started = true;
        await new Promise((resolve) => setTimeout(resolve, ms));
      }
      const next = queue.shift();
      if (next === undefined) {
        controller.close();
      } else {
        controller.enqueue(next);
      }
    },
  });
};

const toolCallParts = (id: string, toolName: string) => [
  { type: 'stream-start' as const, warnings: [] },
  { type: 'tool-input-start' as const, id, toolName },
  { type: 'tool-input-delta' as const, id, delta: '{}' },
  { type: 'tool-input-end' as const, id },
  { type: 'tool-call' as const, toolCallId: id, toolName, input: '{}' },
  { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
];

/** The host's side-utterance door (the nudge): each wait is a fresh promise; `push` resolves the current one. */
class Nudger {
  private waiter: ((input: DrainedInput) => void) | undefined;
  push(input: DrainedInput): void {
    const waiter = this.waiter;
    this.waiter = undefined;
    waiter?.(input);
  }
  wait(): Promise<DrainedInput> {
    return new Promise<DrainedInput>((resolve) => {
      this.waiter = resolve;
    });
  }
}

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const reasoningThenTextStep = (reasoning: string, text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'reasoning-start' as const, id: 'r1' },
    { type: 'reasoning-delta' as const, id: 'r1', delta: reasoning },
    { type: 'reasoning-end' as const, id: 'r1' },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const toolCallStep = (id: string, toolName: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-input-start' as const, id, toolName },
    { type: 'tool-input-delta' as const, id, delta: '{}' },
    { type: 'tool-input-end' as const, id },
    { type: 'tool-call' as const, toolCallId: id, toolName, input: '{}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const errorStream = () =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'error' as const, error: new Error('provider down') },
  ]);

const messageText = (msg: { content: unknown }): string =>
  typeof msg.content === 'string'
    ? msg.content
    : Array.isArray(msg.content)
      ? msg.content
          .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? part.text ?? '' : ''))
          .join('')
      : '';

class Inbox {
  readonly notes: Array<string | DrainedInput> = [];
  private wakers: Array<() => void> = [];

  push(note: string | DrainedInput): void {
    this.notes.push(note);
    const wakers = this.wakers;
    this.wakers = [];
    wakers.forEach((wake) => wake());
  }

  params(): Pick<
    GenerateStreamParams,
    'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes' | 'utterance'
  > {
    return {
      drainInjectedContext: () => this.notes.splice(0, this.notes.length),
      peekInjectedContext: () => this.notes.length > 0,
      inputArrived: () =>
        this.notes.length > 0 ? Promise.resolve() : new Promise<void>((resolve) => this.wakers.push(resolve)),
      absorbExitNotes: true,
      utterance: true,
    };
  }
}

/** The tool's execution hook — a note pushed from inside it lands DURING the call (mid-tool). */
const toolHooks = { onCall: undefined as undefined | (() => void) };

const conversation = (name: string, withTool = false) =>
  new Conversation({
    modelData: fixtureModelData,
    name,
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: withTool
      ? [
          {
            getId: () => 'do-work',
            getName: () => 'DoWork',
            getSystemMessages: () => [],
            getMessageModerators: () => [],
            getFunctions: () => [
              {
                definition: { name: 'doWork', description: 'work', parameters: { type: 'object', properties: {} } },
                call: async () => {
                  toolHooks.onCall?.();
                  return { ok: true };
                },
              },
            ],
          } as never,
        ]
      : [],
  });

async function collect(fullStream: AsyncIterable<unknown>): Promise<{ text: string; parts: Part[] }> {
  let text = '';
  const parts: Part[] = [];
  for await (const part of fullStream as AsyncIterable<Part>) {
    parts.push(part);
    if (part.type === 'text-delta') {
      text += part.textDelta ?? '';
    }
  }
  return { text, parts };
}

/** The framing at the END of a prompt: …, user(input?), assistant(line), user(continue). */
const expectFraming = (prompt: Prompt, line: string, input?: string): void => {
  const tail = prompt.slice(input ? -3 : -2);
  if (input) {
    expect(tail[0].role).toBe('user');
    expect(messageText(tail[0] as never)).toContain(input);
  }
  const lineMessage = tail[tail.length - 2];
  const continueMessage = tail[tail.length - 1];
  expect(lineMessage.role).toBe('assistant');
  expect(messageText(lineMessage as never)).toBe(line);
  expect(continueMessage.role).toBe('user');
  expect(messageText(continueMessage as never)).toBe(Utterance.continueFraming(line));
  // Never a prefill: the prompt ends on a user turn.
  expect(prompt[prompt.length - 1].role).toBe('user');
};

describe('the bounded utterance — one line before every step that takes an input in (FREE_AGENT §M.3 part 2c)', () => {
  test(
    'IDLE PATH: the take-in line is a separate no-tools, no-thinking, 80-token call over the request — streamed first, flagged utterance, then the first step runs under the framing',
    async () => {
      const calls: CallOptions[] = [];
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep('Got it — comparing the two now.') };
          }
          return { stream: reasoningThenTextStep('planning', 'THE ANSWER') };
        },
      });
      const result = await conversation('utterance-idle', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        reasoningEffort: 'auto',
        ...new Inbox().params(),
      });
      const { parts } = await collect(result.fullStream);

      expect(calls).toHaveLength(2);
      // The utterance call: the request with the instruction appended; no tools; thinking off;
      // the small ceiling.
      const utterance = calls[0];
      expect(Utterance.isRequest(utterance.prompt)).toBe(true);
      expect(messageText(utterance.prompt[utterance.prompt.length - 1] as never)).toContain('compare sql and nosql');
      expect(utterance.tools ?? []).toHaveLength(0);
      expect(utterance.maxOutputTokens).toBe(Utterance.MAX_OUTPUT_TOKENS);
      // The main step: tools on, the framing at the tail.
      const main = calls[1];
      expect((main.tools ?? []).length).toBeGreaterThan(0);
      expectFraming(main.prompt, 'Got it — comparing the two now.');
      // The stream: the line as its own step first (flagged), then the main step's parts.
      const firstFinish = parts.findIndex((part) => part.type === 'step-finish');
      expect(parts[firstFinish].utterance).toBe(true);
      expect(
        parts
          .slice(0, firstFinish)
          .filter((part) => part.type === 'text-delta')
          .map((part) => part.textDelta)
          .join('')
      ).toBe('Got it — comparing the two now.');
      expect(parts.slice(firstFinish + 1).some((part) => part.type === 'reasoning-delta')).toBe(true);
      const usageData = await result.usage;
      // Both calls ride the turn's usage.
      expect(usageData.totalTokenUsage.inputTokens).toBe(2);
    },
    TIMEOUT
  );

  test(
    'STEP BOUNDARY: a note drained at prepareStep is uttered there — the line reaches the stream after the finished step and before the next step, and the next step runs under the framing',
    async () => {
      const NOTE = 'Also check the costs.';
      const inbox = new Inbox();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(mainStep === 0 ? 'On it.' : 'Noted — costs too.') };
          }
          mainStep++;
          if (mainStep === 1) {
            return { stream: toolCallStep('tc-1', 'doWork') };
          }
          return { stream: textStep('THE ANSWER WITH COSTS') };
        },
      });
      // The note lands DURING the tool call step 1 makes (a side effect has begun — no restart);
      // the boundary after the call drains it.
      toolHooks.onCall = () => inbox.push(NOTE);
      const result = await conversation('utterance-boundary', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      const { parts } = await collect(result.fullStream);

      // utterance(idle) · step 1 · utterance(note) · step 2
      expect(calls).toHaveLength(4);
      expect(Utterance.isRequest(calls[2].prompt)).toBe(true);
      const noteRequest = calls[2].prompt[calls[2].prompt.length - 1];
      expect(messageText(noteRequest as never)).toContain(NOTE);
      expectFraming(calls[3].prompt, 'Noted — costs too.', NOTE);
      // Order on the stream: … tool-settled, step-finish(tool-calls), text('Noted — costs too.'),
      // step-finish(utterance), then step 2's text.
      toolHooks.onCall = undefined;
      const types = parts.map(
        (part) =>
          `${part.type}${part.utterance ? '(utterance)' : ''}${part.finishReason ? `:${part.finishReason}` : ''}`
      );
      const toolFinish = types.indexOf('step-finish:tool-calls');
      const noteUtterance = types.indexOf('step-finish(utterance):stop', toolFinish);
      expect(toolFinish).toBeGreaterThan(0);
      expect(noteUtterance).toBeGreaterThan(toolFinish);
      expect(
        parts
          .slice(toolFinish + 1, noteUtterance)
          .filter((part) => part.type === 'text-delta')
          .map((part) => part.textDelta)
          .join('')
      ).toBe('Noted — costs too.');
      expect(
        parts
          .slice(noteUtterance + 1)
          .filter((part) => part.type === 'text-delta')
          .map((part) => part.textDelta)
          .join('')
      ).toBe('THE ANSWER WITH COSTS');
    },
    TIMEOUT
  );

  test(
    'RESTART: a note that lands mid-thinking is uttered before the restarted round, which runs under the framing',
    async () => {
      const NOTE = 'Make it about time-series.';
      const inbox = new Inbox();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(mainStep === 0 ? 'On it.' : 'Time-series it is.') };
          }
          mainStep++;
          if (mainStep === 1) {
            inbox.push(NOTE);
            return { stream: reasoningThenTextStep('planning the general answer', 'GENERAL ANSWER') };
          }
          return { stream: textStep('RESHAPED ANSWER') };
        },
      });
      const result = await conversation('utterance-restart').generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      const { text, parts } = await collect(result.fullStream);
      expect(calls).toHaveLength(4);
      expect(text).toBe('On it.Time-series it is.RESHAPED ANSWER');
      expectFraming(calls[3].prompt, 'Time-series it is.', NOTE);
      expect(parts.filter((part) => part.utterance)).toHaveLength(2);
    },
    TIMEOUT
  );

  test(
    'EXIT ABSORPTION: a note after the final text is uttered before the continuation, with the joiner inside the finished step',
    async () => {
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(mainStep === 0 ? 'On it.' : 'Adding costs.') };
          }
          mainStep++;
          return { stream: textStep(mainStep === 1 ? 'FIRST ANSWER' : 'FOLDED IN') };
        },
      });
      const result = await conversation('utterance-exit').generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      let text = '';
      const parts: Part[] = [];
      let pushed = false;
      for await (const part of result.fullStream as AsyncIterable<Part>) {
        parts.push(part);
        if (part.type === 'text-delta') {
          text += part.textDelta ?? '';
          if (!pushed && text.includes('FIRST ANSWER')) {
            pushed = true;
            inbox.push(NOTE);
          }
        }
      }
      expect(calls).toHaveLength(4);
      expect(text).toBe('On it.FIRST ANSWER\n\nAdding costs.FOLDED IN');
      expectFraming(calls[3].prompt, 'Adding costs.', NOTE);
      // The joiner rides the finished step (before its step-finish); the utterance follows it.
      const finishes = parts.map((part, i) => ({ part, i })).filter(({ part }) => part.type === 'step-finish');
      const firstMainFinish = finishes[1];
      expect(firstMainFinish.part.utterance).toBeUndefined();
      expect(parts[firstMainFinish.i - 1].textDelta).toBe('\n\n');
      expect(finishes[2].part.utterance).toBe(true);
    },
    TIMEOUT
  );

  test(
    'a failed utterance call is logged and the step runs without its line — the input still splices as a plain user message',
    async () => {
      const calls: CallOptions[] = [];
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: errorStream() };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-failed').generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...new Inbox().params(),
      });
      const { text, parts } = await collect(result.fullStream);
      expect(calls).toHaveLength(2);
      expect(text).toBe('THE ANSWER');
      expect(parts.some((part) => part.utterance)).toBe(false);
      // No framing: the main step's prompt is the request alone.
      expect(calls[1].prompt[calls[1].prompt.length - 1].role).toBe('user');
      expect(messageText(calls[1].prompt[calls[1].prompt.length - 1] as never)).toBe('compare sql and nosql');
    },
    TIMEOUT
  );

  test('the utterance call runs with thinking OFF on every provider that has it, while the main step keeps its effort', () => {
    type Internals = {
      buildProviderOptions(
        provider: string,
        params: { reasoningEffort?: string },
        modelString?: string
      ): Record<string, any>;
    };
    const internals = conversation('utterance-provider-options') as unknown as Internals;
    expect(
      internals.buildProviderOptions('anthropic', { reasoningEffort: 'none' }, 'claude-opus-4-6').anthropic.thinking
    ).toBeUndefined();
    expect(
      internals.buildProviderOptions('anthropic', { reasoningEffort: 'auto' }, 'claude-opus-4-6').anthropic.thinking
    ).toBeDefined();
    expect(internals.buildProviderOptions('google', { reasoningEffort: 'none' }).google.thinkingConfig).toBeUndefined();
    expect(internals.buildProviderOptions('openai', { reasoningEffort: 'none' }).openai.reasoningEffort).toBe('none');
  });

  test(
    "SIDE UTTERANCE (part 5 — the nudge): the host asks mid-round while the model is silent; the line lands as ONE side-utterance part before the provider's first part, the round is not restarted, and the next step runs under its framing",
    async () => {
      const NUDGE: DrainedInput = {
        text: 'HARNESS NUDGE: the user has heard nothing from you for a while.',
        ask: 'In one short sentence, say what you are doing right now.',
      };
      const LINE = 'Still comparing the two.';
      const inbox = new Inbox();
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            const last = messageText(options.prompt[options.prompt.length - 1] as never);
            return { stream: textStep(last.includes(NUDGE.text) ? LINE : 'On it.') };
          }
          mainStep++;
          if (mainStep === 1) {
            // A silent think: nothing on the wire for 600 ms; the host nudges 100 ms in.
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, toolCallParts('tc-1', 'doWork')) };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-side', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
        sideUtterance: () => nudger.wait(),
      });
      const { parts } = await collect(result.fullStream);

      // take-in utterance · step 1 · the side utterance · step 2 — the round ran on (no restart).
      expect(calls).toHaveLength(4);
      expect(mainStep).toBe(2);
      const side = calls[2];
      expect(Utterance.isRequest(side.prompt)).toBe(true);
      expect(side.tools ?? []).toHaveLength(0);
      const sideText = messageText(side.prompt[side.prompt.length - 1] as never);
      expect(sideText).toContain(NUDGE.text);
      // The input's OWN ask, not the default acknowledgment instruction.
      expect(sideText.trimEnd().endsWith(`${NUDGE.ask} ${Utterance.REPLY_WITH_THE_LINE_ONLY}`)).toBe(true);
      expect(sideText).not.toContain(Utterance.INSTRUCTION);
      // ONE part, mid-round: after the take-in line's step, before step 1's first provider part.
      const types = parts.map((part) => part.type);
      const takeIn = types.indexOf('step-finish');
      const sidePart = types.indexOf('side-utterance');
      const firstToolPart = types.indexOf('tool-call');
      expect(sidePart).toBeGreaterThan(takeIn);
      expect(sidePart).toBeLessThan(firstToolPart);
      expect(parts[sidePart].text).toBe(LINE);
      expect(parts.filter((part) => part.type === 'side-utterance')).toHaveLength(1);
      // Step 2 runs under the framing: the nudge as a user message, the line as the agent's own,
      // the continue instruction — after step 1's tool result, before anything else.
      expectFraming(calls[3].prompt, LINE, NUDGE.text);
      const usageData = await result.usage;
      expect(usageData.totalTokenUsage.inputTokens).toBe(4);
    },
    TIMEOUT
  );

  test(
    'AN INPUT WITH ITS OWN ASK at a boundary is uttered with that ask (the marker holds) and framed by its text alone',
    async () => {
      const NOTE: DrainedInput = { text: 'Also check the costs.', ask: 'Say, in five words, what you will do.' };
      const inbox = new Inbox();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(mainStep === 0 ? 'On it.' : 'Costs next.') };
          }
          mainStep++;
          return { stream: mainStep === 1 ? toolCallStep('tc-1', 'doWork') : textStep('THE ANSWER WITH COSTS') };
        },
      });
      toolHooks.onCall = () => inbox.push(NOTE);
      const result = await conversation('utterance-own-ask', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      await collect(result.fullStream);
      toolHooks.onCall = undefined;
      expect(calls).toHaveLength(4);
      const request = messageText(calls[2].prompt[calls[2].prompt.length - 1] as never);
      expect(request).toContain(NOTE.text);
      expect(request.trimEnd().endsWith(`${NOTE.ask} ${Utterance.REPLY_WITH_THE_LINE_ONLY}`)).toBe(true);
      expect(request).not.toContain(Utterance.INSTRUCTION);
      expectFraming(calls[3].prompt, 'Costs next.', NOTE.text);
      expect(messageText(calls[3].prompt[calls[3].prompt.length - 3] as never)).toBe(NOTE.text);
    },
    TIMEOUT
  );

  test("Utterance.askFor — the default instruction, or the last input's own ask normalized to the marker", () => {
    expect(Utterance.askFor(['a note'])).toBe(Utterance.INSTRUCTION);
    expect(Utterance.askFor([{ text: 'a note' }, { text: 'b' }])).toBe(Utterance.INSTRUCTION);
    expect(Utterance.askFor([{ text: 'a', ask: 'Do X.' }])).toBe(`Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`);
    expect(Utterance.askFor([{ text: 'a', ask: `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}` }])).toBe(
      `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`
    );
    expect(Utterance.askFor([{ text: 'a', ask: 'Do X.' }, { text: 'b' }])).toBe(
      `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`
    );
    expect(Utterance.INSTRUCTION.endsWith(Utterance.REPLY_WITH_THE_LINE_ONLY)).toBe(true);
    const custom = Utterance.request([{ role: 'user', content: 'hello' }] as never[], [{ text: 'n', ask: 'Do X.' }]);
    expect(Utterance.isRequest(custom as never)).toBe(true);
  });

  test('Utterance.request / framing / isRequest — the shapes', () => {
    const transcript = [{ role: 'user', content: 'hello' }] as never[];
    const idle = Utterance.request(transcript, []);
    expect(idle).toHaveLength(1);
    expect(Utterance.isRequest(idle as never)).toBe(true);
    expect(messageText(idle[0] as never)).toBe(`hello\n\n${Utterance.INSTRUCTION}`);
    const withInputs = Utterance.request(transcript, ['a note', 'another']);
    expect(withInputs).toHaveLength(2);
    expect(messageText(withInputs[1] as never)).toBe(`a note\n\nanother\n\n${Utterance.INSTRUCTION}`);
    expect(Utterance.isRequest(transcript as never)).toBe(false);
    const framing = Utterance.framing(['a note'], 'Got it.');
    expect(framing.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(messageText(framing[2] as never)).toContain('Got it.');
  });
});
