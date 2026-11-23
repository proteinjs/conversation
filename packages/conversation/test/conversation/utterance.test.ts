import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance, type DrainedInput } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE SIDE UTTERANCE (plans/FREE_AGENT.md §M.3 part 5 — the nudge's door): when the host asks
 * mid-round, the loop asks the mind for ONE LINE in a separate call over the round's transcript on
 * the step's own tool roster (nothing runnable on it) with the input's OWN ask, and yields it as
 * ONE `side-utterance` part the moment it lands. The round is never restarted for it, and what is
 * said aside stays aside: no step is sent the ask or the line, so the reply is the round's own
 * words, whole. One predicate guards every door against an open server tool (§M.16 found (2)):
 * the side utterance, the cut.
 *
 * THE BOUNDED UTTERANCE IS GONE (2026-11-05, DECISIONS 1099): no separate call precedes a step
 * that takes an input in — at turn start, at a boundary drain, at a restart, a cut or the exit
 * absorption, the inputs splice as plain user messages and the step's own first words are what
 * the person reads about them (the consumer's acknowledgment window). RED at the pre-strip loop:
 * a take-in line was asked for on every door and streamed as its own flagged step.
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

const toolCallStep = (id: string, toolName: string) => convertArrayToReadableStream(toolCallParts(id, toolName));

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
    'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes'
  > {
    return {
      drainInjectedContext: () => this.notes.splice(0, this.notes.length),
      peekInjectedContext: () => this.notes.length > 0,
      inputArrived: () =>
        this.notes.length > 0 ? Promise.resolve() : new Promise<void>((resolve) => this.wakers.push(resolve)),
      absorbExitNotes: true,
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

/**
 * What is said aside stays aside: a step's prompt holds nothing of the side call — not the host's
 * input, not the line — and ends where the round's own transcript ends (a tool result).
 */
const expectNothingOfTheAside = (prompt: Prompt, line: string, input: string): void => {
  expect(prompt.filter((message) => messageText(message as never).includes(input))).toEqual([]);
  expect(prompt.filter((message) => messageText(message as never).includes(line))).toEqual([]);
  expect(prompt[prompt.length - 1].role).toBe('tool');
};

const NUDGE: DrainedInput = {
  text: 'HARNESS NUDGE: the user has heard nothing from you for a while.',
  ask: 'In one short sentence, say what you are doing right now.',
};
const LINE = 'Still comparing the two.';

/** A reply's first section as a model writes one: a heading, a bold line and bullets on consecutive lines — one paragraph. */
const PART_ONE = [
  '### 1. Is it worth it?',
  '**Yes, with one caveat.**',
  '* It pays for itself early.',
  '* The catch is the cost.',
].join('\n');
const PART_TWO = ['### 2. Is switching a big commitment?', 'It depends on which one you drop.'].join('\n');
/** A reply's opening paragraph as prose: one line, far longer than any sentence about what is being done. */
const LONG_OPENING =
  'Yes, it is very manageable for a weekly baker: a fridge-kept starter only needs to be fed once a week with about ' +
  'fifty grams of flour, it forgives several missed weeks and usually revives in two feedings, and the weekly discard ' +
  'costs only a few cents, so for one loaf a week the upkeep is small next to what you gain in flavor and keeping ' +
  'quality, and the only real cost is remembering the feed on the same day you bake.';

/**
 * A model goes on from what its transcript says it has already written: with the reply's first section there as its
 * own words it writes the second only; with nothing of the reply there it writes the reply whole.
 */
const goesOnFromItsTranscript = (prompt: Prompt): string =>
  prompt.some((message) => message.role === 'assistant' && messageText(message as never).includes(PART_ONE))
    ? PART_TWO
    : `${PART_ONE}\n\n${PART_TWO}`;

/** A step that batches web_search (server, deferred — no result in the message) with doWork (client). */
const openOnSearchParts = (serverId: string, clientId: string) => [
  { type: 'stream-start' as const, warnings: [] },
  {
    type: 'tool-call' as const,
    toolCallId: serverId,
    toolName: 'web_search',
    input: '{"query":"frontier models"}',
    providerExecuted: true,
    dynamic: true,
  },
  { type: 'tool-call' as const, toolCallId: clientId, toolName: 'doWork', input: '{}' },
  { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
];
/** The step after: the API ran the deferred search first (its result opens the message), then doWork again. */
const searchResultThenCallParts = (serverId: string, clientId: string) => [
  { type: 'stream-start' as const, warnings: [] },
  {
    type: 'tool-result' as const,
    toolCallId: serverId,
    toolName: 'web_search',
    result: [{ url: 'https://example.com', title: 'Example' }],
  },
  { type: 'tool-call' as const, toolCallId: clientId, toolName: 'doWork', input: '{}' },
  { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
];

/**
 * Anthropic's contract for a server tool the API deferred: the next request may carry ONLY the
 * client tool results after the open assistant message — a user (or assistant) block there ends
 * the assistant turn and the unresolved server call fails the request (the R7 finding-9 400).
 */
const expectNoBlockBeyondToolResultsAfterOpenServerTool = (prompt: Prompt): void => {
  type ToolPart = { type: string; toolCallId?: string; providerExecuted?: boolean };
  const assistantParts = (message: { role: string; content: unknown }): ToolPart[] =>
    message.role === 'assistant' && Array.isArray(message.content) ? (message.content as ToolPart[]) : [];
  // A deferred call is settled by the result block the API puts in its NEXT assistant message.
  const settled = new Set(
    prompt
      .flatMap(assistantParts)
      .filter((part) => part.type === 'tool-result')
      .map((part) => part.toolCallId)
  );
  prompt.forEach((message, i) => {
    const parts = assistantParts(message);
    const open = parts
      .filter((part) => part.type === 'tool-call' && part.providerExecuted === true && !settled.has(part.toolCallId))
      .map((part) => part.toolCallId);
    if (open.length === 0) {
      return;
    }
    const after = prompt.slice(i + 1).map((m) => m.role);
    expect({ open, after }).toEqual({ open, after: after.filter((role) => role === 'tool') });
  });
};

const sideCalls = (calls: CallOptions[]) =>
  calls.filter(
    (call) =>
      Utterance.isRequest(call.prompt) && messageText(call.prompt[call.prompt.length - 1] as never).includes(NUDGE.text)
  );

describe('the side utterance — one line said mid-round at the host’s ask (FREE_AGENT §M.3 part 5)', () => {
  afterEach(() => {
    toolHooks.onCall = undefined;
  });

  test(
    "THE NUDGE: the host asks mid-round while the model is silent; the line lands as ONE side-utterance part before the provider's first part, the round is not restarted, and the next step is sent nothing of it",
    async () => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(LINE) };
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
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { parts } = await collect(result.fullStream);

      // step 1 · the side utterance · step 2 — the round ran on (no restart), and no call preceded
      // step 1 (the bounded utterance is gone).
      expect(calls).toHaveLength(3);
      expect(mainStep).toBe(2);
      expect(Utterance.isRequest(calls[0].prompt)).toBe(false);
      const side = calls[1];
      expect(Utterance.isRequest(side.prompt)).toBe(true);
      // On the round's own roster (the step's cache prefix).
      expect(side.tools).toEqual(calls[0].tools);
      const sideText = messageText(side.prompt[side.prompt.length - 1] as never);
      expect(sideText).toContain(NUDGE.text);
      // The input's OWN ask, normalized to the marker.
      expect(sideText.trimEnd().endsWith(`${NUDGE.ask} ${Utterance.REPLY_WITH_THE_LINE_ONLY}`)).toBe(true);
      // ONE part, mid-round: before step 1's first provider part, before any step-finish.
      const types = parts.map((part) => part.type);
      const sidePart = types.indexOf('side-utterance');
      expect(sidePart).toBeGreaterThanOrEqual(0);
      expect(sidePart).toBeLessThan(types.indexOf('tool-call'));
      expect(sidePart).toBeLessThan(types.indexOf('step-finish'));
      expect(parts[sidePart].text).toBe(LINE);
      expect(parts.filter((part) => part.type === 'side-utterance')).toHaveLength(1);
      // Step 2 is sent what it would have been sent had nothing been asked: step 1's prompt, its
      // tool call and the result — nothing of the nudge, nothing of the line.
      expectNothingOfTheAside(calls[2].prompt, LINE, NUDGE.text);
      expect(calls[2].prompt.slice(0, calls[0].prompt.length)).toEqual(calls[0].prompt);
      expect(calls[2].prompt.slice(calls[0].prompt.length).map((message) => message.role)).toEqual([
        'assistant',
        'tool',
      ]);
      // The side call's usage rides the turn's: three requests, three input tokens.
      const usageData = await result.usage;
      expect(usageData.totalTokenUsage.inputTokens).toBe(3);
    },
    TIMEOUT
  );

  test(
    'NO ACKNOWLEDGMENT CALL ON ANY DOOR: at turn start and at a boundary drain the step itself takes the input in — no separate request precedes it, the input rides the next step as a plain user message, and no step-finish is flagged (RED at the pre-strip loop: the `utterance` switch made a take-in call on both doors)',
    async () => {
      const NOTE = 'also the header';
      const inbox = new Inbox();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            // What the pre-strip loop asked for on every door; nothing asks it now.
            return { stream: textStep('On it.') };
          }
          mainStep++;
          if (mainStep === 1) {
            return { stream: toolCallStep('tc-1', 'doWork') };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      // The note lands DURING the tool call: the boundary after it drains the note into step 2.
      toolHooks.onCall = () => inbox.push(NOTE);
      const result = await conversation('utterance-no-ack-call', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
        // The retired switch, as a pre-strip caller passed it: the loop has no such door any more.
        ...({ utterance: true } as unknown as Partial<GenerateStreamParams>),
      });
      const { text, parts } = await collect(result.fullStream);

      expect(text).toContain('THE ANSWER');
      // Two requests, both main steps — no request asks for a line.
      expect(mainStep).toBe(2);
      expect(calls).toHaveLength(2);
      expect(calls.some((call) => Utterance.isRequest(call.prompt))).toBe(false);
      // The note rides step 2 as a plain user message — the last message, right after the tool
      // result, nothing around it.
      const last = calls[1].prompt[calls[1].prompt.length - 1];
      expect(last.role).toBe('user');
      expect(messageText(last as never)).toBe(NOTE);
      expect(calls[1].prompt[calls[1].prompt.length - 2].role).toBe('tool');
      // No step-finish carries a flag: every step's text is the step's own.
      expect(parts.filter((part) => part.type === 'step-finish')).toHaveLength(2);
      expect(parts.some((part) => 'utterance' in part)).toBe(false);
    },
    TIMEOUT
  );

  test(
    'A FAILED SIDE CALL: nothing is said — no side-utterance part, nothing of the ask in the step after — and the round runs on',
    async () => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: errorStream() };
          }
          mainStep++;
          if (mainStep === 1) {
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, toolCallParts('tc-1', 'doWork')) };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-side-failed', true).generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { text, parts } = await collect(result.fullStream);

      expect(text).toContain('THE ANSWER');
      expect(mainStep).toBe(2);
      expect(sideCalls(calls)).toHaveLength(1);
      expect(parts.some((part) => part.type === 'side-utterance')).toBe(false);
      const step2 = calls[calls.length - 1].prompt;
      expect(step2.some((m) => messageText(m as never).includes(NUDGE.text))).toBe(false);
    },
    TIMEOUT
  );

  test(
    "A SIDE CALL THAT WRITES THE REPLY'S FIRST PART (prod, 2026-11-22: asked what it was doing, a Google model wrote its answer's first section — a heading with a bold line and bullets under it — and the reply a person read began at its second section): a block is never the line, so nothing is said aside, and the reply is whole",
    async () => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            // The side ask answered with the reply itself: its first section, then on to the second.
            return { stream: textStep(`${PART_ONE}\n\n${PART_TWO}`) };
          }
          mainStep++;
          if (mainStep === 1) {
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, toolCallParts('tc-1', 'doWork')) };
          }
          return { stream: textStep(goesOnFromItsTranscript(options.prompt)) };
        },
      });
      const result = await conversation('utterance-side-block', true).generateStream({
        messages: ['is it worth it, and is switching a big commitment?'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { text, parts } = await collect(result.fullStream);

      expect(mainStep).toBe(2);
      expect(sideCalls(calls)).toHaveLength(1);
      // The reply a person reads is the whole answer, from its first section.
      expect(text).toBe(`${PART_ONE}\n\n${PART_TWO}`);
      // And the section is not said aside: a block is not a line.
      expect(parts.filter((part) => part.type === 'side-utterance')).toEqual([]);
    },
    TIMEOUT
  );

  test.each([
    [
      "the reply's first heading on a line of its own",
      `### 1. Is it worth it?\n\n**Yes, with one caveat.**\n\n${PART_TWO}`,
    ],
    ["a whole paragraph of the reply's prose, one long line", `${LONG_OPENING}\n\n${PART_TWO}`],
  ])(
    "A SIDE CALL THAT WRITES THE REPLY'S OPENING ON ONE LINE — %s: one line is not enough to be the line; a heading or a paragraph is the answer begun, so nothing is said aside (it would stand in the timeline and again at the top of the reply), and the reply is whole",
    async (_shape, sideText) => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(sideText) };
          }
          mainStep++;
          if (mainStep === 1) {
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, toolCallParts('tc-1', 'doWork')) };
          }
          return { stream: textStep(goesOnFromItsTranscript(options.prompt)) };
        },
      });
      const result = await conversation('utterance-side-opening', true).generateStream({
        messages: ['is it worth it, and is switching a big commitment?'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { text, parts } = await collect(result.fullStream);

      expect(sideCalls(calls)).toHaveLength(1);
      expect(parts.filter((part) => part.type === 'side-utterance').map((part) => part.text)).toEqual([]);
      expect(text).toBe(`${PART_ONE}\n\n${PART_TWO}`);
    },
    TIMEOUT
  );

  test(
    'THE SIDE CALL CARRIES ITS OWN OUTPUT LIMIT: its request names one and no step of the round does; a call the limit ends has no line — what it wrote so far is never shown cut down — and the round runs on',
    async () => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            // The answer begun in the line's place, stopped by the limit before its first paragraph ended.
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                { type: 'text-start' as const, id: 't1' },
                { type: 'text-delta' as const, id: 't1', delta: 'Yes, it is very manageable for a weekly' },
                { type: 'text-end' as const, id: 't1' },
                { type: 'finish' as const, finishReason: { unified: 'length' as const, raw: 'max_tokens' }, usage },
              ]),
            };
          }
          mainStep++;
          if (mainStep === 1) {
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, toolCallParts('tc-1', 'doWork')) };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-side-limit', true).generateStream({
        messages: ['is it worth it?'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { text, parts } = await collect(result.fullStream);

      const side = sideCalls(calls);
      expect(side).toHaveLength(1);
      expect(side[0].maxOutputTokens).toBeGreaterThan(0);
      expect(calls.filter((call) => !side.includes(call)).map((call) => call.maxOutputTokens)).toEqual([
        undefined,
        undefined,
      ]);
      expect(parts.filter((part) => part.type === 'side-utterance')).toEqual([]);
      expect(text).toBe('THE ANSWER');
    },
    TIMEOUT
  );

  test(
    'SIDE UTTERANCE HELD: a nudge that lands while the transcript is open on a server tool is not asked there (that request 400s) — it is asked at the next closed boundary, its part on the stream, nothing of it in the step after',
    async () => {
      const nudger = new Nudger();
      const calls: CallOptions[] = [];
      let mainStep = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: CallOptions) => {
          calls.push(options);
          if (Utterance.isRequest(options.prompt)) {
            return { stream: textStep(LINE) };
          }
          mainStep++;
          if (mainStep === 1) {
            return { stream: convertArrayToReadableStream(openOnSearchParts('srv-1', 'tc-1')) };
          }
          if (mainStep === 2) {
            // The step that runs the deferred search: silent for 600 ms; the host nudges 100 ms in,
            // while the request this step runs on is still open on the search.
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(600, searchResultThenCallParts('srv-1', 'tc-2')) };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-side-held', true).generateStream({
        messages: ['research the frontier models'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      const { parts } = await collect(result.fullStream);

      for (const call of calls) {
        expectNoBlockBeyondToolResultsAfterOpenServerTool(call.prompt);
      }
      // step 1 · step 2 · the side utterance (at the closed boundary) · step 3
      expect(mainStep).toBe(3);
      const side = sideCalls(calls);
      expect(side).toHaveLength(1);
      expect(calls.indexOf(side[0])).toBe(calls.length - 2);
      // Said over the transcript that carries the search's result — the boundary before step 3.
      const sidePrompt = side[0].prompt;
      const lastAssistant = [...sidePrompt].reverse().find((m) => m.role === 'assistant');
      expect(JSON.stringify(lastAssistant?.content)).toContain('srv-1');
      expect(JSON.stringify(lastAssistant?.content)).toContain('tool-result');
      const sideParts = parts.filter((part) => part.type === 'side-utterance');
      expect(sideParts).toHaveLength(1);
      expect(sideParts[0].text).toBe(LINE);
      expectNothingOfTheAside(calls[calls.length - 1].prompt, LINE, NUDGE.text);
    },
    TIMEOUT
  );

  test("Utterance.askFor — the last input's own ask normalized to the marker; an input set with no ask is an error", () => {
    expect(Utterance.askFor([{ text: 'a', ask: 'Do X.' }])).toBe(`Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`);
    expect(Utterance.askFor([{ text: 'a', ask: `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}` }])).toBe(
      `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`
    );
    expect(Utterance.askFor([{ text: 'a', ask: 'Do X.' }, { text: 'b' }])).toBe(
      `Do X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`
    );
    expect(() => Utterance.askFor(['a note'])).toThrow(/own ask/);
    expect(() => Utterance.askFor([{ text: 'a note' }, { text: 'b' }])).toThrow(/own ask/);
  });

  test('Utterance.request / isRequest / lineEnd / isLine — the shapes; a request with no input is an error', () => {
    const transcript = [{ role: 'user', content: 'hello' }] as never[];
    const request = Utterance.request(transcript, [{ text: 'n', ask: 'Do X.' }]);
    expect(request).toHaveLength(2);
    expect(Utterance.isRequest(request as never)).toBe(true);
    expect(messageText(request[1] as never)).toBe(`n\n\nDo X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`);
    expect(Utterance.isRequest(transcript as never)).toBe(false);
    expect(() => Utterance.request(transcript, [])).toThrow(/none was given/);
    expect(Utterance.lineEnd('Got it.\n\nCaveat')).toBe(7);
    expect(Utterance.lineEnd('\n\nGot it.')).toBe(-1);
    expect(Utterance.lineEnd('Got it — no break yet')).toBe(-1);
    expect(Utterance.isLine(' Got it. \n')).toBe(true);
    expect(Utterance.isLine('### A heading\n* and a list under it')).toBe(false);
  });

  test('Utterance.isLine — a line is a short sentence: emphasis inside it is still a line; a heading, a list item, a table row, a quote or a line that is only bold is the answer begun, and so is a line as long as a paragraph', () => {
    const lines = [
      'Comparing weekly sourdough upkeep and the cost per loaf.',
      '*Reading the sign-in audit.*',
      'Checking the **cost** figures against the three sources.',
      '3 pages read, comparing their prices now.',
    ];
    const notLines = [
      '### 1. Is it worth it?',
      '# Sourdough for one loaf a week',
      '1. Is it worth it?',
      '2) Switching is a small commitment.',
      '- It pays for itself early.',
      '* The catch is the cost.',
      '| Option | Cost per loaf |',
      '> A starter kept in the fridge needs one feeding a week.',
      '**1. Is keeping a starter worth it?**',
      '__Yes, with one caveat.__',
      LONG_OPENING,
    ];
    expect(lines.filter((line) => !Utterance.isLine(line))).toEqual([]);
    expect(notLines.filter((line) => Utterance.isLine(line))).toEqual([]);
  });
});
