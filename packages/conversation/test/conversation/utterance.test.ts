import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance, type DrainedInput } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE SIDE UTTERANCE (plans/FREE_AGENT.md §M.3 part 5 — the nudge's door): when the host asks
 * mid-round, the loop asks the mind for ONE LINE in a separate call over the round's transcript on
 * the step's own tool roster (nothing runnable on it) with the input's OWN ask, yields it as ONE
 * `side-utterance` part the moment it lands, and runs the next step with the input spliced and the
 * line riding as the agent's own prior message + the continue framing. The round is never
 * restarted for it. One predicate guards every door against an open server tool (§M.16 found (2)):
 * the side utterance, the side framing, the cut.
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

const NUDGE: DrainedInput = {
  text: 'HARNESS NUDGE: the user has heard nothing from you for a while.',
  ask: 'In one short sentence, say what you are doing right now.',
};
const LINE = 'Still comparing the two.';

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
    "THE NUDGE: the host asks mid-round while the model is silent; the line lands as ONE side-utterance part before the provider's first part, the round is not restarted, and the next step runs under its framing",
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
      // Step 2 runs under the framing: the nudge as a user message, the line as the agent's own,
      // the continue instruction — after step 1's tool result, before anything else.
      expectFraming(calls[2].prompt, LINE, NUDGE.text);
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
      // The note rides step 2 as a plain user message — the last message, after the tool result,
      // no framing around it.
      const last = calls[1].prompt[calls[1].prompt.length - 1];
      expect(last.role).toBe('user');
      expect(messageText(last as never)).toBe(NOTE);
      expect(calls[1].prompt.some((m) => messageText(m as never).includes('You have already told the user'))).toBe(
        false
      );
      // No step-finish carries a flag: every step's text is the step's own.
      expect(parts.filter((part) => part.type === 'step-finish')).toHaveLength(2);
      expect(parts.some((part) => 'utterance' in part)).toBe(false);
    },
    TIMEOUT
  );

  test(
    'A FAILED SIDE CALL: nothing is said — no side-utterance part, no framing in the step after — and the round runs on',
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
      expect(step2.some((m) => messageText(m as never).includes('You have already told the user'))).toBe(false);
    },
    TIMEOUT
  );

  test(
    'SIDE UTTERANCE HELD: a nudge that lands while the transcript is open on a server tool is not asked there (that request 400s) — it is asked at the next closed boundary, its part on the stream, its framing in the step after',
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
      expectFraming(calls[calls.length - 1].prompt, LINE, NUDGE.text);
    },
    TIMEOUT
  );

  test(
    'SIDE FRAMING HELD: a line said over a closed transcript whose step then batches a server tool with a client tool is not spliced at that open boundary (the main request would 400) — it rides the boundary after the search settles',
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
            // A silent think (closed transcript), the nudge 100 ms in — the line is said; then the
            // step batches the search with a client tool, so the boundary after it is OPEN.
            setTimeout(() => nudger.push(NUDGE), 100);
            return { stream: stalledStream(400, openOnSearchParts('srv-1', 'tc-1')) };
          }
          if (mainStep === 2) {
            return { stream: convertArrayToReadableStream(searchResultThenCallParts('srv-1', 'tc-2')) };
          }
          return { stream: textStep('THE ANSWER') };
        },
      });
      const result = await conversation('utterance-side-framing-held', true).generateStream({
        messages: ['research the frontier models'],
        model: model as never,
        ...new Inbox().params(),
        sideUtterance: () => nudger.wait(),
      });
      await collect(result.fullStream);

      for (const call of calls) {
        expectNoBlockBeyondToolResultsAfterOpenServerTool(call.prompt);
      }
      expect(mainStep).toBe(3);
      expect(sideCalls(calls)).toHaveLength(1);
      const mainPrompts = calls.filter((call) => !Utterance.isRequest(call.prompt)).map((call) => call.prompt);
      // Step 2's request: the client tool result only — the framing waits.
      const step2 = mainPrompts[1];
      expect(step2[step2.length - 1].role).toBe('tool');
      expect(step2.some((m) => messageText(m as never).includes(NUDGE.text))).toBe(false);
      // Step 3's request: the framing, after the settled search and step 2's tool result.
      expectFraming(mainPrompts[2], LINE, NUDGE.text);
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

  test('Utterance.request / framing / isRequest / lineEnd — the shapes; a request with no input is an error', () => {
    const transcript = [{ role: 'user', content: 'hello' }] as never[];
    const request = Utterance.request(transcript, [{ text: 'n', ask: 'Do X.' }]);
    expect(request).toHaveLength(2);
    expect(Utterance.isRequest(request as never)).toBe(true);
    expect(messageText(request[1] as never)).toBe(`n\n\nDo X. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`);
    expect(Utterance.isRequest(transcript as never)).toBe(false);
    expect(() => Utterance.request(transcript, [])).toThrow(/none was given/);
    const framing = Utterance.framing(['a note'], 'Got it.');
    expect(framing.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(messageText(framing[2] as never)).toContain('Got it.');
    expect(Utterance.lineEnd('Got it.\n\nCaveat')).toBe(7);
    expect(Utterance.lineEnd('\n\nGot it.')).toBe(-1);
    expect(Utterance.lineEnd('Got it — no break yet')).toBe(-1);
  });
});
