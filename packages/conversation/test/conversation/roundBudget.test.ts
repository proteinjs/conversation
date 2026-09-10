import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE STEP BUDGET (plans/FREE_AGENT.md §M.3 part 2a/2b — the 10-second bar's thinking and mid-text
 * rows): the executor owns the clock on the STEP, not only on the tool.
 *
 *  (a) Thinking, clock-driven: the round loop races the provider's next part against the input
 *      wake (`inputArrived`), so a note that lands while the model is still thinking — nothing on
 *      the wire yet — restarts the round AT ONCE. RED at the per-part check: a note before the
 *      first part waited for that part (the harness's thinking row read the phase's length).
 *  (b) Text, cut-and-continue: a note that lands while text streams gives the generation N
 *      (`CONVERSATION_TOOL_SOFT_BUDGET_MS`) to finish on its own; past N the round is cut at the
 *      next paragraph break; past N + 2 s (the deadline) at the next BOUNDARY of any kind — a
 *      paragraph break, a sentence end, a line end, a code fence's close — and past the boundary
 *      window (`Conversation.CUT_BOUNDARY_WAIT_MS`) at the next word boundary; NEVER mid-word and
 *      never inside a code fence (plans/FREE_AGENT.md §M.16: the deadline cut landed inside a
 *      sentence, a word and a heading in 3 of 3 live runs). The text so far COMMITS exactly as a
 *      finished round's (the joiner, then a step-finish), and the SAME response continues from that
 *      text with the note spliced. RED at `absorbExitNotes` waiting for the generation's end (the
 *      harness's mid-text row read the generation's length); RED at the deadline cut (mid-word).
 *
 * No network: MockLanguageModelV3 scripts each round; the outgoing prompts prove what each round
 * saw; the parts prove what the consumer was shown, in order.
 */

const TIMEOUT = 30_000;
const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

type Prompt = Array<{ role: string; content: unknown }>;
type Part = { type: string; textDelta?: string; finishReason?: string; utterance?: true };

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** A provider stream that emits parts on a schedule — cut short by the round's abort. */
const scheduledStream = (
  signal: AbortSignal | undefined,
  schedule: Array<{ afterMs: number; parts: unknown[] }>
): ReadableStream<any> => {
  const queue = schedule.map((entry) => ({ ...entry, parts: [...entry.parts] }));
  return new ReadableStream({
    start(controller) {
      controller.enqueue({ type: 'stream-start', warnings: [] });
    },
    async pull(controller) {
      const next = queue.shift();
      if (!next) {
        controller.close();
        return;
      }
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, next.afterMs);
        signal?.addEventListener(
          'abort',
          () => {
            clearTimeout(timer);
            resolve();
          },
          { once: true }
        );
      });
      if (signal?.aborted) {
        controller.error(Object.assign(new Error('aborted'), { name: 'AbortError' }));
        return;
      }
      for (const part of next.parts) {
        controller.enqueue(part);
      }
    },
  });
};

/** A text generation: `deltas` one every `tickMs` after the opening delta, then the finish. */
const generation = (signal: AbortSignal | undefined, opening: string, deltas: string[], tickMs = 50) =>
  scheduledStream(signal, [
    {
      afterMs: 0,
      parts: [
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: opening },
      ],
    },
    ...deltas.map((delta) => ({ afterMs: tickMs, parts: [{ type: 'text-delta', id: 't1', delta }] })),
    {
      afterMs: 0,
      parts: [
        { type: 'text-end', id: 't1' },
        { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage },
      ],
    },
  ]);

const messageText = (msg: { content: unknown }): string =>
  typeof msg.content === 'string'
    ? msg.content
    : Array.isArray(msg.content)
      ? msg.content
          .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? part.text ?? '' : ''))
          .join('')
      : '';

/** The caller's inbox with the wake the loop races (what thought's ChatTurnRegistry provides). */
class Inbox {
  readonly notes: string[] = [];
  private wakers: Array<() => void> = [];

  push(note: string): void {
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

const conversation = (name: string) =>
  new Conversation({ modelData: fixtureModelData, name, logLevel: 'error', limits: { enforceLimits: false } });

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

/** The text the consumer was shown before the first step-finish — the committed cut text. */
const textBeforeFirstFinish = (parts: Part[]): { text: string; finish: Part } => {
  const firstFinish = parts.findIndex((part) => part.type === 'step-finish');
  expect(firstFinish).toBeGreaterThan(0);
  return {
    finish: parts[firstFinish],
    text: parts
      .slice(0, firstFinish)
      .filter((part) => part.type === 'text-delta')
      .map((part) => part.textDelta)
      .join(''),
  };
};

/**
 * The (b) shape with real sentences: `count` words, every `every`th ending a sentence, the note
 * at 500 ms, N = 300 ms — so the deadline (N + 2 s) falls at 2.8 s, inside a sentence.
 */
const sentenceWords = (count: number, every: number) =>
  Array.from({ length: count }, (_, i) => ((i + 1) % every === 0 ? ` word${i}.` : ` word${i}`));

/** One scripted turn of the (b) shape: the note pushed 500 ms in; the prompts stamped. */
const midTextTurn = (
  name: string,
  opening: string,
  deltas: string[],
  extra: Partial<GenerateStreamParams> = {},
  tickMs = 50
) => {
  const NOTE = 'Also mention costs.';
  const inbox = new Inbox();
  const prompts: Array<{ at: number; prompt: Prompt }> = [];
  let pushedAt = 0;
  let call = 0;
  const model = new MockLanguageModelV3({
    doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
      prompts.push({ at: Date.now(), prompt: options.prompt });
      call++;
      if (call === 1) {
        setTimeout(() => {
          pushedAt = Date.now();
          inbox.push(NOTE);
        }, 500);
        return { stream: generation(options.abortSignal, opening, deltas, tickMs) };
      }
      return { stream: textStep('FOLDED IN') };
    },
  });
  const run = async () => {
    const result = await conversation(name).generateStream({
      messages: ['write a long answer'],
      model: model as never,
      ...inbox.params(),
      ...extra,
    });
    const collected = await collect(result.fullStream);
    return { ...collected, prompts, cutLatencyMs: prompts[1].at - pushedAt, NOTE };
  };
  return run;
};

describe('the step budget — a boundary within the bar on every step phase (FREE_AGENT §M.3 part 2a/2b)', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  afterEach(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  test(
    '(a) THINKING: a note that lands while the first part is still 3 s away restarts the round within 100 ms of landing',
    async () => {
      const NOTE = 'Actually make it about time-series data.';
      const inbox = new Inbox();
      const prompts: Array<{ at: number; prompt: Prompt }> = [];
      let pushedAt = 0;
      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
          prompts.push({ at: Date.now(), prompt: options.prompt });
          call++;
          if (call === 1) {
            // The model is "thinking": the connection is open, the first part is 3 s away. The
            // note lands 200 ms in.
            setTimeout(() => {
              pushedAt = Date.now();
              inbox.push(NOTE);
            }, 200);
            return {
              stream: scheduledStream(options.abortSignal, [
                {
                  afterMs: 3_000,
                  parts: [
                    { type: 'reasoning-start', id: 'r1' },
                    { type: 'reasoning-delta', id: 'r1', delta: 'thought it through' },
                    { type: 'reasoning-end', id: 'r1' },
                    { type: 'text-start', id: 't1' },
                    { type: 'text-delta', id: 't1', delta: 'GENERAL ANSWER' },
                    { type: 'text-end', id: 't1' },
                    { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage },
                  ],
                },
              ]),
            };
          }
          return { stream: textStep('RESHAPED ANSWER') };
        },
      });
      const result = await conversation('round-budget-thinking').generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      const { text } = await collect(result.fullStream);

      expect(prompts).toHaveLength(2);
      expect(text).toBe('RESHAPED ANSWER');
      // The restart was clock-driven: the second call went out within 100 ms of the note landing —
      // not 2.8 s later when the first part would have arrived.
      const restartLatencyMs = prompts[1].at - pushedAt;
      expect(restartLatencyMs).toBeLessThanOrEqual(100);
      const round2 = prompts[1].prompt;
      expect(messageText(round2[round2.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );

  test(
    '(b) MID-TEXT: a note during a 4 s generation of sentences (no paragraph breaks) cuts the round at the first SENTENCE END past N + 2 s — never at the deadline, never mid-word — commits the text so far, and continues from it with the note spliced',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      // A sentence every four words (every 200 ms); the deadline (2.8 s) falls inside one.
      const words = sentenceWords(80, 4);
      const run = midTextTurn('round-budget-cut', 'A long answer:', words);
      const { text, parts, prompts, cutLatencyMs, NOTE } = await run();

      expect(prompts).toHaveLength(2);
      // The cut landed past the deadline (N + 2 s = 2.3 s after the note) at the next sentence end
      // — within one sentence (200 ms) of it — not at the deadline itself and not after the 4 s
      // generation.
      expect(cutLatencyMs).toBeGreaterThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS);
      expect(cutLatencyMs).toBeLessThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS + 200 + 250);
      // The text so far was committed as a finished step — the joiner inside it, then a
      // step-finish — and it ends on a complete sentence: the period, then the paragraph joiner.
      const { text: textBeforeCut, finish } = textBeforeFirstFinish(parts);
      expect(finish.finishReason).toBe('stop');
      expect(textBeforeCut.startsWith('A long answer: word0')).toBe(true);
      expect(textBeforeCut).toMatch(/word\d+\.\n\n$/);
      expect(textBeforeCut.length).toBeLessThan('A long answer:'.length + words.join('').length);
      expect(text).toBe(`${textBeforeCut}FOLDED IN`);
      // The continuation ran on the transcript plus the text so far — ending on that sentence, the
      // delta's tail past it dropped for the model to re-say — with the note spliced last.
      const round2 = prompts[1].prompt;
      const assistant = round2.filter((m) => m.role === 'assistant');
      expect(assistant).toHaveLength(1);
      expect(messageText(assistant[0] as never)).toBe(textBeforeCut.trimEnd());
      expect(messageText(assistant[0] as never)).toMatch(/word\d+\.$/);
      expect(messageText(round2[round2.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );

  test(
    '(b) MID-TEXT: with paragraph breaks in the stream, the cut lands at the first break past N — not at the deadline',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      const prompts: Array<{ at: number; prompt: Prompt }> = [];
      let pushedAt = 0;
      let call = 0;
      // A paragraph every 400 ms for 4 s.
      const paragraphs = Array.from({ length: 10 }, (_, i) => `Paragraph ${i}.\n\n`);
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
          prompts.push({ at: Date.now(), prompt: options.prompt });
          call++;
          if (call === 1) {
            // The note lands after paragraph 0 has streamed (text shown → no restart; the cut clock).
            setTimeout(() => {
              pushedAt = Date.now();
              inbox.push(NOTE);
            }, 500);
            return {
              stream: scheduledStream(options.abortSignal, [
                { afterMs: 0, parts: [{ type: 'text-start', id: 't1' }] },
                ...paragraphs.map((paragraph) => ({
                  afterMs: 400,
                  parts: [{ type: 'text-delta', id: 't1', delta: paragraph }],
                })),
                {
                  afterMs: 0,
                  parts: [
                    { type: 'text-end', id: 't1' },
                    { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage },
                  ],
                },
              ]),
            };
          }
          return { stream: textStep('FOLDED IN') };
        },
      });
      const result = await conversation('round-budget-cut-paragraph').generateStream({
        messages: ['write a long answer'],
        model: model as never,
        ...inbox.params(),
      });
      const { text } = await collect(result.fullStream);

      expect(prompts).toHaveLength(2);
      // N = 300 ms after the note (at 500 ms) → the first break past 800 ms is paragraph 1's or
      // 2's (t ≈ 800 / 1200 ms): the cut lands there, well before the N + 2 s deadline.
      const cutLatencyMs = prompts[1].at - pushedAt;
      expect(cutLatencyMs).toBeGreaterThanOrEqual(300);
      expect(cutLatencyMs).toBeLessThan(300 + Conversation.TEXT_CUT_GRACE_MS);
      expect(text).toMatch(/^Paragraph 0\.\n\n(Paragraph \d\.\n\n){0,2}FOLDED IN$/);
    },
    TIMEOUT
  );

  test(
    '(b) BOUNDARY: past the deadline a run-on with no sentence end is NOT cut mid-word — the cut waits for the paragraph break 1 s later and lands there',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      // 55 bare words carry the stream past the deadline (2.8 s); the break lands at ~3.8 s; more
      // words follow so the generation outlives the cut.
      const words = [
        ...Array.from({ length: 74 }, (_, i) => ` word${i}`),
        '\n\nNext paragraph',
        ...Array.from({ length: 20 }, (_, i) => ` more${i}`),
      ];
      const run = midTextTurn('round-budget-boundary-paragraph', 'A long answer:', words);
      const { text, parts, prompts, cutLatencyMs } = await run();

      expect(prompts).toHaveLength(2);
      // Not at the deadline (2.3 s after the note): at the break, ~3.3 s after it.
      expect(cutLatencyMs).toBeGreaterThanOrEqual(3_200);
      expect(cutLatencyMs).toBeLessThanOrEqual(3_200 + 400);
      const { text: textBeforeCut } = textBeforeFirstFinish(parts);
      // Every word before the break is on screen, the break closes the cut text, and nothing of
      // the next paragraph rode ahead of the acknowledgment.
      expect(textBeforeCut).toBe(`A long answer:${words.slice(0, 74).join('')}\n\n`);
      expect(text).toBe(`${textBeforeCut}FOLDED IN`);
      const assistant = prompts[1].prompt.filter((m) => m.role === 'assistant');
      expect(messageText(assistant[0] as never)).toBe(textBeforeCut.trimEnd());
    },
    TIMEOUT
  );

  test(
    '(b) BOUNDARY: the deadline strikes with the text already resting on a boundary — the cut fires at the deadline, waiting for no further part',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      // One sentence and its line end at once, then a 5 s silence before the rest.
      const run = midTextTurn('round-budget-boundary-at-rest', 'First sentence.\n', ['Second sentence.'], {}, 5_000);
      const { text, parts, prompts, cutLatencyMs } = await run();

      expect(prompts).toHaveLength(2);
      expect(cutLatencyMs).toBeGreaterThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS);
      expect(cutLatencyMs).toBeLessThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS + 250);
      const { text: textBeforeCut } = textBeforeFirstFinish(parts);
      expect(textBeforeCut).toBe('First sentence.\n\n');
      expect(text).toBe('First sentence.\n\nFOLDED IN');
    },
    TIMEOUT
  );

  test(
    '(b) BOUNDARY: a stream with no boundary at all inside the window is cut at the first WORD boundary past it — never mid-word',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      // 200 bare words, 10 s: no sentence end, no line end, no paragraph; the window closes at
      // N + 2 s + CUT_BOUNDARY_WAIT_MS = 8.3 s after the note.
      const words = Array.from({ length: 200 }, (_, i) => ` word${i}`);
      const run = midTextTurn('round-budget-boundary-word', 'A long answer:', words);
      const { text, parts, prompts, cutLatencyMs } = await run();

      const windowMs = 300 + Conversation.TEXT_CUT_GRACE_MS + Conversation.CUT_BOUNDARY_WAIT_MS;
      expect(prompts).toHaveLength(2);
      expect(cutLatencyMs).toBeGreaterThanOrEqual(windowMs);
      expect(cutLatencyMs).toBeLessThanOrEqual(windowMs + 300);
      const { text: textBeforeCut } = textBeforeFirstFinish(parts);
      // The cut text ends on a whole word, then the joiner; the word count says the cut came
      // past the window (~166 words at 50 ms), not at the deadline (~55).
      expect(textBeforeCut).toMatch(/ word\d+\n\n$/);
      const shown = textBeforeCut.match(/ word\d+/g)!.length;
      expect(shown).toBeGreaterThanOrEqual(160);
      expect(text).toBe(`${textBeforeCut}FOLDED IN`);
      const assistant = prompts[1].prompt.filter((m) => m.role === 'assistant');
      expect(messageText(assistant[0] as never)).toBe(textBeforeCut.trimEnd());
    },
    TIMEOUT
  );

  test(
    '(b) BOUNDARY: a code fence spanning the deadline is never cut inside — the cut lands after the fence closes, with every line of code on screen',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      // The fence opens at once; 70 lines (3.5 s) carry it past the deadline (2.8 s) — each a line
      // end, each a sentence-shaped statement — then it closes; prose follows.
      const lines = Array.from({ length: 70 }, (_, i) => `const value${i} = compute(${i}). done;\n`);
      const words = [...lines, '```\n', '\nThe code above', ' does the work.', ' More prose follows.'];
      const run = midTextTurn('round-budget-boundary-fence', 'Here is the code:\n\n```ts\n', words);
      const { text, parts, prompts, cutLatencyMs } = await run();

      expect(prompts).toHaveLength(2);
      // At the close (~3.55 s after the start, 3.05 s after the note) — not at the deadline (2.3 s).
      expect(cutLatencyMs).toBeGreaterThanOrEqual(3_000);
      expect(cutLatencyMs).toBeLessThanOrEqual(3_000 + 400);
      const { text: textBeforeCut } = textBeforeFirstFinish(parts);
      expect(textBeforeCut).toBe(`Here is the code:\n\n\`\`\`ts\n${lines.join('')}\`\`\`\n\n`);
      expect(text).toBe(`${textBeforeCut}FOLDED IN`);
      const assistant = prompts[1].prompt.filter((m) => m.role === 'assistant');
      expect(messageText(assistant[0] as never)).toBe(textBeforeCut.trimEnd());
    },
    TIMEOUT
  );

  test(
    '(b) a note during text that finishes within N never cuts — the generation ends on its own, then the exit absorption continues it',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '2000';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      const prompts: Prompt[] = [];
      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
          prompts.push(options.prompt);
          call++;
          if (call === 1) {
            setTimeout(() => inbox.push(NOTE), 100);
            return {
              stream: scheduledStream(options.abortSignal, [
                {
                  afterMs: 0,
                  parts: [
                    { type: 'text-start', id: 't1' },
                    { type: 'text-delta', id: 't1', delta: 'FIRST' },
                  ],
                },
                { afterMs: 300, parts: [{ type: 'text-delta', id: 't1', delta: ' ANSWER' }] },
                {
                  afterMs: 0,
                  parts: [
                    { type: 'text-end', id: 't1' },
                    { type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage },
                  ],
                },
              ]),
            };
          }
          return { stream: textStep('FOLDED IN') };
        },
      });
      const result = await conversation('round-budget-no-cut').generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        ...inbox.params(),
      });
      const { text, parts } = await collect(result.fullStream);
      expect(prompts).toHaveLength(2);
      expect(text).toBe('FIRST ANSWER\n\nFOLDED IN');
      // One step-finish per finished round; nothing was aborted.
      expect(parts.filter((part) => part.type === 'step-finish')).toHaveLength(2);
      await sleep(10);
    },
    TIMEOUT
  );
});

/**
 * THE INTERJECTION SHAPE (plans/FREE_AGENT.md §M.16): `GenerateStreamParams.interjection` names what
 * the loop does with a note that lands mid-text — `'cut-and-continue'` (part 2b, the loop's own shape,
 * absent = this) or `'after-generation'` (the shape before part 2b: the generation runs to its end, the
 * exit absorption continues the same response with the note; the acknowledgment rides there). The
 * host's one owner (thought-server `FreeAgent.interjection()`) reads it from the mid-text-cut kill
 * switch's row so the cut can be retired on prod without a deploy. RED at 6.4.0: the option did not
 * exist — the loop cut.
 */
describe('the interjection shape — after-generation takes the note when the generation ends (FREE_AGENT §M.16)', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  afterEach(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  /** The (b) shape: a 4 s generation of sentences (no paragraph breaks), the note 500 ms in, N = 300 ms. */
  const longGeneration = (inbox: Inbox, note: string, onPush: () => void) => {
    const prompts: Array<{ at: number; prompt: Prompt }> = [];
    const words = sentenceWords(80, 4);
    let call = 0;
    const model = new MockLanguageModelV3({
      doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
        if (Utterance.isRequest(options.prompt)) {
          prompts.push({ at: Date.now(), prompt: options.prompt });
          return { stream: textStep('Taking that in.') };
        }
        prompts.push({ at: Date.now(), prompt: options.prompt });
        call++;
        if (call === 1) {
          setTimeout(() => {
            onPush();
            inbox.push(note);
          }, 500);
          return { stream: generation(options.abortSignal, 'A long answer:', words) };
        }
        return { stream: textStep('FOLDED IN') };
      },
    });
    return { model, prompts, fullText: `A long answer:${words.join('')}` };
  };

  test(
    "'after-generation': the same 4 s generation with the note at 500 ms is NOT cut — every word streams, then the exit absorption continues the same response with the note spliced",
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      let pushedAt = 0;
      const { model, prompts, fullText } = longGeneration(inbox, NOTE, () => {
        pushedAt = Date.now();
      });
      const result = await conversation('interjection-after-generation').generateStream({
        messages: ['write a long answer'],
        model: model as never,
        ...inbox.params(),
        interjection: 'after-generation',
      });
      const { text, parts } = await collect(result.fullStream);

      expect(prompts).toHaveLength(2);
      // The continuation's call went out when the generation ENDED (~3.5 s after the note) — not at
      // the first sentence end past N + 2 s (~2.4 s), where the cut would have fired.
      const continuationLatencyMs = prompts[1].at - pushedAt;
      expect(continuationLatencyMs).toBeGreaterThanOrEqual(3_000);
      // Nothing was lost or aborted: the whole generation, the joiner, the continuation; one
      // step-finish per finished round.
      expect(text).toBe(`${fullText}\n\nFOLDED IN`);
      expect(parts.filter((part) => part.type === 'step-finish')).toHaveLength(2);
      // The continuation ran on the transcript plus the FULL text, with the note spliced last.
      const round2 = prompts[1].prompt;
      const assistant = round2.filter((m) => m.role === 'assistant');
      expect(assistant).toHaveLength(1);
      expect(messageText(assistant[0] as never)).toBe(fullText);
      expect(messageText(round2[round2.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );

  test(
    "'after-generation' under the bounded utterance: the acknowledgment still rides — asked once the generation ended, streamed as its own step between the finished text and the continuation",
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      let pushedAt = 0;
      const { model, prompts, fullText } = longGeneration(inbox, NOTE, () => {
        pushedAt = Date.now();
      });
      const result = await conversation('interjection-after-generation-utterance').generateStream({
        messages: ['write a long answer'],
        model: model as never,
        ...inbox.params(),
        utterance: true,
        interjection: 'after-generation',
      });
      const { text, parts } = await collect(result.fullStream);

      // The calls: the take-in line (the idle path), the generation, the note's line, the continuation.
      expect(prompts.map((p) => (Utterance.isRequest(p.prompt as never) ? 'utterance' : 'step'))).toEqual([
        'utterance',
        'step',
        'utterance',
        'step',
      ]);
      expect(prompts[2].at - pushedAt).toBeGreaterThanOrEqual(3_000);
      // On the stream: the take-in line (its own step; the consumer adds the ack's joiner), the whole
      // generation with the joiner INSIDE its finished step, the note's line as its own flagged step, the
      // continuation.
      expect(text).toBe(`Taking that in.${fullText}\n\nTaking that in.FOLDED IN`);
      const finishes = parts.map((part, index) => ({ part, index })).filter(({ part }) => part.type === 'step-finish');
      expect(finishes.map(({ part }) => !!part.utterance)).toEqual([true, false, true, false]);
      const generationEnd = finishes[1].index;
      const ackEnd = finishes[2].index;
      const between = parts
        .slice(generationEnd + 1, ackEnd)
        .filter((part) => part.type === 'text-delta')
        .map((part) => part.textDelta)
        .join('');
      expect(between).toBe('Taking that in.');
      // The continuation's prompt: … the full text → the note → the line as the agent's own → the continue framing.
      const round = prompts[3].prompt;
      const texts = round.map((m) => `${m.role}: ${messageText(m as never)}`);
      const noteAt = texts.findIndex((t) => t.startsWith('user:') && t.includes(NOTE));
      expect(noteAt).toBeGreaterThan(0);
      expect(texts[noteAt - 1]).toBe(`assistant: ${fullText}`);
      expect(texts[noteAt + 1]).toBe('assistant: Taking that in.');
      expect(texts[noteAt + 2].startsWith('user:')).toBe(true);
    },
    TIMEOUT
  );

  test(
    "'cut-and-continue', named: the cut fires exactly as when the option is absent — at the first sentence end past the deadline",
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      let pushedAt = 0;
      const { model, prompts, fullText } = longGeneration(inbox, NOTE, () => {
        pushedAt = Date.now();
      });
      const result = await conversation('interjection-cut-named').generateStream({
        messages: ['write a long answer'],
        model: model as never,
        ...inbox.params(),
        interjection: 'cut-and-continue',
      });
      const { text, parts } = await collect(result.fullStream);
      expect(prompts).toHaveLength(2);
      const cutLatencyMs = prompts[1].at - pushedAt;
      expect(cutLatencyMs).toBeGreaterThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS);
      expect(cutLatencyMs).toBeLessThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS + 200 + 250);
      expect(text.endsWith('FOLDED IN')).toBe(true);
      expect(text.length).toBeLessThan(fullText.length);
      expect(textBeforeFirstFinish(parts).text).toMatch(/word\d+\.\n\n$/);
    },
    TIMEOUT
  );
});

/**
 * FOUND ON THE WAY (plans/FREE_AGENT.md §M.16 found (2)): the step that runs a DEFERRED server tool
 * (the API stopped at a client tool batched with it; the search runs at the start of this request)
 * runs on a transcript still OPEN on that tool. A cut in that step would continue from that
 * transcript plus the text so far plus the note — an assistant text block and a user block behind
 * the unresolved server call, which Anthropic refuses (the R7 finding-9 400) — and would drop the
 * search's result, which only this step's response carries. No cut there: the note waits for the
 * generation's end and rides the exit absorption, whose transcript carries the result.
 */
describe('no cut while the step is open on a server tool (FREE_AGENT §M.16 found (2))', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  afterEach(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  /** After an assistant message open on a server tool, a request may carry only tool results. */
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

  test(
    'a note mid-text in the step that runs the deferred search does not cut the round — every paragraph streams, the continuation follows the generation with the note last, and no request carries a block behind the open search',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also cover robotics.';
      const inbox = new Inbox();
      const prompts: Prompt[] = [];
      let call = 0;
      const paragraphs = Array.from({ length: 8 }, (_, i) => `Paragraph ${i}.\n\n`);
      const withTool = new Conversation({
        modelData: fixtureModelData,
        name: 'round-budget-open-server-tool',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [
          {
            getId: () => 'do-work',
            getName: () => 'DoWork',
            getSystemMessages: () => [],
            getMessageModerators: () => [],
            getFunctions: () => [
              {
                definition: { name: 'doWork', description: 'work', parameters: { type: 'object', properties: {} } },
                call: async () => ({ ok: true }),
              },
            ],
          } as never,
        ],
      });
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
          prompts.push(options.prompt);
          call++;
          if (call === 1) {
            // web_search (server, deferred) batched with doWork (client): the message stays open.
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
                { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'doWork', input: '{}' },
                { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
              ]),
            };
          }
          if (call === 2) {
            // The step that runs the deferred search: its result first, then a paragraph every
            // 400 ms for 3.2 s; the note lands 500 ms into the text.
            setTimeout(() => inbox.push(NOTE), 550);
            return {
              stream: scheduledStream(options.abortSignal, [
                {
                  afterMs: 0,
                  parts: [
                    {
                      type: 'tool-result',
                      toolCallId: 'srv-1',
                      toolName: 'web_search',
                      result: [{ url: 'https://example.com', title: 'Example' }],
                    },
                    { type: 'text-start', id: 't1' },
                  ],
                },
                ...paragraphs.map((paragraph) => ({
                  afterMs: 400,
                  parts: [{ type: 'text-delta', id: 't1', delta: paragraph }],
                })),
                {
                  afterMs: 0,
                  parts: [
                    { type: 'text-end', id: 't1' },
                    { type: 'finish', finishReason: { unified: 'stop', raw: 'end_turn' }, usage },
                  ],
                },
              ]),
            };
          }
          return { stream: textStep('ROBOTICS') };
        },
      });
      const result = await withTool.generateStream({
        messages: ['research the frontier models'],
        model: model as never,
        ...inbox.params(),
      });
      const { text } = await collect(result.fullStream);

      for (const prompt of prompts) {
        expectNoBlockBeyondToolResultsAfterOpenServerTool(prompt);
      }
      // Every paragraph streamed — the round was never cut — and the note rode the exit absorption.
      expect(text).toBe(`${paragraphs.join('')}ROBOTICS`);
      expect(prompts).toHaveLength(3);
      const continuation = prompts[2];
      expect(messageText(continuation[continuation.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );
});
