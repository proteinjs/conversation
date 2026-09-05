import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
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
 *      next paragraph break (at N + 2 s regardless), the text so far COMMITS exactly as a finished
 *      round's (the joiner, then a step-finish), and the SAME response continues from that text
 *      with the note spliced. RED at `absorbExitNotes` waiting for the generation's end (the
 *      harness's mid-text row read the generation's length).
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
    '(b) MID-TEXT: a note during a 4 s generation (no paragraph breaks) cuts the round at N + 2 s, commits the text so far, and continues from it with the note spliced',
    async () => {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '300';
      const NOTE = 'Also mention costs.';
      const inbox = new Inbox();
      const prompts: Array<{ at: number; prompt: Prompt }> = [];
      let pushedAt = 0;
      let call = 0;
      const words = Array.from({ length: 80 }, (_, i) => ` word${i}`);
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Prompt; abortSignal?: AbortSignal }) => {
          prompts.push({ at: Date.now(), prompt: options.prompt });
          call++;
          if (call === 1) {
            setTimeout(() => {
              pushedAt = Date.now();
              inbox.push(NOTE);
            }, 500);
            return {
              stream: scheduledStream(options.abortSignal, [
                {
                  afterMs: 0,
                  parts: [
                    { type: 'text-start', id: 't1' },
                    { type: 'text-delta', id: 't1', delta: 'A long answer:' },
                  ],
                },
                ...words.map((word) => ({ afterMs: 50, parts: [{ type: 'text-delta', id: 't1', delta: word }] })),
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
      const result = await conversation('round-budget-cut').generateStream({
        messages: ['write a long answer'],
        model: model as never,
        ...inbox.params(),
      });
      const { text, parts } = await collect(result.fullStream);

      expect(prompts).toHaveLength(2);
      // The cut landed past N and no later than N + 2 s after the note (the deadline, no paragraph
      // break to cut at) — the continuation's call went out then, not after the 4 s generation.
      const cutLatencyMs = prompts[1].at - pushedAt;
      expect(cutLatencyMs).toBeGreaterThanOrEqual(300);
      expect(cutLatencyMs).toBeLessThanOrEqual(300 + Conversation.TEXT_CUT_GRACE_MS + 250);
      // The text so far was committed as a finished step — the joiner inside it, then a
      // step-finish — and the continuation follows.
      const firstFinish = parts.findIndex((part) => part.type === 'step-finish');
      expect(firstFinish).toBeGreaterThan(0);
      expect(parts[firstFinish].finishReason).toBe('stop');
      const textBeforeCut = parts
        .slice(0, firstFinish)
        .filter((part) => part.type === 'text-delta')
        .map((part) => part.textDelta)
        .join('');
      expect(textBeforeCut.startsWith('A long answer: word0')).toBe(true);
      expect(textBeforeCut.endsWith('\n\n')).toBe(true);
      expect(textBeforeCut.length).toBeLessThan('A long answer:'.length + words.join('').length);
      expect(text).toBe(`${textBeforeCut}FOLDED IN`);
      // The continuation ran on the transcript plus the text so far, with the note spliced last.
      const round2 = prompts[1].prompt;
      const assistant = round2.filter((m) => m.role === 'assistant');
      expect(assistant).toHaveLength(1);
      expect(messageText(assistant[0] as never)).toBe(textBeforeCut.trimEnd());
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
