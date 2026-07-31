import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';

/**
 * `GenerateStreamParams.peekInjectedContext` — the thinking-phase restart. A note that arrives
 * while the model is still reasoning (nothing user-material streamed yet) aborts the in-flight
 * round and re-plans with the note appended, so the turn produces ONE reshaped answer instead of
 * the composed answer plus the note's answer appended. Once text has streamed, restarts stop and
 * the note rides the exit-note absorption instead.
 *
 * No network: MockLanguageModelV3 scripts each round; outgoing prompts prove what each round saw.
 */

const TIMEOUT = 30_000;

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

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

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const messageText = (msg: { content: unknown }): string => {
  if (typeof msg.content === 'string') {
    return msg.content;
  }
  if (Array.isArray(msg.content)) {
    return msg.content
      .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? (part.text ?? '') : ''))
      .join('');
  }
  return '';
};

async function collectText(fullStream: AsyncIterable<{ type: string }>): Promise<string> {
  let text = '';
  for await (const part of fullStream) {
    if (part.type === 'text-delta') {
      text += (part as { textDelta?: string }).textDelta ?? '';
    }
  }
  return text;
}

describe('Conversation.generateStream — thinking-phase restart (peekInjectedContext)', () => {
  test(
    'a note arriving during reasoning aborts the round and re-plans with the note — one answer, nothing discarded on screen',
    async () => {
      const NOTE = 'Actually make it about time-series data.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          if (call === 1) {
            // The user's second message lands while round 1 is still thinking (its prepareStep
            // already ran, so the note can only be seen by the peek).
            inbox.push(NOTE);
            return { stream: reasoningThenTextStep('planning the general answer', 'GENERAL ANSWER') };
          }
          return { stream: textStep('RESHAPED ANSWER') };
        },
      });

      const conversation = new Conversation({
        name: 'thinking-restart-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
      });

      const result = await conversation.generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        drainInjectedContext: () => inbox.splice(0, inbox.length),
        peekInjectedContext: () => inbox.length > 0,
        absorbExitNotes: true,
      });
      const text = await collectText(result.fullStream as AsyncIterable<{ type: string }>);

      // Round 1 was aborted during reasoning: its text never streamed; the restarted round's
      // answer is the ONLY answer, with no continuation separator prepended.
      expect(capturedPrompts).toHaveLength(2);
      expect(text).toBe('RESHAPED ANSWER');

      // The restarted round saw the note as a user message appended after the original ask.
      const round2 = capturedPrompts[1];
      const noteMessages = round2.filter((m) => m.role === 'user' && messageText(m).includes(NOTE));
      expect(noteMessages).toHaveLength(1);
      expect(messageText(round2[round2.length - 1] as never)).toContain(NOTE);
    },
    TIMEOUT
  );

  test(
    'a note arriving after text has streamed does NOT restart — it continues via exit-note absorption',
    async () => {
      const NOTE = 'Also mention costs.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          return { stream: call === 1 ? textStep('FIRST ANSWER') : textStep('FOLDED IN') };
        },
      });

      const conversation = new Conversation({
        name: 'thinking-restart-after-text-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
      });

      const result = await conversation.generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        drainInjectedContext: () => inbox.splice(0, inbox.length),
        peekInjectedContext: () => inbox.length > 0,
        absorbExitNotes: true,
      });

      let text = '';
      let notePushed = false;
      for await (const part of result.fullStream as AsyncIterable<{ type: string }>) {
        if (part.type === 'text-delta') {
          text += (part as { textDelta?: string }).textDelta ?? '';
          if (!notePushed && text.includes('FIRST ANSWER')) {
            // The note lands only AFTER round 1's answer has visibly streamed.
            notePushed = true;
            inbox.push(NOTE);
          }
        }
      }

      // No restart (the streamed answer stayed); the note continued the same response as a
      // separator-joined continuation round.
      expect(capturedPrompts).toHaveLength(2);
      expect(text).toBe('FIRST ANSWER\n\nFOLDED IN');
      const round2 = capturedPrompts[1];
      expect(round2.filter((m) => m.role === 'user' && messageText(m).includes(NOTE))).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    "a round cut off by the output limit ('length') auto-continues seamlessly — no separator, bounded (2026-07-29: truncated answers are never acceptable)",
    async () => {
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];
      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          if (call === 1) {
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                { type: 'text-start' as const, id: 't1' },
                { type: 'text-delta' as const, id: 't1', delta: 'The answer begins' },
                { type: 'text-end' as const, id: 't1' },
                { type: 'finish' as const, finishReason: { unified: 'length' as const, raw: 'max_tokens' }, usage },
              ]),
            };
          }
          return { stream: textStep(' and here it ends.') };
        },
      });

      const conversation = new Conversation({
        name: 'length-auto-continue-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
      });

      const result = await conversation.generateStream({
        messages: ['write something long'],
        model: model as never,
      });
      const text = await collectText(result.fullStream as AsyncIterable<{ type: string }>);

      // Seamless: the continuation flows on from the cut — NO paragraph separator injected.
      expect(capturedPrompts).toHaveLength(2);
      expect(text).toBe('The answer begins and here it ends.');
      const round2 = capturedPrompts[1];
      // The continuation call carries the partial assistant text and the continue instruction.
      expect(round2.some((m) => m.role === 'assistant' && messageText(m).includes('The answer begins'))).toBe(true);
      expect(messageText(round2[round2.length - 1] as never)).toContain('cut off by the output limit');
    },
    TIMEOUT
  );

  test(
    'a note during a CONTINUATION round does NOT restart once the turn has streamed text (2026-07-29 lost-response class)',
    async () => {
      // The lost-response interleaving: round 1 streams the visible answer; note 1 rides
      // exit-note absorption into continuation round 2; note 2 lands while round 2 is still
      // REASONING. Restart eligibility was round-scoped, so round 2 restarted AFTER the answer
      // had streamed — and the restarted round's tool use made the saver file the whole
      // streamed answer as timeline reasoning with content=''. Eligibility is turn-scoped now:
      // once ANY round streamed text, notes only ever continue, never restart.
      const NOTE_1 = 'Also mention costs.';
      const NOTE_2 = 'And name a vendor.';
      const inbox: string[] = [];
      const capturedPrompts: Array<Array<{ role: string; content: unknown }>> = [];

      let call = 0;
      const model = new MockLanguageModelV3({
        doStream: async (options: { prompt: Array<{ role: string; content: unknown }> }) => {
          capturedPrompts.push(options.prompt);
          call++;
          if (call === 2) {
            // Note 2 lands before round 2's first part reaches the restart check — the exact
            // window where round-scoped eligibility used to abort the continuation.
            inbox.push(NOTE_2);
            return { stream: reasoningThenTextStep('re-thinking with the note', 'ADDENDUM') };
          }
          return { stream: call === 1 ? textStep('MAIN ANSWER') : textStep('SECOND ADDENDUM') };
        },
      });

      const conversation = new Conversation({
        name: 'continuation-no-restart-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
      });

      const result = await conversation.generateStream({
        messages: ['compare sql and nosql'],
        model: model as never,
        drainInjectedContext: () => inbox.splice(0, inbox.length),
        peekInjectedContext: () => inbox.length > 0,
        absorbExitNotes: true,
      });

      let text = '';
      let note1Pushed = false;
      for await (const part of result.fullStream as AsyncIterable<{ type: string }>) {
        if (part.type === 'text-delta') {
          text += (part as { textDelta?: string }).textDelta ?? '';
          if (!note1Pushed && text.includes('MAIN ANSWER')) {
            note1Pushed = true;
            inbox.push(NOTE_1);
          }
        }
      }

      // Round 2 was NOT aborted: its answer streamed in full, and note 2 continued into round 3.
      // (Pre-fix, round 2 restarted and 'ADDENDUM' never streamed.)
      expect(capturedPrompts).toHaveLength(3);
      expect(text).toBe('MAIN ANSWER\n\nADDENDUM\n\nSECOND ADDENDUM');
      const round2 = capturedPrompts[1];
      expect(round2.filter((m) => m.role === 'user' && messageText(m).includes(NOTE_1))).toHaveLength(1);
      const round3 = capturedPrompts[2];
      expect(round3.filter((m) => m.role === 'user' && messageText(m).includes(NOTE_2))).toHaveLength(1);
    },
    TIMEOUT
  );
});
