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
      .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? part.text ?? '' : ''))
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
});
