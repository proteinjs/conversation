import type { ModelMessage } from 'ai';

/**
 * THE SIDE UTTERANCE'S WORDS (plans/FREE_AGENT.md §M.3 part 5 — the nudge's door): when the host
 * wants ONE LINE said mid-round without touching the round — the harness asking the mind what it
 * is doing after a long silence — the loop asks for it in a separate call on the same model over
 * the round's transcript and the step's tool roster (the step's cache prefix; no tool runs on the
 * call), with the input appended and the input's OWN ask. The line lands as one `side-utterance`
 * part, and the next step runs with the input spliced and the line riding as the agent's own prior
 * message followed by the continue framing — roles alternate, so it works with thinking on (a
 * prefill would not), and the mind never says the line twice.
 *
 * One owner for the words: the ask's marker, the request shape and the framing live here; the loop
 * (`Conversation.generateStream`) owns when a side utterance runs (`GenerateStreamParams.sideUtterance`).
 *
 * The BOUNDED utterance (part 2c — a separate no-tools call for a one-line acknowledgment before
 * every step that took an input in, at turn start and at every drain) was retired 2026-11-05
 * (DECISIONS 1099): the screen shows the thinking row and the voice reader waits for the settled
 * message, so a line within 3 s bought nothing, and the call's one-sentence ask was where a
 * planning paragraph landed above the answer (row 3105). The main step's own pre-tool words are the
 * acknowledgment now, committed by the consumer's window (thought-server's `TurnAckWindow`).
 */
/**
 * An input the mind takes in (what `GenerateStreamParams.drainInjectedContext` hands the loop, and
 * what a side utterance is asked for): its text, spliced into the next step as a user message, and
 * — for a side utterance — the instruction the one line is asked with (`ask`: the nudge's "what are
 * you doing right now", plans/FREE_AGENT.md §3.4). Every ask ends with
 * {@link Utterance.REPLY_WITH_THE_LINE_ONLY} — the marker {@link Utterance.isRequest} reads. An
 * input a drain hands the loop carries no ask: it is spliced, never uttered.
 */
export type DrainedInput = { text: string; ask?: string };

export class Utterance {
  /** The sentence every side ask ends with — the request marker scripted models key on. */
  static readonly REPLY_WITH_THE_LINE_ONLY = 'Reply with the line only.';

  /** The break that ends the line: the model's own paragraph break. */
  static readonly PARAGRAPH_BREAK = '\n\n';

  /**
   * The request: the transcript as the next step will see it, plus the input being asked about and
   * its ask — one user message. A side utterance is always asked FOR something: an empty input set
   * is a programming error, never a request.
   */
  static request(transcript: ModelMessage[], inputs: Array<string | DrainedInput>): ModelMessage[] {
    const drained = Utterance.inputs(inputs);
    if (drained.length === 0) {
      throw new Error('A side utterance is asked for an input — none was given');
    }
    const instruction = Utterance.askFor(drained);
    return [
      ...transcript,
      {
        role: 'user',
        content: `${drained.map((input) => input.text).join('\n\n')}\n\n${instruction}`,
      } as ModelMessage,
    ];
  }

  /**
   * The framing the next step runs under once the line is said: the inputs as user messages, the
   * line as the agent's own prior message, then the continue instruction — so the step's prompt
   * ends on a user turn (never a prefill) and the mind knows the line is already on the user's
   * screen.
   */
  static framing(inputs: Array<string | DrainedInput>, line: string): ModelMessage[] {
    return [
      ...Utterance.inputs(inputs).map((input) => ({ role: 'user', content: input.text }) as ModelMessage),
      { role: 'assistant', content: line } as ModelMessage,
      { role: 'user', content: Utterance.continueFraming(line) } as ModelMessage,
    ];
  }

  /** The continue framing's text — what the mind reads after its own line. */
  static continueFraming(line: string): string {
    return (
      `You have already told the user: “${line}” — that line is on their screen as part of this ` +
      `response. Do not repeat it and do not acknowledge again; continue the work from here.`
    );
  }

  /**
   * Whether a provider prompt is a side utterance's request (its last user text ends with the
   * request marker every ask carries, {@link Utterance.REPLY_WITH_THE_LINE_ONLY}) — for scripted
   * models in tests, which answer the side ask with a line and the main step with their script.
   */
  static isRequest(prompt: ReadonlyArray<{ role: string; content: unknown }>): boolean {
    const last = prompt[prompt.length - 1];
    if (!last || last.role !== 'user') {
      return false;
    }
    const text =
      typeof last.content === 'string'
        ? last.content
        : Array.isArray(last.content)
          ? last.content
              .map((part: { type?: string; text?: string }) => (part?.type === 'text' ? part.text ?? '' : ''))
              .join('')
          : '';
    return text.trimEnd().endsWith(Utterance.REPLY_WITH_THE_LINE_ONLY);
  }

  /**
   * The ask a set of inputs is uttered with: the LAST input that carries its own ask names it (the
   * nudge's "what are you doing right now"), normalized to end with the request marker. An input
   * set with no ask has nothing to be asked — a programming error, never a default instruction.
   */
  static askFor(inputs: Array<string | DrainedInput>): string {
    const own = Utterance.inputs(inputs)
      .reverse()
      .find((input) => typeof input.ask === 'string' && input.ask.trim().length > 0);
    if (!own?.ask) {
      throw new Error('A side utterance is asked with the input’s own ask — none of the inputs carries one');
    }
    const ask = own.ask.trim();
    return ask.endsWith(Utterance.REPLY_WITH_THE_LINE_ONLY) ? ask : `${ask} ${Utterance.REPLY_WITH_THE_LINE_ONLY}`;
  }

  /**
   * Where the line ends inside the model's text: the index of the first paragraph break past any
   * leading whitespace, or −1 while none has arrived. The line is the FIRST paragraph — a model
   * that writes a second one (a caveat, a plan) has written past the ask, and that text is never
   * the line; the call is ended there (`Conversation.utter`).
   */
  static lineEnd(text: string): number {
    const lead = text.length - text.trimStart().length;
    return text.indexOf(Utterance.PARAGRAPH_BREAK, lead);
  }

  /** Inputs as {@link DrainedInput}s — a bare string is an input with no ask of its own. */
  static inputs(inputs: Array<string | DrainedInput>): DrainedInput[] {
    return inputs.map((input) => (typeof input === 'string' ? { text: input } : input));
  }
}
