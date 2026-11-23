import type { ModelMessage } from 'ai';

/**
 * THE SIDE UTTERANCE'S WORDS (plans/FREE_AGENT.md §M.3 part 5 — the nudge's door): when the host
 * wants ONE LINE said mid-round without touching the round — the harness asking the mind what it
 * is doing after a long silence — the loop asks for it in a separate call on the same model over
 * the round's transcript and the step's tool roster (the step's cache prefix; no tool runs on the
 * call), with the input appended and the input's OWN ask. The line lands as one `side-utterance`
 * part, and that is all it does.
 *
 * WHAT IS SAID ASIDE STAYS ASIDE. The round's transcript never carries the ask or the line: the
 * steps that follow are sent exactly what they would have been sent had nothing been asked, so the
 * reply is the round's own words, whole, whatever the side call wrote. The line once rode into the
 * next step as the agent's own prior message under "that line is on their screen as part of this
 * response. Do not repeat it" — true of the acknowledgment it was written for (retired below),
 * false of a line a host shows as a passing status, and the way a reply lost its beginning: asked
 * what it was doing, a model wrote its answer's first section; the section was taken as the line,
 * and the round, told the section was already in the response, began the reply at its second
 * (prod, 2026-11-22).
 *
 * One owner for the words: the ask's marker, the request shape and what a line is live here; the loop
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
 * what a side utterance is asked for): its text — spliced into the next step as a user message when
 * a drain hands it over; read by the side call alone when it is a side utterance's — and, for a
 * side utterance, the instruction the one line is asked with (`ask`: the nudge's "what are you
 * doing right now", plans/FREE_AGENT.md §3.4). Every ask ends with
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
   * The most characters a line may have ({@link Utterance.isLine}). Chosen from what real models
   * wrote when asked for the line (2026-11-23: Gemini 4 Hydrogen, Claude Opus 5, Claude Fable 5.1,
   * GPT-5.6 Sol and GPT-5.5 Pro, under three wordings of the ask): the sentences about what was
   * being done ran 26 to 123 characters; the paragraphs of an answer written in the line's place
   * ran 323 to 349. 140 clears every sentence seen and is under half the shortest of those
   * paragraphs. A line over it is not shown and not cut down: the length is guidance in the ask,
   * never a cut.
   */
  static readonly MAX_LINE_CHARS = 140;

  /**
   * The most output tokens the side call's own request allows (`Conversation.utter`) — a ceiling
   * on what a model that begins its answer in the line's place can spend there, never what shapes
   * a line: a call the limit ends has no line at all. A provider counts a model's thinking against
   * the same limit as its words (seen on Google and on OpenAI), and the least thinking some models
   * offer is not none, so the ceiling cannot sit at a line's own size: the largest real call seen
   * was 342 tokens, 331 of them a model's thinking before a line of 33 characters. 1024 is three
   * times that.
   */
  static readonly MAX_CALL_OUTPUT_TOKENS = 1024;

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

  /**
   * Whether the model's first paragraph IS a line — the ONE rule for what is said aside. A line is
   * a short sentence:
   *  - words on one line. A paragraph that runs over several lines — a heading with a list under
   *    it, a table, a code fence — is a block: the model began something else in the line's place
   *    (the reply itself, a plan);
   *  - at most {@link Utterance.MAX_LINE_CHARS} characters. A paragraph of prose is one line too,
   *    and it is no sentence about what the mind is doing;
   *  - not opening as the answer's own structure: a heading, a list item, a table row, a quote, or
   *    a line that is nothing but bold (a heading in all but name).
   * Whole or nothing: what is not a line is never said aside and never cut down to fit (a block's
   * first line would be its heading, a long paragraph's first words its opening — shown as if they
   * were what the mind is doing, and again where the reply begins) — the call has no line.
   */
  static isLine(paragraph: string): boolean {
    const line = paragraph.trim();
    return !/[\r\n]/.test(line) && line.length <= Utterance.MAX_LINE_CHARS && !Utterance.opensAsAnswerStructure(line);
  }

  /** Inputs as {@link DrainedInput}s — a bare string is an input with no ask of its own. */
  static inputs(inputs: Array<string | DrainedInput>): DrainedInput[] {
    return inputs.map((input) => (typeof input === 'string' ? { text: input } : input));
  }

  /**
   * Whether one line opens the way a part of an answer does, in the markdown models write: a
   * heading (`# `), a list item (`- `, `* `, `+ `, `1. `, `1) `), a table row (`|`), a quote (`>`),
   * or one bold span and nothing else (`**…**`, `__…__`). Emphasis inside a sentence is not
   * structure, and neither is a sentence set in italics.
   */
  private static opensAsAnswerStructure(line: string): boolean {
    if (/^(?:#{1,6}(?:\s|$)|(?:[-*+]|\d{1,9}[.)])\s|\||>)/.test(line)) {
      return true;
    }
    const bold = /^(\*\*|__)(.+)\1$/.exec(line);
    return !!bold && !bold[2].includes(bold[1]);
  }
}
