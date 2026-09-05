import type { ModelMessage } from 'ai';

/**
 * The BOUNDED UTTERANCE (plans/FREE_AGENT.md §M.3 part 2c — the load-bearing guarantee of the
 * 10-second bar): before the mind takes an input into its next step, the harness asks it for ONE
 * LINE — a separate call on the same model over the same transcript (a prompt-cache hit) with the
 * input appended and this instruction, no tools, thinking off, a small output ceiling. The line is
 * the acknowledgment the user is owed at once, in the agent's own words; its turnaround is a
 * no-thinking completion on a cached prefix (1–2 s), whatever the main step's adaptive thinking
 * decides. The main step then runs with the input spliced and the line riding as the agent's OWN
 * prior utterance followed by the continue framing — roles alternate, so it works with thinking
 * on (a prefill would not), and the mind never acknowledges twice.
 *
 * One owner for the words: the instruction, the framing, and the request shape live here; the
 * loop (`Conversation.generateStream`) owns when the utterance runs — at turn start (the idle
 * path), and at every drain that hands inputs to the model (a step boundary, a thinking-phase
 * restart, a mid-text cut, the exit absorption).
 */
/**
 * An input the mind takes in (what `GenerateStreamParams.drainInjectedContext` hands the loop, and
 * what a side utterance is asked for): its text, spliced into the next step as a user message, and
 * — when the input kind has its own utterance ask — the instruction the one line is asked with in
 * place of the default {@link Utterance.INSTRUCTION}. A user note and a job result ride the
 * default (acknowledge what you are taking in); a nudge (plans/FREE_AGENT.md §3.4) asks for what
 * the mind is doing right now. Every ask ends with {@link Utterance.REPLY_WITH_THE_LINE_ONLY} —
 * the marker {@link Utterance.isRequest} reads.
 */
export type DrainedInput = { text: string; ask?: string };

export class Utterance {
  /** The sentence every utterance ask ends with — the request marker scripted models key on. */
  static readonly REPLY_WITH_THE_LINE_ONLY = 'Reply with the line only.';

  /**
   * The F7 instruction, appended to the input the mind is about to take in. The trailing sentence
   * is the marker {@link Utterance.isRequest} reads — a scripted model can tell the utterance call
   * from the main step by it.
   */
  static readonly INSTRUCTION =
    'Before you continue: acknowledge, in one short line and in your own words, what you are taking ' +
    'in from the message above and that it is handled — nothing about your plan, your tools, or what ' +
    `comes next, and no time estimate. That single line reaches the user right away; you continue ` +
    `the work after it. ${Utterance.REPLY_WITH_THE_LINE_ONLY}`;

  /** The output ceiling of the utterance call — one line, never a paragraph. */
  static readonly MAX_OUTPUT_TOKENS = 80;

  /**
   * The request: the transcript as the next step will see it, plus the inputs being taken in and
   * the instruction — one user message. With no inputs (the idle path: the transcript's own last
   * message is the request), the instruction rides that last user message instead.
   */
  static request(transcript: ModelMessage[], inputs: Array<string | DrainedInput>): ModelMessage[] {
    const drained = Utterance.inputs(inputs);
    if (drained.length > 0) {
      const instruction = Utterance.askFor(drained);
      return [
        ...transcript,
        {
          role: 'user',
          content: `${drained.map((input) => input.text).join('\n\n')}\n\n${instruction}`,
        } as ModelMessage,
      ];
    }
    const last = transcript[transcript.length - 1];
    if (!last || last.role !== 'user') {
      return [...transcript, { role: 'user', content: Utterance.INSTRUCTION } as ModelMessage];
    }
    const withInstruction: ModelMessage =
      typeof last.content === 'string'
        ? ({ ...last, content: `${last.content}\n\n${Utterance.INSTRUCTION}` } as ModelMessage)
        : ({ ...last, content: [...last.content, { type: 'text', text: Utterance.INSTRUCTION }] } as ModelMessage);
    return [...transcript.slice(0, -1), withInstruction];
  }

  /**
   * The framing the main step runs under once the line is said: the inputs as user messages, the
   * line as the agent's own prior message, then the continue instruction — so the step's prompt
   * ends on a user turn (never a prefill) and the mind knows the acknowledgment is already on the
   * user's screen.
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
   * Whether a provider prompt is an utterance request (its last user text ends with the request
   * marker every ask carries, {@link Utterance.REPLY_WITH_THE_LINE_ONLY}) — for scripted models in
   * tests, which answer the utterance with a line and the main step with their script.
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
   * The ask a set of inputs is uttered with: the LAST input that carries its own ask names it (an
   * input kind with its own question — the nudge's "what are you doing right now"), else the
   * default acknowledgment instruction. Normalized to end with the request marker.
   */
  static askFor(inputs: Array<string | DrainedInput>): string {
    const own = Utterance.inputs(inputs)
      .reverse()
      .find((input) => typeof input.ask === 'string' && input.ask.trim().length > 0);
    if (!own?.ask) {
      return Utterance.INSTRUCTION;
    }
    const ask = own.ask.trim();
    return ask.endsWith(Utterance.REPLY_WITH_THE_LINE_ONLY) ? ask : `${ask} ${Utterance.REPLY_WITH_THE_LINE_ONLY}`;
  }

  /** Inputs as {@link DrainedInput}s — a bare string is an input with the default ask. */
  static inputs(inputs: Array<string | DrainedInput>): DrainedInput[] {
    return inputs.map((input) => (typeof input === 'string' ? { text: input } : input));
  }
}
