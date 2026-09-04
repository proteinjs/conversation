import { ChatCompletionFunctionTool } from 'openai/resources/chat';

/**
 * A tool call's timeline subject: display text plus an optional app deep-link href that lets the
 * rendering layer make the timeline detail clickable (e.g. `thought://nav?...` opens the edited
 * document). Plain-string details remain valid — most tools have nothing to link to.
 *
 * `glyph` carries the acted-on entity's TYPE identity as a serializable FontAwesome
 * string-lookup icon plus its canonical hue (e.g. a thought's ThoughtType record icon/color),
 * so rendering layers can show typed identity without importing the producing domain's code.
 */
export type ToolTimelineDetail = {
  text: string;
  href?: string;
  glyph?: { icon: string; style?: 'solid' | 'regular' | 'light'; color?: string };
};

/**
 * A long call's progress, in the tool's own present-progressive words ("Setting up the
 * workspace" — "getting the code (n3xa app)"). Reported through {@link ToolCallContext.onPhase};
 * the harness names the current phase in its yield sentence when the call converts to a
 * background job and moves the job's node in place afterwards (plans/FREE_AGENT.md §2.2 `phases`).
 */
export type ToolPhase = { on: string; detail?: string };

/**
 * What the harness hands a tool beside its arguments (plans/FREE_AGENT.md §M.3 part 1 — the
 * tool-call budget). Present only under a budgeted executor (`GenerateStreamParams.toolBudget`);
 * a tool never depends on it.
 */
export type ToolCallContext = {
  /**
   * The call's OWN abort signal: fired by the background job's Stop (D9) or its HARD budget once
   * the call has converted — never by the turn's yield. A tool that can stop honours it.
   */
  signal: AbortSignal;
  /** Report a phase (see {@link ToolPhase}). Cheap; safe to call before or after a conversion. */
  onPhase: (phase: ToolPhase) => void;
};

export interface Function {
  definition: ChatCompletionFunctionTool['function'];
  call(obj: any, ctx?: ToolCallContext): Promise<any>;
  instructions?: string[];
  /**
   * Optional budget hints (plans/FREE_AGENT.md §2.2, D2 — "declarations are optional and only an
   * optimization"; a missing or wrong one costs nothing, the budget catches it):
   * `background` — the call is ALWAYS long: the executor converts it to a background job at t = 0
   * instead of after the SOFT budget, so the yield is immediate.
   */
  background?: boolean;
  /** The tool's own generous HARD ceiling in ms (D8); the host arms it on the job. Default 30 min. */
  hardBudgetMs?: number;
  /**
   * §2.8 idempotency: `false` = a repeat call IS the intended act (a second createDevelopmentTask
   * with the same request is a second task) — never deduped against a running job.
   */
  dedupe?: boolean;
  /** The repeat-call identity when the arguments' hash is not it (a tool whose repeat is unsafe). */
  dedupeKey?(args: any): string;
  /**
   * Optional: produce a short, human-meaningful subject for a call to this
   * tool — typically the acted-on entity's name/title — to personalize the
   * call's node in the chat thinking timeline. May do a lookup. Best-effort:
   * the framework swallows errors and falls back to a generic detail.
   * Return a `ToolTimelineDetail` to also carry a deep-link href.
   */
  getTimelineDetail?(
    args: any
  ): string | ToolTimelineDetail | undefined | Promise<string | ToolTimelineDetail | undefined>;
  /**
   * Optional: outcome-aware re-labeling of the just-finished call's timeline node. The
   * call-time node is named from the INPUT alone, so a call that ends up applying nothing
   * (e.g. an edit refused by a freshness/lock fence) would still render as done — this hook
   * inspects the RESULT and relabels. Best-effort: the framework swallows errors and keeps
   * the call-time labeling. `name` REPLACES the node's tool name — use the `tool:variant`
   * suffix convention (e.g. `editThoughts:deferred`) so presentation maps can target it;
   * `detail` replaces the node's detail (omit to keep the call-time one). Return undefined
   * (or an empty object) for no relabel.
   *
   * `ok: false` marks the call SETTLED-FAILED for the timeline's status lifecycle: tools that
   * report failure by RETURNING an error message to the model (the LLM-friendly convention —
   * the SDK sees a successful result either way) use it so their node settles `errored`
   * instead of rendering as done, and a follow-up same-tool call is treated as the retry of
   * the same intent. Purely presentational — it never changes what the model receives.
   */
  getTimelineOutcome?(
    args: any,
    result: any
  ):
    | { name?: string; detail?: string | ToolTimelineDetail; ok?: boolean }
    | undefined
    | Promise<{ name?: string; detail?: string | ToolTimelineDetail; ok?: boolean } | undefined>;
}
