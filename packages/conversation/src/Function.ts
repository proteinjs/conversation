import { ChatCompletionFunctionTool } from 'openai/resources/chat';

/**
 * The acted-on entity's TYPE identity as its renderer draws it — a FACE: the type's icon carried
 * WHOLE (its name, its style, and every drawing term the producing domain declares on it — a term
 * this package does not know rides through untouched; the index signature is that promise), the
 * type's canonical hue, and the type's id when the producer has one (a renderer never needs it —
 * two faces of one drawing draw alike with or without it). Serializable: it rides the timeline
 * event, so a rendering layer shows typed identity without importing the producing domain's code.
 */
export type ToolTimelineGlyph = {
  id?: string;
  icon: { name: string; style?: 'solid' | 'regular' | 'light'; [term: string]: unknown };
  color?: string;
};

/**
 * The FLAT glyph shape details carried before the face (the icon's NAME on `icon`, its style and hue
 * beside it): admitted by the type so a producer still composing it compiles, and lifted by the
 * rendering layer into a {@link ToolTimelineGlyph} with no drawing terms. This package never reads a
 * glyph — the detail passes through the harness whole — so the admission is the type's alone.
 * Compatibility, dated 2026-10-05: removable once every producer hands the face.
 */
export type ToolTimelineFlatGlyph = { icon: string; style?: 'solid' | 'regular' | 'light'; color?: string };

/**
 * A tool call's timeline subject: display text plus an optional app deep-link href that lets the
 * rendering layer make the timeline detail clickable (e.g. `thought://nav?...` opens the edited
 * document). Plain-string details remain valid — most tools have nothing to link to.
 *
 * `glyph` carries the acted-on entity's TYPE identity as a face ({@link ToolTimelineGlyph}; the
 * flat form {@link ToolTimelineFlatGlyph} is admitted for its dated window).
 */
export type ToolTimelineDetail = {
  text: string;
  href?: string;
  glyph?: ToolTimelineGlyph | ToolTimelineFlatGlyph;
};

/**
 * A long call's progress, in the tool's own present-progressive words ("Setting up the
 * workspace" — "getting the code (n3xa app)"). Reported through {@link ToolCallContext.onPhase};
 * the harness names the phase a kind-declared call reported at its start in its yield sentence and
 * moves the job's node in place afterwards (plans/FREE_AGENT.md §2.2 `phases`).
 */
export type ToolPhase = { on: string; detail?: string };

/**
 * THE KINDS OF WORK THAT RUN ON THEIR OWN (plans/DECISIONS.md 630 — the one-response class rule;
 * plans/AGENT_ON_THE_BOX_DESIGN_2026-10-01.md §B the kinds): a spawned agent, a shell on a box, a
 * line-notifying watcher, an orchestration of agents, a routine. A function whose call IS one of
 * these declares it ({@link Function.job}) and the executor converts the call into a job of that
 * kind at its start; every other call is awaited whole. The chat's row vocabulary (chat-common
 * `ChatJobKind`) holds these and the pre-rule `tool` reading.
 */
export type ToolJobKind = 'agent' | 'shell' | 'watcher' | 'orchestration' | 'routine';

/**
 * What the harness hands a tool beside its arguments (plans/FREE_AGENT.md §M.3 part 1; DECISIONS
 * 630). Present only under a budgeted executor (`GenerateStreamParams.toolBudget`); a tool never
 * depends on it.
 */
export type ToolCallContext = {
  /**
   * The call's OWN abort signal: for a kind-declared call, fired by the job's Stop (D9) or its
   * HARD budget — never by the turn's yield. An awaited call's signal is minted the same way and
   * nothing fires it yet (the per-call Stop on an awaited call's node is the next build; the
   * turn's own Stop is the person's handle meanwhile). A tool that can stop honors it.
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
   * THE ONE JOB DECLARATION (plans/DECISIONS.md 630 — the one-response class rule): this call IS
   * a job of the named kind — work that genuinely runs on its own ({@link ToolJobKind}) — and the
   * executor converts it at its start, the kind riding to the host. A function that declares
   * nothing is AWAITED WHOLE, however long it runs: a deck, a document, a picture, a catalog check
   * lands in the reply the person asked in. No per-tool opt-in exists in either direction: no
   * "always awaited" hint, no "always background" hint — the class is the rule. A tool that mints
   * its own job through the turn's job door and returns at once (a spawnAgent, a StartJob)
   * declares nothing here: its call is short; the job it made is the door's.
   */
  job?: { kind: ToolJobKind };
  /** A kind-declared call's own generous HARD ceiling in ms (D8); the host arms it on the job. Default 30 min. */
  hardBudgetMs?: number;
  /**
   * §2.8 idempotency: `false` = a repeat call IS the intended act (a second createDevelopmentTask
   * with the same request is a second task) — never deduped against a running job.
   */
  dedupe?: boolean;
  /** The repeat-call identity when the arguments' hash is not it (a tool whose repeat is unsafe). */
  dedupeKey?(args: any): string;
  /**
   * Optional, for a DISPATCHING function — one that invokes another function chosen by its own
   * arguments (a lazy-loading catalog's "use this tool of that skill"): the function this call
   * reaches, with the arguments it receives. The executor reads the declarations off the TARGET
   * (`job`, `hardBudgetMs`, `dedupe`, `dedupeKey`) and keys the call's identity on the target's
   * own name and arguments, so a tool keeps its declared class whether it is called directly or
   * through the dispatcher — a declaration is made once, at the tool. A target that dispatches in
   * turn is followed to the function that does the work. Return undefined when the arguments
   * name no reachable target: the call then runs under this function's own declarations.
   */
  dispatchTarget?(args: any): { fn: Function; args: any } | undefined;
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
