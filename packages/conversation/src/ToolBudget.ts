import { createHash } from 'crypto';
import type { Function, ToolJobKind, ToolPhase, ToolTimelineDetail } from './Function';

/**
 * A call handed to the host at its conversion (plans/FREE_AGENT.md §2.5 the job record; DECISIONS
 * 630 the class rule): everything the host needs to own the job from here — the kind the function
 * declared, the running promise, its abort control, the phase relay, the task's visible name, and
 * the identity the §2.8 dedupe keys on.
 */
export type ToolBudgetConversion = {
  toolCallId: string;
  /**
   * The name of the function that does the work: the called function's, or — for a call made
   * through a dispatching function (`Function.dispatchTarget`) — its TARGET's, so the job and the
   * §2.8 identity read as the tool they are whichever road the call took.
   */
  name: string;
  /** The arguments of the function that does the work (a dispatch's target's), WITHOUT the harness's `task` label. */
  args: unknown;
  /** What runs behind the promise — the kind the function declared (`Function.job`); the row's kind. */
  kind: ToolJobKind;
  /**
   * The job's visible NAME (the founder's ruling 2026-09-04 22:00Z, FREE_AGENT §M "promoted calls
   * are NAMED TASKS"): what the agent is trying to do, in plain English — the model's own `task`
   * label for the call, else the tool's own vocabulary ({@link ToolBudget.title}). Never the tool's
   * name; nothing that says it began as a tool call. The row, the timeline node and any
   * notification carry exactly this.
   */
  title: string;
  /** The model's own `task` label for the call, when it gave one. */
  task?: string;
  /** The call's timeline detail text (the tool's `getTimelineDetail`), when any. */
  detail?: string;
  /** The phase the tool reported at its start, BEFORE the conversion, when any. */
  phase?: ToolPhase;
  /** The HARD ceiling to arm (the tool's own declaration, else the default — D8). */
  hardBudgetMs: number;
  /** The §2.8 identity of this call among the chat's running jobs; absent for `dedupe: false` tools. */
  dedupeKey?: string;
  /**
   * The running call: resolves/rejects with the tool's own result. The executor never awaits it
   * again (a rejection here is the host's to deliver as a failed job, never an unhandled one).
   */
  promise: Promise<unknown>;
  /** The call's own abort signal — the one the tool received as `ToolCallContext.signal`. */
  signal: AbortSignal;
  /** The job's Stop (D9) and its HARD kill: fires `signal`. */
  abort: () => void;
  /** Subscribe to the phases the tool reports AFTER the conversion (the job's node moves in place). */
  onPhase: (listener: (phase: ToolPhase) => void) => void;
  /**
   * The called function's outcome hook (`Function.getTimelineOutcome`), bound to this call's
   * arguments — a dispatcher's hook reaches its target's. A converted call's result part carries
   * the hand-off, not the tool's result, so the executor's stream never relabels its node; the
   * host reads the outcome here when the job settles with the tool's own result, and settles the
   * node with what the call made. Absent when the function declares no hook.
   */
  outcome?: (
    result: unknown
  ) => Promise<{ name?: string; detail?: string | ToolTimelineDetail; ok?: boolean } | undefined>;
};

/**
 * A call as the host's call hooks name it: the tool-call id (the timeline node's id — the handle a
 * per-node Stop names beside the response message) and the name the call's `tool-call` part carries
 * (the called function's — a dispatcher's own, never its target's).
 */
export type ToolBudgetCall = { toolCallId: string; name: string };

/**
 * The host side of the job conversion (`GenerateStreamParams.toolBudget`) — implemented by the
 * chat turn (thought-server wires it to `ChatJobRegistry`); the conversation package cannot know
 * what a job is, only that one exists once `convert` returns.
 *
 * THE CALL HOOKS (`started` / `phase` / `settled` — plans/DECISIONS.md 630's second half: "the reply
 * stays open with the job's progress and its stop handle until the deliverable lands"): the host
 * observes every call the executor runs under it, so an AWAITED call — a deck, a document, a catalog
 * check landing in this reply — has its progress words on its running node and a Stop of its own,
 * the way a converted call has them through its job. One owner per concept: an awaited call's abort
 * is handed at `started` and released at `settled`; a converted call's abort rides the conversion
 * (D9) and its phases after the conversion reach the job's `onPhase` alone.
 */
export type ToolBudgetHost = {
  /** Convert a kind-declared call IN PLACE into a job of its kind; the id names it to the model. */
  convert(call: ToolBudgetConversion): Promise<{ jobId: string }>;
  /**
   * §2.8: the job already RUNNING for the same (tool, dedupeKey), if any — a repeat call returns
   * "already running as job {id}" to the model and never executes a second time.
   */
  running?(call: { name: string; args: unknown; dedupeKey: string }): Promise<{ jobId: string } | undefined>;
  /**
   * An AWAITED call started: the host learns its abort at its start — the per-node Stop's handle.
   * `abort` fires the call's own `ToolCallContext.signal`; a tool that honors it settles the call as
   * a failure the model reads through the executor's failure path (the tool's own rejection, as any
   * failed call's). Released at `settled`. Never fires for a kind-declared call: its abort rides the
   * conversion.
   */
  started?(call: ToolBudgetCall & { abort: () => void; signal: AbortSignal }): void;
  /**
   * A phase the tool reported while the call is NOT a job — every phase of an awaited call, and a
   * declared call's phases before its conversion (the hand-off carries the last of those as `phase`):
   * the running node's words, in the tool's own present-progressive voice. After the conversion the
   * phases reach the job through the conversion's `onPhase` alone. Nothing fires after the call
   * settled.
   */
  phase?(call: ToolBudgetCall, phase: ToolPhase): void;
  /** An AWAITED call settled — its result or its failure is the model's: the host releases what `started` gave it. */
  settled?(call: ToolBudgetCall): void;
};

/** What a call came to: the tool's own result, or its conversion (the text the model reads). */
export type ToolBudgetOutcome =
  | { kind: 'settled'; result: unknown }
  | { kind: 'converted'; jobId: string; title: string; text: string; deduped?: boolean };

/**
 * THE ONE-RESPONSE CLASS RULE at the executor (plans/DECISIONS.md 630, ruled 2026-10-17; plans/
 * FREE_AGENT.md §M.23 stage C): a call converts into a job ONLY when the function it reaches
 * declares its job kind (`Function.job` — spawned agents, shells, watchers, orchestrations,
 * routines: the things that genuinely run on their own, jobs by what they ARE), and it converts
 * AT ITS START — no clock: the running promise is handed to the host with its kind, its abort
 * control and its phase relay, and the model reads a typed "started in the background — job {id}"
 * result at the step boundary. EVERY OTHER CALL IS AWAITED WHOLE: a deck, a document, a picture,
 * a catalog check — whatever the agent makes for the person — lands in the reply they asked in;
 * nothing they wait on is ever split into "I started that" and a second message later (the third
 * leak of that shape was his 1.32.0 smoke finding 6). No per-tool opt-in, no allowlist of
 * always-awaited declarations: the class is the rule. One instance per call.
 *
 * N ({@link ToolBudget.softBudgetMs}) is no longer a clock on any tool call. It remains the
 * ROUND's number (§M.3 part 2b — the time a streaming round gets to finish on its own once a
 * mid-turn note lands, `Conversation`'s cut-and-continue) and the number the job row records.
 *
 * The job's NAME is a task in plain English (the founder's ruling, FREE_AGENT §M): under a budgeted
 * executor every tool's schema offers the model a `task` label ({@link ToolBudget.TASK_PARAMETER}
 * — "what you are doing with this call, for the user"); the executor strips it before the tool
 * runs and {@link ToolBudget.title} derives the visible name from it, else from the tool's own
 * vocabulary — never from the tool's name.
 */
export class ToolBudget {
  /** The per-call label every budgeted tool accepts beside its own arguments. */
  static readonly TASK_PARAMETER = 'task';

  private lastPhase: ToolPhase | undefined;
  private readonly phaseListeners: Array<(phase: ToolPhase) => void> = [];
  /** The call's road once decided: a job's phases reach its listeners; before that, and for an awaited call, the host's `phase` hook. */
  private converted = false;
  /** An awaited call that resolved or rejected: a phase reported after it reaches no one. */
  private settled = false;

  constructor(
    private readonly call: {
      host: ToolBudgetHost;
      fn: Function;
      toolCallId: string;
      /** The tool's arguments, the `task` label already split off. */
      input: unknown;
      task?: string;
      detail?: string;
    }
  ) {}

  /**
   * N — the ROUND's soft budget (§M.2, §M.3 part 2b): env `CONVERSATION_TOOL_SOFT_BUDGET_MS`,
   * default 5 000 ms. Read by the round loop's cut-and-continue clock and recorded on the job row;
   * never a clock on a tool call (the class rule above).
   */
  static softBudgetMs(): number {
    return Number(process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS || 5_000);
  }

  /** The HARD default (D8, §2.2): env `CONVERSATION_TOOL_HARD_BUDGET_MS`, default 30 min. */
  static hardBudgetMs(): number {
    return Number(process.env.CONVERSATION_TOOL_HARD_BUDGET_MS || 1_800_000);
  }

  /**
   * Run the call under the class rule: a function that declares no job kind is awaited whole —
   * the host is never asked; a kind-declared function is checked against the §2.8 running jobs
   * first (a repeat never executes) and otherwise converts at its start.
   */
  async run(): Promise<ToolBudgetOutcome> {
    const { host, fn, toolCallId, input, task, detail } = this.call;
    // The function that does the work — the called one, or through a dispatching function its
    // TARGET: its declaration is this call's class and its name + arguments this call's identity,
    // so a tool behaves as it declared on either road (a dispatcher declares nothing of its own).
    const target = ToolBudget.reach(fn, input);
    const controller = new AbortController();
    const ctx = {
      signal: controller.signal,
      onPhase: (phase: ToolPhase) => this.reportPhase(phase),
    };
    const job = target.fn.job;
    if (!job) {
      // AWAITED WHOLE: the deliverable lands in this reply; a rejection propagates to the
      // executor's failure path exactly as it always has. The host holds the call's abort for its
      // life (the Stop on its node — DECISIONS 630's second half) and is told when it settled.
      host.started?.({ ...this.ref(), abort: () => controller.abort(), signal: controller.signal });
      try {
        return { kind: 'settled', result: await fn.call(input, ctx) };
      } finally {
        this.settled = true;
        host.settled?.(this.ref());
      }
    }
    const name = target.fn.definition.name;
    const hints = target.fn;
    const dedupeKey =
      hints.dedupe === false ? undefined : hints.dedupeKey?.(target.args) ?? ToolBudget.argsKey(target.args);
    if (dedupeKey !== undefined && host.running) {
      const running = await host.running({ name, args: target.args, dedupeKey });
      if (running) {
        const title = ToolBudget.title({ task, detail });
        return {
          kind: 'converted',
          jobId: running.jobId,
          title,
          deduped: true,
          text: ToolBudget.alreadyRunningSentence({ title, jobId: running.jobId }),
        };
      }
    }
    // A JOB BY KIND: the call starts and is handed to the host at once — the host observes the
    // settlement (done or failed); never an unhandled rejection here.
    const promise = (async () => fn.call(input, ctx))();
    promise.catch(() => undefined);
    // A JOB BY KIND from here: what the tool said at its start reached the host's `phase` hook and
    // rides the hand-off below; every later phase is the job's (the listeners the conversion registers).
    this.converted = true;
    const title = ToolBudget.title({ task, phase: this.lastPhase, detail });
    // The called function's outcome hook rides the conversion (a dispatcher's reaches its target's):
    // the host settles the job's node with what the call made, read from the tool's own result.
    const outcome = fn.getTimelineOutcome;
    const { jobId } = await host.convert({
      toolCallId,
      name,
      args: target.args,
      kind: job.kind,
      title,
      ...(task ? { task } : {}),
      ...(detail ? { detail } : {}),
      ...(this.lastPhase ? { phase: this.lastPhase } : {}),
      hardBudgetMs: hints.hardBudgetMs ?? ToolBudget.hardBudgetMs(),
      ...(dedupeKey !== undefined ? { dedupeKey } : {}),
      promise,
      signal: controller.signal,
      abort: () => controller.abort(),
      onPhase: (listener) => this.phaseListeners.push(listener),
      ...(outcome ? { outcome: async (result: unknown) => (await outcome.call(fn, input, result)) || undefined } : {}),
    });
    return {
      kind: 'converted',
      jobId,
      title,
      text: ToolBudget.yieldSentence({ title, detail: ToolBudget.phaseDetail(title, this.lastPhase), jobId }),
    };
  }

  /**
   * The tool's schema with the `task` label offered beside its own arguments — what a budgeted
   * executor sends the model for every in-process tool. Never touches a schema that is not an
   * object, never marks the label required, never overrides a tool's own `task` property.
   */
  static withTaskParameter(schema: Record<string, any>): Record<string, any> {
    if (!schema || schema.type !== 'object') {
      return schema;
    }
    const properties = (schema.properties ?? {}) as Record<string, unknown>;
    if (ToolBudget.TASK_PARAMETER in properties) {
      return schema;
    }
    return {
      ...schema,
      properties: {
        ...properties,
        [ToolBudget.TASK_PARAMETER]: {
          type: 'string',
          description:
            'What you are doing with this call, for the user, in a few plain words — a task in ' +
            'plain English such as "Reading the sign-in audit" or "Building the workspace". It ' +
            'names this work wherever the user sees it. Never a tool name.',
        },
      },
    };
  }

  /**
   * Split the model's `task` label off a call's arguments: the tool receives its own arguments
   * exactly as declared (the label is the harness's, not the tool's), and the label — trimmed,
   * empty = absent — names the work.
   */
  static splitTask(args: unknown): { args: unknown; task?: string } {
    if (!args || typeof args !== 'object' || Array.isArray(args) || !(ToolBudget.TASK_PARAMETER in args)) {
      return { args };
    }
    const { [ToolBudget.TASK_PARAMETER]: raw, ...rest } = args as Record<string, unknown>;
    const task = typeof raw === 'string' ? raw.trim() : '';
    return task ? { args: rest, task } : { args: rest };
  }

  /**
   * The visible name of the work (the founder's ruling): the model's own task label; else the
   * tool's own words for the call — its last reported phase, or "Working on {the call's subject}"
   * from the timeline detail; else the plain "Working in the background". Never the tool's name,
   * never a word that says it is a tool call.
   */
  static title(args: { task?: string; phase?: ToolPhase; detail?: string }): string {
    const task = args.task?.trim();
    if (task) {
      return task;
    }
    if (args.phase?.on.trim()) {
      return args.phase.on.trim();
    }
    const detail = args.detail?.trim();
    return detail ? `Working on ${detail}` : 'Working in the background';
  }

  /**
   * The phase words as the detail line under a job's name: the tool's phase in full ("Setting up
   * the workspace — getting the code") unless the name already IS the phase's own words (the
   * title came from the phase), in which case only the phase's detail is new.
   */
  static phaseDetail(title: string, phase: ToolPhase | undefined): string | undefined {
    if (!phase) {
      return undefined;
    }
    if (phase.on.trim() === title.trim()) {
      return phase.detail;
    }
    return phase.detail ? `${phase.on} — ${phase.detail}` : phase.on;
  }

  /**
   * The harness → the agent at a conversion (§2.4, one voice): plain and complete — what started,
   * the job id, that the result arrives as a separate message, never re-call to wait, and the
   * one-line acknowledgment the user is owed (D3's refinement; the same-reply rule for a question
   * sent meanwhile). The agent's own sentence to the user is its own (D4).
   */
  static yieldSentence(args: { title: string; detail?: string; jobId: string }): string {
    const what = args.detail ? `${args.title} — ${args.detail}` : args.title;
    return (
      `Started in the background: ${what} — job ${args.jobId}. It keeps running; its result ` +
      `will reach you as a separate message when it finishes, and the user can see it running. Do not assume ` +
      `its result and do not call this tool again to wait for it. Tell the user in one plain line what has ` +
      `started (no time estimate), then continue with anything else you can do for them now — a question the ` +
      `user sent meanwhile gets its answer in this same reply.`
    );
  }

  /** The §2.8 repeat: the same voice, pointing at the job that already runs. */
  static alreadyRunningSentence(args: { title: string; jobId: string }): string {
    return (
      `Already running as job ${args.jobId}: ${args.title}. It was started earlier in this chat and ` +
      `is still going; its result will reach you as a separate message when it finishes. Do not start it again ` +
      `and do not wait for it — continue with anything else you can do for the user now.`
    );
  }

  /** The default §2.8 identity: a stable hash of the tool's arguments (key order never matters). */
  static argsKey(input: unknown): string {
    return createHash('sha1').update(ToolBudget.stableJson(input)).digest('hex');
  }

  // ─── helpers ────────────────────────────────────────────────────────────────

  /**
   * The function a call reaches: the called one, or — through dispatching functions
   * (`Function.dispatchTarget`), followed until one does the work — the target with its own
   * arguments. A hook that throws or loops names no target: the call is its own.
   */
  private static reach(fn: Function, input: unknown): { fn: Function; args: unknown } {
    let at = { fn, args: input };
    const seen = new Set<Function>([fn]);
    for (;;) {
      let next: { fn: Function; args: unknown } | undefined;
      try {
        next = at.fn.dispatchTarget?.(at.args);
      } catch {
        return at;
      }
      if (!next || seen.has(next.fn)) {
        return at;
      }
      seen.add(next.fn);
      at = next;
    }
  }

  /**
   * A phase the tool reported: a job's (after its conversion) reaches the listeners the conversion
   * registered; an awaited call's — and a declared call's before its conversion — reaches the host's
   * `phase` hook under the call's id. After an awaited call settled, nothing: the node is settled.
   */
  private reportPhase(phase: ToolPhase): void {
    if (this.settled) {
      return;
    }
    this.lastPhase = phase;
    if (!this.converted) {
      this.call.host.phase?.(this.ref(), phase);
      return;
    }
    for (const listener of this.phaseListeners) {
      listener(phase);
    }
  }

  /** The call as the host's hooks name it. */
  private ref(): ToolBudgetCall {
    return { toolCallId: this.call.toolCallId, name: this.call.fn.definition.name };
  }

  private static stableJson(value: unknown): string {
    if (Array.isArray(value)) {
      return `[${value.map((v) => ToolBudget.stableJson(v)).join(',')}]`;
    }
    if (value && typeof value === 'object') {
      const record = value as Record<string, unknown>;
      return `{${Object.keys(record)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${ToolBudget.stableJson(record[key])}`)
        .join(',')}}`;
    }
    return JSON.stringify(value ?? null);
  }
}
