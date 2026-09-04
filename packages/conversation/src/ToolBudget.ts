import { createHash } from 'crypto';
import type { Function, ToolPhase } from './Function';

/**
 * A call handed to the host at its conversion (plans/FREE_AGENT.md §2.2 SOFT, §2.5 the job record):
 * everything the host needs to own the job from here — the running promise, its abort control,
 * the phase relay, the task's visible name, and the identity the §2.8 dedupe keys on.
 */
export type ToolBudgetConversion = {
  toolCallId: string;
  name: string;
  /** The tool's arguments WITHOUT the harness's `task` label (what the tool itself received). */
  args: unknown;
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
  /** The last phase the tool reported BEFORE the conversion, when any. */
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
};

/**
 * The host side of the tool-call budget (`GenerateStreamParams.toolBudget`) — implemented by the
 * chat turn (thought-server wires it to `ChatJobRegistry`); the conversation package cannot know
 * what a job is, only that one exists once `convert` returns.
 */
export type ToolBudgetHost = {
  /** N for this loop's calls; default {@link ToolBudget.softBudgetMs} (the one place). */
  softBudgetMs?: number;
  /** Convert an over-budget call IN PLACE into a background job; the id names it to the model. */
  convert(call: ToolBudgetConversion): Promise<{ jobId: string }>;
  /**
   * §2.8: the job already RUNNING for the same (tool, dedupeKey), if any — a repeat call returns
   * "already running as job {id}" to the model and never executes a second time.
   */
  running?(call: { name: string; args: unknown; dedupeKey: string }): Promise<{ jobId: string } | undefined>;
};

/** What a budgeted call came to: the tool's own result, or its conversion (the text the model reads). */
export type ToolBudgetOutcome =
  | { kind: 'settled'; result: unknown }
  | { kind: 'converted'; jobId: string; title: string; text: string; deduped?: boolean };

/**
 * The tool-call BUDGET (plans/FREE_AGENT.md §2 and §M.3 part 1): the harness owns the clock on
 * every in-process tool call. A call that has not resolved by N — SOFT, default 5 s, the ONE
 * number every layer reads through {@link ToolBudget.softBudgetMs} — is CONVERTED IN PLACE into
 * a background job: the running promise is handed to the host with its abort control and phase
 * relay, and the model reads a typed "started in the background — job {id}" result at the step
 * boundary, so the round continues and the next input reaches the one mind within the bar. The
 * promise runs on; its later settlement is the host's to deliver as an input (I3). No per-tool
 * special case and no timer-as-fallback: the budget IS the contract; a tool's optional hints
 * (`background`, `hardBudgetMs`, `dedupe`/`dedupeKey`) only tune it (D2). One instance per call.
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

  /** N (§M.2): env `CONVERSATION_TOOL_SOFT_BUDGET_MS`, default 5 000 ms. The bar is not tunable; N is. */
  static softBudgetMs(): number {
    return Number(process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS || 5_000);
  }

  /** The HARD default (D8, §2.2): env `CONVERSATION_TOOL_HARD_BUDGET_MS`, default 30 min. */
  static hardBudgetMs(): number {
    return Number(process.env.CONVERSATION_TOOL_HARD_BUDGET_MS || 1_800_000);
  }

  /**
   * Run the call under the budget: the §2.8 running-job check first (a repeat never executes),
   * then the call raced against N (`background: true` → t = 0); past N the call converts.
   */
  async run(): Promise<ToolBudgetOutcome> {
    const { host, fn, toolCallId, input, task, detail } = this.call;
    const name = fn.definition.name;
    const dedupeKey = fn.dedupe === false ? undefined : fn.dedupeKey?.(input) ?? ToolBudget.argsKey(input);
    if (dedupeKey !== undefined && host.running) {
      const running = await host.running({ name, args: input, dedupeKey });
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
    const controller = new AbortController();
    const promise = (async () =>
      fn.call(input, {
        signal: controller.signal,
        onPhase: (phase) => this.reportPhase(phase),
      }))();
    const budgetMs = fn.background ? 0 : host.softBudgetMs ?? ToolBudget.softBudgetMs();
    const raced = await ToolBudget.race(promise, budgetMs);
    if (raced.settled) {
      return { kind: 'settled', result: raced.value };
    }
    // Handed off: the host observes the settlement (done or failed) — never an unhandled rejection here.
    promise.catch(() => undefined);
    const title = ToolBudget.title({ task, phase: this.lastPhase, detail });
    const { jobId } = await host.convert({
      toolCallId,
      name,
      args: input,
      title,
      ...(task ? { task } : {}),
      ...(detail ? { detail } : {}),
      ...(this.lastPhase ? { phase: this.lastPhase } : {}),
      hardBudgetMs: fn.hardBudgetMs ?? ToolBudget.hardBudgetMs(),
      ...(dedupeKey !== undefined ? { dedupeKey } : {}),
      promise,
      signal: controller.signal,
      abort: () => controller.abort(),
      onPhase: (listener) => this.phaseListeners.push(listener),
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

  private reportPhase(phase: ToolPhase): void {
    this.lastPhase = phase;
    for (const listener of this.phaseListeners) {
      listener(phase);
    }
  }

  /**
   * The race: the call's own settlement wins under N; the budget wins past it (the call keeps
   * running — nothing is cancelled by the yield, §2.3 #1). A rejection under N propagates to the
   * executor's failure path exactly as an unbudgeted call's would.
   */
  private static race<T>(
    promise: Promise<T>,
    budgetMs: number
  ): Promise<{ settled: true; value: T } | { settled: false }> {
    if (budgetMs <= 0) {
      return Promise.resolve({ settled: false });
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => resolve({ settled: false }), budgetMs);
      promise.then(
        (value) => {
          clearTimeout(timer);
          resolve({ settled: true, value });
        },
        (error: unknown) => {
          clearTimeout(timer);
          reject(error);
        }
      );
    });
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
