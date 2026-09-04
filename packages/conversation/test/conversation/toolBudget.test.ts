import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type StreamPart } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function, type ToolCallContext, type ToolPhase } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { ToolBudget, type ToolBudgetConversion, type ToolBudgetHost } from '../../src/ToolBudget';
import { fixtureModelData } from './fixtureModelData';

/**
 * The tool-call BUDGET (plans/FREE_AGENT.md §M.3 part 1; §6 #1 and #6): the executor races every
 * in-process tool call against N (SOFT, default 5 s — `ToolBudget.softBudgetMs()`, the one place)
 * and converts a call still running past N IN PLACE into a background job the host owns:
 *  - a 6 s tool converts at 5 s — the loop reaches its next step boundary at N, not at 6 s; the
 *    model reads the typed "started in the background — job {id}" result; the promise runs on
 *    and the host receives it (its later settlement carries the tool's own value, counter = 1);
 *  - a 2 s tool never converts (D5: under N a tool returns inline exactly as before);
 *  - `background: true` converts at t = 0 (D2's one optimization);
 *  - §2.8 dedupe: a repeat of a running call never executes — the model reads "already running
 *    as job {id}"; `dedupe: false` opts out;
 *  - phases reported before the conversion name the yield sentence; phases after it reach the host;
 *  - the job's Stop reaches the tool through its own signal; a rejection after the hand-off is the
 *    host's to observe, never an unhandled rejection.
 * No network: a MockLanguageModelV3 scripts the loop (the injectedContext suite's idiom).
 *
 * RED at the pre-fix executor (`result = await f.call(args)`, unbounded): the boundary arrives when
 * the tool ends (6 s), the model reads the tool's own result, no host is ever called.
 */

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const toolCallStep = (toolName: string, ids: string[], input = '{}') =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    ...ids.map((id) => ({ type: 'tool-call' as const, toolCallId: id, toolName, input })),
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

type Prompt = Array<{ role: string; content: unknown }>;

/** Every tool-result text a step's prompt carries (the model's view of what the tool returned). */
const toolResultTexts = (prompt: Prompt): string[] =>
  prompt
    .filter((m) => m.role === 'tool' && Array.isArray(m.content))
    .flatMap((m) =>
      (m.content as Array<{ type?: string; output?: { type?: string; value?: unknown } }>)
        .filter((p) => p?.type === 'tool-result')
        .map((p) => {
          const out = p.output;
          return out?.type === 'text' ? String(out.value) : JSON.stringify(out?.value ?? out);
        })
    );

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const skill = (fns: Function[]): ConversationSkill => ({
  getId: () => 'tool-budget-test-skill',
  getName: () => 'ToolBudgetTestSkill',
  getSystemMessages: () => [],
  getFunctions: () => fns,
  getMessageModerators: () => [] as MessageModerator[],
});

/** A host that records conversions and dedupes on the key the executor hands it (the registry's twin). */
const recordingHost = (opts?: { softBudgetMs?: number }) => {
  const conversions: ToolBudgetConversion[] = [];
  const runningByKey = new Map<string, string>();
  const runningLookups: string[] = [];
  const host: ToolBudgetHost = {
    ...(opts?.softBudgetMs !== undefined ? { softBudgetMs: opts.softBudgetMs } : {}),
    convert: async (call) => {
      conversions.push(call);
      const jobId = `job-${conversions.length}`;
      if (call.dedupeKey) {
        runningByKey.set(`${call.name}|${call.dedupeKey}`, jobId);
      }
      return { jobId };
    },
    running: async ({ name, dedupeKey }) => {
      runningLookups.push(dedupeKey);
      const jobId = runningByKey.get(`${name}|${dedupeKey}`);
      return jobId ? { jobId } : undefined;
    },
  };
  return { host, conversions, runningLookups };
};

/**
 * Run a scripted loop: `steps` scripts each model call; returns the prompts each step saw (with
 * the wall-clock ms of the call relative to `t0`), the streamed parts, the final text and the
 * recorded tool invocations.
 */
const runLoop = async (args: {
  fns: Function[];
  host?: ToolBudgetHost;
  steps: Array<() => ReturnType<typeof toolCallStep> | ReturnType<typeof textStep>>;
}) => {
  const prompts: Array<{ atMs: number; prompt: Prompt }> = [];
  const t0 = Date.now();
  let call = 0;
  const model = new MockLanguageModelV3({
    doStream: async (options: { prompt: Prompt }) => {
      prompts.push({ atMs: Date.now() - t0, prompt: options.prompt });
      const step = args.steps[Math.min(call, args.steps.length - 1)];
      call++;
      return { stream: step() };
    },
  });
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'tool-budget-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [skill(args.fns)],
  });
  const result = await conversation.generateStream({
    messages: ['do the work'],
    model: model as never,
    ...(args.host ? { toolBudget: args.host } : {}),
  });
  const parts: StreamPart[] = [];
  for await (const part of result.fullStream) {
    parts.push(part);
  }
  return {
    prompts,
    parts,
    text: await result.text,
    toolInvocations: await result.toolInvocations,
    t0,
  };
};

describe('the tool-call budget — tool → job in place at the executor (FREE_AGENT §M.3 part 1)', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;

  beforeEach(() => {
    // The default N is the number under test — no env override.
    delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  });

  afterAll(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  test('N is one number: 5 s by default, env-tunable', () => {
    expect(ToolBudget.softBudgetMs()).toBe(5_000);
    process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '250';
    expect(ToolBudget.softBudgetMs()).toBe(250);
    delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  });

  test('§6 #1: a 6 s tool converts at N = 5 s — the boundary fires at N, the model reads the typed hand-off, the promise runs on and the host receives it (counter = 1 later)', async () => {
    let counter = 0;
    let settledAtMs = 0;
    const slow: Function = {
      definition: { name: 'slowTool', description: 'slow', parameters: { type: 'object', properties: {} } },
      call: async () => {
        await sleep(6_000);
        counter++;
        settledAtMs = Date.now();
        return { done: true };
      },
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [slow],
      host,
      steps: [() => toolCallStep('slowTool', ['tc-1']), () => textStep('carried on')],
    });

    // The boundary (step 2's model call) arrived at N, not when the tool ended.
    expect(run.prompts).toHaveLength(2);
    const boundaryAtMs = run.prompts[1].atMs;
    expect(boundaryAtMs).toBeGreaterThanOrEqual(4_900);
    expect(boundaryAtMs).toBeLessThan(5_900);
    // The loop was free to continue: the final text landed before the tool had finished.
    expect(run.text).toBe('carried on');
    expect(counter).toBe(0);

    // What the model read as the tool's result: the harness's yield sentence, naming the job.
    const toolResults = toolResultTexts(run.prompts[1].prompt);
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toContain('Started in the background: Working in the background — job job-1');
    expect(toolResults[0]).toContain('do not call this tool again to wait for it');
    expect(toolResults[0]).toContain('one plain line');
    expect(toolResults[0]).toContain('a question the user sent meanwhile gets its answer in this same reply');

    // ONE conversion, with the call's identity, its HARD default and its dedupe key.
    expect(conversions).toHaveLength(1);
    expect(conversions[0].toolCallId).toBe('tc-1');
    expect(conversions[0].name).toBe('slowTool');
    expect(conversions[0].hardBudgetMs).toBe(ToolBudget.hardBudgetMs());
    expect(conversions[0].dedupeKey).toBe(ToolBudget.argsKey({}));

    // The invocation record and the settle part both carry the conversion (part 3 paints it).
    expect(run.toolInvocations).toHaveLength(1);
    expect(run.toolInvocations[0].converted).toEqual({ jobId: 'job-1', title: 'Working in the background' });
    expect(run.toolInvocations[0].ok).toBe(true);
    const settled = run.parts.find((p) => p.type === 'tool-settled');
    expect(settled).toMatchObject({
      type: 'tool-settled',
      id: 'tc-1',
      ok: true,
      converted: { jobId: 'job-1', title: 'Working in the background' },
    });

    // The promise ran ON (§2.3 #1): the host's copy settles with the tool's own value, once.
    await expect(conversions[0].promise).resolves.toEqual({ done: true });
    expect(counter).toBe(1);
    expect(settledAtMs - run.t0).toBeGreaterThanOrEqual(5_900);
  }, 20_000);

  test('D5: a 2 s tool never converts — the model reads its own result inline, no host call', async () => {
    let counter = 0;
    const quick: Function = {
      definition: { name: 'quickTool', description: 'quick', parameters: { type: 'object', properties: {} } },
      call: async () => {
        await sleep(2_000);
        counter++;
        return { done: true };
      },
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [quick],
      host,
      steps: [() => toolCallStep('quickTool', ['tc-1']), () => textStep('inline')],
    });
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(1_900);
    expect(conversions).toHaveLength(0);
    expect(counter).toBe(1);
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toContain('"done":true');
    expect(run.toolInvocations[0].converted).toBeUndefined();
    expect(run.parts.find((p) => p.type === 'tool-settled')).toEqual({
      type: 'tool-settled',
      id: 'tc-1',
      toolName: 'quickTool',
      ok: true,
    });
  }, 20_000);

  test('D2: `background: true` converts at t = 0 — the yield is immediate', async () => {
    let resolveTool!: (v: unknown) => void;
    const always: Function = {
      definition: { name: 'alwaysLong', description: 'long', parameters: { type: 'object', properties: {} } },
      background: true,
      hardBudgetMs: 60_000,
      call: () => new Promise((resolve) => (resolveTool = resolve)),
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [always],
      host,
      steps: [() => toolCallStep('alwaysLong', ['tc-1']), () => textStep('went on')],
    });
    expect(run.prompts[1].atMs).toBeLessThan(500);
    expect(conversions).toHaveLength(1);
    // The tool's own HARD declaration rides the conversion (D8).
    expect(conversions[0].hardBudgetMs).toBe(60_000);
    resolveTool('late');
    await expect(conversions[0].promise).resolves.toBe('late');
  });

  test('the host may set N for its loop (a constructor/option, not a second env)', async () => {
    const slowish: Function = {
      definition: { name: 'slowish', description: 'slowish', parameters: { type: 'object', properties: {} } },
      call: async () => {
        await sleep(1_200);
        return 'ok';
      },
    };
    const { host, conversions } = recordingHost({ softBudgetMs: 300 });
    const run = await runLoop({
      fns: [slowish],
      host,
      steps: [() => toolCallStep('slowish', ['tc-1']), () => textStep('x')],
    });
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(280);
    expect(run.prompts[1].atMs).toBeLessThan(1_100);
    expect(conversions).toHaveLength(1);
  });

  test('without a host the executor awaits the call to completion (an unbudgeted caller — unchanged behaviour)', async () => {
    const slowish: Function = {
      definition: { name: 'slowish', description: 'slowish', parameters: { type: 'object', properties: {} } },
      call: async () => {
        await sleep(600);
        return 'ok';
      },
    };
    const run = await runLoop({
      fns: [slowish],
      steps: [() => toolCallStep('slowish', ['tc-1']), () => textStep('x')],
    });
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(580);
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toBe('ok');
  });

  test('§6 #6 dedupe: the same tool + args while the job runs → "already running as job {id}"; the side effect stays at 1; `dedupe: false` opts out', async () => {
    let counter = 0;
    const hold = new Promise<unknown>(() => undefined); // never settles: the job keeps running
    const slow: Function = {
      definition: { name: 'slowTool', description: 'slow', parameters: { type: 'object', properties: {} } },
      call: async () => {
        counter++;
        return hold;
      },
    };
    const { host, conversions, runningLookups } = recordingHost({ softBudgetMs: 100 });
    const run = await runLoop({
      fns: [slow],
      host,
      steps: [
        () => toolCallStep('slowTool', ['tc-1'], '{"repo":"a"}'),
        () => toolCallStep('slowTool', ['tc-2'], '{"repo":"a"}'),
        () => textStep('x'),
      ],
    });
    expect(conversions).toHaveLength(1);
    expect(counter).toBe(1);
    // Both calls asked the host; the second found the first's job by the same key.
    expect(runningLookups).toHaveLength(2);
    expect(runningLookups[0]).toBe(runningLookups[1]);
    const second = toolResultTexts(run.prompts[2].prompt).find((t) => t.startsWith('Already running'));
    expect(second).toContain('Already running as job job-1: Working in the background');
    expect(second).toContain('Do not start it again');
    expect(run.toolInvocations.map((i) => i.converted)).toEqual([
      { jobId: 'job-1', title: 'Working in the background' },
      { jobId: 'job-1', title: 'Working in the background', deduped: true },
    ]);
    const settles = run.parts.filter((p) => p.type === 'tool-settled');
    expect(settles[1]).toMatchObject({ id: 'tc-2', converted: { jobId: 'job-1', deduped: true } });

    // Opt-out: a tool whose repeat IS the intended act runs twice.
    let twice = 0;
    const repeatable: Function = {
      definition: { name: 'createTask', description: 'task', parameters: { type: 'object', properties: {} } },
      dedupe: false,
      call: async () => {
        twice++;
        return hold;
      },
    };
    const again = recordingHost({ softBudgetMs: 100 });
    await runLoop({
      fns: [repeatable],
      host: again.host,
      steps: [
        () => toolCallStep('createTask', ['tc-1'], '{"request":"x"}'),
        () => toolCallStep('createTask', ['tc-2'], '{"request":"x"}'),
        () => textStep('x'),
      ],
    });
    expect(twice).toBe(2);
    expect(again.conversions).toHaveLength(2);
    expect(again.conversions.map((c) => c.dedupeKey)).toEqual([undefined, undefined]);
    expect(again.runningLookups).toHaveLength(0);
  });

  test('a declared dedupeKey is the identity, and the arguments hash is key-order independent', () => {
    expect(ToolBudget.argsKey({ a: 1, b: [1, { c: 2 }] })).toBe(ToolBudget.argsKey({ b: [1, { c: 2 }], a: 1 }));
    expect(ToolBudget.argsKey({ a: 1 })).not.toBe(ToolBudget.argsKey({ a: 2 }));
  });

  test('phases: reported before the conversion they name the yield sentence; after it they reach the host', async () => {
    let ctxSeen: ToolCallContext | undefined;
    const phased: Function = {
      definition: { name: 'materialize', description: 'm', parameters: { type: 'object', properties: {} } },
      getTimelineDetail: () => 'n3xa app',
      call: async (_args, ctx) => {
        ctxSeen = ctx;
        ctx?.onPhase({ on: 'Setting up the workspace', detail: 'getting a machine to work on' });
        await sleep(400);
        ctx?.onPhase({ on: 'Setting up the workspace', detail: 'getting the code (n3xa app)' });
        await sleep(200);
        return { path: '/repo' };
      },
    };
    const phasesAfter: ToolPhase[] = [];
    const { host, conversions } = recordingHost({ softBudgetMs: 150 });
    const hostWithListener: ToolBudgetHost = {
      ...host,
      convert: async (call) => {
        call.onPhase((phase) => phasesAfter.push(phase));
        return host.convert(call);
      },
    };
    const run = await runLoop({
      fns: [phased],
      host: hostWithListener,
      steps: [() => toolCallStep('materialize', ['tc-1']), () => textStep('x')],
    });
    expect(ctxSeen?.signal).toBeInstanceOf(AbortSignal);
    expect(conversions[0].phase).toEqual({ on: 'Setting up the workspace', detail: 'getting a machine to work on' });
    expect(conversions[0].detail).toBe('n3xa app');
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toContain(
      'Started in the background: Setting up the workspace — getting a machine to work on — job job-1'
    );
    await conversions[0].promise;
    expect(phasesAfter).toEqual([{ on: 'Setting up the workspace', detail: 'getting the code (n3xa app)' }]);
  });

  test("the job's NAME is a task in plain English (founder ruling 2026-09-04 22:00Z): the model's task label, else the tool's own words — never the tool's name, never 'tool' or 'call'", () => {
    expect(ToolBudget.title({ task: ' Reading the sign-in audit ' })).toBe('Reading the sign-in audit');
    expect(
      ToolBudget.title({ phase: { on: 'Setting up the workspace', detail: 'getting the code' }, detail: 'n3xa app' })
    ).toBe('Setting up the workspace');
    expect(ToolBudget.title({ detail: 'tldraw' })).toBe('Working on tldraw');
    expect(ToolBudget.title({})).toBe('Working in the background');
    // THE PIN: no derivation ever reaches for the tool's name or says it is a tool call.
    const names = ['lookAtCodebase', 'web_search', 'slowTool', 'createDevelopmentTask', 'bash'];
    const inputs: Array<{ task?: string; phase?: ToolPhase; detail?: string }> = [
      {},
      { detail: 'the sign-in audit' },
      { phase: { on: 'Searching the web', detail: 'Spanner pricing' } },
      { task: 'Building the workspace' },
    ];
    for (const name of names) {
      for (const input of inputs) {
        const title = ToolBudget.title(input).toLowerCase();
        expect(title).not.toContain(name.toLowerCase());
        expect(title).not.toMatch(/\btool\b/);
        expect(title).not.toMatch(/\bcall\b/);
      }
    }
    expect(ToolBudget.yieldSentence({ title: 'Building the workspace', jobId: 'job-7' })).toContain(
      'Started in the background: Building the workspace — job job-7'
    );
    // The phase words ride under the name: in full under a model-given name, the detail alone
    // when the name IS the phase's own words.
    const phase = { on: 'Setting up the workspace', detail: 'getting the code' };
    expect(ToolBudget.phaseDetail('Looking at the tldraw code', phase)).toBe(
      'Setting up the workspace — getting the code'
    );
    expect(ToolBudget.phaseDetail('Setting up the workspace', phase)).toBe('getting the code');
    expect(ToolBudget.phaseDetail('Looking at the code', { on: 'Cloning' })).toBe('Cloning');
    expect(ToolBudget.phaseDetail('Looking at the code', undefined)).toBeUndefined();
  });

  test("the model's `task` label rides every budgeted tool's schema, is split off before the tool runs, names the job, and never changes the dedupe identity", async () => {
    let received: unknown;
    const slow: Function = {
      definition: {
        name: 'slowTool',
        description: 'slow',
        parameters: { type: 'object', properties: { repo: { type: 'string' } }, required: ['repo'] },
      },
      call: async (args) => {
        received = args;
        return new Promise(() => undefined);
      },
    };
    const { host, conversions, runningLookups } = recordingHost({ softBudgetMs: 100 });
    const run = await runLoop({
      fns: [slow],
      host,
      steps: [
        () => toolCallStep('slowTool', ['tc-1'], '{"repo":"a","task":"Reading the sign-in audit"}'),
        () => toolCallStep('slowTool', ['tc-2'], '{"repo":"a","task":"Reading the audit again"}'),
        () => textStep('x'),
      ],
    });
    // The tool never saw the label; the record carries the tool's own arguments.
    expect(received).toEqual({ repo: 'a' });
    expect(run.toolInvocations[0].input).toEqual({ repo: 'a' });
    // The conversion names the work by the label, and the model reads that name.
    expect(conversions).toHaveLength(1);
    expect(conversions[0].task).toBe('Reading the sign-in audit');
    expect(conversions[0].title).toBe('Reading the sign-in audit');
    expect(conversions[0].args).toEqual({ repo: 'a' });
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toContain(
      'Started in the background: Reading the sign-in audit — job job-1'
    );
    expect(run.toolInvocations[0].converted).toEqual({ jobId: 'job-1', title: 'Reading the sign-in audit' });
    expect(run.parts.find((p) => p.type === 'tool-settled')).toMatchObject({
      converted: { jobId: 'job-1', title: 'Reading the sign-in audit' },
    });
    // A repeat with the same arguments and a DIFFERENT label is the same work (§2.8): deduped.
    expect(runningLookups[0]).toBe(runningLookups[1]);
    expect(run.toolInvocations[1].converted).toMatchObject({ jobId: 'job-1', deduped: true });
  });

  test('a budgeted executor offers the `task` label on the schema; an unbudgeted one sends the schema untouched', () => {
    const schema = { type: 'object', properties: { repo: { type: 'string' } }, required: ['repo'] };
    const offered = ToolBudget.withTaskParameter(schema);
    expect(Object.keys(offered.properties)).toEqual(['repo', 'task']);
    expect(offered.required).toEqual(['repo']);
    expect(offered.properties.task.description).toContain('plain English');
    // A tool whose own schema already has `task` keeps it; a non-object schema is untouched.
    const own = { type: 'object', properties: { task: { type: 'number' } } };
    expect(ToolBudget.withTaskParameter(own)).toBe(own);
    const notObject = { type: 'string' };
    expect(ToolBudget.withTaskParameter(notObject)).toBe(notObject);
    expect(ToolBudget.splitTask({ a: 1, task: '  ' })).toEqual({ args: { a: 1 } });
    expect(ToolBudget.splitTask('text')).toEqual({ args: 'text' });
  });

  test("D9: the job's Stop reaches the tool through its own signal; a rejection after the hand-off is the host's, never unhandled", async () => {
    const aborted: string[] = [];
    const stoppable: Function = {
      definition: { name: 'stoppable', description: 's', parameters: { type: 'object', properties: {} } },
      call: (_args, ctx) =>
        new Promise((_resolve, reject) => {
          ctx?.signal.addEventListener('abort', () => {
            aborted.push('signal');
            reject(new Error('stopped by the job'));
          });
        }),
    };
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      const { host, conversions } = recordingHost({ softBudgetMs: 100 });
      const run = await runLoop({
        fns: [stoppable],
        host,
        steps: [() => toolCallStep('stoppable', ['tc-1']), () => textStep('x')],
      });
      expect(run.text).toBe('x');
      expect(conversions).toHaveLength(1);
      expect(conversions[0].signal.aborted).toBe(false);
      conversions[0].abort();
      expect(conversions[0].signal.aborted).toBe(true);
      expect(aborted).toEqual(['signal']);
      await expect(conversions[0].promise).rejects.toThrow('stopped by the job');
      await sleep(50);
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  test('a call that fails UNDER N follows the failure path exactly as an unbudgeted call would', async () => {
    const failing: Function = {
      definition: { name: 'failing', description: 'f', parameters: { type: 'object', properties: {} } },
      call: async () => {
        throw new Error('boom');
      },
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [failing],
      host,
      steps: [() => toolCallStep('failing', ['tc-1']), () => textStep('recovered')],
    });
    expect(conversions).toHaveLength(0);
    expect(run.toolInvocations[0].ok).toBe(false);
    expect(run.toolInvocations[0].error?.message).toBe('boom');
    expect(run.parts.find((p) => p.type === 'tool-settled')).toMatchObject({ id: 'tc-1', ok: false });
  });
});
