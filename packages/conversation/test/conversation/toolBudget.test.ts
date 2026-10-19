import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type StreamPart } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function, type ToolCallContext, type ToolPhase } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { ToolBudget, type ToolBudgetConversion, type ToolBudgetHost } from '../../src/ToolBudget';
import { fixtureModelData } from './fixtureModelData';
import { SkillDispatcherSkill } from '../../src/SkillDispatcherSkill';

/**
 * THE ONE-RESPONSE CLASS RULE at the executor (plans/DECISIONS.md 630; plans/FREE_AGENT.md §M.23
 * stage C): a call converts into a job ONLY when the function it reaches declares its job kind
 * (`Function.job`), and then AT ITS START — no clock; every other call is AWAITED WHOLE, however
 * long it runs, and the model reads the tool's own result at the boundary:
 *  - a 1.5 s tool that declares nothing is awaited under a 1 ms N — the boundary comes after it,
 *    the host is never asked (RED at the pre-rule executor: the race converted it at N);
 *  - a `job: { kind }` tool converts at t = 0 — the yield is immediate, the conversion carries the
 *    kind, the call's identity, its HARD declaration and its dedupe key; the promise runs on and
 *    the host receives it (its later settlement carries the tool's own value);
 *  - §2.8 dedupe: a repeat of a running kind-declared call never executes — the model reads
 *    "already running as job {id}"; `dedupe: false` opts out;
 *  - the phase a kind-declared call reports at its start names the yield sentence; later phases
 *    reach the host;
 *  - the job's Stop reaches the tool through its own signal; a rejection after the hand-off is the
 *    host's to observe, never an unhandled rejection.
 * No network: a MockLanguageModelV3 scripts the loop (the injectedContext suite's idiom).
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
const recordingHost = () => {
  const conversions: ToolBudgetConversion[] = [];
  const runningByKey = new Map<string, string>();
  const runningLookups: string[] = [];
  const host: ToolBudgetHost = {
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

describe('the one-response class rule — a call converts only by its declared kind (DECISIONS 630)', () => {
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

  test('N is one number — the round’s (§M.3 part 2b): 5 s by default, env-tunable', () => {
    expect(ToolBudget.softBudgetMs()).toBe(5_000);
    process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '250';
    expect(ToolBudget.softBudgetMs()).toBe(250);
    delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  });

  test('THE RULE: a tool that declares nothing is AWAITED WHOLE under a 1 ms N — the boundary waits for it, the model reads the tool’s own result, the host is never asked (RED at the pre-rule executor: converted at N)', async () => {
    process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = '1';
    let counter = 0;
    const slow: Function = {
      definition: { name: 'slowTool', description: 'slow', parameters: { type: 'object', properties: {} } },
      call: async () => {
        await sleep(1_500);
        counter++;
        return { done: true };
      },
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [slow],
      host,
      steps: [() => toolCallStep('slowTool', ['tc-1']), () => textStep('here it is')],
    });

    // The boundary (step 2's model call) arrived when the tool ended — 1 500× past the old N.
    expect(run.prompts).toHaveLength(2);
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(1_400);
    expect(counter).toBe(1);
    expect(conversions).toHaveLength(0);
    // What the model read as the tool's result: the tool's own value, never a hand-off.
    const toolResults = toolResultTexts(run.prompts[1].prompt);
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toContain('"done":true');
    expect(toolResults[0]).not.toContain('Started in the background');
    expect(run.toolInvocations[0].converted).toBeUndefined();
    expect(run.toolInvocations[0].data).toEqual({ done: true });
    expect(run.parts.find((p) => p.type === 'tool-settled')).toEqual({
      type: 'tool-settled',
      id: 'tc-1',
      toolName: 'slowTool',
      ok: true,
    });
    expect(run.text).toBe('here it is');
  }, 20_000);

  test('a kind-declared tool converts AT ITS START — the yield is immediate; the conversion carries the kind, the call’s identity, its HARD declaration and its dedupe key; the promise runs on and the host receives its value', async () => {
    let counter = 0;
    let resolveTool!: (v: unknown) => void;
    const shell: Function = {
      definition: { name: 'runOnTheBox', description: 'long', parameters: { type: 'object', properties: {} } },
      job: { kind: 'shell' },
      hardBudgetMs: 60_000,
      call: () =>
        new Promise((resolve) => {
          resolveTool = (v) => {
            counter++;
            resolve(v);
          };
        }),
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [shell],
      host,
      steps: [() => toolCallStep('runOnTheBox', ['tc-1']), () => textStep('carried on')],
    });
    // The boundary came at once — no clock ran.
    expect(run.prompts).toHaveLength(2);
    expect(run.prompts[1].atMs).toBeLessThan(500);
    expect(run.text).toBe('carried on');
    expect(counter).toBe(0);

    // What the model read as the tool's result: the harness's yield sentence, naming the job.
    const toolResults = toolResultTexts(run.prompts[1].prompt);
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toContain('Started in the background: Working in the background — job job-1');
    expect(toolResults[0]).toContain('do not call this tool again to wait for it');
    expect(toolResults[0]).toContain('one plain line');
    expect(toolResults[0]).toContain('a question the user sent meanwhile gets its answer in this same reply');

    // ONE conversion, with the kind, the call's identity, its HARD declaration (D8) and its dedupe key.
    expect(conversions).toHaveLength(1);
    expect(conversions[0].kind).toBe('shell');
    expect(conversions[0].toolCallId).toBe('tc-1');
    expect(conversions[0].name).toBe('runOnTheBox');
    expect(conversions[0].hardBudgetMs).toBe(60_000);
    expect(conversions[0].dedupeKey).toBe(ToolBudget.argsKey({}));

    // The invocation record and the settle part both carry the conversion (part 3 paints it).
    expect(run.toolInvocations).toHaveLength(1);
    expect(run.toolInvocations[0].converted).toEqual({ jobId: 'job-1', title: 'Working in the background' });
    expect(run.toolInvocations[0].ok).toBe(true);
    expect(run.parts.find((p) => p.type === 'tool-settled')).toMatchObject({
      type: 'tool-settled',
      id: 'tc-1',
      ok: true,
      converted: { jobId: 'job-1', title: 'Working in the background' },
    });

    // The promise ran ON (§2.3 #1): the host's copy settles with the tool's own value, once.
    resolveTool({ done: true });
    await expect(conversions[0].promise).resolves.toEqual({ done: true });
    expect(counter).toBe(1);
  });

  test('a kind-declared call with no HARD declaration converts with the HARD default', async () => {
    const agent: Function = {
      definition: { name: 'spawn', description: 'a', parameters: { type: 'object', properties: {} } },
      job: { kind: 'agent' },
      call: () => new Promise(() => undefined),
    };
    const { host, conversions } = recordingHost();
    await runLoop({ fns: [agent], host, steps: [() => toolCallStep('spawn', ['tc-1']), () => textStep('x')] });
    expect(conversions).toHaveLength(1);
    expect(conversions[0].kind).toBe('agent');
    expect(conversions[0].hardBudgetMs).toBe(ToolBudget.hardBudgetMs());
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

  test('§6 #6 dedupe: the same kind-declared tool + args while the job runs → "already running as job {id}"; the side effect stays at 1; `dedupe: false` opts out', async () => {
    let counter = 0;
    const hold = new Promise<unknown>(() => undefined); // never settles: the job keeps running
    const slow: Function = {
      definition: { name: 'slowTool', description: 'slow', parameters: { type: 'object', properties: {} } },
      job: { kind: 'watcher' },
      call: async () => {
        counter++;
        return hold;
      },
    };
    const { host, conversions, runningLookups } = recordingHost();
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
      job: { kind: 'orchestration' },
      dedupe: false,
      call: async () => {
        twice++;
        return hold;
      },
    };
    const again = recordingHost();
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

  test('an awaited call never asks the §2.8 running-job lookup — a repeat of an awaited tool simply runs again', async () => {
    let twice = 0;
    const read: Function = {
      definition: { name: 'readIt', description: 'r', parameters: { type: 'object', properties: {} } },
      call: async () => {
        twice++;
        return 'content';
      },
    };
    const { host, conversions, runningLookups } = recordingHost();
    await runLoop({
      fns: [read],
      host,
      steps: [
        () => toolCallStep('readIt', ['tc-1'], '{"id":"a"}'),
        () => toolCallStep('readIt', ['tc-2'], '{"id":"a"}'),
        () => textStep('x'),
      ],
    });
    expect(twice).toBe(2);
    expect(conversions).toHaveLength(0);
    expect(runningLookups).toHaveLength(0);
  });

  test('a declared dedupeKey is the identity, and the arguments hash is key-order independent', () => {
    expect(ToolBudget.argsKey({ a: 1, b: [1, { c: 2 }] })).toBe(ToolBudget.argsKey({ b: [1, { c: 2 }], a: 1 }));
    expect(ToolBudget.argsKey({ a: 1 })).not.toBe(ToolBudget.argsKey({ a: 2 }));
  });

  test('phases: the one a kind-declared call reports at its start names the yield sentence; later ones reach the host', async () => {
    let ctxSeen: ToolCallContext | undefined;
    const phased: Function = {
      definition: { name: 'materialize', description: 'm', parameters: { type: 'object', properties: {} } },
      job: { kind: 'shell' },
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
    const { host, conversions } = recordingHost();
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
      job: { kind: 'shell' },
      call: async (args) => {
        received = args;
        return new Promise(() => undefined);
      },
    };
    const { host, conversions, runningLookups } = recordingHost();
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

  test('the `task` label is split off an AWAITED call too — the tool receives its own arguments exactly as declared', async () => {
    let received: unknown;
    const read: Function = {
      definition: {
        name: 'readIt',
        description: 'r',
        parameters: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
      },
      call: async (args) => {
        received = args;
        return 'content';
      },
    };
    const run = await runLoop({
      fns: [read],
      host: recordingHost().host,
      steps: [() => toolCallStep('readIt', ['tc-1'], '{"id":"a","task":"Reading the note"}'), () => textStep('x')],
    });
    expect(received).toEqual({ id: 'a' });
    expect(run.toolInvocations[0].input).toEqual({ id: 'a' });
    expect(run.toolInvocations[0].converted).toBeUndefined();
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
      job: { kind: 'shell' },
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
      const { host, conversions } = recordingHost();
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

  test('a failing AWAITED call follows the failure path exactly as an unbudgeted call would', async () => {
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

/**
 * EVERY DELIVERABLE IS AWAITED WHOLE — the old per-tool "always awaited" declaration
 * (`softBudgetMs: Infinity`, §M.19's allowlist) retired with the hint: a document write, a deck, a
 * picture, a catalog check needs NO declaration to land in the reply. The set is the class, never
 * a per-tool number.
 *
 * RED at the pre-rule executor under a 250 ms N: the undeclared write converted at N and the
 * model read the hand-off.
 */
describe('every deliverable is awaited whole — no per-tool declaration (the §M.19 allowlist retired)', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  const OLD_N_MS = 250;

  beforeEach(() => {
    process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = String(OLD_N_MS);
  });

  afterAll(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  const writeTool = (opts: { ms: number; job?: Function['job']; name?: string }) => {
    const state = { settled: 0 };
    const fn: Function = {
      definition: {
        name: opts.name ?? 'writeDocument',
        description: 'writes the document the user asked for',
        parameters: { type: 'object', properties: {} },
      },
      ...(opts.job ? { job: opts.job } : {}),
      call: async () => {
        await sleep(opts.ms);
        state.settled++;
        return { written: true, id: 'doc-1' };
      },
    };
    return { fn, state };
  };

  test('a 1.5 s document write with NO declaration is AWAITED: the boundary comes after the write, the model reads the write’s own result, the host is never called', async () => {
    const { fn, state } = writeTool({ ms: 1_500 });
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: [fn],
      host,
      steps: [() => toolCallStep('writeDocument', ['tc-1']), () => textStep('here it is')],
    });

    expect(run.prompts).toHaveLength(2);
    // The boundary came AFTER the write (6× past the old N), not at N.
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(1_400);
    expect(state.settled).toBe(1);
    expect(conversions).toHaveLength(0);
    const toolResults = toolResultTexts(run.prompts[1].prompt);
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toContain('"written":true');
    expect(toolResults[0]).not.toContain('Started in the background');
    expect(run.toolInvocations[0].converted).toBeUndefined();
    expect(run.toolInvocations[0].data).toEqual({ written: true, id: 'doc-1' });
    expect(run.text).toBe('here it is');
  }, 20_000);

  test('a kind-declared call beside it converts at its start — the set is the class, never a per-tool number', async () => {
    const routine = writeTool({ ms: 1_000, job: { kind: 'routine' }, name: 'runTheRoutine' });
    const routineHost = recordingHost();
    const routineRun = await runLoop({
      fns: [routine.fn],
      host: routineHost.host,
      steps: [() => toolCallStep('runTheRoutine', ['tc-1']), () => textStep('started')],
    });
    expect(routineHost.conversions).toHaveLength(1);
    expect(routineHost.conversions[0].kind).toBe('routine');
    expect(routineRun.prompts[1].atMs).toBeLessThan(500);
    await expect(routineHost.conversions[0].promise).resolves.toEqual({ written: true, id: 'doc-1' });

    const plain = writeTool({ ms: 1_000, name: 'plainTool' });
    const plainHost = recordingHost();
    const plainRun = await runLoop({
      fns: [plain.fn],
      host: plainHost.host,
      steps: [() => toolCallStep('plainTool', ['tc-2']), () => textStep('done')],
    });
    expect(plainHost.conversions).toHaveLength(0);
    expect(plainRun.prompts[1].atMs).toBeGreaterThanOrEqual(900);
    expect(plain.state.settled).toBe(1);
  }, 20_000);
});

/**
 * A call made THROUGH A DISPATCHING FUNCTION takes its TARGET's declaration and identity
 * (`Function.dispatchTarget`): a lazy-loading skill catalog's `useSkill` names the tool it reaches;
 * the executor reads `job` / `hardBudgetMs` and the dedupe identity off that tool with its own
 * arguments, and the node settles under that tool's name with its outcome intact — so a tool keeps
 * its class on its first use through the catalog exactly as it does once pinned and called
 * directly (§M.23 stage A, carried onto the rule).
 *
 * RED at the pre-rule executor: a target with no declaration converted at the loop's N through the
 * dispatcher, named `useSkill` and keyed on the dispatch's whole input.
 */
describe('a dispatching function takes its TARGET’s declaration and identity (`Function.dispatchTarget`)', () => {
  const savedSoft = process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
  const OLD_N_MS = 250;

  beforeEach(() => {
    process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = String(OLD_N_MS);
  });

  afterAll(() => {
    if (savedSoft === undefined) {
      delete process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS;
    } else {
      process.env.CONVERSATION_TOOL_SOFT_BUDGET_MS = savedSoft;
    }
  });

  /** The catalog: one deferred skill behind the real dispatcher, its tools reached through `useSkill`. */
  const catalog = (fns: Function[]) =>
    new SkillDispatcherSkill([
      {
        getId: () => 'writer',
        getName: () => 'Writer',
        getSystemMessages: () => '',
        getFunctions: () => fns,
        getMessageModerators: () => [] as MessageModerator[],
      },
    ]);
  const dispatch = (tool: string, args: Record<string, unknown>, task?: string) =>
    JSON.stringify({ skill: 'writer', tool, args, ...(task ? { task } : {}) });
  const tool = (name: string, properties: Record<string, unknown> = {}): Function['definition'] => ({
    name,
    description: name,
    parameters: { type: 'object', properties },
  });

  test('a target with no declaration is AWAITED through the dispatcher: the boundary waits for it, the model reads the tool’s own result, the host is never asked', async () => {
    const state = { settled: 0 };
    const write: Function = {
      definition: tool('writeDocument', { title: { type: 'string' } }),
      call: async ({ title }: { title: string }) => {
        await sleep(1_500);
        state.settled++;
        return { written: true, title };
      },
    };
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns: catalog([write]).getFunctions(),
      host,
      steps: [
        () => toolCallStep('useSkill', ['tc-1'], dispatch('writeDocument', { title: 'Notes' })),
        () => textStep('here it is'),
      ],
    });

    expect(run.prompts).toHaveLength(2);
    // The boundary came AFTER the write (6× past the old N), not at N.
    expect(run.prompts[1].atMs).toBeGreaterThanOrEqual(1_400);
    expect(state.settled).toBe(1);
    expect(conversions).toHaveLength(0);
    const toolResults = toolResultTexts(run.prompts[1].prompt);
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toContain('"written": true');
    expect(toolResults[0]).not.toContain('Started in the background');
    expect(run.toolInvocations[0].converted).toBeUndefined();
    expect(run.text).toBe('here it is');
  }, 20_000);

  test('a `job`-declared target converts at its start through the dispatcher — the conversion named and keyed by the TARGET (its name, its own arguments, its kind, its HARD ceiling), and a repeat through the dispatcher is the same work', async () => {
    let resolveTool!: (v: unknown) => void;
    const long: Function = {
      definition: tool('longSearch', { q: { type: 'string' } }),
      job: { kind: 'agent' },
      hardBudgetMs: 60_000,
      call: () => new Promise((resolve) => (resolveTool = resolve)),
    };
    const { host, conversions, runningLookups } = recordingHost();
    const run = await runLoop({
      fns: catalog([long]).getFunctions(),
      host,
      steps: [
        () => toolCallStep('useSkill', ['tc-1'], dispatch('longSearch', { q: 'a' })),
        () => toolCallStep('useSkill', ['tc-2'], dispatch('longSearch', { q: 'a' }, 'Searching again')),
        () => textStep('went on'),
      ],
    });
    expect(run.prompts[1].atMs).toBeLessThan(500);
    expect(conversions).toHaveLength(1);
    expect(conversions[0].name).toBe('longSearch');
    expect(conversions[0].kind).toBe('agent');
    expect(conversions[0].args).toEqual({ q: 'a' });
    expect(conversions[0].hardBudgetMs).toBe(60_000);
    expect(conversions[0].dedupeKey).toBe(ToolBudget.argsKey({ q: 'a' }));
    // The repeat through the dispatcher is deduped on the target's identity (§2.8), label or not.
    expect(runningLookups).toHaveLength(2);
    expect(runningLookups[0]).toBe(runningLookups[1]);
    expect(run.toolInvocations[1].converted).toMatchObject({ jobId: 'job-1', deduped: true });
    resolveTool('late');
    await expect(conversions[0].promise).resolves.toBe('late');
  });

  test('the node settles under the TARGET’s name with its outcome intact: the settle relabels `useSkill` to the tool that ran (the tool’s own relabel when it gave one), the detail rides through, the invocation record carries the same', async () => {
    const save: Function = {
      definition: tool('saveNote', { title: { type: 'string' } }),
      getTimelineDetail: (args: { title?: string }) => args.title,
      getTimelineOutcome: (_args: unknown, result: unknown) =>
        typeof result === 'string' && result.startsWith('Saved')
          ? { detail: { text: 'Monday', href: 'app://notes/1' } }
          : { ok: false, name: 'saveNote:refused' },
      call: async ({ title }: { title: string }) => (title === 'Monday' ? `Saved "${title}"` : 'Refused'),
    };
    const plain: Function = { definition: tool('plainTool'), call: async () => 'ok' };
    const fns = catalog([save, plain]).getFunctions();

    const saved = await runLoop({
      fns,
      host: recordingHost().host,
      steps: [() => toolCallStep('useSkill', ['tc-1'], dispatch('saveNote', { title: 'Monday' })), () => textStep('x')],
    });
    expect(saved.parts.find((p) => p.type === 'tool-settled')).toMatchObject({
      id: 'tc-1',
      toolName: 'useSkill',
      ok: true,
    });
    expect(saved.parts.find((p) => p.type === 'tool-outcome')).toEqual({
      type: 'tool-outcome',
      toolName: 'useSkill',
      name: 'saveNote',
      detail: { text: 'Monday', href: 'app://notes/1' },
    });
    expect(saved.toolInvocations[0].name).toBe('useSkill');
    expect(saved.toolInvocations[0].outcome).toEqual({
      name: 'saveNote',
      detail: { text: 'Monday', href: 'app://notes/1' },
    });

    // The target's own relabel wins over its name; its refusal settles the node errored; an outcome without a detail
    // settles with the target's OWN subject (its `getTimelineDetail` — the words a direct call shows).
    const refused = await runLoop({
      fns,
      host: recordingHost().host,
      steps: [() => toolCallStep('useSkill', ['tc-2'], dispatch('saveNote', { title: 'Nope' })), () => textStep('x')],
    });
    expect(refused.parts.find((p) => p.type === 'tool-settled')).toMatchObject({ id: 'tc-2', ok: false });
    expect(refused.parts.find((p) => p.type === 'tool-outcome')).toEqual({
      type: 'tool-outcome',
      toolName: 'useSkill',
      name: 'saveNote:refused',
      detail: 'Nope',
    });

    // A target with no outcome hook and no subject of its own still settles under its own name — with NO subject
    // (the empty detail): the call-time line was the dispatcher's ("Skill → plainTool"), never the target's.
    const named = await runLoop({
      fns,
      host: recordingHost().host,
      steps: [() => toolCallStep('useSkill', ['tc-3'], dispatch('plainTool', {})), () => textStep('x')],
    });
    expect(named.parts.find((p) => p.type === 'tool-outcome')).toEqual({
      type: 'tool-outcome',
      toolName: 'useSkill',
      name: 'plainTool',
      detail: '',
    });
  }, 20_000);

  test('the conversion carries the call’s outcome hook bound to its arguments (a dispatcher’s reaches its target’s) for the host to settle the job’s node with; a dispatch that names no tool runs under the dispatcher’s own name and declaration', async () => {
    const slow: Function = {
      definition: tool('slowWrite'),
      job: { kind: 'agent' },
      getTimelineOutcome: (_args: unknown, result: unknown) => ({ detail: `made ${String(result)}` }),
      call: async () => {
        await sleep(600);
        return 'doc-1';
      },
    };
    const fns = catalog([slow]).getFunctions();
    const { host, conversions } = recordingHost();
    const run = await runLoop({
      fns,
      host,
      steps: [() => toolCallStep('useSkill', ['tc-1'], dispatch('slowWrite', {})), () => textStep('started')],
    });
    expect(run.prompts[1].atMs).toBeLessThan(500);
    expect(conversions).toHaveLength(1);
    expect(conversions[0].name).toBe('slowWrite');
    expect(conversions[0].kind).toBe('agent');
    const result = await conversions[0].promise;
    expect(result).toBe('doc-1');
    await expect(conversions[0].outcome!(result)).resolves.toEqual({ name: 'slowWrite', detail: 'made doc-1' });

    // A direct call's conversion carries its own hook; a tool without one carries none.
    const direct: Function = {
      definition: tool('directWrite'),
      job: { kind: 'shell' },
      getTimelineOutcome: (_args: unknown, result: unknown) => ({ detail: `kept ${String(result)}` }),
      call: async () => {
        await sleep(600);
        return 'k-1';
      },
    };
    const bare: Function = {
      definition: tool('bareWrite'),
      job: { kind: 'shell' },
      call: async () => {
        await sleep(600);
        return 'b-1';
      },
    };
    const directHost = recordingHost();
    await runLoop({
      fns: [direct, bare],
      host: directHost.host,
      steps: [
        () => toolCallStep('directWrite', ['tc-4']),
        () => toolCallStep('bareWrite', ['tc-5']),
        () => textStep('started'),
      ],
    });
    expect(directHost.conversions.map((c) => c.name)).toEqual(['directWrite', 'bareWrite']);
    await expect(directHost.conversions[0].outcome!('k-1')).resolves.toEqual({ detail: 'kept k-1' });
    expect(directHost.conversions[1].outcome).toBeUndefined();

    // No tool reached: the dispatcher's own name and class (awaited), its own message as the result, no relabel.
    const missing = await runLoop({
      fns,
      host: recordingHost().host,
      steps: [() => toolCallStep('useSkill', ['tc-9'], dispatch('nothing', {})), () => textStep('x')],
    });
    expect(toolResultTexts(missing.prompts[1].prompt)[0]).toContain('has no dispatcher-reachable tool named "nothing"');
    expect(missing.parts.find((p) => p.type === 'tool-outcome')).toBeUndefined();
    expect(missing.toolInvocations[0].converted).toBeUndefined();
  }, 20_000);
});

/**
 * THE SECOND HALF OF THE CLASS RULE (plans/DECISIONS.md 630 — "the reply stays open with the job's progress
 * and its stop handle until the deliverable lands"; row 2301): an AWAITED call's progress words and its own
 * Stop. The host observes every call the executor runs through its call hooks (`ToolBudgetHost.started` /
 * `phase` / `settled`):
 *  - an awaited tool's phases reach the host's `phase` hook in order, under the call's id — the node's words;
 *  - a converted call's phases keep flowing to the JOB (the conversion's `onPhase`) unchanged — the host's
 *    hook sees only what the tool said before the conversion (the hand-off's own `phase`);
 *  - a phase reported after the call settled reaches no one — nothing is written after the settle;
 *  - `started` hands the host an awaited call's abort at its start: firing it fires the call's own
 *    `ToolCallContext.signal`, the tool that honors it rejects, the call settles as a failure the model reads
 *    (the executor's failure path), and `settled` releases the handle;
 *  - a kind-declared call never reaches `started` / `settled`: its abort rides the conversion (D9).
 * RED at the pre-build executor: no hook exists — the host's records stay empty.
 */
describe('the awaited call’s progress words and its own Stop (DECISIONS 630, the second half)', () => {
  type CallRef = { toolCallId: string; name: string };
  type StartedCall = CallRef & { abort: () => void; signal: AbortSignal };
  /** The recording host with the call hooks — what thought-server's turn supplies. */
  const observingHost = () => {
    const base = recordingHost();
    const started: StartedCall[] = [];
    const phases: Array<{ call: CallRef; phase: ToolPhase }> = [];
    const settled: CallRef[] = [];
    const host = {
      ...base.host,
      started: (call: StartedCall) => started.push(call),
      phase: (call: CallRef, phase: ToolPhase) => phases.push({ call, phase }),
      settled: (call: CallRef) => settled.push(call),
    } as ToolBudgetHost;
    return { ...base, host, started, phases, settled };
  };

  test('an AWAITED tool reporting two phases reaches the host’s `phase` hook in order, under the call’s id; the host never converts; the model reads the tool’s own result', async () => {
    const deck: Function = {
      definition: { name: 'makeDeck', description: 'd', parameters: { type: 'object', properties: {} } },
      call: async (_args, ctx) => {
        ctx?.onPhase({ on: 'Laying out the slides', detail: '1 of 8' });
        await sleep(40);
        ctx?.onPhase({ on: 'Laying out the slides', detail: '3 of 8' });
        return { deckId: 'deck-1' };
      },
    };
    const { host, conversions, phases, started, settled } = observingHost();
    const run = await runLoop({
      fns: [deck],
      host,
      steps: [() => toolCallStep('makeDeck', ['tc-1']), () => textStep('here is the deck')],
    });
    expect(phases).toEqual([
      { call: { toolCallId: 'tc-1', name: 'makeDeck' }, phase: { on: 'Laying out the slides', detail: '1 of 8' } },
      { call: { toolCallId: 'tc-1', name: 'makeDeck' }, phase: { on: 'Laying out the slides', detail: '3 of 8' } },
    ]);
    expect(conversions).toHaveLength(0);
    expect(started.map((c) => ({ toolCallId: c.toolCallId, name: c.name }))).toEqual([
      { toolCallId: 'tc-1', name: 'makeDeck' },
    ]);
    expect(settled).toEqual([{ toolCallId: 'tc-1', name: 'makeDeck' }]);
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toContain('"deckId":"deck-1"');
    expect(run.text).toBe('here is the deck');
  });

  test('a converted call’s phases keep flowing to the JOB unchanged — the host’s `phase` hook sees only the phase said before the conversion', async () => {
    const phased: Function = {
      definition: { name: 'materialize', description: 'm', parameters: { type: 'object', properties: {} } },
      job: { kind: 'shell' },
      call: async (_args, ctx) => {
        ctx?.onPhase({ on: 'Setting up the workspace', detail: 'getting a machine to work on' });
        await sleep(300);
        ctx?.onPhase({ on: 'Setting up the workspace', detail: 'getting the code (n3xa app)' });
        await sleep(100);
        return { path: '/repo' };
      },
    };
    const phasesAfter: ToolPhase[] = [];
    const observed = observingHost();
    const host = {
      ...observed.host,
      convert: async (call: ToolBudgetConversion) => {
        call.onPhase((phase) => phasesAfter.push(phase));
        return observed.host.convert(call);
      },
    } as ToolBudgetHost;
    await runLoop({ fns: [phased], host, steps: [() => toolCallStep('materialize', ['tc-1']), () => textStep('x')] });
    await sleep(500);
    expect(observed.conversions[0].phase).toEqual({
      on: 'Setting up the workspace',
      detail: 'getting a machine to work on',
    });
    expect(phasesAfter).toEqual([{ on: 'Setting up the workspace', detail: 'getting the code (n3xa app)' }]);
    expect(observed.phases).toEqual([
      {
        call: { toolCallId: 'tc-1', name: 'materialize' },
        phase: { on: 'Setting up the workspace', detail: 'getting a machine to work on' },
      },
    ]);
    // A job by kind: its abort rides the conversion — the awaited-call hooks never fire for it.
    expect(observed.started).toEqual([]);
    expect(observed.settled).toEqual([]);
  });

  test('a phase reported after the call settled reaches no one — nothing is written after the settle', async () => {
    let late: (() => void) | undefined;
    const sloppy: Function = {
      definition: { name: 'sloppy', description: 's', parameters: { type: 'object', properties: {} } },
      call: async (_args, ctx) => {
        ctx?.onPhase({ on: 'Checking the catalog' });
        late = () => ctx?.onPhase({ on: 'Still checking', detail: 'after the fact' });
        return 'checked';
      },
    };
    const { host, phases, settled } = observingHost();
    await runLoop({ fns: [sloppy], host, steps: [() => toolCallStep('sloppy', ['tc-1']), () => textStep('x')] });
    expect(settled).toHaveLength(1);
    late!();
    expect(phases.map((p) => p.phase)).toEqual([{ on: 'Checking the catalog' }]);
  });

  test('the per-node Stop: the host’s `started` abort fires the call’s own signal; the tool that honors it rejects; the call settles as a failure the model reads; `settled` releases the handle', async () => {
    const seen: string[] = [];
    const deck: Function = {
      definition: { name: 'makeDeck', description: 'd', parameters: { type: 'object', properties: {} } },
      call: (_args, ctx) =>
        new Promise((_resolve, reject) => {
          ctx?.signal.addEventListener('abort', () => {
            seen.push('signal');
            reject(new Error('Stopped laying out the slides'));
          });
        }),
    };
    const { host, conversions, started, settled } = observingHost();
    const loop = runLoop({
      fns: [deck],
      host,
      steps: [() => toolCallStep('makeDeck', ['tc-1']), () => textStep('all right, no deck')],
    });
    await sleep(100);
    expect(started).toHaveLength(1);
    expect(started[0].signal.aborted).toBe(false);
    expect(settled).toEqual([]);
    started[0].abort();
    const run = await loop;
    expect(started[0].signal.aborted).toBe(true);
    expect(seen).toEqual(['signal']);
    expect(settled).toEqual([{ toolCallId: 'tc-1', name: 'makeDeck' }]);
    expect(conversions).toHaveLength(0);
    expect(run.parts.find((p) => p.type === 'tool-settled')).toMatchObject({ id: 'tc-1', ok: false });
    expect(run.toolInvocations[0].ok).toBe(false);
    expect(run.toolInvocations[0].error?.message).toBe('Stopped laying out the slides');
    expect(toolResultTexts(run.prompts[1].prompt)[0]).toContain('Stopped laying out the slides');
    expect(run.text).toBe('all right, no deck');
  });
});
