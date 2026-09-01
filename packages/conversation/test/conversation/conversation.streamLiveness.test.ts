import { Conversation } from '../../src/Conversation';

/**
 * Deterministic unit tests for the stream liveness guard (guardStreamLiveness).
 *
 * Root-caused live on the R5 demo estate (2026-09-01): the SDK fullStream goes quiet while a
 * LOCALLY-EXECUTED tool's execute() runs (editThoughts queued behind a live editor lease, a dev
 * tool running a build) — there is no model connection in that window, but the guard counted the
 * quiet as model silence and aborted every tool execution that outlived the idle window at
 * exactly 300s as a fake "silent connection loss".
 *
 * Liveness must mean MODEL-stream liveness only:
 *  - a local tool call outstanding (tool-call seen, final tool-result not yet) SUSPENDS the race;
 *  - real silence — no outstanding local tool — still aborts honestly (the 2026-07-10 hang class);
 *  - provider server-executed tools (not in the local set) stay guarded.
 */

type GuardInternals = {
  guardStreamLiveness(
    stream: AsyncIterable<any>,
    controller: AbortController,
    modelString: string,
    locallyExecutedToolNames?: ReadonlySet<string>
  ): AsyncIterable<any>;
};

const IDLE_MS = 150;
const OUTLIVE_MS = 400; // comfortably past the idle window

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function* streamWithGap(parts: any[], gapBeforeIndex: number, gapMs: number): AsyncIterable<any> {
  for (let i = 0; i < parts.length; i++) {
    if (i === gapBeforeIndex) {
      await sleep(gapMs);
    }
    yield parts[i];
  }
}

async function collect(iterable: AsyncIterable<any>): Promise<any[]> {
  const out: any[] = [];
  for await (const part of iterable) {
    out.push(part);
  }
  return out;
}

describe('Conversation stream liveness guard', () => {
  let priorIdle: string | undefined;

  beforeAll(() => {
    priorIdle = process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
    process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = String(IDLE_MS);
  });

  afterAll(() => {
    if (priorIdle === undefined) {
      delete process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS;
    } else {
      process.env.CONVERSATION_STREAM_IDLE_TIMEOUT_MS = priorIdle;
    }
  });

  const guard = (
    stream: AsyncIterable<any>,
    controller: AbortController,
    localTools?: ReadonlySet<string>
  ): AsyncIterable<any> =>
    (new Conversation({ name: 'liveness-test' }) as unknown as GuardInternals).guardStreamLiveness(
      stream,
      controller,
      'test-model',
      localTools
    );

  test('a local tool execution outliving the idle window does NOT abort the stream (the R5 stall repro)', async () => {
    const parts = [
      { type: 'start' },
      { type: 'tool-input-start', id: 't1' },
      { type: 'tool-input-end', id: 't1' },
      { type: 'tool-call', toolCallId: 't1', toolName: 'editThoughts', input: {} },
      // gap here = execute() running (fence queue-wait / long dev tool) — longer than the window
      { type: 'tool-result', toolCallId: 't1', toolName: 'editThoughts', output: 'ok' },
      { type: 'finish-step', finishReason: 'tool-calls' },
      { type: 'finish' },
    ];
    const controller = new AbortController();
    const collected = await collect(
      guard(streamWithGap(parts, 4, OUTLIVE_MS), controller, new Set(['editThoughts']))
    );
    expect(collected.map((p) => p.type)).toEqual(parts.map((p) => p.type));
    expect(controller.signal.aborted).toBe(false);
  });

  test('real silence with NO outstanding local tool still aborts honestly (the guard still bites)', async () => {
    const parts = [
      { type: 'start' },
      { type: 'text-delta', delta: 'hi' },
      // gap here = genuine model silence
      { type: 'text-delta', delta: ' there' },
    ];
    const controller = new AbortController();
    await expect(collect(guard(streamWithGap(parts, 2, OUTLIVE_MS), controller))).rejects.toThrow(
      /Model stream stalled: no parts from test-model/
    );
    expect(controller.signal.aborted).toBe(true);
  });

  test('a provider server-executed tool (not in the local set) stays guarded', async () => {
    const parts = [
      { type: 'start' },
      { type: 'tool-call', toolCallId: 'w1', toolName: 'web_search', input: {} },
      // gap here = the provider connection dying mid server-tool — must still abort
      { type: 'tool-result', toolCallId: 'w1', toolName: 'web_search', output: 'results' },
    ];
    const controller = new AbortController();
    await expect(
      collect(guard(streamWithGap(parts, 2, OUTLIVE_MS), controller, new Set(['editThoughts'])))
    ).rejects.toThrow(/Model stream stalled/);
    expect(controller.signal.aborted).toBe(true);
  });

  test('the guard RE-ARMS after the local tool settles — silence after tool-result still aborts', async () => {
    const parts = [
      { type: 'start' },
      { type: 'tool-call', toolCallId: 't1', toolName: 'editThoughts', input: {} },
      { type: 'tool-result', toolCallId: 't1', toolName: 'editThoughts', output: 'ok' },
      // gap here = model silence on the NEXT step's call — guard must be re-armed
      { type: 'text-delta', delta: 'late' },
    ];
    const controller = new AbortController();
    await expect(
      collect(guard(streamWithGap(parts, 3, OUTLIVE_MS), controller, new Set(['editThoughts'])))
    ).rejects.toThrow(/Model stream stalled/);
    expect(controller.signal.aborted).toBe(true);
  });

  test('a preliminary tool-result does not re-arm the guard mid-execution', async () => {
    const parts = [
      { type: 'start' },
      { type: 'tool-call', toolCallId: 't1', toolName: 'editThoughts', input: {} },
      { type: 'tool-result', toolCallId: 't1', toolName: 'editThoughts', preliminary: true, output: 'partial' },
      // gap here = execution still running after a preliminary result — must NOT abort
      { type: 'tool-result', toolCallId: 't1', toolName: 'editThoughts', output: 'final' },
      { type: 'finish' },
    ];
    const controller = new AbortController();
    const collected = await collect(
      guard(streamWithGap(parts, 3, OUTLIVE_MS), controller, new Set(['editThoughts']))
    );
    expect(collected).toHaveLength(parts.length);
    expect(controller.signal.aborted).toBe(false);
  });
});
