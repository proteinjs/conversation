import { createAnthropic } from '@ai-sdk/anthropic';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance, type DrainedInput } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE SIDE UTTERANCE SHARES THE STEP'S CACHE PREFIX. The side utterance ({@link Utterance} — the
 * nudge's line, said mid-round) is a second request over the round's transcript, sent beside the
 * step it is said during. A provider's prompt cache is keyed on the request's prefix bytes in
 * order — the tool roster first, then the system prefix, then the messages — so the side call
 * reads the entry the step wrote only when both requests carry the same roster and the same system
 * prefix up to the same breakpoint. Sent without the roster, the side call writes a prefix no step
 * ever reads, and the whole system region is paid at the cache-write price again.
 *
 * The proof is at the wire: the real Anthropic provider over a scripted transport that records
 * every request body and answers as the provider's cache would — a system-tier prefix (the model,
 * the roster, the system blocks up to the breakpoint) seen before is a cache READ of its size, a
 * new one a cache WRITE — and the turn's usage rows (`UsageData.steps`) carry what each request
 * paid.
 *
 * RED before the envelope: the side call carried no tools, so its prefix matched nothing the step
 * sent — it wrote the system region again instead of reading what the step wrote.
 */

const TIMEOUT = 30_000;
const LINE = 'Still comparing the two.';
const ANSWER = 'THE ANSWER';
const NUDGE: DrainedInput = {
  text: 'HARNESS NUDGE: the user has heard nothing from you for a while.',
  ask: 'In one short sentence, say what you are doing right now.',
};
/** How long the first step is silent on the wire — the nudge lands inside it. */
const STALL_MS = 600;

type SystemBlock = { type: string; text: string; cache_control?: unknown };
type WireBody = {
  model: string;
  tools?: unknown[];
  tool_choice?: unknown;
  system?: SystemBlock[];
  messages: Array<{ role: string; content: unknown }>;
};

/** One server-sent event, as the provider's stream parser reads it. */
const event = (payload: Record<string, unknown>): string =>
  `event: ${String(payload.type)}\ndata: ${JSON.stringify(payload)}\n\n`;

const messageStart = (model: string, usage: { read: number; write: number }): string =>
  event({
    type: 'message_start',
    message: {
      id: 'msg_scripted',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      usage: {
        input_tokens: 12,
        cache_creation_input_tokens: usage.write,
        cache_read_input_tokens: usage.read,
        output_tokens: 1,
      },
    },
  });

/** A text answer: one text block, the model's own stop. */
const textAnswer = (model: string, text: string, usage: { read: number; write: number }): string =>
  [
    messageStart(model, usage),
    event({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    event({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }),
    event({ type: 'content_block_stop', index: 0 }),
    event({
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 5 },
    }),
    event({ type: 'message_stop' }),
  ].join('');

/** A tool-call answer: the model asks for a tool instead of writing text. */
const toolCallAnswer = (model: string, toolName: string, usage: { read: number; write: number }): string =>
  [
    messageStart(model, usage),
    event({
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'tool_use', id: 'toolu_scripted', name: toolName, input: {} },
    }),
    event({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{}' } }),
    event({ type: 'content_block_stop', index: 0 }),
    event({
      type: 'message_delta',
      delta: { stop_reason: 'tool_use', stop_sequence: null },
      usage: { output_tokens: 3 },
    }),
    event({ type: 'message_stop' }),
  ].join('');

/**
 * The transport under the real provider: records every request body; answers the FIRST step with
 * a tool call after a silence (the nudge lands inside it), the side ask with the line (or, when
 * told to, a tool call), and the step after the tool with the answer; bills as the provider's
 * cache does — one entry per system-tier prefix.
 */
class ScriptedAnthropicTransport {
  readonly requests: WireBody[] = [];
  /** Answer the side ask with a call to this tool instead of a line. */
  answerSideWithTool?: string;
  private readonly entries = new Map<string, number>();
  private steps = 0;

  readonly fetch = async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body)) as WireBody;
    this.requests.push(body);
    const usage = this.bill(body);
    const side = ScriptedAnthropicTransport.isSideRequest(body);
    let answer: string;
    if (side) {
      answer = this.answerSideWithTool
        ? toolCallAnswer(body.model, this.answerSideWithTool, usage)
        : textAnswer(body.model, LINE, usage);
    } else if (this.steps++ === 0) {
      // The silent first step: nothing on the wire for the stall, then the tool call.
      await new Promise((resolve) => setTimeout(resolve, STALL_MS));
      answer = toolCallAnswer(body.model, 'doWork', usage);
    } else {
      answer = textAnswer(body.model, ANSWER, usage);
    }
    return new Response(answer, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };

  /** Whether a request body is the side ask (its last user text ends with the request marker). */
  static isSideRequest(body: WireBody): boolean {
    const last = body.messages[body.messages.length - 1];
    if (!last || last.role !== 'user') {
      return false;
    }
    const text =
      typeof last.content === 'string'
        ? last.content
        : (last.content as Array<{ type?: string; text?: string }>)
            .map((p) => (p.type === 'text' ? p.text ?? '' : ''))
            .join('');
    return text.trimEnd().endsWith(Utterance.REPLY_WITH_THE_LINE_ONLY);
  }

  /**
   * The provider's billing for one request, the documented cache tiers mirrored: a cache entry is
   * keyed on the model, the tool roster and the system blocks up to the system breakpoint (in that
   * order — the first bytes of the rendered prompt); a prefix seen before is read, a new one is
   * written. Tokens are sized by the prefix's bytes. The messages tier is not modeled: a changed
   * tool choice or thinking setting never touches the tools and system entry.
   */
  private bill(body: WireBody): { read: number; write: number } {
    const prefix = ScriptedAnthropicTransport.systemTierPrefix(body);
    if (!prefix) {
      return { read: 0, write: 0 };
    }
    const tokens = Math.ceil(prefix.length / 4);
    if (this.entries.has(prefix)) {
      return { read: tokens, write: 0 };
    }
    this.entries.set(prefix, tokens);
    return { read: 0, write: tokens };
  }

  private static systemTierPrefix(body: WireBody): string | undefined {
    const system = body.system ?? [];
    let breakpoint = -1;
    system.forEach((block, i) => {
      if (block.cache_control) {
        breakpoint = i;
      }
    });
    if (breakpoint < 0) {
      return undefined;
    }
    return JSON.stringify({ model: body.model, tools: body.tools ?? null, system: system.slice(0, breakpoint + 1) });
  }
}

const toolCalls = { count: 0 };

const conversation = (name: string) =>
  new Conversation({
    modelData: fixtureModelData,
    name,
    logLevel: 'error',
    limits: { enforceLimits: false },
    // Two skills, so the system prefix is two blocks: the breakpoint lands on the LAST one, and a
    // mark moved to the first is a different prefix (the mutation the suite must catch).
    skills: [
      {
        getId: () => 'persona',
        getName: () => 'Persona',
        getSystemMessages: () => ['You are a careful assistant. Compare things fairly and keep answers short.'],
        getMessageModerators: () => [],
        getFunctions: () => [],
      } as never,
      {
        getId: () => 'do-work',
        getName: () => 'DoWork',
        getSystemMessages: () => ['When two things are alike, say so before listing their differences.'],
        getMessageModerators: () => [],
        getFunctions: () => [
          {
            definition: { name: 'doWork', description: 'work', parameters: { type: 'object', properties: {} } },
            call: async () => {
              toolCalls.count++;
              return { ok: true };
            },
          },
        ],
      } as never,
    ],
  });

/** A turn with nothing arriving mid-turn. */
const idleTurn = (): Pick<
  GenerateStreamParams,
  'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes'
> => ({
  drainInjectedContext: () => [],
  peekInjectedContext: () => false,
  inputArrived: () => new Promise<void>(() => {}),
  absorbExitNotes: true,
});

type Part = { type: string; text?: string };

async function drain(fullStream: AsyncIterable<unknown>): Promise<Part[]> {
  const parts: Part[] = [];
  for await (const part of fullStream as AsyncIterable<Part>) {
    parts.push(part);
  }
  return parts;
}

/** One turn: the nudge lands 100 ms into the silent first step; the side line is said beside it. */
async function oneTurn(transport: ScriptedAnthropicTransport, name: string) {
  const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })('claude-opus-5');
  let nudged = false;
  const result = await conversation(name).generateStream({
    messages: ['compare sql and nosql'],
    model: model as never,
    reasoningEffort: 'auto',
    ...idleTurn(),
    sideUtterance: () =>
      nudged
        ? new Promise<DrainedInput>(() => {})
        : new Promise<DrainedInput>((resolve) => {
            nudged = true;
            setTimeout(() => resolve(NUDGE), 100);
          }),
  });
  const parts = await drain(result.fullStream);
  const usage = await result.usage;
  return { parts, usage };
}

describe('the side utterance rides the same cache prefix as the step it is said beside', () => {
  beforeEach(() => {
    toolCalls.count = 0;
  });

  test(
    'THE ENVELOPE: the side request and the step requests carry byte-identical tool rosters and system prefixes up to the system breakpoint',
    async () => {
      const transport = new ScriptedAnthropicTransport();
      const { parts } = await oneTurn(transport, 'side-prefix-envelope');

      // step 1 (silent, then the tool) · the side ask (inside the silence) · step 2 (the answer).
      expect(transport.requests).toHaveLength(3);
      const [step1, side, step2] = transport.requests;
      expect(ScriptedAnthropicTransport.isSideRequest(step1)).toBe(false);
      expect(ScriptedAnthropicTransport.isSideRequest(side)).toBe(true);
      expect(ScriptedAnthropicTransport.isSideRequest(step2)).toBe(false);
      // The step's roster is real (the skill's function and the provider's search tool).
      expect((step1.tools ?? []).length).toBeGreaterThan(0);
      // The same model, the same roster, the same system blocks (breakpoints included) — the
      // bytes the cache key begins with.
      expect(side.model).toBe(step1.model);
      expect(JSON.stringify(side.tools)).toBe(JSON.stringify(step1.tools));
      expect(JSON.stringify(side.system)).toBe(JSON.stringify(step1.system));
      // The breakpoint is there to read under: the system prefix is several blocks and the LAST
      // one carries the mark on both — nothing earlier does.
      const system = (body: WireBody) => body.system ?? [];
      expect(system(step1).length).toBeGreaterThan(1);
      for (const body of [step1, side]) {
        const marks = system(body).map((block) => (block.cache_control ? 'mark' : '-'));
        expect(marks[marks.length - 1]).toBe('mark');
        expect(marks.slice(0, -1).every((mark) => mark === '-')).toBe(true);
      }
      // The line landed as its own part, never a step.
      const sideParts = parts.filter((part) => part.type === 'side-utterance');
      expect(sideParts).toHaveLength(1);
      expect(sideParts[0].text).toBe(LINE);
    },
    TIMEOUT
  );

  test(
    "THE USAGE ROWS: the side call's cache READ covers the step's cache WRITE — the second request reads what the first wrote, nothing of the system region is written twice",
    async () => {
      const transport = new ScriptedAnthropicTransport();
      const { usage } = await oneTurn(transport, 'side-prefix-usage');

      const steps = usage.steps ?? [];
      expect(steps).toHaveLength(3);
      // One request paid the write (a cold cache; nothing was in it) — the first step's.
      const writers = steps.filter((step) => (step.cacheWriteTokens ?? 0) > 0);
      expect(writers).toHaveLength(1);
      const written = writers[0].cacheWriteTokens ?? 0;
      expect(writers[0].cachedInputTokens ?? 0).toBe(0);
      // Every other request — the side call and the step after the tool — read it back whole.
      for (const step of steps.filter((step) => step !== writers[0])) {
        expect(step.cachedInputTokens ?? 0).toBeGreaterThanOrEqual(written);
        expect(step.cacheWriteTokens ?? 0).toBe(0);
      }
    },
    TIMEOUT
  );

  test(
    'NOTHING RUNS ON THE ASK: a tool-call answer to the side ask runs no tool — no line is said, and the step after carries no framing',
    async () => {
      const transport = new ScriptedAnthropicTransport();
      transport.answerSideWithTool = 'doWork';
      const { parts } = await oneTurn(transport, 'side-prefix-tool-answer');

      expect(transport.requests).toHaveLength(3);
      // The roster rode the ask; the call it drew ran nothing — the one run is step 1's own call.
      expect(toolCalls.count).toBe(1);
      // No line, no part; step 2's request carries no framing (nothing about the nudge, no
      // continue instruction).
      expect(parts.some((part) => part.type === 'side-utterance')).toBe(false);
      const step2 = transport.requests[2];
      const body = JSON.stringify(step2.messages);
      expect(body).not.toContain(NUDGE.text);
      expect(body).not.toContain('You have already told the user');
    },
    TIMEOUT
  );
});
