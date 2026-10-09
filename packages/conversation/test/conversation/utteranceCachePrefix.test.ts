import { createAnthropic } from '@ai-sdk/anthropic';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { Utterance } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE ACKNOWLEDGMENT SHARES THE STEP'S CACHE PREFIX. The bounded utterance ({@link Utterance}) is
 * a second request over the same transcript, sent just before the step that takes the input in.
 * A provider's prompt cache is keyed on the request's prefix bytes in order — the tool roster
 * first, then the system prefix, then the messages — so the utterance's cache write serves the
 * step that follows only when both requests carry the same roster and the same system prefix up
 * to the same breakpoint. Sent without the roster, the utterance writes a prefix no step ever
 * reads, and the whole system region is paid at the cache-write price twice per turn.
 *
 * The proof is at the wire: the real Anthropic provider over a scripted transport that records
 * every request body and answers as the provider's cache would — a system-tier prefix (the model,
 * the roster, the system blocks up to the breakpoint) seen before is a cache READ of its size, a
 * new one a cache WRITE — and the turn's usage rows (`UsageData.steps`) carry what each request
 * paid.
 *
 * RED before the fix: the utterance carried no tools, so its prefix matched nothing the step
 * sent — the step read 0 of what the utterance wrote.
 */

const TIMEOUT = 30_000;
const LINE = 'Got it — comparing the two now.';
const ANSWER = 'THE ANSWER';

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

/** A tool-call answer: the model asks for a tool instead of writing a line. */
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
 * The transport under the real provider: records every request body, answers the utterance ask
 * with the line (or, when told to, a tool call) and the step with the answer, and bills as the
 * provider's cache does — one entry per system-tier prefix.
 */
class ScriptedAnthropicTransport {
  readonly requests: WireBody[] = [];
  /** Answer the utterance ask with a call to this tool instead of a line. */
  answerUtteranceWithTool?: string;
  private readonly entries = new Map<string, number>();

  readonly fetch = async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body)) as WireBody;
    this.requests.push(body);
    const usage = this.bill(body);
    const utterance = ScriptedAnthropicTransport.isUtteranceRequest(body);
    const answer =
      utterance && this.answerUtteranceWithTool
        ? toolCallAnswer(body.model, this.answerUtteranceWithTool, usage)
        : textAnswer(body.model, utterance ? LINE : ANSWER, usage);
    return new Response(answer, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };

  /** Whether a request body is the utterance ask (its last user text ends with the request marker). */
  static isUtteranceRequest(body: WireBody): boolean {
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

/** A turn with the bounded utterance on and nothing arriving mid-turn. */
const idleTurn = (): Pick<
  GenerateStreamParams,
  'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes' | 'utterance'
> => ({
  drainInjectedContext: () => [],
  peekInjectedContext: () => false,
  inputArrived: () => new Promise<void>(() => {}),
  absorbExitNotes: true,
  utterance: true,
});

async function drain(fullStream: AsyncIterable<unknown>): Promise<Array<{ type: string; utterance?: true }>> {
  const parts: Array<{ type: string; utterance?: true }> = [];
  for await (const part of fullStream as AsyncIterable<{ type: string; utterance?: true }>) {
    parts.push(part);
  }
  return parts;
}

async function oneTurn(transport: ScriptedAnthropicTransport, name: string) {
  const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })('claude-opus-5');
  const result = await conversation(name).generateStream({
    messages: ['compare sql and nosql'],
    model: model as never,
    reasoningEffort: 'auto',
    ...idleTurn(),
  });
  const parts = await drain(result.fullStream);
  const usage = await result.usage;
  return { parts, usage };
}

describe('the bounded utterance rides the same cache prefix as the step it precedes', () => {
  beforeEach(() => {
    toolCalls.count = 0;
  });

  test(
    'THE ENVELOPE: the utterance request and the step request carry byte-identical tool rosters and system prefixes up to the system breakpoint',
    async () => {
      const transport = new ScriptedAnthropicTransport();
      const { parts } = await oneTurn(transport, 'utterance-prefix-envelope');

      expect(transport.requests).toHaveLength(2);
      const [utterance, step] = transport.requests;
      expect(ScriptedAnthropicTransport.isUtteranceRequest(utterance)).toBe(true);
      expect(ScriptedAnthropicTransport.isUtteranceRequest(step)).toBe(false);
      // The step's roster is real (the skill's function and the provider's search tool).
      expect((step.tools ?? []).length).toBeGreaterThan(0);
      // The same model, the same roster, the same system blocks (breakpoints included) — the
      // bytes the cache key begins with.
      expect(utterance.model).toBe(step.model);
      expect(JSON.stringify(utterance.tools)).toBe(JSON.stringify(step.tools));
      expect(JSON.stringify(utterance.system)).toBe(JSON.stringify(step.system));
      // The breakpoint is there to write under: the system prefix is several blocks and the LAST
      // one carries the mark on both — nothing earlier does.
      const system = (body: WireBody) => body.system ?? [];
      expect(system(step).length).toBeGreaterThan(1);
      for (const body of [utterance, step]) {
        const marks = system(body).map((block) => (block.cache_control ? 'mark' : '-'));
        expect(marks[marks.length - 1]).toBe('mark');
        expect(marks.slice(0, -1).every((mark) => mark === '-')).toBe(true);
      }
      // The line still streams first, as its own step.
      const firstFinish = parts.findIndex((part) => part.type === 'step-finish');
      expect(parts[firstFinish].utterance).toBe(true);
    },
    TIMEOUT
  );

  test(
    "THE USAGE ROWS: the step's cache READ covers the utterance's cache WRITE — the second request reads what the first wrote",
    async () => {
      const transport = new ScriptedAnthropicTransport();
      const { usage } = await oneTurn(transport, 'utterance-prefix-usage');

      const steps = usage.steps ?? [];
      expect(steps).toHaveLength(2);
      const [utterance, step] = steps;
      // The utterance paid the write (a cold cache; nothing was in it).
      expect(utterance.cacheWriteTokens).toBeGreaterThan(0);
      expect(utterance.cachedInputTokens).toBe(0);
      // The step read it back: nothing of the system region was written twice.
      expect(step.cachedInputTokens).toBeGreaterThanOrEqual(utterance.cacheWriteTokens);
      expect(step.cacheWriteTokens).toBe(0);
    },
    TIMEOUT
  );

  test(
    'NOTHING RUNS ON THE ASK: a tool-call answer to the utterance runs no tool — the step runs without its line',
    async () => {
      const transport = new ScriptedAnthropicTransport();
      transport.answerUtteranceWithTool = 'doWork';
      const { parts } = await oneTurn(transport, 'utterance-prefix-tool-answer');

      expect(transport.requests).toHaveLength(2);
      // The roster rode the ask; the call it drew ran nothing.
      expect(toolCalls.count).toBe(0);
      // No line, no utterance step; the step's request carries no framing (its last message is
      // the user's own, not the continue instruction).
      expect(parts.some((part) => part.type === 'step-finish' && part.utterance)).toBe(false);
      const step = transport.requests[1];
      const last = step.messages[step.messages.length - 1];
      expect(last.role).toBe('user');
      expect(JSON.stringify(last.content)).toContain('compare sql and nosql');
      expect(JSON.stringify(last.content)).not.toContain('You have already told the user');
    },
    TIMEOUT
  );
});
