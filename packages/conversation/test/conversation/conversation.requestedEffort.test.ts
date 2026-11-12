import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { createXai } from '@ai-sdk/xai';
import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Conversation, type GenerateStreamParams, type ReasoningEffort } from '../../src/Conversation';
import { RequestedEffort } from '../../src/RequestedEffort';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE REQUESTED EFFORT FOLLOWS THE MODEL — no network, no keys. A MockLanguageModelV3 stands in
 * as the provider (Conversation takes a model instance straight through resolveModel), so these
 * run the REAL wiring: Conversation → LlmTransportRetry.wrap → RequestedEffort.follow →
 * ForcedToolChoice.follow → model.
 *
 * The refusal is the provider's, recorded live 2026-09-22 (OpenAI, `gpt-6-astra`, the Responses
 * API): a request at `reasoning.effort: none` — then the bounded utterance's, the acknowledgment
 * call retired 2026-11-05; here the main step asked at that level — and the model answered HTTP
 * 400 with the clause below. Before this rule EVERY such request failed and paid a refused
 * request each time, and no per-level probe could see it (a probe runs only the levels a model
 * claims).
 *
 * Since 2026-09-29 the re-issue OMITS the effort (the provider's own default applies) instead of
 * picking the nearest listed level — the ruling: "leave the current functionality of omitting
 * effort so the api can default" — and the omission is a `step-start` warning on the stream.
 */

const TIMEOUT = 30_000;

/** The provider's verdict, in its own words. */
const ASTRA_CLAUSE =
  "Unsupported value: 'none' is not supported with the 'gpt-6-astra' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.";

/** xAI's verdict for grok-4.5, recorded live 2026-09-29 (`reasoning.effort: none` on the Responses API). */
const GROK_CLAUSE = 'This model does not support `reasoning_effort` value `none`.';

/** OpenAI's 400 for an unsupported parameter value, as the SDK surfaces it (the parsed body on `data`). */
const openAiRefusal = (message: string, param = 'reasoning.effort') => {
  const body = { error: { message, type: 'invalid_request_error', param, code: 'unsupported_value' } };
  return new APICallError({
    message,
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody: JSON.stringify(body),
    isRetryable: false,
    data: body,
  });
};

const otherBadRequest = () =>
  new APICallError({
    message: "Invalid value: 'flex'. Supported values are: 'auto', 'default', and 'priority'.",
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody: '',
    isRetryable: false,
    data: {
      error: { message: "Invalid value: 'flex'.", type: 'invalid_request_error', param: 'service_tier', code: null },
    },
  });

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const reasonedStep = (reasoning: string, text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'reasoning-start' as const, id: 'r1' },
    { type: 'reasoning-delta' as const, id: 'r1', delta: reasoning },
    { type: 'reasoning-end' as const, id: 'r1' },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

type Call = {
  prompt: Array<{ role: string; content: unknown }>;
  providerOptions?: Record<string, Record<string, unknown>>;
  toolChoice?: unknown;
};

const effortOf = (call: Call): unknown => call.providerOptions?.openai?.reasoningEffort;

/**
 * A model shaped like the provider's adapter: refuses the efforts in `refuses` with the provider's
 * own clause, answers the step with a reasoned answer.
 */
const scriptedModel = (opts: {
  provider: string;
  modelId: string;
  refuses?: (effort: unknown, call: Call) => Error | undefined;
}) =>
  new MockLanguageModelV3({
    provider: opts.provider,
    modelId: opts.modelId,
    doStream: async (call) => {
      const refusal = opts.refuses?.(effortOf(call as never), call as never);
      if (refusal) {
        throw refusal;
      }
      return { stream: reasonedStep('planning', 'THE ANSWER') };
    },
  });

/** GPT-6 Astra as the provider answered live: no `none`; every other listed level accepted. */
const astra = () =>
  scriptedModel({
    provider: 'openai.responses',
    modelId: 'gpt-6-astra',
    refuses: (effort) => (effort === 'none' ? openAiRefusal(ASTRA_CLAUSE) : undefined),
  });

/** A streaming turn's doors: one no-input inbox, nothing arriving mid-turn. */
const turnParams = (): Pick<
  GenerateStreamParams,
  'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes'
> => ({
  drainInjectedContext: () => [],
  peekInjectedContext: () => false,
  inputArrived: () => new Promise<void>(() => undefined),
  absorbExitNotes: true,
});

const newConversation = (name: string) =>
  new Conversation({
    modelData: fixtureModelData,
    name,
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

type Part = { type: string; textDelta?: string; warnings?: Array<Record<string, unknown>>; carriedEffort?: string };

async function collect(fullStream: AsyncIterable<unknown>): Promise<Part[]> {
  const parts: Part[] = [];
  for await (const part of fullStream as AsyncIterable<Part>) {
    parts.push(part);
  }
  return parts;
}

/** Every warning the stream's `step-start` parts carried, in order. */
const streamWarnings = (parts: Part[]): Array<Record<string, unknown>> =>
  parts.filter((part) => part.type === 'step-start').flatMap((part) => part.warnings ?? []);

/**
 * A streaming turn through the real wiring, the step asked at `reasoningEffort` — THE CALLER'S WORD (DECISIONS 1019,
 * row 3082): the library applies no default of its own; 'none' is the least an OpenAI row that lists it (Sol) and
 * an Anthropic family with an off switch (its explicit off — Sonnet 5) offer; a Google caller, and an Anthropic family
 * with no off switch (Opus 5.5), says 'low' (see the Anthropic and Google case below). The SDK's warning log is
 * captured.
 */
const turn = async (
  name: string,
  model: MockLanguageModelV3,
  reasoningEffort: 'none' | 'low' | 'high' | 'auto' = 'none',
  extra: Partial<GenerateStreamParams> = {}
) => {
  const logged: Array<{ warnings: unknown[]; model?: string }> = [];
  const g = globalThis as { AI_SDK_LOG_WARNINGS?: unknown };
  const prior = g.AI_SDK_LOG_WARNINGS;
  g.AI_SDK_LOG_WARNINGS = (entry: { warnings: unknown[]; model?: string }) => logged.push(entry);
  try {
    const result = await newConversation(name).generateStream({
      messages: ['A train leaves at 9:40…'],
      model: model as never,
      reasoningEffort,
      ...turnParams(),
      ...extra,
    });
    const parts = await collect(result.fullStream);
    return { parts, text: await result.text, warnings: logged.flatMap((entry) => entry.warnings) };
  } finally {
    if (prior === undefined) {
      delete g.AI_SDK_LOG_WARNINGS;
    } else {
      g.AI_SDK_LOG_WARNINGS = prior;
    }
  }
};

/** The requests a model received, as JSON — what the provider would be sent (the abort signal is not a request field). */
const requestsJson = (model: MockLanguageModelV3): string =>
  JSON.stringify(
    model.doStreamCalls.map(({ abortSignal: _signal, ...call }) => call),
    null,
    1
  );

const dump = (name: string, json: string): void => {
  const dir = process.env.EFFORT_CHECK_DUMP_DIR;
  if (dir) {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, `${name}.json`), json);
  }
};

type WireProvider = 'openai' | 'anthropic' | 'google' | 'xai';
type WireRoad = 'generate' | 'stream';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type WireBody = Record<string, any>;

/** The answer every scripted wire gives: the object's JSON on a generate call, its text on a stream call. */
const WIRE_TEXT = '{"answer":"ok"}';
const ANSWER_SCHEMA = {
  type: 'object',
  properties: { answer: { type: 'string' } },
  required: ['answer'],
  additionalProperties: false,
};

/** The installed provider package's own version — the one the transport loads. */
const installedVersion = (packageName: string): string =>
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require(`${packageName}/package.json`).version;

/** The level word at the one place each provider's body carries it — read here from the body that reached fetch. */
const levelAtFetch = (provider: WireProvider, body: WireBody | undefined): unknown =>
  provider === 'openai' || provider === 'xai'
    ? body?.reasoning?.effort
    : provider === 'anthropic'
      ? body?.output_config?.effort
      : body?.generationConfig?.thinkingConfig?.thinkingLevel;

/**
 * Each provider's REAL package over a scripted `fetch`: every body that reached the wire kept, a minimal answer served
 * in that provider's own wire shape for the road the call took — JSON for a generate call, server-sent events for a
 * stream call. What the library reports is then compared with what left. No network, no key.
 */
class ProviderWire {
  readonly bodies: WireBody[] = [];

  constructor(
    private readonly provider: WireProvider,
    private readonly road: WireRoad
  ) {}

  readonly fetch = async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body)) as WireBody;
    this.bodies.push(body);
    return this.road === 'stream'
      ? new Response(ProviderWire.events(this.provider), {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        })
      : new Response(JSON.stringify(ProviderWire.message(this.provider, body)), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
  };

  model(modelId: string): unknown {
    const settings = { apiKey: 'scripted', fetch: this.fetch as never };
    if (this.provider === 'openai') {
      return createOpenAI(settings).responses(modelId);
    }
    if (this.provider === 'xai') {
      return createXai(settings).responses(modelId);
    }
    if (this.provider === 'anthropic') {
      return createAnthropic(settings)(modelId);
    }
    return createGoogleGenerativeAI(settings)(modelId);
  }

  /** A generate call's answer. Anthropic's object road may be the adapter's json tool — answered on the road the body took. */
  private static message(provider: WireProvider, body: WireBody): WireBody {
    if (provider === 'google') {
      return {
        candidates: [{ content: { role: 'model', parts: [{ text: WIRE_TEXT }] }, finishReason: 'STOP', index: 0 }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3, totalTokenCount: 8 },
      };
    }
    if (provider === 'anthropic') {
      const viaTool = (body.tools ?? []).some((tool: { name?: string }) => tool.name === 'json');
      return {
        id: 'msg_wire',
        type: 'message',
        role: 'assistant',
        model: body.model,
        content: viaTool
          ? [{ type: 'tool_use', id: 'toolu_json', name: 'json', input: JSON.parse(WIRE_TEXT) }]
          : [{ type: 'text', text: WIRE_TEXT }],
        stop_reason: viaTool ? 'tool_use' : 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 5, output_tokens: 3 },
      };
    }
    return ProviderWire.response('completed', body.model, [ProviderWire.item('completed')]);
  }

  /** A stream call's answer, as the provider streams it. */
  private static events(provider: WireProvider): string {
    const data = (event: WireBody) => `data: ${JSON.stringify(event)}\n\n`;
    if (provider === 'google') {
      return data({
        candidates: [{ content: { role: 'model', parts: [{ text: WIRE_TEXT }] }, finishReason: 'STOP', index: 0 }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3, totalTokenCount: 8 },
      });
    }
    if (provider === 'anthropic') {
      const named = (event: WireBody) => `event: ${String(event.type)}\n${data(event)}`;
      return [
        named({
          type: 'message_start',
          message: {
            id: 'msg_wire',
            type: 'message',
            role: 'assistant',
            model: 'scripted',
            content: [],
            stop_reason: null,
            usage: { input_tokens: 5, output_tokens: 1 },
          },
        }),
        named({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
        named({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: WIRE_TEXT } }),
        named({ type: 'content_block_stop', index: 0 }),
        named({
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null },
          usage: { output_tokens: 3 },
        }),
        named({ type: 'message_stop' }),
      ].join('');
    }
    // OpenAI's and xAI's Responses wire.
    let sequence = 0;
    const event = (fields: WireBody) => data({ ...fields, sequence_number: sequence++ });
    const at = { item_id: 'msg_wire', output_index: 0, content_index: 0 };
    return [
      event({ type: 'response.created', response: ProviderWire.response('in_progress', 'scripted', []) }),
      event({ type: 'response.output_item.added', output_index: 0, item: ProviderWire.item('in_progress') }),
      event({ type: 'response.output_text.delta', ...at, delta: WIRE_TEXT, logprobs: [] }),
      event({ type: 'response.output_text.done', ...at, text: WIRE_TEXT, logprobs: [] }),
      event({ type: 'response.output_item.done', output_index: 0, item: ProviderWire.item('completed') }),
      event({
        type: 'response.completed',
        response: ProviderWire.response('completed', 'scripted', [ProviderWire.item('completed')]),
      }),
    ].join('');
  }

  private static response(status: string, model: string, output: WireBody[]): WireBody {
    return {
      id: 'resp_wire',
      object: 'response',
      created_at: 1762900000,
      status,
      incomplete_details: null,
      model,
      output,
      service_tier: 'default',
      usage:
        status === 'completed'
          ? {
              input_tokens: 5,
              input_tokens_details: { cached_tokens: 0 },
              output_tokens: 3,
              output_tokens_details: { reasoning_tokens: 0 },
              total_tokens: 8,
            }
          : null,
    };
  }

  private static item(status: string): WireBody {
    return {
      id: 'msg_wire',
      type: 'message',
      role: 'assistant',
      status,
      content: status === 'completed' ? [{ type: 'output_text', text: WIRE_TEXT, annotations: [], logprobs: [] }] : [],
    };
  }
}

/**
 * One call through the library on one road over the provider's real package: the level word the library REPORTS
 * for the request (the step's `carriedEffort` on a stream call, the result's on a generate call) beside the word in
 * the body that reached fetch.
 */
const reportedAndFetched = async (
  provider: WireProvider,
  modelId: string,
  effort: string,
  road: WireRoad
): Promise<{ reported: unknown[]; fetched: unknown[] }> => {
  const wire = new ProviderWire(provider, road);
  const conversation = newConversation(`carried-${provider}-${modelId}-${effort}-${road}`);
  const model = wire.model(modelId) as never;
  const reasoningEffort = effort as ReasoningEffort;
  if (road === 'generate') {
    const result = await conversation.generateObject<{ answer: string }>({
      messages: ['Say ok.'],
      model,
      reasoningEffort,
      schema: ANSWER_SCHEMA,
      recordInHistory: false,
    });
    expect(result.object.answer).toBe('ok');
    return {
      reported: [result.carriedEffort],
      fetched: wire.bodies.map((body) => levelAtFetch(provider, body)),
    };
  }
  const result = await conversation.generateStream({ messages: ['Say ok.'], model, reasoningEffort });
  const parts = await collect(result.fullStream);
  expect(await result.text).toBe(WIRE_TEXT);
  return {
    reported: parts.filter((part) => part.type === 'step-start').map((part) => part.carriedEffort),
    fetched: wire.bodies.map((body) => levelAtFetch(provider, body)),
  };
};

/**
 * Every named level of the 10 live source rows with levels (chat-common's modelCatalog.declaration.json at chat
 * 0ec951c1, read 2026-11-12: 47 levels) and gemini-4-hydrogen's four (Google's thinking page, 2026-11-12).
 */
const LIVE_LEVELS: Array<[WireProvider, string, string[]]> = [
  ['anthropic', 'claude-fable-5-1', ['low', 'medium', 'high', 'xhigh', 'max']],
  ['anthropic', 'claude-opus-4-8', ['low', 'medium', 'high', 'xhigh', 'max']],
  ['anthropic', 'claude-opus-5', ['low', 'medium', 'high', 'xhigh', 'max']],
  ['anthropic', 'claude-sonnet-5', ['low', 'medium', 'high', 'xhigh', 'max']],
  ['google', 'gemini-3.1-pro-preview', ['low', 'medium', 'high']],
  ['openai', 'gpt-5.5-pro', ['medium', 'high', 'xhigh']],
  ['openai', 'gpt-5.6-sol', ['none', 'low', 'medium', 'high', 'xhigh', 'max']],
  ['openai', 'gpt-5.6-terra', ['none', 'low', 'medium', 'high', 'xhigh', 'max']],
  ['xai', 'grok-4.3', ['none', 'low', 'medium', 'high', 'xhigh']],
  ['xai', 'grok-4.5', ['low', 'medium', 'high', 'xhigh']],
  ['google', 'gemini-4-hydrogen', ['low', 'medium', 'high', 'xhigh']],
];

/** Silences the SDK's own warning log for the length of a call (the warnings themselves are read off the step). */
const quietly = async <T>(run: () => Promise<T>): Promise<T> => {
  const g = globalThis as { AI_SDK_LOG_WARNINGS?: unknown };
  const prior = g.AI_SDK_LOG_WARNINGS;
  g.AI_SDK_LOG_WARNINGS = false;
  try {
    return await run();
  } finally {
    if (prior === undefined) {
      delete g.AI_SDK_LOG_WARNINGS;
    } else {
      g.AI_SDK_LOG_WARNINGS = prior;
    }
  }
};

beforeEach(() => RequestedEffort.forgetAll());

describe('the requested effort follows the model (a step asked at a level the provider refuses runs with the effort omitted)', () => {
  test(
    'the provider refuses `none` — the step is re-issued with the effort OMITTED (the provider’s default), the answer arrives, and the omission rides the stream as a step-start warning',
    async () => {
      const model = astra();
      const { parts, text, warnings } = await turn('effort-refused-once', model);

      expect(text).toContain('THE ANSWER');
      // Two requests: the step at none (refused, nothing billed), the step with NO effort field
      // (the provider applies its own default — never a level of this library's choosing).
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', undefined]);
      expect(model.doStreamCalls[1].providerOptions?.openai).not.toHaveProperty('reasoningEffort');
      // The omission is surfaced ONCE, as a warning on the stream (the SDK's warning channel),
      // never as an error to the person — and the conversation forwards it on its own stream.
      expect(warnings).toHaveLength(1);
      expect(JSON.stringify(warnings[0])).toMatch(/none/);
      expect(JSON.stringify(warnings[0])).toMatch(/omitted/);
      expect(JSON.stringify(warnings[0])).toMatch(/gpt-6-astra/);
      expect(RequestedEffort.omits('gpt-6-astra', 'none')).toBe(true);
      const forwarded = streamWarnings(parts);
      expect(forwarded).toHaveLength(1);
      expect(forwarded[0]).toMatchObject({ type: 'compatibility', feature: 'reasoningEffort' });
      expect(String(forwarded[0].details)).toMatch(/gpt-6-astra does not accept reasoning effort 'none'/);
      expect(String(forwarded[0].details)).toMatch(/omitted/);
    },
    TIMEOUT
  );

  test(
    'the omission is remembered for the process — a second turn on the same model never pays the refused request, and warns no more',
    async () => {
      await turn('effort-remembered-first', astra());

      const second = astra();
      const { parts, warnings } = await turn('effort-remembered-second', second);

      expect(parts.some((part) => part.type === 'text-delta')).toBe(true);
      expect(second.doStreamCalls.map((call) => effortOf(call as never))).toEqual([undefined]);
      expect(warnings).toHaveLength(0);
      expect(streamWarnings(parts)).toHaveLength(0);
    },
    TIMEOUT
  );

  // A caller that must hear the provider itself (the catalog's level check) forgets the one refusal it is about to
  // ask again: before this, a second check of a refused level in the same process left with no level and no warning,
  // and passed (the attack's finding 1a, 2026-11-12). Forgetting is per model and per level.
  test(
    'a remembered refusal, forgotten for that model and level, is asked again — the next request carries the level',
    async () => {
      await turn('effort-forget-first', astra());
      expect(RequestedEffort.omits('gpt-6-astra', 'none')).toBe(true);

      // Another level, or another model, forgets nothing of it.
      RequestedEffort.forget('gpt-6-astra', 'low');
      RequestedEffort.forget('gpt-6-sol', 'none');
      const still = astra();
      await turn('effort-forget-other', still);
      expect(still.doStreamCalls.map((call) => effortOf(call as never))).toEqual([undefined]);

      RequestedEffort.forget('gpt-6-astra', 'none');
      const again = astra();
      const { parts } = await turn('effort-forget-again', again);
      // The provider is asked at the level, refuses it in its own words, and the omission is said once more.
      expect(again.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', undefined]);
      expect(streamWarnings(parts)).toHaveLength(1);
      expect(String(streamWarnings(parts)[0].details)).toContain(ASTRA_CLAUSE);
    },
    TIMEOUT
  );

  test(
    'a model that accepts `none` is untouched — one request, at none, byte-identical',
    async () => {
      const model = scriptedModel({ provider: 'openai.responses', modelId: 'gpt-5.6-sol' });
      const { text, warnings } = await turn('effort-accepted', model);

      expect(text).toContain('THE ANSWER');
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none']);
      expect(warnings).toHaveLength(0);
      dump('openai-accepts-none', requestsJson(model));
    },
    TIMEOUT
  );

  test(
    'a caller that knows the model’s levels asks at a listed level — nothing is refused, nothing is remembered',
    async () => {
      const model = astra();
      const { text, warnings } = await turn('effort-listed-level', model, 'low');

      expect(text).toContain('THE ANSWER');
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['low']);
      expect(warnings).toHaveLength(0);
      expect(RequestedEffort.omits('gpt-6-astra', 'none')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'Anthropic and Google requests hear no refusal, nothing substituted — the step at the least each provider offers: Sonnet 5 at none (the explicit off, thinking disabled, no effort field), Opus 5.5 at low (no off switch: adaptive at low), Google at low (no off switch on Gemini: a thinkingLevel, never no config)',
    async () => {
      // Where each provider carries the effort — the level is the caller's word (the owner's answer per provider:
      // 'none' on Anthropic sends the family's explicit off, thinking disabled, and NO effort field — a request with no
      // thinking block THINKS on Sonnet 5 (AnthropicModelRules); 'low' on Google sends thinkingLevel 'low'), sent as
      // asked (the dumps are md5-compared across the fix). A family with no off switch (Opus 5.5) is asked at its
      // least, 'low': adaptive thinking at effort low, nothing refused.
      const effortField = {
        anthropic: (call: Call) => call.providerOptions?.anthropic?.effort,
        google: (call: Call) =>
          (call.providerOptions?.google?.thinkingConfig as { thinkingLevel?: unknown } | undefined)?.thinkingLevel,
      };
      const adaptive = { type: 'adaptive', display: 'summarized' };
      for (const [provider, modelId, name, least, field, thinking] of [
        ['anthropic.messages', 'claude-sonnet-5', 'anthropic', 'none', undefined, { type: 'disabled' }],
        ['anthropic.messages', 'claude-opus-5-5', 'anthropic', 'low', 'low', adaptive],
        ['google.generative-ai', 'gemini-3-pro', 'google', 'low', 'low', undefined],
      ] as const) {
        const model = scriptedModel({ provider, modelId });
        const { text, warnings } = await turn(`effort-${name}-${modelId}`, model, least);
        expect(text).toContain('THE ANSWER');
        expect(model.doStreamCalls.map((call) => effortField[name](call as never))).toEqual([field]);
        expect(model.doStreamCalls.map((call) => call.providerOptions?.anthropic?.thinking)).toEqual([thinking]);
        expect(warnings).toHaveLength(0);
        dump(`${name}-${modelId}`, requestsJson(model));
      }
    },
    TIMEOUT
  );

  test(
    'any other 400 surfaces untouched — the step fails as before, nothing re-issued, nothing remembered',
    async () => {
      const model = scriptedModel({
        provider: 'openai.responses',
        modelId: 'gpt-6-astra',
        refuses: () => otherBadRequest(),
      });
      await expect(turn('effort-other-400', model)).rejects.toThrow();

      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none']);
      expect(RequestedEffort.omits('gpt-6-astra', 'none')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'the re-issue is ONCE — a model that refuses the effort-less request too surfaces that refusal, nothing remembered',
    async () => {
      const model = scriptedModel({
        provider: 'openai.responses',
        modelId: 'gpt-6-nova',
        refuses: (effort) =>
          effort === 'none'
            ? openAiRefusal(
                "Unsupported value: 'none' is not supported with the 'gpt-6-nova' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'."
              )
            : effort === undefined
              ? otherBadRequest()
              : undefined,
      });
      await expect(turn('effort-refused-twice', model)).rejects.toThrow();

      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', undefined]);
      expect(RequestedEffort.omits('gpt-6-nova', 'none')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'the generate path hears the same verdict — a refused top-level effort is re-issued with the effort omitted, the warning on the result',
    async () => {
      // A model without `xhigh` (Anthropic: "Not every model that supports max supports xhigh"),
      // refusing in the provider's shape — the field path leads the message, no `param` field.
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-sonnet-5-5',
        doGenerate: async (call) => {
          const effort = call.providerOptions?.anthropic?.effort;
          if (effort === 'xhigh') {
            const body = {
              type: 'error',
              error: {
                type: 'invalid_request_error',
                message:
                  "output_config.effort: 'xhigh' is not supported for this model. Supported values are: 'low', 'medium', 'high', and 'max'.",
              },
            };
            throw new APICallError({
              message: body.error.message,
              url: 'https://api.anthropic.com/v1/messages',
              requestBodyValues: {},
              statusCode: 400,
              responseHeaders: {},
              responseBody: JSON.stringify(body),
              isRetryable: false,
              data: body,
            });
          }
          return {
            content: [{ type: 'text' as const, text: '{"answer":4}' }],
            finishReason: { unified: 'stop' as const, raw: 'end_turn' },
            usage,
            warnings: [],
          };
        },
      });
      const result = await newConversation('effort-generate').generateObject<{ answer: number }>({
        messages: ['2+2?'],
        model: model as never,
        reasoningEffort: 'xhigh',
        schema: { type: 'object', properties: { answer: { type: 'number' } }, required: ['answer'] },
      });

      expect(result.object.answer).toBe(4);
      // The second request carries no effort at all — the provider's own default (high) applies.
      expect(model.doGenerateCalls.map((call) => call.providerOptions?.anthropic?.effort)).toEqual([
        'xhigh',
        undefined,
      ]);
      expect(model.doGenerateCalls[1].providerOptions?.anthropic).not.toHaveProperty('effort');
      expect(RequestedEffort.omits('claude-sonnet-5-5', 'xhigh')).toBe(true);
    },
    TIMEOUT
  );
});

describe('a level the installed SDK cannot carry is a refusal too (the transport’s, before any request)', () => {
  /**
   * The SDK's own refusal, as `parseProviderOptions` raises it (an AI_InvalidArgumentError whose
   * message names only the provider; the effort is named in its AI_TypeValidationError cause).
   */
  const sdkRefusal = (provider: string, value: Record<string, unknown>) => {
    const cause = Object.assign(
      new Error(
        `Type validation failed: Value: ${JSON.stringify(value)}.\nError message: Invalid option: expected one of "none"|"low"|"medium"|"high"`
      ),
      { name: 'AI_TypeValidationError', value }
    );
    return Object.assign(new Error(`invalid ${provider} provider options`), {
      name: 'AI_InvalidArgumentError',
      argument: 'providerOptions',
      cause,
    });
  };

  // The installed package's own options schema refuses a word its list lacks before any request leaves (xAI's
  // reasoningEffort lists none…xhigh; Google's thinkingLevel minimal…xhigh from 3.0.146): the provider was never
  // asked. The re-issue is the same as for the provider's refusal; the warning names the package and the version
  // that refused — the fact a catalog turns into "save the row without it" — never the provider. RED at 9261af0a:
  // the library sent xAI's max as xhigh and Google's as high, so the package was never reached at max; at
  // the package's refusal the warning said the model did not accept the level.
  test.each([
    ['xai', 'grok-4.5', '@ai-sdk/xai'],
    ['google', 'gemini-4-hydrogen', '@ai-sdk/google'],
  ] as Array<[WireProvider, string, string]>)(
    "%s's installed package lacks max (%s): no request leaves at max, the step is re-issued without the level, and the warning names the package and its version",
    async (provider, modelId, packageName) => {
      const wire = new ProviderWire(provider, 'stream');
      const result = await newConversation(`package-refusal-${provider}`).generateStream({
        messages: ['A train leaves at 9:40…'],
        model: wire.model(modelId) as never,
        reasoningEffort: 'max',
      });
      const parts = await collect(result.fullStream);

      expect(await result.text).toBe(WIRE_TEXT);
      // One request reached the wire — the re-issue, carrying no level; the step reports none carried.
      expect(wire.bodies).toHaveLength(1);
      expect(levelAtFetch(provider, wire.bodies[0])).toBeUndefined();
      const starts = parts.filter((part) => part.type === 'step-start');
      expect(starts.map((part) => part.carriedEffort)).toEqual([undefined]);
      const forwarded = streamWarnings(parts);
      expect(forwarded).toHaveLength(1);
      expect(forwarded[0]).toMatchObject({ type: 'compatibility', feature: 'reasoningEffort' });
      const details = String(forwarded[0].details);
      expect(details).toContain(`${packageName} ${installedVersion(packageName)}`);
      expect(details).toContain("'max'");
      expect(details).not.toContain('does not accept');
      expect(RequestedEffort.omits(modelId, 'max')).toBe(true);
    },
    TIMEOUT
  );

  test('the verdict reads the cause: the SDK’s option refusal names the effort only in its cause', () => {
    expect(RequestedEffort.isRefusal(sdkRefusal('xai', { reasoningEffort: 'xhigh' }), 'xhigh')).toBe(true);
    expect(RequestedEffort.isRefusal(sdkRefusal('xai', { reasoningEffort: 'none' }), 'none')).toBe(true);
    // Another option the schema refuses is not the effort's refusal.
    expect(RequestedEffort.isRefusal(sdkRefusal('xai', { searchParameters: 'x' }), 'xhigh')).toBe(false);
  });
});

describe('each request reports the level word its body carried — read from the provider package’s own report of the body', () => {
  // The fact a catalog's check proves a level by: the word that LEFT, never the word asked (a substitution, a package
  // that drops a level, an id the library says does not reason, a remembered refusal all leave something else).
  // RED at 9261af0a: nothing was reported on either road.
  test('on a generate call and on a stream call, for each of the four providers, the reported word is the word in the body at fetch, and that word is the row’s', async () => {
    const misses: string[] = [];
    let cells = 0;
    await quietly(async () => {
      for (const [provider, modelId, levels] of LIVE_LEVELS) {
        for (const level of levels) {
          for (const road of ['generate', 'stream'] as const) {
            cells += 1;
            const { reported, fetched } = await reportedAndFetched(provider, modelId, level, road);
            if (fetched.length !== 1 || fetched[0] !== level || reported.length !== 1 || reported[0] !== fetched[0]) {
              misses.push(
                `${provider} ${modelId} at ${level} (${road}): at fetch ${JSON.stringify(fetched)}, reported ${JSON.stringify(reported)}`
              );
            }
          }
        }
      }
    });
    expect(cells).toBe(102);
    expect(misses).toEqual([]);
  }, 120_000);

  test(
    'an id the library says does not reason (o5): the body carries no level and the reported word is absent, on both roads',
    async () => {
      await quietly(async () => {
        for (const road of ['generate', 'stream'] as const) {
          const { reported, fetched } = await reportedAndFetched('openai', 'o5', 'high', road);
          expect(fetched).toEqual([undefined]);
          expect(reported).toEqual([undefined]);
        }
      });
    },
    TIMEOUT
  );
});

describe('RequestedEffort — the verdict', () => {
  test('only the effort refusal is heard: by the named parameter, by the field path in the message, or by the quoted value — never another 400, never a 5xx', () => {
    expect(RequestedEffort.isRefusal(openAiRefusal(ASTRA_CLAUSE), 'none')).toBe(true);
    // xAI's grammar (grok-4.5, 2026-09-29): the parameter named in the message, no `param` field.
    expect(RequestedEffort.isRefusal(new Error(GROK_CLAUSE), 'none')).toBe(true);
    // OpenAI names the parameter: a different parameter's refusal is not ours, whatever the message says.
    expect(RequestedEffort.isRefusal(otherBadRequest(), 'none')).toBe(false);
    // No `param`: the message names the field path …
    expect(
      RequestedEffort.isRefusal(new Error("output_config.effort: 'xhigh' is not supported for this model."), 'xhigh')
    ).toBe(true);
    expect(RequestedEffort.isRefusal(new Error("Invalid value for 'thinking_level': 'medium'."), 'medium')).toBe(true);
    // … or quotes the value this request sent.
    expect(
      RequestedEffort.isRefusal(new Error("Unsupported value: 'none' is not supported with this model."), 'none')
    ).toBe(true);
    expect(
      RequestedEffort.isRefusal(new Error("Unsupported value: 'none' is not supported with this model."), 'low')
    ).toBe(false);
    // Another 400 in the same words about something else, and a transient failure, are not heard.
    expect(RequestedEffort.isRefusal(new Error('messages: text content blocks must be non-empty'), 'none')).toBe(false);
    expect(
      RequestedEffort.isRefusal(new Error('"thinking.type.disabled" is not supported for this model.'), 'none')
    ).toBe(false);
    expect(
      RequestedEffort.isRefusal(
        new APICallError({
          message: "Unsupported value: 'none' is not supported with this model right now.",
          url: 'https://api.openai.com/v1/responses',
          requestBodyValues: {},
          statusCode: 500,
          responseHeaders: {},
          responseBody: '',
          isRetryable: true,
        }),
        'none'
      )
    ).toBe(false);
  });
});
