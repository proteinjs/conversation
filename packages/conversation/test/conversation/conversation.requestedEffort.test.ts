import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
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

type Part = { type: string; textDelta?: string; warnings?: Array<Record<string, unknown>> };

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

  test(
    "xAI's provider options refuse 'xhigh' (recorded live 2026-09-29, @ai-sdk/xai 3.0.92 on grok-4.5): the step is re-issued with the effort OMITTED, the omission rides the stream as the same step-start warning, and it is remembered",
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'xai',
        modelId: 'grok-4.5',
        doStream: async (call) => {
          const effort = (call as Call).providerOptions?.xai?.reasoningEffort;
          if (effort === 'xhigh') {
            throw sdkRefusal('xai', { reasoningEffort: 'xhigh' });
          }
          return { stream: reasonedStep('planning', 'THE ANSWER') };
        },
      });
      const result = await newConversation('sdk-refusal-xhigh').generateStream({
        messages: ['A train leaves at 9:40…'],
        model: model as never,
        reasoningEffort: 'xhigh',
      });
      const parts = await collect(result.fullStream);
      expect(await result.text).toContain('THE ANSWER');
      const sent = model.doStreamCalls.map((call) => (call as Call).providerOptions?.xai?.reasoningEffort);
      expect(sent).toEqual(['xhigh', undefined]);
      expect(model.doStreamCalls[1].providerOptions?.xai).not.toHaveProperty('reasoningEffort');
      const forwarded = streamWarnings(parts);
      expect(forwarded).toHaveLength(1);
      expect(forwarded[0]).toMatchObject({ type: 'compatibility', feature: 'reasoningEffort' });
      expect(String(forwarded[0].details)).toMatch(/grok-4.5 does not accept reasoning effort 'xhigh'/);
      expect(String(forwarded[0].details)).toMatch(/invalid xai provider options/);
      expect(RequestedEffort.omits('grok-4.5', 'xhigh')).toBe(true);
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
