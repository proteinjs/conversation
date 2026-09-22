import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Conversation, type GenerateStreamParams } from '../../src/Conversation';
import { RequestedEffort } from '../../src/RequestedEffort';
import { Utterance } from '../../src/Utterance';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE REQUESTED EFFORT FOLLOWS THE MODEL — no network, no keys. A MockLanguageModelV3 stands in
 * as the provider (Conversation takes a model instance straight through resolveModel), so these
 * run the REAL wiring: Conversation → LlmTransportRetry.wrap → RequestedEffort.follow →
 * ForcedToolChoice.follow → model.
 *
 * The refusal is the provider's, recorded live 2026-09-22 (OpenAI, `gpt-6-astra`, the Responses
 * API): the bounded utterance — the acknowledgment line a turn speaks first — asks for
 * `reasoning.effort: none`, and the model answered HTTP 400 with the clause below. Before this
 * rule EVERY GPT-6 Astra turn ran without its acknowledgment line and paid a refused request
 * each time, and the catalog's probes could not see it (they run only the claimed efforts).
 */

const TIMEOUT = 30_000;

/** The provider's verdict, in its own words — the ladder it accepts is IN the clause. */
const ASTRA_CLAUSE =
  "Unsupported value: 'none' is not supported with the 'gpt-6-astra' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.";

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

const LINE = 'On it — checking the timetable now.';

const textStep = (text: string, warnings: unknown[] = []) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: warnings as never },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

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
 * own clause (listing the efforts it accepts), answers the utterance with one line, the step with
 * a reasoned answer.
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
      return {
        stream: Utterance.isRequest(call.prompt as never) ? textStep(LINE) : reasonedStep('planning', 'THE ANSWER'),
      };
    },
  });

/** GPT-6 Astra as the catalog check met it: no `none`; every other listed level accepted. */
const astra = () =>
  scriptedModel({
    provider: 'openai.responses',
    modelId: 'gpt-6-astra',
    refuses: (effort) => (effort === 'none' ? openAiRefusal(ASTRA_CLAUSE) : undefined),
  });

/** The bounded utterance's door (the idle path): one no-input inbox, `utterance` on. */
const utteranceParams = (): Pick<
  GenerateStreamParams,
  'drainInjectedContext' | 'peekInjectedContext' | 'inputArrived' | 'absorbExitNotes' | 'utterance'
> => ({
  drainInjectedContext: () => [],
  peekInjectedContext: () => false,
  inputArrived: () => new Promise<void>(() => undefined),
  absorbExitNotes: true,
  utterance: true,
});

const newConversation = (name: string) =>
  new Conversation({
    modelData: fixtureModelData,
    name,
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

type Part = { type: string; textDelta?: string; utterance?: true };

async function collect(fullStream: AsyncIterable<unknown>): Promise<Part[]> {
  const parts: Part[] = [];
  for await (const part of fullStream as AsyncIterable<Part>) {
    parts.push(part);
  }
  return parts;
}

/** The acknowledgment line the consumer saw: the text before the step-finish flagged `utterance`, or nothing. */
const utteredLine = (parts: Part[]): string | undefined => {
  const at = parts.findIndex((part) => part.type === 'step-finish' && part.utterance);
  if (at < 0) {
    return undefined;
  }
  return parts
    .slice(0, at)
    .filter((part) => part.type === 'text-delta')
    .map((part) => part.textDelta ?? '')
    .join('');
};

/** A turn with the bounded utterance leading it, through the real wiring; the SDK's warning log captured. */
const turn = async (name: string, model: MockLanguageModelV3, reasoningEffort: 'high' | 'auto' = 'high') => {
  const logged: Array<{ warnings: unknown[]; model?: string }> = [];
  const g = globalThis as { AI_SDK_LOG_WARNINGS?: unknown };
  const prior = g.AI_SDK_LOG_WARNINGS;
  g.AI_SDK_LOG_WARNINGS = (entry: { warnings: unknown[]; model?: string }) => logged.push(entry);
  try {
    const result = await newConversation(name).generateStream({
      messages: ['A train leaves at 9:40…'],
      model: model as never,
      reasoningEffort,
      ...utteranceParams(),
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

describe('the requested effort follows the model (the bounded utterance asks for none; a model without none gets its floor)', () => {
  test(
    'the provider refuses `none` — the utterance is re-issued at the nearest level the clause lists, and the acknowledgment line arrives',
    async () => {
      const model = astra();
      const { parts, text, warnings } = await turn('effort-refused-once', model);

      // The line arrived: its own step, flagged, before the answer.
      expect(utteredLine(parts)).toBe(LINE);
      expect(text).toContain('THE ANSWER');
      // Three requests: the utterance at none (refused, nothing billed), the utterance at the
      // floor the clause lists (`low` — the nearest to none), the step at its own effort.
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', 'low', 'high']);
      expect(Utterance.isRequest(model.doStreamCalls[1].prompt as never)).toBe(true);
      // The substitution is surfaced ONCE, as a warning on the stream (the SDK's warning
      // channel), never as an error to the person.
      expect(warnings).toHaveLength(1);
      expect(JSON.stringify(warnings[0])).toMatch(/none/);
      expect(JSON.stringify(warnings[0])).toMatch(/low/);
      expect(JSON.stringify(warnings[0])).toMatch(/gpt-6-astra/);
      expect(RequestedEffort.substituteFor('gpt-6-astra', 'none')).toBe('low');
    },
    TIMEOUT
  );

  test(
    'the substitution is remembered for the process — a second turn on the same model never pays the refused request, and warns no more',
    async () => {
      await turn('effort-remembered-first', astra());

      const second = astra();
      const { parts, warnings } = await turn('effort-remembered-second', second);

      expect(utteredLine(parts)).toBe(LINE);
      expect(second.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['low', 'high']);
      expect(warnings).toHaveLength(0);
    },
    TIMEOUT
  );

  test(
    'a model that accepts `none` is untouched — one utterance request, at none, byte-identical',
    async () => {
      const model = scriptedModel({ provider: 'openai.responses', modelId: 'gpt-5.6-sol' });
      const { parts, warnings } = await turn('effort-accepted', model);

      expect(utteredLine(parts)).toBe(LINE);
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', 'high']);
      expect(warnings).toHaveLength(0);
      dump('openai-accepts-none', requestsJson(model));
    },
    TIMEOUT
  );

  test(
    'Anthropic and Google requests are byte-identical before and after — no effort field to hear, nothing substituted',
    async () => {
      // Where each provider carries the effort — the utterance's `none` sends NO field to these
      // two, and the step's `high` is sent as asked (the dumps are md5-compared across the fix).
      const effortField = {
        anthropic: (call: Call) => call.providerOptions?.anthropic?.effort,
        google: (call: Call) =>
          (call.providerOptions?.google?.thinkingConfig as { thinkingLevel?: unknown } | undefined)?.thinkingLevel,
      };
      for (const [provider, modelId, name] of [
        ['anthropic.messages', 'claude-opus-5-5', 'anthropic'],
        ['google.generative-ai', 'gemini-3-pro', 'google'],
      ] as const) {
        const model = scriptedModel({ provider, modelId });
        const { parts, warnings } = await turn(`effort-${name}`, model);
        expect(utteredLine(parts)).toBe(LINE);
        expect(model.doStreamCalls.map((call) => effortField[name](call as never))).toEqual([undefined, 'high']);
        expect(warnings).toHaveLength(0);
        dump(name, requestsJson(model));
      }
    },
    TIMEOUT
  );

  test(
    'any other 400 surfaces untouched — the utterance fails as before, nothing re-issued, nothing remembered',
    async () => {
      const model = scriptedModel({
        provider: 'openai.responses',
        modelId: 'gpt-6-astra',
        refuses: (_effort, call) => (Utterance.isRequest(call.prompt as never) ? otherBadRequest() : undefined),
      });
      const { parts, text, warnings } = await turn('effort-other-400', model);

      expect(utteredLine(parts)).toBeUndefined();
      expect(text).toContain('THE ANSWER');
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', 'high']);
      expect(warnings).toHaveLength(0);
      expect(RequestedEffort.substituteFor('gpt-6-astra', 'none')).toBeUndefined();
    },
    TIMEOUT
  );

  test(
    'the re-issue is ONCE — a model that refuses the substitute too surfaces that refusal, nothing remembered',
    async () => {
      const model = scriptedModel({
        provider: 'openai.responses',
        modelId: 'gpt-6-nova',
        refuses: (effort) =>
          effort === 'none' || effort === 'low'
            ? openAiRefusal(
                `Unsupported value: '${effort}' is not supported with the 'gpt-6-nova' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.`
              )
            : undefined,
      });
      const { parts, text } = await turn('effort-refused-twice', model);

      expect(utteredLine(parts)).toBeUndefined();
      expect(text).toContain('THE ANSWER');
      expect(model.doStreamCalls.map((call) => effortOf(call as never))).toEqual(['none', 'low', 'high']);
      expect(RequestedEffort.substituteFor('gpt-6-nova', 'none')).toBeUndefined();
    },
    TIMEOUT
  );

  test(
    'the generate path hears the same verdict — a refused top-level effort is re-issued at the nearest listed level, the warning on the result',
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
      // xhigh sits between high and max: a tie, and the tie goes to the lower level.
      expect(model.doGenerateCalls.map((call) => call.providerOptions?.anthropic?.effort)).toEqual(['xhigh', 'high']);
      expect(RequestedEffort.substituteFor('claude-sonnet-5-5', 'xhigh')).toBe('high');
    },
    TIMEOUT
  );
});

describe('RequestedEffort — the verdict and the ladder', () => {
  test('the nearest level: none → low on the listed ladder; the floor when nothing is listed; a tie goes lower; never the refused value', () => {
    expect(RequestedEffort.nearest('none', ['low', 'medium', 'high', 'xhigh', 'max'])).toBe('low');
    expect(RequestedEffort.nearest('max', ['low', 'medium', 'high', 'xhigh'])).toBe('xhigh');
    expect(RequestedEffort.nearest('xhigh', ['low', 'medium', 'high', 'max'])).toBe('high');
    expect(RequestedEffort.nearest('medium', ['low', 'high'])).toBe('low');
    expect(RequestedEffort.nearest('none', [RequestedEffort.FLOOR])).toBe('low');
    expect(RequestedEffort.nearest('low', ['low'])).toBeUndefined();
    expect(RequestedEffort.nearest('none', [])).toBeUndefined();
  });

  test('the ladder is read from the clause in each provider’s grammar', () => {
    expect(RequestedEffort.listedLadder(ASTRA_CLAUSE)).toEqual(['low', 'medium', 'high', 'xhigh', 'max']);
    expect(
      RequestedEffort.listedLadder(
        "The value 'medium' is not supported for 'thinking_level'. Supported values: 'low', 'high'."
      )
    ).toEqual(['low', 'high']);
    expect(RequestedEffort.listedLadder('The server had an error.')).toEqual([]);
  });

  test('only the effort refusal is heard: by the named parameter, by the field path in the message, or by the quoted value — never another 400, never a 5xx', () => {
    expect(RequestedEffort.isRefusal(openAiRefusal(ASTRA_CLAUSE), 'none')).toBe(true);
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
