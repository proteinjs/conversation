import { inspect } from 'util';
import { NoObjectGeneratedError } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { Logger, Log } from '@proteinjs/logger';
import { Conversation } from '../../src/Conversation';
import {
  ObjectGenerationError,
  ObjectParseError,
  ObjectRefusedError,
  ObjectTruncatedError,
} from '../../src/ObjectGenerationError';
import { fixtureModelData } from './fixtureModelData';

/**
 * The classification table for a structured answer that was not the object, row by row over the
 * client library's own error shapes (`NoObjectGeneratedError` with a finish reason, a usage, a
 * response body carrying the provider's stop reason, and the text) — the rows the fake-model
 * suite (conversation.objectFailures.test.ts) does not cover, plus the two laws every row shares:
 *
 * - `length` at the model's own ceiling WITH reasoning tokens is a truncation that is NOT a
 *   runaway (the model reasoned; it converged on nothing, but it did not run away);
 * - `length` with no cap known at all names its owner `unknown`;
 * - an error with NO finish reason at all is a parse miss whose finish reads `unknown`;
 * - every class carries the schema's title, the token counts, the provider's raw stop reason and
 *   the client library's error as `cause`;
 * - a fenced JSON answer (```json … ```) parses locally through the repair hook — the object
 *   comes back and the model was called exactly once;
 * - a log line handed a refusal or a parse miss carries the facts and never the text or headers.
 */

const TIMEOUT = 30_000;
const TEXT_MARKER = 'rst_text_5c2e9d1a7b3f4e6d8a9c0b1d2e3f4a5b';
const HEADER_MARKER = 'rst_header_0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c';
const HAIKU_CEILING = 64_000;

const schema = {
  title: 'RevisionLabel',
  type: 'object',
  properties: { title: { type: 'string' }, description: { type: 'string' } },
  required: ['title', 'description'],
};

const usage = (args: { output: number; reasoning?: number }) =>
  ({
    inputTokens: { total: 1258, noCache: 1258, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: args.output, text: args.output - (args.reasoning ?? 0), reasoning: args.reasoning ?? 0 },
  }) as never;

/** The client library's error exactly as it raises it: the text, the finish reason, the usage, the response. */
const sdkError = (args: {
  text?: string;
  finishReason: string | undefined;
  stopReason?: string;
  output: number;
  reasoning?: number;
}) =>
  new NoObjectGeneratedError({
    message: 'No object generated: could not parse the response.',
    text: args.text ?? `{"title":"${TEXT_MARKER}`,
    response: {
      id: 'msg_1',
      modelId: 'claude-haiku-4-5',
      timestamp: new Date(0),
      headers: { 'request-id': HEADER_MARKER },
      body: args.stopReason ? { id: 'msg_1', stop_reason: args.stopReason } : { id: 'msg_1' },
    } as never,
    usage: usage({ output: args.output, reasoning: args.reasoning }),
    finishReason: args.finishReason as never,
  });

/** A model whose one answer is `text`, finished as `finishReason`, with the provider's own `stopReason` on the body. */
const answering = (args: {
  text: string;
  finishReason: 'length' | 'content-filter' | 'stop';
  stopReason: string;
  output: number;
  reasoning?: number;
}) =>
  new MockLanguageModelV3({
    provider: 'anthropic.messages',
    modelId: 'claude-haiku-4-5',
    doGenerate: async () => ({
      content: [{ type: 'text' as const, text: args.text }],
      finishReason: { unified: args.finishReason, raw: args.stopReason } as never,
      usage: usage({ output: args.output, reasoning: args.reasoning }),
      warnings: [],
      response: {
        id: 'msg_1',
        modelId: 'claude-haiku-4-5',
        timestamp: new Date(0),
        headers: { 'request-id': HEADER_MARKER },
        body: { id: 'msg_1', stop_reason: args.stopReason },
      },
    }),
  });

class CapturedLines {
  private readonly logs: Log[] = [];
  readonly logger = new Logger({
    name: 'Caller',
    logWriter: { write: (log: Log) => void this.logs.push(log) } as never,
  });

  logWhole(error: unknown): string {
    this.logger.error({ message: 'The label call failed', error: error as Error });
    this.logger.warn({ message: 'The label call failed; the fallback stays', obj: { step: 'label', error } });
    return this.logs
      .map((log) => [
        JSON.stringify({ message: log.message, error: log.error, obj: log.obj }),
        inspect({ error: log.error, obj: log.obj }, { depth: 10, maxStringLength: 5000 }),
      ])
      .flat()
      .join('\n');
  }

  printed(): { [key: string]: unknown } {
    const error = this.logs[0].error as Error;
    return JSON.parse(JSON.stringify({ ...error, name: error.name, message: error.message }));
  }
}

const conversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'object-failures-classification-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

const caught = async (work: () => Promise<unknown>): Promise<unknown> => {
  try {
    await work();
  } catch (error) {
    return error;
  }
  throw new Error('the call did not fail');
};

describe('ObjectGenerationError.fromSdk — the classification table over the client library`s shapes', () => {
  const call = { modelId: 'claude-haiku-4-5', provider: 'anthropic', modelMaxTokens: HAIKU_CEILING, schema };

  test('finish length under the caller`s cap (below the model`s) → ObjectTruncatedError owned by the caller, no runaway', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ finishReason: 'length', stopReason: 'max_tokens', output: 64 }),
      {
        ...call,
        requestedMaxTokens: 64,
      }
    );
    expect(ObjectTruncatedError.isInstance(error)).toBe(true);
    expect(error).toMatchObject({ name: 'ObjectTruncatedError', cap: 64, capOwner: 'caller', runaway: false });
    expect(error.message).toMatch(/cut off at 64 output tokens \(the call's own cap\)/);
    expect(error.message).not.toMatch(/runaway/);
  });

  test('finish length at the model`s own cap with ZERO reasoning tokens → owned by the model, runaway true', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ finishReason: 'length', stopReason: 'max_tokens', output: HAIKU_CEILING, reasoning: 0 }),
      call
    );
    expect(error).toMatchObject({
      name: 'ObjectTruncatedError',
      cap: HAIKU_CEILING,
      capOwner: 'model',
      runaway: true,
      outputTokens: HAIKU_CEILING,
      reasoningTokens: 0,
    });
    expect(error.message).toMatch(/the model's own ceiling/);
    expect(error.message).toMatch(/runaway generation/);
  });

  test('finish length at the model`s cap WITH reasoning tokens → owned by the model, runaway FALSE (the model reasoned; it did not run away)', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ finishReason: 'length', stopReason: 'max_tokens', output: HAIKU_CEILING, reasoning: 1_200 }),
      call
    );
    expect(ObjectTruncatedError.isInstance(error)).toBe(true);
    expect(error).toMatchObject({
      cap: HAIKU_CEILING,
      capOwner: 'model',
      runaway: false,
      outputTokens: HAIKU_CEILING,
      reasoningTokens: 1_200,
    });
    expect(error.message).toMatch(/the model's own ceiling/);
    expect(error.message).not.toMatch(/runaway/);
  });

  test('finish length with no cap known at all → owned by nobody: capOwner unknown, no cap, no runaway', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ finishReason: 'length', stopReason: 'max_tokens', output: 500 }),
      {
        modelId: 'some-model',
        provider: 'other',
        schema,
      }
    );
    expect(ObjectTruncatedError.isInstance(error)).toBe(true);
    expect(error).toMatchObject({ capOwner: 'unknown', runaway: false, finishReason: 'length' });
    expect((error as ObjectTruncatedError).cap).toBeUndefined();
    expect(error.message).toMatch(/cut off at the output limit \(finish reason "length", stop reason "max_tokens"\)/);
  });

  test('finish content-filter → ObjectRefusedError (never a truncation, never a parse miss)', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ finishReason: 'content-filter', stopReason: 'refusal', output: 39 }),
      call
    );
    expect(ObjectRefusedError.isInstance(error)).toBe(true);
    expect(ObjectTruncatedError.isInstance(error)).toBe(false);
    expect(ObjectParseError.isInstance(error)).toBe(false);
    expect(error).toMatchObject({
      name: 'ObjectRefusedError',
      finishReason: 'content-filter',
      rawStopReason: 'refusal',
    });
  });

  test('finish stop with text that is not the object → ObjectParseError', () => {
    const error = ObjectGenerationError.fromSdk(
      sdkError({ text: 'not json at all', finishReason: 'stop', stopReason: 'end_turn', output: 4 }),
      call
    );
    expect(ObjectParseError.isInstance(error)).toBe(true);
    expect(ObjectTruncatedError.isInstance(error)).toBe(false);
    expect(error).toMatchObject({ name: 'ObjectParseError', finishReason: 'stop', rawStopReason: 'end_turn' });
  });

  test('an error with NO finish reason at all → ObjectParseError whose finish reads "unknown" (never a throw, never a truncation)', () => {
    const error = ObjectGenerationError.fromSdk(sdkError({ finishReason: undefined, output: 4 }), call);
    expect(ObjectParseError.isInstance(error)).toBe(true);
    expect(ObjectTruncatedError.isInstance(error)).toBe(false);
    expect(error).toMatchObject({ name: 'ObjectParseError', finishReason: 'unknown' });
    expect(error.rawStopReason).toBeUndefined();
    expect(error.message).toMatch(/\(finish reason "unknown"\) for schema "RevisionLabel" on claude-haiku-4-5\./);
  });

  test('every class carries the schema`s title, the token counts, the provider`s raw stop reason and the client library`s error as cause', () => {
    const rows: Array<[string, NoObjectGeneratedError, string, number]> = [
      [
        'ObjectTruncatedError',
        sdkError({ finishReason: 'length', stopReason: 'max_tokens', output: 64 }),
        'max_tokens',
        64,
      ],
      [
        'ObjectRefusedError',
        sdkError({ finishReason: 'content-filter', stopReason: 'refusal', output: 39 }),
        'refusal',
        39,
      ],
      ['ObjectParseError', sdkError({ finishReason: 'stop', stopReason: 'end_turn', output: 12 }), 'end_turn', 12],
    ];
    for (const [name, sdk, stopReason, outputTokens] of rows) {
      const error = ObjectGenerationError.fromSdk(sdk, { ...call, requestedMaxTokens: 64 });
      expect(error.name).toBe(name);
      expect(error).toMatchObject({
        schemaTitle: 'RevisionLabel',
        modelId: 'claude-haiku-4-5',
        inputTokens: 1258,
        outputTokens,
        reasoningTokens: 0,
        rawStopReason: stopReason,
      });
      expect(error.cause).toBe(sdk);
      expect(error.textHead).toContain(TEXT_MARKER);
      expect(error.message).toMatch(/for schema "RevisionLabel" on claude-haiku-4-5\.$/);
    }
  });
});

describe('generateObject against a fake provider — the rows the classification runs through the one owner', () => {
  beforeEach(() => {
    delete process.env.DEVELOPMENT;
    delete process.env.CONVERSATION_LOG_PROVIDER_PAYLOADS;
  });

  test(
    'finish length at the model`s ceiling WITH reasoning tokens → ObjectTruncatedError owned by the model, runaway false, one call',
    async () => {
      const model = answering({
        text: `{"title":"${TEXT_MARKER}`,
        finishReason: 'length',
        stopReason: 'max_tokens',
        output: HAIKU_CEILING,
        reasoning: 2_048,
      });

      const error = await caught(() =>
        conversation().generateObject({ messages: ['label this revision'], model: model as never, schema })
      );

      expect(ObjectTruncatedError.isInstance(error)).toBe(true);
      expect(error).toMatchObject({
        cap: HAIKU_CEILING,
        capOwner: 'model',
        runaway: false,
        outputTokens: HAIKU_CEILING,
        reasoningTokens: 2_048,
        rawStopReason: 'max_tokens',
      });
      expect(model.doGenerateCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'a fenced JSON answer (```json … ```) parses LOCALLY through the repair hook: the object comes back, the model was called once',
    async () => {
      const model = answering({
        text: '```json\n{"title":"Reworded the intro","description":"The intro reads differently."}\n```',
        finishReason: 'stop',
        stopReason: 'end_turn',
        output: 24,
      });

      const result = await conversation().generateObject<{ title: string; description: string }>({
        messages: ['label this revision'],
        model: model as never,
        schema,
      });

      expect(result.object).toEqual({ title: 'Reworded the intro', description: 'The intro reads differently.' });
      expect(model.doGenerateCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'the log line for a REFUSAL and for a PARSE MISS: the finish reason, the stop reason, the counts, the schema and the model — never the text or the headers',
    async () => {
      const rows: Array<{ finishReason: 'content-filter' | 'stop'; stopReason: string; name: string }> = [
        { finishReason: 'content-filter', stopReason: 'refusal', name: 'ObjectRefusedError' },
        { finishReason: 'stop', stopReason: 'end_turn', name: 'ObjectParseError' },
      ];
      for (const row of rows) {
        const model = answering({
          text: `{"title":"${TEXT_MARKER}`,
          finishReason: row.finishReason,
          stopReason: row.stopReason,
          output: 39,
        });
        const error = await caught(() =>
          conversation().generateObject({ messages: ['label this revision'], model: model as never, schema })
        );
        expect(model.doGenerateCalls).toHaveLength(1);
        // The text rides the error itself (the cause), for whoever classifies on it…
        expect(inspect(error, { depth: 10 })).toContain(TEXT_MARKER);
        // …and never a line about it.
        const lines = new CapturedLines();
        const text = lines.logWhole(error);
        expect(text).not.toContain(TEXT_MARKER);
        expect(text).not.toContain(HEADER_MARKER);
        expect(lines.printed()).toMatchObject({
          name: row.name,
          finishReason: row.finishReason,
          rawStopReason: row.stopReason,
          inputTokens: 1258,
          outputTokens: 39,
          reasoningTokens: 0,
          textLength: expect.any(Number),
          schemaTitle: 'RevisionLabel',
          modelId: 'claude-haiku-4-5',
          provider: 'anthropic',
        });
        expect(lines.printed()).not.toHaveProperty('textHead');
        expect(lines.printed()).not.toHaveProperty('cause');
      }
    },
    TIMEOUT
  );
});
