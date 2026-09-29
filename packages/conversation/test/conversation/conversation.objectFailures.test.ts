import { inspect } from 'util';
import { APICallError } from 'ai';
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
 * A structured answer that was not the object leaves `generateObject` as the library's OWN typed
 * error, named by the finish reason the client library attached — never as the client library's
 * opaque "No object generated: could not parse the response." Three kinds, each read from a
 * stubbed model (no network, no keys; the fixture strings are markers, nobody's content):
 *
 * - `length` with the model's own ceiling in force and no reasoning → ObjectTruncatedError, the
 *   cap and its owner named, `runaway` true — the ops signal that the provider ran away;
 * - `length` under the caller's own `maxTokens` → ObjectTruncatedError owned by the caller, no runaway;
 * - `content-filter` → ObjectRefusedError;
 * - `stop` with text that is not JSON → ObjectParseError.
 *
 * In every case the model is called EXACTLY ONCE: the library re-issues nothing and runs no repair
 * round through the model. The typed error keeps the client library's error whole as `cause` (the
 * text rides it) while a log line handed the typed error carries the facts, never the text.
 */

const TIMEOUT = 30_000;
const TEXT_MARKER = 'rst_text_2b1d7f4c9e8a4b6c8d1e2f3a4b5c6d7e';
const HEADER_MARKER = 'rst_header_9a0364b9e99bb480dd25e1f0284c8555';
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

/** A runaway: one key outside the schema, then noise, to the ceiling. */
const runawayText = (length: number) => `{"cunningOf":"${TEXT_MARKER}`.padEnd(length, '_]~@[|}^~_[|]@~^_') as string;

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

const caught = async (work: () => Promise<unknown>): Promise<unknown> => {
  try {
    await work();
  } catch (error) {
    return error;
  }
  throw new Error('the call did not fail');
};

const generateAgainst = (model: MockLanguageModelV3, maxTokens?: number) =>
  caught(() =>
    new Conversation({
      modelData: fixtureModelData,
      name: 'object-failures-test',
      logLevel: 'error',
      limits: { enforceLimits: false },
    }).generateObject<{ title: string; description: string }>({
      messages: ['label this revision'],
      model: model as never,
      schema,
      ...(maxTokens !== undefined ? { maxTokens } : {}),
    })
  );

describe('generateObject: a structured answer that was not the object is a typed error, named by finish reason', () => {
  beforeEach(() => {
    delete process.env.DEVELOPMENT;
    delete process.env.CONVERSATION_LOG_PROVIDER_PAYLOADS;
  });

  test(
    'a runaway to the model`s own ceiling (finish length, no caller cap, no reasoning) → ObjectTruncatedError, runaway, one call',
    async () => {
      const model = answering({
        text: runawayText(64_011),
        finishReason: 'length',
        stopReason: 'max_tokens',
        output: HAIKU_CEILING,
      });

      const error = await generateAgainst(model);

      expect(ObjectTruncatedError.isInstance(error)).toBe(true);
      expect(ObjectGenerationError.isInstance(error)).toBe(true);
      const truncated = error as ObjectTruncatedError;
      expect(truncated).toMatchObject({
        name: 'ObjectTruncatedError',
        finishReason: 'length',
        rawStopReason: 'max_tokens',
        cap: HAIKU_CEILING,
        capOwner: 'model',
        runaway: true,
        outputTokens: HAIKU_CEILING,
        reasoningTokens: 0,
        inputTokens: 1258,
        schemaTitle: 'RevisionLabel',
        modelId: 'claude-haiku-4-5',
        textLength: 64_011,
      });
      expect(truncated.textHead).toHaveLength(200);
      expect(truncated.textHead).toContain(TEXT_MARKER);
      expect(truncated.textTail).toHaveLength(200);
      expect(truncated.message).toMatch(/cut off at 64000 output tokens \(the model's own ceiling\)/);
      expect(truncated.message).toMatch(/runaway/);
      expect(truncated.message).toMatch(/for schema "RevisionLabel" on claude-haiku-4-5/);
      // The client library's error rides whole as the cause — the text for whoever classifies on it.
      expect((truncated.cause as { name?: string }).name).toBe('AI_NoObjectGeneratedError');
      expect(inspect(truncated, { depth: 10 })).toContain(TEXT_MARKER);
      // Nothing was re-issued.
      expect(model.doGenerateCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'the caller`s own cap that bit (finish length under maxTokens) → ObjectTruncatedError owned by the caller, no runaway, one call',
    async () => {
      const model = answering({
        text: `{"title":"Reworded the intro","description":"The intro ${TEXT_MARKER}`,
        finishReason: 'length',
        stopReason: 'max_tokens',
        output: 64,
      });

      const error = await generateAgainst(model, 64);

      expect(ObjectTruncatedError.isInstance(error)).toBe(true);
      expect(error).toMatchObject({
        cap: 64,
        capOwner: 'caller',
        runaway: false,
        outputTokens: 64,
        finishReason: 'length',
      });
      expect((error as Error).message).toMatch(/cut off at 64 output tokens \(the call's own cap\)/);
      expect((error as Error).message).not.toMatch(/runaway/);
      expect(model.doGenerateCalls).toHaveLength(1);
      expect(model.doGenerateCalls[0].maxOutputTokens).toBe(64);
    },
    TIMEOUT
  );

  test(
    'a refusal (finish content-filter) → ObjectRefusedError carrying the provider`s own stop reason, one call',
    async () => {
      const model = answering({
        text: `{�0183174402014322${TEXT_MARKER}`,
        finishReason: 'content-filter',
        stopReason: 'refusal',
        output: 39,
      });

      const error = await generateAgainst(model);

      expect(ObjectRefusedError.isInstance(error)).toBe(true);
      expect(ObjectTruncatedError.isInstance(error)).toBe(false);
      expect(error).toMatchObject({
        name: 'ObjectRefusedError',
        finishReason: 'content-filter',
        rawStopReason: 'refusal',
        outputTokens: 39,
        schemaTitle: 'RevisionLabel',
      });
      expect((error as Error).message).toMatch(
        /declined the structured answer \(finish reason "content-filter", stop reason "refusal"\)/
      );
      expect(model.doGenerateCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'an answer that finished on its own and is not JSON (finish stop) → ObjectParseError, one call, no repair round',
    async () => {
      const model = answering({
        text: `{"title": "x" ${TEXT_MARKER}`,
        finishReason: 'stop',
        stopReason: 'end_turn',
        output: 12,
      });

      const error = await generateAgainst(model);

      expect(ObjectParseError.isInstance(error)).toBe(true);
      expect(ObjectRefusedError.isInstance(error)).toBe(false);
      expect(error).toMatchObject({
        name: 'ObjectParseError',
        finishReason: 'stop',
        rawStopReason: 'end_turn',
        schemaTitle: 'RevisionLabel',
      });
      expect((error as ObjectParseError).textHead).toContain(TEXT_MARKER);
      expect(model.doGenerateCalls).toHaveLength(1);
    },
    TIMEOUT
  );

  test(
    'a schema without a title names the call by its property names',
    async () => {
      const error = await caught(() =>
        new Conversation({
          modelData: fixtureModelData,
          name: 'object-failures-test',
          logLevel: 'error',
        }).generateObject({
          messages: ['label this revision'],
          model: answering({ text: 'not json', finishReason: 'stop', stopReason: 'end_turn', output: 3 }) as never,
          schema: { type: 'object', properties: { title: { type: 'string' }, description: { type: 'string' } } },
        })
      );

      expect(error).toMatchObject({ schemaProperties: ['title', 'description'] });
      expect((error as Error).message).toMatch(/for the schema with "title", "description"/);
    },
    TIMEOUT
  );

  test(
    'a transport error (APICallError) still passes through as itself',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-haiku-4-5',
        doGenerate: async () => {
          throw new APICallError({
            message: 'prompt is too long',
            url: 'https://provider.test/v1/messages',
            requestBodyValues: {},
            statusCode: 400,
            responseHeaders: {},
            responseBody: '{"type":"error","error":{"type":"invalid_request_error"}}',
            isRetryable: false,
          });
        },
      });

      const error = await generateAgainst(model);

      expect(APICallError.isInstance(error)).toBe(true);
      expect(ObjectGenerationError.isInstance(error)).toBe(false);
    },
    TIMEOUT
  );

  test(
    'the log line: the finish reason, the cap, the runaway flag and the model — never the text or the headers',
    async () => {
      const error = await generateAgainst(
        answering({
          text: runawayText(64_011),
          finishReason: 'length',
          stopReason: 'max_tokens',
          output: HAIKU_CEILING,
        })
      );

      const lines = new CapturedLines();
      const text = lines.logWhole(error);
      expect(text).not.toContain(TEXT_MARKER);
      expect(text).not.toContain(HEADER_MARKER);
      expect(lines.printed()).toMatchObject({
        name: 'ObjectTruncatedError',
        finishReason: 'length',
        rawStopReason: 'max_tokens',
        cap: HAIKU_CEILING,
        capOwner: 'model',
        runaway: true,
        outputTokens: HAIKU_CEILING,
        reasoningTokens: 0,
        schemaTitle: 'RevisionLabel',
        modelId: 'claude-haiku-4-5',
        provider: 'anthropic',
      });
      expect(lines.printed().message).toMatch(/^The structured answer was cut off at 64000 output tokens/);
    },
    TIMEOUT
  );
});
