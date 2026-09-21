import { inspect } from 'util';
import { APICallError } from 'ai';
import { APIError as OpenAiSdkError } from 'openai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Logger, Log } from '@proteinjs/logger';
import { Conversation } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';
import { TransientProviderError } from '../../src/TransientProviderError';
import { ProviderBillingError } from '../../src/ProviderBillingError';
import { ProviderFailureLine } from '../../src/ProviderFailureLine';
import { fixtureModelData } from './fixtureModelData';

/**
 * A provider's error never prints what the request carried.
 *
 * The client library's `APICallError` keeps the whole request body (`requestBodyValues` — the
 * conversation itself), the response body and the response headers as ordinary fields, so any
 * log line that is handed the error WHOLE carries them. The library marks every provider error
 * that leaves it for the logger's boundary: a line then carries the error's name, the HTTP
 * status, the provider, the model, the vendor's error code and a sentence of the library's own.
 *
 * Every scenario first checks its premise — the error the caller CATCHES still carries the body
 * (what is thrown is nobody's to rewrite) — and then reads the line as two writers render it:
 * a structured one (`JSON.stringify`, what a deployed server writes) and the default one
 * (`util.inspect`). No network, no keys: the fixtures are markers, not anybody's content.
 */

const TIMEOUT = 30_000;
const REQUEST_MARKER = 'rst_request_5d41402abc4b2a76b9719d911017c592';
const RESPONSE_MARKER = 'rst_response_7d793037a0760186574b0282f2f435e7';
const HEADER_MARKER = 'rst_header_9a0364b9e99bb480dd25e1f0284c8555';
const MARKERS = [REQUEST_MARKER, RESPONSE_MARKER, HEADER_MARKER];

const apiError = (args: { message: string; statusCode: number; type: string; isRetryable?: boolean }) =>
  new APICallError({
    message: args.message,
    url: 'https://provider.test/v1/messages',
    requestBodyValues: { model: 'claude-test', messages: [{ role: 'user', content: REQUEST_MARKER }] },
    statusCode: args.statusCode,
    responseHeaders: { 'request-id': HEADER_MARKER, 'retry-after': '0' },
    responseBody: JSON.stringify({ type: 'error', error: { type: args.type, message: RESPONSE_MARKER } }),
    isRetryable: args.isRetryable,
    data: { type: 'error', error: { type: args.type, message: RESPONSE_MARKER } },
  });

/** What a caller's lines read as, under both writers. */
class CapturedLines {
  private readonly logs: Log[] = [];
  readonly logger = new Logger({
    name: 'Caller',
    logWriter: { write: (log: Log) => void this.logs.push(log) } as never,
  });

  /** The error as the line's `error`, and inside `obj` on a warn line: the two shapes callers use. */
  logWhole(error: unknown): string {
    this.logger.error({ message: 'The model call failed', error: error as Error });
    this.logger.warn({ message: 'The model call failed; carrying on', obj: { step: 'summarize', error } });
    return this.text();
  }

  text(): string {
    return this.logs
      .map((log) => [
        JSON.stringify({ message: log.message, error: log.error, obj: log.obj }),
        inspect({ error: log.error, obj: log.obj }, { depth: 10, maxStringLength: 2000 }),
      ])
      .flat()
      .join('\n');
  }

  structured(): { error?: { [key: string]: unknown }; obj?: { error?: { [key: string]: unknown } } }[] {
    return this.logs.map((log) => JSON.parse(JSON.stringify({ error: this.plain(log.error), obj: log.obj })));
  }

  /** An error's message and name do not enumerate: read them the way a structured writer does. */
  private plain(error: unknown) {
    return error instanceof Error ? { ...error, name: error.name, message: error.message } : error;
  }
}

/** Everything written to the console from here on, as a writer would render it. */
const consoleWrites = (): string[] => {
  const written: string[] = [];
  for (const level of ['error', 'warn', 'info', 'log'] as const) {
    jest
      .spyOn(console, level)
      .mockImplementation(
        (...parts: unknown[]) => void written.push(parts.map((part) => inspect(part, { depth: 10 })).join(' '))
      );
  }
  return written;
};

const expectNoMarker = (text: string) => MARKERS.forEach((marker) => expect(text).not.toContain(marker));

const caught = async (work: () => Promise<unknown>): Promise<unknown> => {
  try {
    await work();
  } catch (error) {
    return error;
  }
  throw new Error('the call did not fail');
};

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'provider-error-lines-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

const generateObjectAgainst = (model: MockLanguageModelV3) =>
  caught(() =>
    newConversation().generateObject<{ answer: string }>({
      messages: ['give me the answer'],
      model: model as never,
      schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
    })
  );

describe('a provider error on a log line never carries the request, the response body or the headers', () => {
  const env = { ...process.env };
  afterEach(() => {
    process.env = { ...env };
    jest.restoreAllMocks();
  });
  beforeEach(() => {
    delete process.env.DEVELOPMENT;
    delete process.env.CONVERSATION_LOG_PROVIDER_PAYLOADS;
  });

  test(
    'generateObject: the APICallError a caller catches and logs whole',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doGenerate: async () => {
          throw apiError({ message: 'prompt is too long', statusCode: 400, type: 'invalid_request_error' });
        },
      });

      const error = await generateObjectAgainst(model);

      expect(APICallError.isInstance(error)).toBe(true);
      expect(JSON.stringify((error as APICallError).requestBodyValues)).toContain(REQUEST_MARKER);
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      const [onTheLine, insideObj] = lines.structured();
      for (const printed of [onTheLine.error, insideObj.obj?.error]) {
        expect(printed).toMatchObject({
          statusCode: 400,
          provider: 'anthropic',
          modelId: 'claude-test',
          code: 'invalid_request_error',
        });
      }
      expect(onTheLine.error).toMatchObject({
        name: 'AI_APICallError',
        message:
          'The model call failed (HTTP 400, invalid_request_error) on claude-test: the provider rejected the request as invalid',
      });
    },
    TIMEOUT
  );

  test(
    'the library`s own line about a failure that outlived the retry budget, and the typed error it throws',
    async () => {
      const written: string[] = [];
      jest.spyOn(console, 'error').mockImplementation((...parts: unknown[]) => void written.push(parts.join(' ')));
      jest.spyOn(console, 'warn').mockImplementation((...parts: unknown[]) => void written.push(parts.join(' ')));
      const retry = new LlmTransportRetry({ budgetMs: 1 });

      const error = await caught(() =>
        retry.run(
          () => {
            throw apiError({ message: 'Overloaded', statusCode: 529, type: 'overloaded_error', isRetryable: true });
          },
          { isRetryable: (each) => APICallError.isInstance(each) && each.isRetryable === true, modelId: 'claude-test' }
        )
      );

      expect(TransientProviderError.isInstance(error)).toBe(true);
      expect(JSON.stringify(((error as TransientProviderError).cause as APICallError).requestBodyValues)).toContain(
        REQUEST_MARKER
      );
      expect(written.join('\n')).toContain('retry budget exhausted');
      expectNoMarker(written.join('\n'));
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({
        name: 'TransientProviderError',
        statusCode: 529,
        modelId: 'claude-test',
        code: 'overloaded_error',
      });
    },
    TIMEOUT
  );

  test(
    'the library`s own line about a billing failure, and the typed error it throws',
    async () => {
      const written: string[] = [];
      jest.spyOn(console, 'error').mockImplementation((...parts: unknown[]) => void written.push(parts.join(' ')));

      const error = await caught(() =>
        new LlmTransportRetry({ budgetMs: 5_000 }).run(
          () => {
            throw apiError({ message: 'billing problem', statusCode: 402, type: 'billing_error' });
          },
          { isRetryable: () => false, modelId: 'claude-test' }
        )
      );

      expect(ProviderBillingError.isInstance(error)).toBe(true);
      expect(written.join('\n')).toContain('billing/credit failure');
      expectNoMarker(written.join('\n'));
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({
        name: 'ProviderBillingError',
        statusCode: 402,
        modelId: 'claude-test',
        code: 'billing_error',
      });
    },
    TIMEOUT
  );

  test(
    'a stream: the error a read throws once output has begun',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'openai.responses',
        modelId: 'gpt-test',
        doStream: async () => ({
          stream: new ReadableStream({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] });
              controller.enqueue({ type: 'text-start', id: '1' });
              controller.enqueue({ type: 'text-delta', id: '1', delta: 'a first word' });
            },
            pull() {
              throw apiError({ message: 'The server had an error', statusCode: 500, type: 'server_error' });
            },
          }),
        }),
      });
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();

      const error = await caught(async () => {
        for (;;) {
          if ((await reader.read()).done) {
            return;
          }
        }
      });

      expect(APICallError.isInstance(error)).toBe(true);
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({ statusCode: 500, provider: 'openai', modelId: 'gpt-test' });
    },
    TIMEOUT
  );

  test(
    'a stream: the error an error part carries',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            {
              type: 'error' as const,
              error: apiError({ message: 'not allowed', statusCode: 403, type: 'permission_error' }),
            },
          ]),
        }),
      });
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();
      const parts: { type: string; error?: unknown }[] = [];
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        parts.push(read.value as { type: string; error?: unknown });
      }

      const carried = parts.find((part) => part.type === 'error')?.error;

      expect(APICallError.isInstance(carried)).toBe(true);
      expectNoMarker(new CapturedLines().logWhole(carried));
    },
    TIMEOUT
  );

  /** A stream that begins its output and then carries a provider's RAW error payload on an error part. */
  const rawPayloadMidStream = () =>
    new MockLanguageModelV3({
      provider: 'anthropic.messages',
      modelId: 'claude-test',
      doStream: async () => ({
        stream: convertArrayToReadableStream([
          { type: 'stream-start' as const, warnings: [] },
          { type: 'text-start' as const, id: '1' },
          { type: 'text-delta' as const, id: '1', delta: 'a first word' },
          {
            type: 'error' as const,
            error: { type: 'error', error: { type: 'overloaded_error', message: `Overloaded ${RESPONSE_MARKER}` } },
          },
        ]),
      }),
    });

  test(
    'generateStream: a raw payload mid-stream — the Error the library words from it',
    async () => {
      const result = await newConversation().generateStream({
        messages: ['say something'],
        model: rawPayloadMidStream() as never,
      });

      const error = await caught(async () => {
        for await (const _part of result.fullStream) {
          // read to the failure
        }
      });

      expect(error).toBeInstanceOf(Error);
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({ provider: 'anthropic', code: 'overloaded_error' });
    },
    TIMEOUT
  );

  test(
    'generateStream: a failed stream is never printed by the client library itself (its default prints the error whole)',
    async () => {
      const written = consoleWrites();
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doStream: async () => {
          throw apiError({ message: 'prompt is too long', statusCode: 400, type: 'invalid_request_error' });
        },
      });
      const result = await new Conversation({
        modelData: fixtureModelData,
        name: 'provider-error-lines-test',
        logLevel: 'warn',
        limits: { enforceLimits: false },
      }).generateStream({ messages: ['say something'], model: model as never });

      const error = await caught(async () => {
        for await (const _part of result.fullStream) {
          // read to the failure
        }
      });

      expect(APICallError.isInstance(error)).toBe(true);
      expect(written.join('\n')).toContain('The model stream reported an error');
      expectNoMarker(written.join('\n'));
    },
    TIMEOUT
  );

  test(
    'generateObject through the tool loop: a raw payload mid-stream',
    async () => {
      const written = consoleWrites();
      const error = await caught(() =>
        newConversation().generateObject<{ answer: string }>({
          messages: ['look, then answer'],
          model: rawPayloadMidStream() as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
          maxToolCalls: 3,
          tools: [
            {
              definition: {
                name: 'look',
                description: 'Reads one thing.',
                parameters: { type: 'object', properties: {} },
              },
              call: async () => ({ seen: true }),
            },
          ],
        })
      );

      expect(error).toBeInstanceOf(Error);
      expectNoMarker(written.join('\n'));
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({
        provider: 'anthropic',
        modelId: 'claude-test',
        code: 'overloaded_error',
      });
    },
    TIMEOUT
  );

  test(
    'the bounded utterance: its failed stream is not printed by the client library either',
    async () => {
      const written = consoleWrites();
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doStream: async () => {
          throw apiError({ message: 'prompt is too long', statusCode: 400, type: 'invalid_request_error' });
        },
      });
      type UtteranceInternals = {
        utter(args: {
          model: unknown;
          transcript: unknown[];
          inputs: { text: string }[];
          provider: string;
          modelString: string;
          abortSignal: AbortSignal;
          onResult: (result: unknown) => void;
        }): AsyncGenerator<unknown, string | undefined>;
      };
      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'provider-error-lines-test',
        logLevel: 'warn',
        limits: { enforceLimits: false },
      }) as unknown as UtteranceInternals;

      const parts = conversation.utter({
        model: new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never),
        transcript: [{ role: 'user', content: 'and the login page?' }],
        inputs: [{ text: 'also the header' }],
        provider: 'anthropic',
        modelString: 'claude-test',
        abortSignal: new AbortController().signal,
        onResult: () => undefined,
      });
      for (let next = await parts.next(); !next.done; next = await parts.next()) {
        // no line is expected: the call failed
      }

      expect(written.join('\n')).toContain('The model stream reported an error');
      expectNoMarker(written.join('\n'));
    },
    TIMEOUT
  );

  test(
    'a stream: the error an error part carries once output has begun',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            { type: 'text-start' as const, id: '1' },
            { type: 'text-delta' as const, id: '1', delta: 'a first word' },
            {
              type: 'error' as const,
              error: apiError({ message: 'Overloaded', statusCode: 529, type: 'overloaded_error' }),
            },
          ]),
        }),
      });
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();
      const parts: { type: string; error?: unknown }[] = [];
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        parts.push(read.value as { type: string; error?: unknown });
      }

      const carried = parts.find((part) => part.type === 'error')?.error;

      expect(APICallError.isInstance(carried)).toBe(true);
      expectNoMarker(new CapturedLines().logWhole(carried));
    },
    TIMEOUT
  );

  test(
    'a stream: the RAW payload an error part carries once output has begun, logged as it was carried',
    async () => {
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(rawPayloadMidStream() as never);
      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();
      const parts: { type: string; error?: unknown }[] = [];
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        parts.push(read.value as { type: string; error?: unknown });
      }

      const carried = parts.find((part) => part.type === 'error')?.error;

      expect(JSON.stringify(carried)).toContain(RESPONSE_MARKER);
      expectNoMarker(new CapturedLines().logWhole(carried));
    },
    TIMEOUT
  );

  test(
    'a typed error marked by anyone: what it wraps is marked with it, though it never met the transport',
    async () => {
      const wrapped = apiError({ message: 'Overloaded', statusCode: 529, type: 'overloaded_error', isRetryable: true });

      const error = ProviderFailureLine.mark(TransientProviderError.wrap(wrapped, 'claude-test'));

      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error.cause));
      expect(lines.structured()[0].error).toMatchObject({ statusCode: 529, modelId: 'claude-test' });
    },
    TIMEOUT
  );

  test(
    'a vendor code rides the line only as an identifier — prose in a code field does not',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doGenerate: async () => {
          throw apiError({ message: 'odd', statusCode: 400, type: 'prose in a code field' });
        },
      });

      const error = await generateObjectAgainst(model);

      const lines = new CapturedLines();
      expect(lines.logWhole(error)).not.toContain('prose in a code field');
      expect(lines.structured()[0].error).toMatchObject({ statusCode: 400, code: 400 });
    },
    TIMEOUT
  );

  test(
    'the OpenAI SDK`s error through the plain retried call: its headers and body stay off the line',
    async () => {
      const sdkError = new OpenAiSdkError(
        400,
        { message: RESPONSE_MARKER, type: 'invalid_request_error', code: 'context_length_exceeded' },
        'too long',
        new Headers({ 'x-request-id': HEADER_MARKER })
      );

      const error = await caught(() =>
        new LlmTransportRetry({ budgetMs: 5_000 }).run(
          () => {
            throw sdkError;
          },
          { isRetryable: () => false, modelId: 'gpt-test' }
        )
      );

      expect(error).toBe(sdkError);
      expect(inspect(error, { depth: 10 })).toContain(RESPONSE_MARKER);
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({
        statusCode: 400,
        modelId: 'gpt-test',
        code: 'context_length_exceeded',
      });
    },
    TIMEOUT
  );

  test(
    'generateObject: an answer that does not parse — the error keeps the model`s text, the line does not',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doGenerate: async () => ({
          content: [{ type: 'text' as const, text: `not json at all ${RESPONSE_MARKER}` }],
          finishReason: 'stop' as never,
          usage: {
            inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 1, text: 1, reasoning: 0 },
          } as never,
          warnings: [],
          response: {
            id: 'r1',
            modelId: 'claude-test',
            timestamp: new Date(0),
            headers: { 'request-id': HEADER_MARKER },
          },
        }),
      });

      const error = await generateObjectAgainst(model);

      expect((error as Error).name).toBe('AI_NoObjectGeneratedError');
      expect(inspect(error, { depth: 10 })).toContain(RESPONSE_MARKER);
      const lines = new CapturedLines();
      expectNoMarker(lines.logWhole(error));
      expect(lines.structured()[0].error).toMatchObject({
        name: 'AI_NoObjectGeneratedError',
        modelId: 'claude-test',
        message:
          'The model call failed (AI_NoObjectGeneratedError) on claude-test: the answer did not parse into the requested shape',
      });
    },
    TIMEOUT
  );

  describe('the development switch', () => {
    const failing = () =>
      new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-test',
        doGenerate: async () => {
          throw apiError({ message: 'prompt is too long', statusCode: 400, type: 'invalid_request_error' });
        },
      });

    test(
      'both gates open: the error prints as it is',
      async () => {
        const error = await generateObjectAgainst(failing());
        process.env.DEVELOPMENT = 'true';
        process.env.CONVERSATION_LOG_PROVIDER_PAYLOADS = '1';

        expect(new CapturedLines().logWhole(error)).toContain(REQUEST_MARKER);
      },
      TIMEOUT
    );

    test.each([
      ['the development gate alone', { DEVELOPMENT: 'true' }],
      ['the payload switch alone', { CONVERSATION_LOG_PROVIDER_PAYLOADS: '1' }],
      ['the payload switch set to anything but 1', { DEVELOPMENT: 'true', CONVERSATION_LOG_PROVIDER_PAYLOADS: 'true' }],
    ])(
      '%s: the line stays clean',
      async (_gate, vars) => {
        const error = await generateObjectAgainst(failing());
        Object.assign(process.env, vars);

        expectNoMarker(new CapturedLines().logWhole(error));
      },
      TIMEOUT
    );
  });

  test(
    'an error that is not the provider`s keeps its own words',
    async () => {
      const error = await caught(() =>
        new LlmTransportRetry({ budgetMs: 5_000 }).run(
          () => {
            throw new TypeError('fetch failed');
          },
          { isRetryable: () => false, modelId: 'claude-test' }
        )
      );

      expect(new CapturedLines().logWhole(error)).toContain('fetch failed');
    },
    TIMEOUT
  );
});
