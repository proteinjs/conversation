import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';
import { TransientProviderError } from '../../src/TransientProviderError';

/**
 * Transport-retry layer tests — no network, no API keys. A MockLanguageModelV3 stands in as the
 * transport; `Conversation` accepts a model instance directly (resolveModel passes instances through),
 * so these exercise the REAL wiring: Conversation → LlmTransportRetry.wrap → model.
 */

const TIMEOUT = 30_000;

const newConversation = () =>
  new Conversation({ name: 'transport-retry-test', logLevel: 'error', limits: { enforceLimits: false } });

const transientError = () =>
  new APICallError({
    message: 'server exploded',
    url: 'https://example.test',
    requestBodyValues: {},
    statusCode: 500,
    responseHeaders: {},
    responseBody: '',
  });

const semanticError = () =>
  new APICallError({
    message: 'bad request',
    url: 'https://example.test',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody: '',
  });

/**
 * The payload the @ai-sdk/openai Responses transport enqueues on an `error` stream part when the
 * provider fails mid-stream: the raw SSE error chunk, NOT an APICallError — there's no HTTP status.
 * This exact shape (nested `error.type: 'server_error'`) killed a CI release run before any output.
 */
const openAiServerErrorPart = () => ({
  type: 'error' as const,
  error: {
    type: 'error',
    sequence_number: 0,
    error: { type: 'server_error', code: 'server_error', message: 'The server had an error.', param: null },
  },
});

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const objectResult = (json: string) => ({
  content: [{ type: 'text' as const, text: json }],
  finishReason: { unified: 'stop' as const, raw: 'stop' },
  usage,
  warnings: [],
});

const goodStream = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    {
      type: 'finish' as const,
      finishReason: { unified: 'stop' as const, raw: 'stop' },
      usage,
    },
  ]);

const collectText = async (result: { fullStream: AsyncIterable<{ type: string; textDelta?: string }> }) => {
  let text = '';
  for await (const part of result.fullStream) {
    if (part.type === 'text-delta') {
      text += part.textDelta;
    }
  }
  return text;
};

describe('LlmTransportRetry via Conversation', () => {
  test(
    'generateObject: transient failures retry invisibly, then succeed',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          calls++;
          if (calls <= 2) {
            throw transientError();
          }
          return objectResult('{"answer":"ok"}');
        },
      });

      const result = await newConversation().generateObject<{ answer: string }>({
        messages: ['give me the answer'],
        model: model as never,
        schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
      });

      expect(calls).toBe(3);
      expect(result.object.answer).toBe('ok');
    },
    TIMEOUT
  );

  test(
    'generateObject: semantic errors are never retried',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          calls++;
          throw semanticError();
        },
      });

      await expect(
        newConversation().generateObject({
          messages: ['x'],
          model: model as never,
          schema: { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] },
        })
      ).rejects.toThrow('bad request');
      expect(calls).toBe(1);
    },
    TIMEOUT
  );

  test(
    'generateStream: transient stream-initiation failures retry invisibly, then stream fully',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls <= 2) {
            throw transientError();
          }
          return { stream: goodStream('hello world') };
        },
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      const text = await collectText(result);

      expect(calls).toBe(3);
      expect(text).toBe('hello world');
    },
    TIMEOUT
  );

  test(
    'generateStream: a transient error PART before any output retries invisibly, then streams fully',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                openAiServerErrorPart(),
              ]),
            };
          }
          return { stream: goodStream('hello world') };
        },
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      const text = await collectText(result);

      expect(calls).toBe(2);
      expect(text).toBe('hello world');
    },
    TIMEOUT
  );

  test(
    'generateStream: a transient error THROWN before any output retries invisibly, then streams fully',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            return {
              stream: new ReadableStream({
                start(controller) {
                  controller.enqueue({ type: 'stream-start', warnings: [] });
                  controller.error(transientError());
                },
              }),
            };
          }
          return { stream: goodStream('hello world') };
        },
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      const text = await collectText(result);

      expect(calls).toBe(2);
      expect(text).toBe('hello world');
    },
    TIMEOUT
  );

  test(
    'generateStream: a semantic (non-retryable) error PART surfaces as a thrown error — no retry',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              // A 400 (e.g. prompt too long) must surface even pre-output — never retried.
              { type: 'error' as const, error: semanticError() },
            ]),
          };
        },
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      await expect(collectText(result)).rejects.toThrow('bad request');
      expect(calls).toBe(1);
    },
    TIMEOUT
  );

  test(
    'generateStream: an error AFTER output has flowed surfaces immediately, even when transient-shaped',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              { type: 'text-start' as const, id: 't1' },
              { type: 'text-delta' as const, id: 't1', delta: 'partial' },
              openAiServerErrorPart(),
            ]),
          };
        },
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      // Partial output already reached the consumer — replaying would duplicate it, so no retry.
      const streamed: string[] = [];
      await expect(
        (async () => {
          for await (const part of result.fullStream) {
            if (part.type === 'text-delta') {
              streamed.push(part.textDelta);
            }
          }
        })()
      ).rejects.toThrow();
      expect(streamed.join('')).toBe('partial');
      expect(calls).toBe(1);
    },
    TIMEOUT
  );
});

describe('LlmTransportRetry.wrap — stream part accounting', () => {
  const readAllParts = async (stream: ReadableStream<{ type: string }>) => {
    const parts: Array<{ type: string }> = [];
    const reader = stream.getReader();
    for (;;) {
      const result = await reader.read();
      if (result.done) {
        break;
      }
      parts.push(result.value);
    }
    return parts;
  };

  test(
    "doStream: a failed attempt's preamble parts are discarded — the consumer sees exactly one attempt",
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls === 1) {
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                { type: 'response-metadata' as const, id: 'failed-attempt' },
                openAiServerErrorPart(),
              ]),
            };
          }
          return { stream: goodStream('ok') };
        },
      });

      const wrapped = new LlmTransportRetry().wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] });
      const parts = await readAllParts(stream);

      expect(calls).toBe(2);
      expect(parts.map((p) => p.type)).toEqual(['stream-start', 'text-start', 'text-delta', 'text-end', 'finish']);
    },
    TIMEOUT
  );

  test(
    'doStream: budget exhaustion on a pre-output transient error surfaces the error part (preamble intact)',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              openAiServerErrorPart(),
            ]),
          };
        },
      });

      // budget of 0ms can't fit any backoff delay → exactly one attempt, error passes through.
      const wrapped = new LlmTransportRetry({ budgetMs: 0 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] });
      const parts = await readAllParts(stream);

      expect(calls).toBe(1);
      expect(parts.map((p) => p.type)).toEqual(['stream-start', 'error']);
    },
    TIMEOUT
  );
});

describe('LlmTransportRetry.run', () => {
  test(
    'wall-clock budget bounds retries and surfaces the last error',
    async () => {
      const retry = new LlmTransportRetry({ budgetMs: 1 });
      let calls = 0;
      await expect(
        retry.run(
          async () => {
            calls++;
            throw new Error('always transient');
          },
          { isRetryable: () => true }
        )
      ).rejects.toThrow('always transient');
      // budget of 1ms can't fit any backoff delay → exactly one attempt
      expect(calls).toBe(1);
    },
    TIMEOUT
  );

  test(
    'abort wins immediately — no retry after an aborted call',
    async () => {
      const retry = new LlmTransportRetry();
      const controller = new AbortController();
      let calls = 0;
      await expect(
        retry.run(
          async () => {
            calls++;
            controller.abort();
            throw new Error('boom');
          },
          { abortSignal: controller.signal, isRetryable: () => true }
        )
      ).rejects.toThrow('boom');
      expect(calls).toBe(1);
    },
    TIMEOUT
  );
});

/**
 * Budget exhaustion of a TRANSIENT-classified error surfaces a typed `TransientProviderError` (the
 * flow runner's provider-outage taxonomy routes on it); semantic errors and aborts surface the
 * ORIGINAL error untouched. Classification happens exactly once, here — the layer with ground truth.
 */
describe('LlmTransportRetry — budget-exhausted transient errors surface typed', () => {
  const overloadedError = () =>
    new APICallError({
      message: 'Overloaded',
      url: 'https://api.anthropic.test',
      requestBodyValues: {},
      statusCode: 529,
      responseHeaders: {},
      responseBody: '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
    });

  test(
    'doGenerate: a 529 that outlives the budget rejects with TransientProviderError carrying the original',
    async () => {
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          throw overloadedError();
        },
      });

      // budget of 0ms can't fit any backoff delay → exactly one attempt, surfaced typed.
      const wrapped = new LlmTransportRetry({ budgetMs: 0 }).wrap(model as never);
      let caught: unknown;
      try {
        await wrapped.doGenerate({ prompt: [] });
      } catch (error: unknown) {
        caught = error;
      }
      expect(TransientProviderError.isInstance(caught)).toBe(true);
      const typed = caught as TransientProviderError;
      expect(typed.statusCode).toBe(529);
      expect(typed.message).toContain('Overloaded');
      expect(APICallError.isInstance(typed.cause)).toBe(true);
    },
    TIMEOUT
  );

  test(
    'doGenerate: a semantic 400 surfaces the ORIGINAL error — never wrapped',
    async () => {
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          throw semanticError();
        },
      });

      const wrapped = new LlmTransportRetry({ budgetMs: 0 }).wrap(model as never);
      let caught: unknown;
      try {
        await wrapped.doGenerate({ prompt: [] });
      } catch (error: unknown) {
        caught = error;
      }
      expect(TransientProviderError.isInstance(caught)).toBe(false);
      expect(APICallError.isInstance(caught)).toBe(true);
    },
    TIMEOUT
  );

  test(
    'doStream: a budget-exhausted transient error PART carries the type tag on the surfaced part',
    async () => {
      const model = new MockLanguageModelV3({
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            openAiServerErrorPart(),
          ]),
        }),
      });

      const wrapped = new LlmTransportRetry({ budgetMs: 0 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] });
      const reader = stream.getReader();
      const parts: Array<{ type: string; error?: unknown }> = [];
      for (;;) {
        const result = await reader.read();
        if (result.done) {
          break;
        }
        parts.push(result.value as { type: string; error?: unknown });
      }

      const errorPart = parts.find((p) => p.type === 'error');
      expect(errorPart).toBeTruthy();
      expect(TransientProviderError.isInstance(errorPart!.error)).toBe(true);
      // The original raw provider payload rides along as the cause.
      expect((errorPart!.error as TransientProviderError).cause).toEqual(openAiServerErrorPart().error);
    },
    TIMEOUT
  );

  test(
    'run(): budget exhaustion wraps; an abort surfaces the original untouched',
    async () => {
      let caught: unknown;
      try {
        await new LlmTransportRetry({ budgetMs: 1 }).run(
          async () => {
            throw overloadedError();
          },
          { isRetryable: () => true }
        );
      } catch (error: unknown) {
        caught = error;
      }
      expect(TransientProviderError.isInstance(caught)).toBe(true);

      const controller = new AbortController();
      let abortCaught: unknown;
      try {
        await new LlmTransportRetry({ budgetMs: 1 }).run(
          async () => {
            controller.abort();
            throw overloadedError();
          },
          { abortSignal: controller.signal, isRetryable: () => true }
        );
      } catch (error: unknown) {
        abortCaught = error;
      }
      expect(TransientProviderError.isInstance(abortCaught)).toBe(false);
      expect(APICallError.isInstance(abortCaught)).toBe(true);
    },
    TIMEOUT
  );
});
