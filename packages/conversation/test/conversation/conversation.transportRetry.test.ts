import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';

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
      let text = '';
      for await (const part of result.fullStream) {
        if (part.type === 'text-delta') {
          text += part.textDelta;
        }
      }

      expect(calls).toBe(3);
      expect(text).toBe('hello world');
    },
    TIMEOUT
  );

  test(
    'generateStream: an error PART surfaces as a thrown error (no silent empty result)',
    async () => {
      const model = new MockLanguageModelV3({
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            { type: 'error' as const, error: new Error('transport died mid-stream') },
          ]),
        }),
      });

      const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
      await expect(
        (async () => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          for await (const _part of result.fullStream) {
            // consume
          }
        })()
      ).rejects.toThrow('transport died mid-stream');
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
