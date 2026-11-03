import { inspect } from 'util';
import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE ROUND'S ERROR LINE NAMES ITS CAUSE (n3xa row 3119, 2026-11-03). Error Reporting groups
 * entries by the message's text, and "The round ended on an error" was the same words for every
 * provider failure — one group in the founder's ledger carried a Sonnet 400, a Gemini 400 and a
 * dev cluster's reproduction under one record, with a triage written for the first cause. The
 * line now carries the cause the marked error already knows — the provider family, the model, the
 * HTTP status, the vendor's code — so each cause is its own group and its own record. The words
 * are identifiers only: never the provider's clause (which may quote the request).
 */
const REQUEST_MARKER = 'REQUEST-BODY-MARKER';
const RESPONSE_MARKER = 'the provider explains at length what was wrong with the request';

const apiError = (args: { statusCode: number; type: string }) =>
  new APICallError({
    message: 'prompt is too long',
    url: 'https://provider.test/v1/messages',
    requestBodyValues: { model: 'claude-test', messages: [{ role: 'user', content: REQUEST_MARKER }] },
    statusCode: args.statusCode,
    responseHeaders: {},
    responseBody: JSON.stringify({ type: 'error', error: { type: args.type, message: RESPONSE_MARKER } }),
    isRetryable: false,
    data: { type: 'error', error: { type: args.type, message: RESPONSE_MARKER } },
  });

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

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'round-error-line-cause',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

const drain = async (model: MockLanguageModelV3) => {
  try {
    const result = await newConversation().generateStream({ messages: ['hi'], model: model as never });
    for await (const _part of result.fullStream) {
      // read to the failure
    }
  } catch {
    // the egress throws; the log line is what these tests read
  }
};

afterEach(() => jest.restoreAllMocks());

describe('the round-ended line names the cause so Error Reporting groups by it', () => {
  test('a provider refusal: the provider family, the model, the HTTP status and the vendor code — never the clause, never the request', async () => {
    const written = consoleWrites();
    await drain(
      new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-sonnet-5-5',
        doStream: async () => {
          throw apiError({ statusCode: 400, type: 'invalid_request_error' });
        },
      })
    );
    const logged = written.join('\n');
    expect(logged).toContain('The round ended on an error: anthropic claude-sonnet-5-5 HTTP 400 invalid_request_error');
    expect(logged).not.toContain(RESPONSE_MARKER);
    expect(logged).not.toContain(REQUEST_MARKER);
  });

  test('a raw error part mid-stream (no HTTP status): the provider family, the model and the vendor code', async () => {
    const written = consoleWrites();
    // A code the transport never retries (an overload would be retried past the test's clock).
    const errorPart: LanguageModelV3StreamPart = {
      type: 'error',
      error: { type: 'error', error: { type: 'invalid_prompt', code: 'invalid_prompt', message: RESPONSE_MARKER } },
    };
    await drain(
      new MockLanguageModelV3({
        provider: 'google.generative-ai',
        modelId: 'gemini-4-argon',
        doStream: async () => ({
          stream: convertArrayToReadableStream([{ type: 'stream-start', warnings: [] }, errorPart]),
        }),
      })
    );
    const logged = written.join('\n');
    expect(logged).toContain('The round ended on an error: google gemini-4-argon invalid_prompt');
    expect(logged).not.toContain(RESPONSE_MARKER);
  });

  test('two causes on one model are two lines; one cause on two models are two lines', async () => {
    const written = consoleWrites();
    await drain(
      new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-sonnet-5-5',
        doStream: async () => {
          throw apiError({ statusCode: 529, type: 'overloaded_error' });
        },
      })
    );
    await drain(
      new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-haiku-5',
        doStream: async () => {
          throw apiError({ statusCode: 400, type: 'invalid_request_error' });
        },
      })
    );
    const lines = written.filter((line) => line.includes('The round ended on an error'));
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain('anthropic claude-sonnet-5-5 HTTP 529 overloaded_error');
    expect(lines[1]).toContain('anthropic claude-haiku-5 HTTP 400 invalid_request_error');
  });
});
