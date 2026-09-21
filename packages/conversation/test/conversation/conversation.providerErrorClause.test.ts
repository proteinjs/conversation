import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import type { LanguageModelV3StreamPart } from '@ai-sdk/provider';
import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE PROVIDER'S CLAUSE SURVIVES A PLAIN-OBJECT ERROR. A Responses stream can end on an `error`
 * event whose payload is a plain object, not an Error (the SDK enqueues it as-is: `{ type: 'error',
 * sequence_number, error: { type, code, message } }`); its type is not one the transport retries,
 * so it reaches the round. Every place that turns that failure into words — the buffered read's
 * throw, the streaming egress's throw, the round's log line — used `String(error)`, which reads
 * "[object Object]": the person saw a house line and the log could not name what the provider said
 * (2026-09-22, live: a pro-class model refused one turn in six with a clause nobody could read).
 * The THROWS carry the provider's words; the LOG LINE carries the error marked — the vendor's code
 * and the model, never the words (ProviderFailureLine: a provider's message may quote the request).
 */
const usage = {
  inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 0, text: 0, reasoning: 0 },
};

const sseErrorPart = (): LanguageModelV3StreamPart => ({
  type: 'error',
  error: {
    type: 'error',
    sequence_number: 3,
    error: {
      type: 'invalid_prompt',
      code: 'invalid_prompt',
      message: 'The provider said no to this prompt.',
      param: null,
    },
  },
});

const failingModel = () =>
  new MockLanguageModelV3({
    provider: 'openai.responses',
    modelId: 'gpt-6-astra',
    doStream: async () => ({
      stream: convertArrayToReadableStream([{ type: 'stream-start', warnings: [] }, sseErrorPart()]),
    }),
  });

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'provider-error-clause',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

describe('a provider error that is a plain object keeps its clause', () => {
  test('the buffered read throws the provider’s own words, never "[object Object]"', async () => {
    await expect(newConversation().generateResponse({ messages: ['hi'], model: failingModel() })).rejects.toMatchObject(
      {
        message: 'The provider said no to this prompt.',
      }
    );
  });

  test('the streaming egress throws the provider’s own words from fullStream', async () => {
    const result = await newConversation().generateStream({ messages: ['hi'], model: failingModel() });
    const drain = async () => {
      for await (const _part of result.fullStream) {
        // nothing arrives before the error
      }
    };
    await expect(drain()).rejects.toMatchObject({ message: 'The provider said no to this prompt.' });
    // `failure` stays the raw error the round ended on (the contract); the words are the throw's.
    expect(await result.failure).toMatchObject({ error: { message: 'The provider said no to this prompt.' } });
  });

  test('the round’s log line names the failure by its vendor code and model — never the clause, never "[object Object]"', async () => {
    const lines: string[] = [];
    const spy = jest.spyOn(console, 'error').mockImplementation((...parts: unknown[]) => {
      lines.push(parts.map((p) => (typeof p === 'string' ? p : JSON.stringify(p))).join(' '));
    });
    try {
      const result = await newConversation().generateStream({ messages: ['hi'], model: failingModel() });
      await Promise.resolve(result.failure);
      for await (const _part of result.fullStream) {
        // drains to the error
      }
    } catch {
      // the egress throws; the log line is what this test reads
    } finally {
      spy.mockRestore();
    }
    const logged = lines.join('\n');
    expect(logged).toContain('The round ended on an error');
    // The line carries the error MARKED (ProviderFailureLine): the vendor's code and the model
    // name the failure; the provider's own words — which may quote the request — never ride it.
    expect(logged).toContain('invalid_prompt');
    expect(logged).toContain('gpt-6-astra');
    expect(logged).not.toContain('The provider said no to this prompt.');
    expect(logged).not.toContain('[object Object]');
  });
});
