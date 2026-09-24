import { inspect } from 'util';
import { createOpenAI } from '@ai-sdk/openai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Logger, Log } from '@proteinjs/logger';
import { Conversation, type StreamPart, type StreamSource } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE PROVIDER'S WORDS STAY OFF THE LIBRARY'S OWN LINES, on the stream path as it is now: the
 * round's `onError` line, the buffered read's throw, the bounded utterance's failure line and the
 * streaming egress under a web search. A provider words its error itself and may quote the request
 * in it (a refused prompt, a value it could not parse), so the words a person's card carries — the
 * THROW's, `providerErrorClause` — are not the words a log line carries: the line carries the error
 * marked, and prints as the vendor's code, the model and a sentence of the library's.
 *
 * Every scenario checks its premise first — what is thrown or handed over still carries the words —
 * and then reads the lines: the library's own (what its writer hands the console) and a caller's
 * (a Logger with its own writer, the error whole on the line and inside `obj`).
 *
 * No network, no key: the sources scenario feeds the real OpenAI Responses adapter a wire-shaped
 * event stream (as the sources suites do); the others use the AI SDK's mock model.
 */
const TIMEOUT = 30_000;
/** What a provider's message quotes back: the request. Never on a line. */
const MARKER = 'rst_quoted_prompt_0cc175b9c0f1b6a831c399e269772661';
const PAGE_A = 'https://docs.example.com/releases/26.10';
const PAGE_B = 'https://docs.example.com/releases/';

/** Everything the console is handed from here on, as a writer would render it. */
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

/** What a caller's lines read as, under a structured writer and the default one. */
class CapturedLines {
  private readonly logs: Log[] = [];
  readonly logger = new Logger({
    name: 'Caller',
    logWriter: { write: (log: Log) => void this.logs.push(log) } as never,
  });

  logWhole(error: unknown): string {
    this.logger.error({ message: 'The model call failed', error: error as Error });
    this.logger.warn({ message: 'The model call failed; carrying on', obj: { step: 'answer', failure: error } });
    return this.logs
      .map((log) => [
        JSON.stringify({ message: log.message, error: log.error, obj: log.obj }),
        inspect({ error: log.error, obj: log.obj }, { depth: 10, maxStringLength: 2000 }),
      ])
      .flat()
      .join('\n');
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

const newConversation = (logLevel: 'error' | 'warn' = 'error') =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'provider-error-lines-stream-path-test',
    logLevel,
    limits: { enforceLimits: false },
  });

/** One Responses turn over the wire: a search that returns two pages, then the provider's error event. */
function responsesEventsEndingOnError(): Array<Record<string, unknown>> {
  let seq = 0;
  const ev = (event: Record<string, unknown>) => ({ ...event, sequence_number: seq++ });
  const response = {
    id: 'resp_error_after_search',
    object: 'response',
    created_at: 1790236000,
    status: 'in_progress',
    incomplete_details: null,
    model: 'gpt-6-astra',
    output: [],
    service_tier: 'default',
    usage: null,
  };
  const action = {
    type: 'search',
    query: 'the release',
    sources: [
      { type: 'url', url: PAGE_A },
      { type: 'url', url: PAGE_B },
    ],
  };
  return [
    ev({ type: 'response.created', response }),
    ev({ type: 'response.in_progress', response }),
    ev({
      type: 'response.output_item.added',
      output_index: 0,
      item: { id: 'ws_1', type: 'web_search_call', status: 'in_progress' },
    }),
    ev({ type: 'response.web_search_call.in_progress', output_index: 0, item_id: 'ws_1' }),
    ev({ type: 'response.web_search_call.searching', output_index: 0, item_id: 'ws_1' }),
    ev({ type: 'response.web_search_call.completed', output_index: 0, item_id: 'ws_1' }),
    ev({
      type: 'response.output_item.done',
      output_index: 0,
      item: { id: 'ws_1', type: 'web_search_call', status: 'completed', action },
    }),
    ev({
      type: 'error',
      error: {
        type: 'server_error',
        code: 'server_error',
        message: `The server had an error while processing ${MARKER}`,
        param: null,
      },
    }),
  ];
}

function responsesModelEndingOnError() {
  const provider = createOpenAI({
    apiKey: 'test-key-never-used',
    fetch: async () => {
      const body = responsesEventsEndingOnError()
        .map((event) => `data: ${JSON.stringify(event)}\n\n`)
        .join('');
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(body));
            controller.close();
          },
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    },
  });
  return provider.responses('gpt-6-astra');
}

/** A stream that carries a provider's RAW error payload before any text — a 400-class type the transport surfaces at once. */
const rawPayloadBeforeText = () =>
  new MockLanguageModelV3({
    provider: 'anthropic.messages',
    modelId: 'claude-test',
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: 'stream-start' as const, warnings: [] },
        {
          type: 'error' as const,
          error: { type: 'error', error: { type: 'invalid_request_error', message: `prompt is too long: ${MARKER}` } },
        },
      ]),
    }),
  });

const sourcesOf = (parts: StreamPart[]): StreamSource[] =>
  parts.filter((p): p is Extract<StreamPart, { type: 'source' }> => p.type === 'source').map((p) => p.source);

describe('the provider`s words stay off the library`s own lines on the stream path', () => {
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
    'under a web search: the pages a search returned are the turn`s sources before the provider`s error event ends the stream; the round`s line names the failure by its code, never by the provider`s words',
    async () => {
      const written = consoleWrites();
      const result = await newConversation().generateStream({
        messages: ['Look it up.'],
        model: responsesModelEndingOnError(),
      });
      const parts: StreamPart[] = [];
      const error = await caught(async () => {
        for await (const part of result.fullStream) {
          parts.push(part);
        }
      });

      // The sources-live path held: both pages reached the stream at the search's settle.
      expect(sourcesOf(parts)).toEqual([{ url: PAGE_A }, { url: PAGE_B }]);
      // The premise — what is THROWN carries the provider's words (the person's card reads them) …
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain(MARKER);
      // … and what the round hands over as its failure is the provider's own event, untouched.
      const failure = await result.failure;
      expect(JSON.stringify(failure)).toContain(MARKER);
      // The library's own line: the code and the model, never the words; and no other console
      // write (the client library's default print of a failed stream) carries them either.
      const console_ = written.join('\n');
      expect(console_).toContain('The round ended on an error');
      expect(console_).toContain('server_error');
      expect(console_).toContain('gpt-6-astra');
      expect(console_).not.toContain(MARKER);
      // A caller's line about the failure it was handed: the same.
      expect(new CapturedLines().logWhole(failure)).not.toContain(MARKER);
      expect(new CapturedLines().logWhole(error)).not.toContain(MARKER);
    },
    TIMEOUT
  );

  test(
    'the buffered read: the error it throws carries the provider`s words; the same error on a line carries the code, the provider and the model instead',
    async () => {
      const written = consoleWrites();

      const error = await caught(() =>
        newConversation().generateResponse({ messages: ['say something'], model: rawPayloadBeforeText() as never })
      );

      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain(MARKER);
      expect(written.join('\n')).toContain('The round ended on an error');
      expect(written.join('\n')).not.toContain(MARKER);
      const lines = new CapturedLines();
      const text = lines.logWhole(error);
      expect(text).not.toContain(MARKER);
      expect(text).toContain('invalid_request_error');
      expect(text).toContain('claude-test');
      expect(text).toContain('anthropic');
    },
    TIMEOUT
  );

  test(
    'the bounded utterance: its own line about a failed line names the failure`s code, never the provider`s words',
    async () => {
      const written = consoleWrites();
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
      const conversation = newConversation('warn') as unknown as UtteranceInternals;

      const parts = conversation.utter({
        model: new LlmTransportRetry({ budgetMs: 5_000 }).wrap(rawPayloadBeforeText() as never),
        transcript: [{ role: 'user', content: 'and the login page?' }],
        inputs: [{ text: 'also the header' }],
        provider: 'anthropic',
        modelString: 'claude-test',
        abortSignal: new AbortController().signal,
        onResult: () => undefined,
      });
      let line: string | undefined;
      for (let next = await parts.next(); ; next = await parts.next()) {
        if (next.done) {
          line = next.value;
          break;
        }
      }

      expect(line).toBeUndefined();
      const console_ = written.join('\n');
      expect(console_).toContain('The bounded utterance has no line');
      expect(console_).toContain('invalid_request_error');
      expect(console_).not.toContain(MARKER);
    },
    TIMEOUT
  );

  describe('the development switch, at the library`s own line', () => {
    const roundLine = async (): Promise<string> => {
      const written = consoleWrites();
      const result = await newConversation().generateStream({
        messages: ['say something'],
        model: rawPayloadBeforeText() as never,
      });
      await caught(async () => {
        for await (const _part of result.fullStream) {
          // read to the failure
        }
      });
      return written.join('\n');
    };

    test(
      'both gates open: the round`s line prints the error as it is',
      async () => {
        process.env.DEVELOPMENT = 'true';
        process.env.CONVERSATION_LOG_PROVIDER_PAYLOADS = '1';

        const line = await roundLine();

        expect(line).toContain('The round ended on an error');
        expect(line).toContain(MARKER);
      },
      TIMEOUT
    );

    test.each([
      ['the development gate alone', { DEVELOPMENT: 'true' }],
      ['the payload switch alone', { CONVERSATION_LOG_PROVIDER_PAYLOADS: '1' }],
      ['the payload switch set to anything but 1', { DEVELOPMENT: 'true', CONVERSATION_LOG_PROVIDER_PAYLOADS: 'true' }],
    ])(
      '%s: the round`s line stays clean',
      async (_gate, vars) => {
        Object.assign(process.env, vars);

        const line = await roundLine();

        expect(line).toContain('The round ended on an error');
        expect(line).not.toContain(MARKER);
      },
      TIMEOUT
    );
  });
});
