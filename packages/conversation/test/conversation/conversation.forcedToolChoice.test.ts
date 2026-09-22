import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ForcedToolChoice } from '../../src/ForcedToolChoice';
import { fixtureModelData } from './fixtureModelData';

/**
 * FORCED TOOL CHOICE FOLLOWS THE MODEL — no network, no keys. A MockLanguageModelV3 stands in as
 * the provider (Conversation takes a model instance straight through resolveModel), so these run
 * the REAL wiring: Conversation → LlmTransportRetry.wrap → ForcedToolChoice.follow → model.
 *
 * The refusal is the provider's, recorded live 2026-09-22 (Anthropic, `claude-opus-5-5`, request
 * `req_011CfJxJWCjBQDurxM2y5t2z`): a forced `tool_choice: { type: 'tool', name: 'web_search' }`
 * answered with HTTP 400 and the body below — the same clause Claude Fable 5.1 has answered with
 * since 2026-09-01. Before this rule the library kept a list of ids that refuse forcing and
 * missed every new one: the web-search toggle killed the whole turn on Opus 5.5.
 */

const TIMEOUT = 30_000;
const REFUSAL_BODY =
  '{"type":"error","error":{"type":"invalid_request_error","message":"tool_choice: type \\"tool\\" and \\"any\\" are not supported for this model."},"request_id":"req_011CfJxJWCjBQDurxM2y5t2z"}';

const refusal = () =>
  new APICallError({
    message: 'tool_choice: type "tool" and "any" are not supported for this model.',
    url: 'https://api.anthropic.com/v1/messages',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody: REFUSAL_BODY,
    isRetryable: false,
    data: {
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'tool_choice: type "tool" and "any" are not supported for this model.',
      },
    },
  });

const otherBadRequest = () =>
  new APICallError({
    message: 'messages: text content blocks must be non-empty',
    url: 'https://api.anthropic.com/v1/messages',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody: '',
    isRetryable: false,
  });

const usage = {
  inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 5, text: 5, reasoning: 0 },
};

/** A searched answer: one source, then the text. */
const searchedAnswer = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    {
      type: 'source' as const,
      sourceType: 'url' as const,
      id: 's1',
      url: 'https://example.test/today',
      title: 'Today',
    },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'end_turn' }, usage },
  ]);

const plainAnswer = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'end_turn' }, usage },
  ]);

const clientToolCall = (id: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'tool-call' as const, toolCallId: id, toolName: 'echo', input: '{"value":"x"}' },
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'forced-tool-choice-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

/** An Anthropic-shaped model that refuses forcing the way the provider does, and answers on auto. */
const refusingModel = (modelId: string) =>
  new MockLanguageModelV3({
    provider: 'anthropic.messages',
    modelId,
    doStream: async ({ toolChoice }) => {
      if (toolChoice?.type === 'tool' || toolChoice?.type === 'required') {
        throw refusal();
      }
      return { stream: searchedAnswer('Today: the headline.') };
    },
  });

const toolNames = (call: { tools?: Array<{ name: string }> }) => (call.tools ?? []).map((t) => t.name);

beforeEach(() => ForcedToolChoice.forgetAll());

describe('ForcedToolChoice via Conversation', () => {
  test(
    'the provider refuses the forced search — the request is re-issued with auto, the tool still attached, and the answer flows',
    async () => {
      const model = refusingModel('claude-opus-5-5');

      const result = await newConversation().generateResponse({
        messages: ['What is a news headline from today? Include the source URL.'],
        model: model as never,
        webSearch: true,
      });

      expect(result.text).toBe('Today: the headline.');
      expect(result.sources).toEqual([{ url: 'https://example.test/today', title: 'Today' }]);
      expect(model.doStreamCalls).toHaveLength(2);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(model.doStreamCalls[1].toolChoice).toEqual({ type: 'auto' });
      // The softened request carries the same tools — only the forcing is dropped.
      expect(toolNames(model.doStreamCalls[1])).toEqual(toolNames(model.doStreamCalls[0]));
      expect(toolNames(model.doStreamCalls[1])).toContain('web_search');
      expect(ForcedToolChoice.refusesForcing('claude-opus-5-5')).toBe(true);
    },
    TIMEOUT
  );

  test(
    'the refusal is remembered for the process — the next conversation on that model sends auto from its first request',
    async () => {
      const first = refusingModel('claude-fable-5-1');
      await newConversation().generateResponse({ messages: ['news?'], model: first as never, webSearch: true });
      expect(first.doStreamCalls).toHaveLength(2);

      const second = refusingModel('claude-fable-5-1');
      const result = await newConversation().generateResponse({
        messages: ['news?'],
        model: second as never,
        webSearch: true,
      });

      expect(result.text).toBe('Today: the headline.');
      expect(second.doStreamCalls).toHaveLength(1);
      expect(second.doStreamCalls[0].toolChoice).toEqual({ type: 'auto' });
      expect(toolNames(second.doStreamCalls[0])).toContain('web_search');
    },
    TIMEOUT
  );

  test(
    'a model that accepts forcing is never softened',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-sonnet-5',
        doStream: async () => ({ stream: searchedAnswer('Searched.') }),
      });

      const result = await newConversation().generateResponse({
        messages: ['news?'],
        model: model as never,
        webSearch: true,
      });

      expect(result.text).toBe('Searched.');
      expect(model.doStreamCalls).toHaveLength(1);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(ForcedToolChoice.refusesForcing('claude-sonnet-5')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'any other 400 surfaces untouched — never softened, never remembered',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-opus-5-5',
        doStream: async () => {
          throw otherBadRequest();
        },
      });

      await expect(
        newConversation().generateResponse({ messages: ['news?'], model: model as never, webSearch: true })
      ).rejects.toThrow('text content blocks must be non-empty');
      expect(model.doStreamCalls).toHaveLength(1);
      expect(ForcedToolChoice.refusesForcing('claude-opus-5-5')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'the forced search is the first step’s only — a client tool answered on step 1 runs step 2 on the model’s own choice',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-sonnet-5',
        doStream: async () => ({ stream: calls++ === 0 ? clientToolCall('tc-1') : plainAnswer('Done: x') }),
      });
      const echo = {
        definition: {
          name: 'echo',
          description: 'Echoes the value.',
          parameters: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] },
        },
        call: async (input: { value: string }) => `echo:${input.value}`,
      };

      const result = await newConversation().generateResponse({
        messages: ['echo x'],
        model: model as never,
        webSearch: true,
        tools: [echo],
      });

      expect(result.text).toBe('Done: x');
      expect(model.doStreamCalls).toHaveLength(2);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(model.doStreamCalls[1].toolChoice).toEqual({ type: 'auto' });
    },
    TIMEOUT
  );
});

describe('Conversation.generateResponse on a refused request', () => {
  test(
    'a provider 400 with no answer THROWS its clause — never an empty answer',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-opus-5-5',
        doStream: async () => {
          throw otherBadRequest();
        },
      });

      await expect(newConversation().generateResponse({ messages: ['hi'], model: model as never })).rejects.toThrow(
        'messages: text content blocks must be non-empty'
      );
    },
    TIMEOUT
  );

  test(
    'a clean round resolves with no failure',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-opus-5-5',
        doStream: async () => ({ stream: plainAnswer('4') }),
      });
      const stream = await newConversation().generateStream({ messages: ['2+2?'], model: model as never });
      await expect(stream.text).resolves.toBe('4');
      await expect(stream.failure).resolves.toBeUndefined();
    },
    TIMEOUT
  );
});
