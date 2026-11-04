import { createAnthropic } from '@ai-sdk/anthropic';
import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ForcedToolChoice } from '../../src/ForcedToolChoice';
import { fixtureModelData } from './fixtureModelData';
import { PromptCacheScriptedTransport, type WireBody } from './PromptCacheScriptedTransport';

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
 *
 * TWO WORDS SOFTEN A FORCED CHOICE, one owner: the vendor's own table (`@ai-sdk/anthropic`'s
 * `getModelCapabilities` marks Sonnet 5.5, Opus 5.5 and Fable 5.1 at ^3.0.138) before any request,
 * and the provider's refusal — heard once, remembered — for an id the table does not know yet. The
 * refusal road here runs on ids the table does not know (`claude-opus-6`, `claude-fable-6`: models
 * newer than the adapter), which is exactly the case that road exists for.
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

/** A client tool beside the search tool — the roster a softened request must keep whole. */
const echo = {
  definition: {
    name: 'echo',
    description: 'Echoes the value.',
    parameters: { type: 'object', properties: { value: { type: 'string' } } },
  },
  call: async (input: { value?: string }) => `echo:${input.value ?? ''}`,
};

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

describe('ForcedToolChoice via Conversation — the refusal road, for an id the vendor’s table does not know', () => {
  test(
    'the provider refuses the forced search — the request is re-issued with auto, the tool still attached, and the answer flows',
    async () => {
      const model = refusingModel('claude-opus-6');

      const result = await newConversation().generateResponse({
        messages: ['What is a news headline from today? Include the source URL.'],
        model: model as never,
        webSearch: true,
        tools: [echo],
      });

      expect(result.text).toBe('Today: the headline.');
      expect(result.sources).toEqual([{ url: 'https://example.test/today', title: 'Today' }]);
      expect(model.doStreamCalls).toHaveLength(2);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(model.doStreamCalls[1].toolChoice).toEqual({ type: 'auto' });
      // The softened request carries the same tools — only the forcing is dropped.
      expect(toolNames(model.doStreamCalls[1])).toEqual(toolNames(model.doStreamCalls[0]));
      expect(toolNames(model.doStreamCalls[1])).toEqual(['echo', 'web_search']);
      expect(ForcedToolChoice.heardRefusal('claude-opus-6')).toBe(true);
      expect(ForcedToolChoice.refusesForcing('claude-opus-6')).toBe(true);
    },
    TIMEOUT
  );

  test(
    'the refusal is remembered for the process — the next conversation on that model sends auto from its first request',
    async () => {
      const first = refusingModel('claude-fable-6');
      await newConversation().generateResponse({ messages: ['news?'], model: first as never, webSearch: true });
      expect(first.doStreamCalls).toHaveLength(2);

      const second = refusingModel('claude-fable-6');
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
    'any other 400 surfaces untouched — never softened by it, never remembered',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-opus-6',
        doStream: async () => {
          throw otherBadRequest();
        },
      });

      await expect(
        newConversation().generateResponse({ messages: ['news?'], model: model as never, webSearch: true })
      ).rejects.toThrow('text content blocks must be non-empty');
      expect(model.doStreamCalls).toHaveLength(1);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(ForcedToolChoice.heardRefusal('claude-opus-6')).toBe(false);
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

/**
 * THE VENDOR'S OWN VERDICT, BEFORE ANY REQUEST. `@ai-sdk/anthropic` (^3.0.138) keeps a per-model table —
 * `getModelCapabilities`, exported from `@ai-sdk/anthropic/internal` — that marks the ids rejecting forced
 * tool use (Sonnet 5.5, Opus 5.5, Fable 5.1), and its own `prepareTools` would soften such a request itself,
 * narrower: a named choice becomes `auto` with THAT TOOL ALONE on the request (every client tool dropped, and
 * with the first tool the house's cache breakpoint on the roster), one warning on the stream. The house asks
 * the table first and softens with the WHOLE roster before the adapter sees the request — nothing narrowed,
 * nothing warned, nothing heard — and the same word covers budget thinking, under which the vendor refuses
 * forcing on any model (the adapter narrows there too, 3.0.131).
 */
describe('ForcedToolChoice — the vendor’s table softens before any request, with the whole roster', () => {
  test.each(['claude-sonnet-5-5', 'claude-opus-5-5', 'claude-fable-5-1'])(
    '%s: the forced search leaves as auto with every tool on the first request — no refusal to hear',
    async (modelId) => {
      const model = refusingModel(modelId);

      const result = await newConversation().generateResponse({
        messages: ['news?'],
        model: model as never,
        webSearch: true,
        tools: [echo],
      });

      expect(result.text).toBe('Today: the headline.');
      expect(model.doStreamCalls).toHaveLength(1);
      expect(model.doStreamCalls[0].toolChoice).toEqual({ type: 'auto' });
      expect(toolNames(model.doStreamCalls[0])).toEqual(['echo', 'web_search']);
      expect(ForcedToolChoice.refusesForcing(modelId)).toBe(true);
      expect(ForcedToolChoice.heardRefusal(modelId)).toBe(false);
    },
    TIMEOUT
  );

  /** The middleware alone on a mock, with the thinking shape the house would have built. */
  const forcedUnder = async (modelId: string, thinking: Record<string, string | number>) => {
    const inner = new MockLanguageModelV3({
      provider: 'anthropic.messages',
      modelId,
      doStream: async () => ({ stream: plainAnswer('ok') }),
    });
    await ForcedToolChoice.follow(inner).doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'news?' }] }],
      tools: [
        { type: 'function', name: 'echo', inputSchema: { type: 'object', properties: {} } },
        { type: 'function', name: 'web_search', inputSchema: { type: 'object', properties: {} } },
      ],
      toolChoice: { type: 'tool', toolName: 'web_search' },
      providerOptions: { anthropic: { thinking } },
    });
    expect(inner.doStreamCalls).toHaveLength(1);
    return inner.doStreamCalls[0];
  };

  test(
    'budget thinking refuses forcing on any model that takes a budget — softened whole; adaptive thinking, or a model that swaps the budget for adaptive, forces as sent',
    async () => {
      const budgeted = await forcedUnder('claude-sonnet-4-5', { type: 'enabled', budgetTokens: 5000 });
      expect(budgeted.toolChoice).toEqual({ type: 'auto' });
      expect(toolNames(budgeted)).toEqual(['echo', 'web_search']);

      const adaptive = await forcedUnder('claude-sonnet-4-5', { type: 'adaptive', display: 'summarized' });
      expect(adaptive.toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });

      // Haiku 5.5 rejects a budget: the adapter sends adaptive thinking in its place, under which forcing holds.
      const swapped = await forcedUnder('claude-haiku-5-5', { type: 'enabled', budgetTokens: 5000 });
      expect(swapped.toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
      expect(ForcedToolChoice.heardRefusal('claude-sonnet-4-5')).toBe(false);
    },
    TIMEOUT
  );
});

/**
 * ON THE WIRE, THROUGH THE REAL ADAPTER over a scripted transport that bills as the provider's prompt cache
 * does. Pinned on the request the provider would receive: the roster whole and the choice auto on an id the
 * table marks, with zero warnings; a named choice with the roster on an id that accepts forcing; and the
 * billing a two-step turn draws — the second step READS the prefix the first wrote, because both carry the
 * same roster (the narrowing this replaced left step 1 with the search tool alone: step 2 read nothing).
 */
describe('ForcedToolChoice on the wire at the floor', () => {
  type ToolBody = WireBody & { tool_choice?: Record<string, unknown> };

  /** A search-toggled turn with one client tool, through the real adapter: the body sent, the step's warnings. */
  async function wire(modelId: string): Promise<{ body: ToolBody; warnings: Array<Record<string, unknown>> }> {
    const transport = new PromptCacheScriptedTransport();
    const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })(modelId);
    const result = await newConversation().generateStream({
      messages: ['news?'],
      model: model as never,
      webSearch: true,
      tools: [echo],
    });
    const warnings: Array<Record<string, unknown>> = [];
    for await (const part of result.fullStream as AsyncIterable<{
      type: string;
      warnings?: Array<Record<string, unknown>>;
    }>) {
      if (part.type === 'step-start') {
        warnings.push(...(part.warnings ?? []));
      }
    }
    await result.usage;
    expect(transport.requests).toHaveLength(1);
    return { body: transport.requests[0] as ToolBody, warnings };
  }

  test(
    'Sonnet 5.5: the forced search leaves as auto with the whole roster and no warning — the adapter’s narrowing is never reached',
    async () => {
      const { body, warnings } = await wire('claude-sonnet-5-5');
      expect(body.tool_choice).toEqual({ type: 'auto' });
      // The whole roster — the client tool first, then the search tool, as the conversation lays them out.
      expect((body.tools ?? []).map((tool) => tool.name)).toEqual(['echo', 'web_search']);
      expect(warnings).toEqual([]);
      expect(ForcedToolChoice.heardRefusal('claude-sonnet-5-5')).toBe(false);
    },
    TIMEOUT
  );

  test(
    'Sonnet 5: the forced search leaves as sent — the named choice with the whole roster, no warning',
    async () => {
      const { body, warnings } = await wire('claude-sonnet-5');
      expect(body.tool_choice).toEqual({ type: 'tool', name: 'web_search' });
      expect((body.tools ?? []).map((tool) => tool.name)).toEqual(['echo', 'web_search']);
      expect(warnings).toEqual([]);
    },
    TIMEOUT
  );

  test(
    'Sonnet 5.5, a two-step turn: both steps carry the same roster, and the second reads the prompt cache the first wrote',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      transport.editTool = 'echo';
      const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })('claude-sonnet-5-5');

      const result = await newConversation().generateResponse({
        messages: ['[edit] echo it'],
        model: model as never,
        webSearch: true,
        tools: [echo],
      });

      expect(result.text).toBe('Done.');
      expect(transport.requests).toHaveLength(2);
      // THE BILLING: step 2 reads the prefix step 1 wrote (the narrowing this replaced read 0 here).
      expect(transport.billings[1].read).toBeGreaterThan(0);
      // Because both steps carry the same roster, the client tool's breakpoint first.
      const rosters = transport.requests.map((request) => (request.tools ?? []).map((tool) => tool.name));
      expect(rosters).toEqual([
        ['echo', 'web_search'],
        ['echo', 'web_search'],
      ]);
      expect(transport.requests.map((request) => (request as ToolBody).tool_choice)).toEqual([
        { type: 'auto' },
        { type: 'auto' },
      ]);
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
