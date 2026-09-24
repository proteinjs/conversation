import { createOpenAI } from '@ai-sdk/openai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation, type StreamPart, type StreamSource } from '../../src/Conversation';
import type { ConversationSkill } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import type { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE EDGES OF A WEB SEARCH'S SOURCES (the check of `conversation.webSearchSourcesLive`): what the
 * translator yields when a search returns nothing, when a page comes back from two searches, when
 * the answer cites a page no search returned, when a search runs after the answer began, when a
 * result is not a search's at all — and the one admission rule's key: exactly OpenAI's own
 * `utm_source=openai` is set aside when matching a citation to the page a search returned, and
 * nothing else about the address is touched. The buffered read (`StreamResult.sources`) is pinned to
 * the stream in every case here, including the round that spans two steps.
 *
 * No network, no key: the real `@ai-sdk/openai` Responses adapter reads a wire-shaped event stream
 * (as the live suite does); the two-step round and the Anthropic pin use the AI SDK's mock model.
 */
const TIMEOUT = 30_000;

type Page = { type: 'url'; url: string } | { type: 'api'; name: string };
type Search = { id: string; query: string; sources?: Page[]; action?: Record<string, unknown> };
type Citation = { url: string; title: string };
type Turn = {
  /** Searches before the answer's first word. */
  before: Search[];
  text: string;
  citations: Citation[];
  /** A search that runs after the answer began (the text continues after it). */
  after?: { search: Search; text: string; citations: Citation[] };
};

/** One Responses turn over the wire, in the event shapes the live suite uses. */
function responsesEvents(turn: Turn): Array<Record<string, unknown>> {
  let seq = 0;
  let index = 0;
  const ev = (event: Record<string, unknown>) => ({ ...event, sequence_number: seq++ });
  const response = (status: string, extra: Record<string, unknown> = {}) => ({
    id: 'resp_edges',
    object: 'response',
    created_at: 1790236000,
    status,
    incomplete_details: null,
    model: 'gpt-6-astra',
    output: [],
    service_tier: 'default',
    usage: null,
    ...extra,
  });
  const search = (s: Search) => {
    const i = index++;
    const action = s.action ?? { type: 'search', query: s.query, ...(s.sources ? { sources: s.sources } : {}) };
    return [
      ev({
        type: 'response.output_item.added',
        output_index: i,
        item: { id: s.id, type: 'web_search_call', status: 'in_progress' },
      }),
      ev({ type: 'response.web_search_call.in_progress', output_index: i, item_id: s.id }),
      ev({ type: 'response.web_search_call.searching', output_index: i, item_id: s.id }),
      ev({ type: 'response.web_search_call.completed', output_index: i, item_id: s.id }),
      ev({
        type: 'response.output_item.done',
        output_index: i,
        item: { id: s.id, type: 'web_search_call', status: 'completed', action },
      }),
    ];
  };
  const message = (msg: string, text: string, citations: Citation[], offset: number) => {
    const i = index++;
    const annotations = citations.map((c, k) => ({
      type: 'url_citation',
      url: c.url,
      title: c.title,
      start_index: offset + k,
      end_index: offset + k + 1,
    }));
    return [
      ev({
        type: 'response.output_item.added',
        output_index: i,
        item: { id: msg, type: 'message', status: 'in_progress', content: [], role: 'assistant' },
      }),
      ev({
        type: 'response.content_part.added',
        item_id: msg,
        output_index: i,
        content_index: 0,
        part: { type: 'output_text', annotations: [], logprobs: [], text: '' },
      }),
      ev({
        type: 'response.output_text.delta',
        item_id: msg,
        output_index: i,
        content_index: 0,
        delta: text,
        logprobs: [],
      }),
      ...annotations.map((annotation, k) =>
        ev({
          type: 'response.output_text.annotation.added',
          item_id: msg,
          output_index: i,
          content_index: 0,
          annotation_index: k,
          annotation,
        })
      ),
      ev({ type: 'response.output_text.done', item_id: msg, output_index: i, content_index: 0, text, logprobs: [] }),
      ev({
        type: 'response.content_part.done',
        item_id: msg,
        output_index: i,
        content_index: 0,
        part: { type: 'output_text', annotations, logprobs: [], text },
      }),
      ev({
        type: 'response.output_item.done',
        output_index: i,
        item: {
          id: msg,
          type: 'message',
          status: 'completed',
          role: 'assistant',
          content: [{ type: 'output_text', annotations, logprobs: [], text }],
        },
      }),
    ];
  };
  return [
    ev({ type: 'response.created', response: response('in_progress') }),
    ev({ type: 'response.in_progress', response: response('in_progress') }),
    ...turn.before.flatMap(search),
    ...message('msg_1', turn.text, turn.citations, 0),
    ...(turn.after
      ? [...search(turn.after.search), ...message('msg_2', turn.after.text, turn.after.citations, turn.text.length)]
      : []),
    ev({
      type: 'response.completed',
      response: response('completed', {
        usage: {
          input_tokens: 900,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens: 40,
          output_tokens_details: { reasoning_tokens: 0 },
          total_tokens: 940,
        },
      }),
    }),
  ];
}

function responsesModel(turn: Turn) {
  const provider = createOpenAI({
    apiKey: 'test-key-never-used',
    fetch: async () => {
      const body = responsesEvents(turn)
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

const newConversation = (skills: ConversationSkill[] = []) =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'web-search-sources-edges-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills,
  });

async function run(turn: Turn) {
  const result = await newConversation().generateStream({ messages: ['Look it up.'], model: responsesModel(turn) });
  const parts: StreamPart[] = [];
  for await (const part of result.fullStream) {
    parts.push(part);
  }
  return { parts, result };
}

const allSources = (parts: StreamPart[]): StreamSource[] =>
  parts.filter((p): p is Extract<StreamPart, { type: 'source' }> => p.type === 'source').map((p) => p.source);
const sourcesBefore = (parts: StreamPart[], at: (part: StreamPart) => boolean): StreamSource[] => {
  const held: StreamSource[] = [];
  for (const part of parts) {
    if (at(part)) {
      return held;
    }
    if (part.type === 'source') {
      held.push(part.source);
    }
  }
  throw new Error('the moment never came');
};
const settled = (id: string) => (part: StreamPart) => part.type === 'tool-settled' && part.id === id;
const firstText = (part: StreamPart) => part.type === 'text-delta';
const url = (u: string): Page => ({ type: 'url', url: u });
const cited = (u: string) => `${u}${u.includes('?') ? '&' : '?'}utm_source=openai`;

const A = 'https://docs.example.com/releases/26.10';
const B = 'https://docs.example.com/releases/';
const C = 'https://news.example.org/26-10';

describe('a web search’s sources — the edges (OpenAI Responses, the real adapter)', () => {
  test(
    'a search that returns no page yields no source; the buffered read is the same empty list',
    async () => {
      const { parts, result } = await run({
        before: [
          { id: 'ws_empty', query: 'nothing', sources: [] },
          // The include flag absent from the response: no `sources` key at all.
          { id: 'ws_bare', query: 'nothing either' },
          // A feed, not a page.
          { id: 'ws_api', query: 'a feed', sources: [{ type: 'api', name: 'oai-news' }] },
        ],
        text: 'Nothing to report.',
        citations: [],
      });
      expect(parts.filter((p) => p.type === 'tool-settled')).toHaveLength(3);
      expect(allSources(parts)).toEqual([]);
      expect(await result.sources).toEqual([]);
    },
    TIMEOUT
  );

  test(
    'a result that is not a search’s (an opened page, a find-in-page) carries no source',
    async () => {
      const { parts, result } = await run({
        before: [
          { id: 'ws_open', query: '', action: { type: 'open_page', url: A } },
          { id: 'ws_find', query: '', action: { type: 'find_in_page', url: A, pattern: 'release' } },
        ],
        text: 'Read it.',
        citations: [],
      });
      expect(allSources(parts)).toEqual([]);
      expect(await result.sources).toEqual([]);
    },
    TIMEOUT
  );

  test(
    'the same page from two searches is one source, on the first search; a page returned bare has no title key',
    async () => {
      const { parts, result } = await run({
        before: [
          { id: 'ws_1', query: 'one', sources: [url(A), url(B)] },
          { id: 'ws_2', query: 'two', sources: [url(B), url(C), url(A)] },
        ],
        text: 'Both.',
        citations: [],
      });
      expect(sourcesBefore(parts, settled('ws_2'))).toEqual([{ url: A }, { url: B }]);
      expect(sourcesBefore(parts, firstText)).toEqual([{ url: A }, { url: B }, { url: C }]);
      expect(Object.keys(allSources(parts)[0])).toEqual(['url']);
      expect(await result.sources).toEqual(allSources(parts));
    },
    TIMEOUT
  );

  test(
    'a page cited but never returned is added as cited; a later search returning that page bare adds nothing (and never strips the cited form already shown)',
    async () => {
      const { parts, result } = await run({
        before: [{ id: 'ws_1', query: 'one', sources: [url(A)] }],
        text: 'First.',
        citations: [{ url: cited(C), title: 'C, cited' }],
        after: {
          search: { id: 'ws_3', query: 'more', sources: [url(C), url(B)] },
          text: ' Then more.',
          citations: [{ url: cited(B), title: 'B, cited' }],
        },
      });
      expect(allSources(parts)).toEqual([
        { url: A },
        { url: cited(C), title: 'C, cited' },
        // C again, bare, from the later search: nothing; B is new, bare …
        { url: B },
        // … and then named by the answer, under the address the search returned.
        { url: B, title: 'B, cited' },
      ]);
      // The search after the answer began settled its page live, before the second run of words.
      const secondWords = (part: StreamPart) => part.type === 'text-delta' && part.textDelta.includes('Then more');
      expect(sourcesBefore(parts, secondWords).map((s) => s.url)).toEqual([A, cited(C), B]);
      expect(await result.sources).toEqual(allSources(parts));
    },
    TIMEOUT
  );

  test(
    'the key sets aside exactly `utm_source=openai` — a page’s own utm_source, its utm_medium, its encoding and its bare parameters all stay',
    async () => {
      const NEWSLETTER = 'https://docs.example.com/p?utm_source=newsletter';
      const MEDIUM = 'https://docs.example.com/q?utm_medium=social';
      const ENCODED = 'https://docs.example.com/w?title=Release%20Notes';
      const BARE_PARAM = 'https://docs.example.com/r?a=1&b';
      const OPENAI2 = 'https://docs.example.com/s?utm_source=openai2';
      const { parts, result } = await run({
        before: [
          {
            id: 'ws_1',
            query: 'one',
            sources: [url(NEWSLETTER), url(MEDIUM), url(ENCODED), url(BARE_PARAM), url(OPENAI2)],
          },
        ],
        text: 'All five.',
        citations: [
          { url: cited(NEWSLETTER), title: 'Newsletter page' },
          { url: cited(MEDIUM), title: 'Medium page' },
          { url: cited(ENCODED), title: 'Encoded page' },
          { url: cited(BARE_PARAM), title: 'Bare-param page' },
          { url: cited(OPENAI2), title: 'openai2 page' },
          // A page that IS utm-tagged by its owner, cited with OpenAI's tag in front.
          {
            url: 'https://docs.example.com/p?utm_source=openai&utm_source=newsletter',
            title: 'Newsletter page, tag first',
          },
        ],
      });
      const sources = allSources(parts);
      // Five pages returned bare, then each one named ONCE under the address the search returned —
      // never a second entry under the cited address.
      expect(sources.map((s) => s.url)).toEqual([
        NEWSLETTER,
        MEDIUM,
        ENCODED,
        BARE_PARAM,
        OPENAI2,
        NEWSLETTER,
        MEDIUM,
        ENCODED,
        BARE_PARAM,
        OPENAI2,
      ]);
      expect(sources.slice(5).map((s) => s.title)).toEqual([
        'Newsletter page',
        'Medium page',
        'Encoded page',
        'Bare-param page',
        'openai2 page',
      ]);
      expect(await result.sources).toEqual(sources);
    },
    TIMEOUT
  );
});

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

function buildSkill(fn: Function): ConversationSkill {
  return {
    getId: () => 'sources-edges-test-skill',
    getName: () => 'SourcesEdgesTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [fn],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

describe('a web search’s sources across a round of two steps (OpenAI)', () => {
  test(
    'a search in the first step and a citation in the second: the stream carries both; the buffered read is the same list',
    async () => {
      let call = 0;
      const model = new MockLanguageModelV3({
        provider: 'openai.responses',
        modelId: 'gpt-6-astra',
        doStream: async () => {
          call++;
          if (call === 1) {
            // Step 1: the search settles with its pages, then the model calls a house function.
            return {
              stream: convertArrayToReadableStream([
                { type: 'stream-start' as const, warnings: [] },
                {
                  type: 'tool-call' as const,
                  toolCallId: 'ws_1',
                  toolName: 'web_search',
                  input: '{}',
                  providerExecuted: true,
                  dynamic: true,
                },
                {
                  type: 'tool-result' as const,
                  toolCallId: 'ws_1',
                  toolName: 'web_search',
                  result: { action: { type: 'search', query: 'one' }, sources: [url(A), url(B)] },
                  providerExecuted: true,
                  dynamic: true,
                },
                { type: 'tool-call' as const, toolCallId: 'tc_1', toolName: 'doWork', input: '{}' },
                { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
              ]),
            };
          }
          // Step 2: the answer, citing the first page.
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              { type: 'text-start' as const, id: 't1' },
              { type: 'text-delta' as const, id: 't1', delta: 'Done.' },
              { type: 'source' as const, sourceType: 'url' as const, id: 's1', url: cited(A), title: 'A, cited' },
              { type: 'text-end' as const, id: 't1' },
              { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
            ]),
          };
        },
      });
      const work: Function = {
        definition: {
          name: 'doWork',
          description: 'Does one unit of work.',
          parameters: { type: 'object', properties: {} },
        },
        call: async () => ({ ok: true }),
      };
      const result = await newConversation([buildSkill(work)]).generateStream({
        messages: ['Look it up, then work.'],
        model,
      });
      const parts: StreamPart[] = [];
      for await (const part of result.fullStream) {
        parts.push(part);
      }
      expect(call).toBe(2);
      expect(allSources(parts)).toEqual([{ url: A }, { url: B }, { url: A, title: 'A, cited' }]);
      expect(await result.sources).toEqual(allSources(parts));
    },
    TIMEOUT
  );
});

describe('an Anthropic turn’s sources are untouched (the adapter’s own parts, one for one)', () => {
  test(
    'every source part passes through as it arrived — repeats, bare pages and titles alike; the buffered read is the adapter’s list',
    async () => {
      const model = new MockLanguageModelV3({
        provider: 'anthropic.messages',
        modelId: 'claude-fixture',
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            {
              type: 'tool-call' as const,
              toolCallId: 'srv_1',
              toolName: 'web_search',
              input: '{"query":"one"}',
              providerExecuted: true,
              dynamic: true,
            },
            { type: 'source' as const, sourceType: 'url' as const, id: 's1', url: A, title: 'A' },
            { type: 'source' as const, sourceType: 'url' as const, id: 's2', url: B },
            { type: 'source' as const, sourceType: 'url' as const, id: 's3', url: A, title: 'A again' },
            {
              type: 'source' as const,
              sourceType: 'url' as const,
              id: 's4',
              url: `${A}?utm_source=openai`,
              title: 'A, tagged',
            },
            {
              type: 'tool-result' as const,
              toolCallId: 'srv_1',
              toolName: 'web_search',
              result: [{ url: A, title: 'A' }],
              providerExecuted: true,
              dynamic: true,
            },
            { type: 'text-start' as const, id: 't1' },
            { type: 'text-delta' as const, id: 't1', delta: 'Done.' },
            { type: 'text-end' as const, id: 't1' },
            { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'end_turn' }, usage },
          ]),
        }),
      });
      const result = await newConversation().generateStream({ messages: ['Look it up.'], model });
      const parts: StreamPart[] = [];
      for await (const part of result.fullStream) {
        parts.push(part);
      }
      const expected = [
        { url: A, title: 'A' },
        { url: B, title: undefined },
        { url: A, title: 'A again' },
        { url: `${A}?utm_source=openai`, title: 'A, tagged' },
      ];
      expect(allSources(parts)).toEqual(expected);
      expect(JSON.stringify(allSources(parts))).toBe(JSON.stringify(expected));
      expect(await result.sources).toEqual(expected);
    },
    TIMEOUT
  );
});
