import { createOpenAI } from '@ai-sdk/openai';
import { Conversation, type StreamPart, type StreamSource } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * A WEB SEARCH'S SOURCES ARRIVE WHEN THE SEARCH SETTLES — for OpenAI too (ask 842, the founder's R10.1
 * smoke: "for astra, source icons didn't show up in real time as searches happened. they did show up
 * after the fact when the response completed").
 *
 * Anthropic's adapter mints one `source` part per search result right after the search's result, so a
 * Claude turn's site icons fill in as each search lands. OpenAI's adapter hands a search's results over
 * only inside the provider-executed tool's RESULT (`{ action, sources }` — it asks the Responses API for
 * `web_search_call.action.sources` whenever the web search tool rides the request) and mints `source`
 * parts only for the answer's `url_citation` annotations, which arrive with the final text. Before the
 * fix the library's translator (`mapFullStream`) passed only the latter on, so an OpenAI turn (GPT-6
 * Astra: minutes of reasoning before the text) showed no source until its answer was being written.
 *
 * No network, no key: the REAL `@ai-sdk/openai` Responses adapter reads a Responses event stream shaped
 * exactly like the wire (the event types and fields of the AI SDK's own recorded web-search fixture,
 * with this suite's own pages), so every assertion below is what a chat consumer holds at a given
 * point of the stream.
 */
const TIMEOUT = 30_000;

const NODE_NOTES = 'https://nodejs.org/en/blog/release/v26.10.0';
const NODE_INDEX = 'https://nodejs.org/en/about/previous-releases';
const NODE_NEWS = 'https://example.com/news/node-26-10';
const PY_DOWNLOADS = 'https://www.python.org/downloads/';
const PY_NEWS = 'https://example.org/python-3-14-7';
const PY_RELEASE = 'https://www.python.org/downloads/release/python-3147/';

/** One Responses turn: reasoning, a search (3 pages + an API source), reasoning, a second search (overlapping one page), the cited answer. */
function responsesEvents(): Array<Record<string, unknown>> {
  let seq = 0;
  const ev = (event: Record<string, unknown>) => ({ ...event, sequence_number: seq++ });
  const response = (status: string, extra: Record<string, unknown> = {}) => ({
    id: 'resp_sources_live',
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
  const reasoning = (index: number, id: string) => [
    ev({ type: 'response.output_item.added', output_index: index, item: { id, type: 'reasoning', summary: [] } }),
    ev({ type: 'response.output_item.done', output_index: index, item: { id, type: 'reasoning', summary: [] } }),
  ];
  const search = (index: number, id: string, query: string, sources: Array<Record<string, string>>) => [
    ev({
      type: 'response.output_item.added',
      output_index: index,
      item: { id, type: 'web_search_call', status: 'in_progress' },
    }),
    ev({ type: 'response.web_search_call.in_progress', output_index: index, item_id: id }),
    ev({ type: 'response.web_search_call.searching', output_index: index, item_id: id }),
    ev({ type: 'response.web_search_call.completed', output_index: index, item_id: id }),
    ev({
      type: 'response.output_item.done',
      output_index: index,
      item: { id, type: 'web_search_call', status: 'completed', action: { type: 'search', query, sources } },
    }),
  ];
  const text1 = 'Node.js 26.10.0 shipped on September 22.';
  const text2 = ' Python 3.14.7 is the latest stable release.';
  const citation = (url: string, title: string, start: number, end: number) => ({
    type: 'url_citation',
    url,
    title,
    start_index: start,
    end_index: end,
  });
  // OpenAI appends its own `utm_source=openai` to every cited url — the same page a search returned
  // without it.
  const citeNode = citation(`${NODE_NOTES}?utm_source=openai`, 'Node.js 26.10.0 release notes', 0, text1.length);
  const citePy = citation(
    `${PY_RELEASE}?utm_source=openai`,
    'Python Release Python 3.14.7 | Python.org',
    text1.length,
    text1.length + text2.length
  );
  const msg = 'msg_sources_live';
  return [
    ev({ type: 'response.created', response: response('in_progress') }),
    ev({ type: 'response.in_progress', response: response('in_progress') }),
    ...reasoning(0, 'rs_1'),
    ...search(1, 'ws_node', 'latest Node.js release', [
      { type: 'url', url: NODE_NOTES },
      { type: 'url', url: NODE_INDEX },
      { type: 'api', name: 'oai-news' },
      { type: 'url', url: NODE_NEWS },
    ]),
    ...reasoning(2, 'rs_2'),
    ...search(3, 'ws_python', 'latest Python release', [
      { type: 'url', url: PY_DOWNLOADS },
      // The same page the first search returned — one source, never two.
      { type: 'url', url: NODE_INDEX },
      { type: 'url', url: PY_NEWS },
    ]),
    ev({
      type: 'response.output_item.added',
      output_index: 4,
      item: { id: msg, type: 'message', status: 'in_progress', content: [], role: 'assistant' },
    }),
    ev({
      type: 'response.content_part.added',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      part: { type: 'output_text', annotations: [], logprobs: [], text: '' },
    }),
    ev({
      type: 'response.output_text.delta',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      delta: text1,
      logprobs: [],
    }),
    ev({
      type: 'response.output_text.annotation.added',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      annotation_index: 0,
      annotation: citeNode,
    }),
    ev({
      type: 'response.output_text.delta',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      delta: text2,
      logprobs: [],
    }),
    ev({
      type: 'response.output_text.annotation.added',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      annotation_index: 1,
      annotation: citePy,
    }),
    // The first page cited at a second claim — nothing new to say about it.
    ev({
      type: 'response.output_text.annotation.added',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      annotation_index: 2,
      annotation: citeNode,
    }),
    ev({
      type: 'response.output_text.done',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      text: text1 + text2,
      logprobs: [],
    }),
    ev({
      type: 'response.content_part.done',
      item_id: msg,
      output_index: 4,
      content_index: 0,
      part: { type: 'output_text', annotations: [citeNode, citePy, citeNode], logprobs: [], text: text1 + text2 },
    }),
    ev({
      type: 'response.output_item.done',
      output_index: 4,
      item: {
        id: msg,
        type: 'message',
        status: 'completed',
        role: 'assistant',
        content: [
          { type: 'output_text', annotations: [citeNode, citePy, citeNode], logprobs: [], text: text1 + text2 },
        ],
      },
    }),
    ev({
      type: 'response.completed',
      response: response('completed', {
        usage: {
          input_tokens: 9_000,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens: 400,
          output_tokens_details: { reasoning_tokens: 300 },
          total_tokens: 9_400,
        },
      }),
    }),
  ];
}

/** The Responses API over the wire: server-sent events, one `data:` line per event. Records each request body. */
function responsesModel(requests: Array<Record<string, unknown>>) {
  const provider = createOpenAI({
    apiKey: 'test-key-never-used',
    fetch: async (_url, init) => {
      requests.push(JSON.parse(String(init?.body ?? '{}')));
      const body = responsesEvents()
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

const newConversation = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'web-search-sources-live-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });

async function run() {
  const requests: Array<Record<string, unknown>> = [];
  const result = await newConversation().generateStream({
    messages: ['What are the latest Node.js and Python releases?'],
    model: responsesModel(requests),
  });
  const parts: StreamPart[] = [];
  for await (const part of result.fullStream) {
    parts.push(part);
  }
  return { requests, parts, result };
}

/** The sources a consumer holds (every `source` part so far, in arrival order) at each moment named by `at`. */
function sourcesAt(parts: StreamPart[], at: (part: StreamPart) => boolean): StreamSource[] {
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
}

const settled = (id: string) => (part: StreamPart) => part.type === 'tool-settled' && part.id === id;
const called = (id: string) => (part: StreamPart) => part.type === 'tool-call' && part.id === id;
const firstText = (part: StreamPart) => part.type === 'text-delta';
const urls = (sources: StreamSource[]) => sources.map((s) => s.url);
const allSources = (parts: StreamPart[]) =>
  parts.filter((p): p is Extract<StreamPart, { type: 'source' }> => p.type === 'source').map((p) => p.source);

describe('a web search’s sources arrive when the search settles (OpenAI Responses, the real adapter)', () => {
  test(
    'the request asks the Responses API for each search’s sources',
    async () => {
      const { requests } = await run();
      expect(requests).toHaveLength(1);
      expect(requests[0].include).toEqual(expect.arrayContaining(['web_search_call.action.sources']));
    },
    TIMEOUT
  );

  test(
    'by the time the second search starts, the first search’s pages are the turn’s sources — in the order it returned them',
    async () => {
      const { parts } = await run();
      // Nothing before the first search settles …
      expect(sourcesAt(parts, settled('ws_node'))).toEqual([]);
      // … its three pages (the API source carries no page) once it has — before the next search begins.
      expect(urls(sourcesAt(parts, called('ws_python')))).toEqual([NODE_NOTES, NODE_INDEX, NODE_NEWS]);
    },
    TIMEOUT
  );

  test(
    'the second search adds only the pages the turn did not already hold, before the answer’s first word',
    async () => {
      const { parts } = await run();
      expect(urls(sourcesAt(parts, firstText))).toEqual([NODE_NOTES, NODE_INDEX, NODE_NEWS, PY_DOWNLOADS, PY_NEWS]);
    },
    TIMEOUT
  );

  test(
    'a citation of a page a search returned names it (its title rides the same address); a page no search returned is added; a repeat citation adds nothing',
    async () => {
      const { parts } = await run();
      const answered = allSources(parts).slice(5);
      expect(answered).toEqual([
        // The page search 1 returned bare, now titled — under the SAME address, so every list keyed by
        // address holds one entry for it.
        { url: NODE_NOTES, title: 'Node.js 26.10.0 release notes' },
        // Cited, never returned by a search: new, as the provider cited it.
        { url: `${PY_RELEASE}?utm_source=openai`, title: 'Python Release Python 3.14.7 | Python.org' },
      ]);
    },
    TIMEOUT
  );

  test(
    'the buffered read of the round’s sources is the list the stream carried',
    async () => {
      const { parts, result } = await run();
      expect(await result.sources).toEqual(allSources(parts));
    },
    TIMEOUT
  );
});
