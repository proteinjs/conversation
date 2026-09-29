import fs from 'fs';
import path from 'path';
import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE VETO ON THE WIRE. `conversation.webSearchVeto.test.ts` pins what the two helpers return; this suite pins
 * what the PROVIDER RECEIVES — the exact request body each provider client posts, captured at `fetch` (the
 * models are the real ones the library routes a model string to; the fake fetch answers 400 and no request
 * leaves the process).
 *
 * (a) THE VETO IS TOTAL: a conversation built with `webSearch: 'off'` posts no web-search tool to Anthropic,
 *     OpenAI, xAI or Google — and no tool choice naming one — even when the request's own flag asks for a search.
 * (b) THE DEFAULT IS UNTOUCHED: a conversation built without the option posts what it always did — the tool-use
 *     providers carry the search on every request (forced on the ask); Google carries it on the ask only.
 *
 * With `WEB_SEARCH_REQUEST_RECORD_DIR` set, every captured body is written there byte-for-byte as the provider
 * received it (`<provider>-<case>.json`), so a default request can be compared across versions of the library.
 */
type RequestBody = Record<string, any>;
type Captured = { url: string; raw: string; body: RequestBody };

type ProviderCase = { provider: string; model: string; key: string };
const PROVIDERS: ProviderCase[] = [
  { provider: 'anthropic', model: 'claude-opus-4-8', key: 'ANTHROPIC_API_KEY' },
  { provider: 'openai', model: 'gpt-5.5', key: 'OPENAI_API_KEY' },
  { provider: 'xai', model: 'grok-4.3', key: 'XAI_API_KEY' },
  { provider: 'google', model: 'gemini-3.5-flash', key: 'GOOGLE_GENERATIVE_AI_API_KEY' },
];

/** Every spelling a provider's web-search tool has on the wire (the tool's type, the tool's name, Google's grounding tool). */
const SEARCH_MARKS = ['web_search', 'googleSearch', 'google_search'];

const originalFetch = globalThis.fetch;
const previousKeys = new Map<string, string | undefined>();

beforeAll(() => {
  // The provider clients read their key at request time; the fake fetch below answers every request in-process.
  for (const { key } of PROVIDERS) {
    previousKeys.set(key, process.env[key]);
    process.env[key] = 'test-key-never-used';
  }
});

afterAll(() => {
  globalThis.fetch = originalFetch;
  previousKeys.forEach((value, key) => {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  });
});

/** The one request a turn posts: captured at fetch, answered 400 so the round ends at once (the body is what matters). */
async function capture(conversation: Conversation, model: string, asked?: boolean): Promise<Captured> {
  const captured: Captured[] = [];
  globalThis.fetch = (async (url: unknown, init?: RequestInit) => {
    const raw = String(init?.body);
    captured.push({ url: String(url), raw, body: JSON.parse(raw) });
    return new Response(JSON.stringify({ error: { type: 'invalid_request_error', message: 'captured by the test' } }), {
      status: 400,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;
  try {
    const result = await conversation.generateStream({
      messages: ['What is in the news today?'],
      model,
      ...(asked === undefined ? {} : { webSearch: asked }),
    });
    try {
      for await (const part of result.fullStream) {
        void part;
      }
    } catch {
      // the 400 surfaces here; the request was already captured
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
  expect(captured).toHaveLength(1);
  return captured[0];
}

function record(name: string, captured: Captured): void {
  const dir = process.env.WEB_SEARCH_REQUEST_RECORD_DIR;
  if (!dir) {
    return;
  }
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `${name}.json`), captured.raw);
}

const newConversation = (veto: boolean) =>
  new Conversation({
    modelData: fixtureModelData,
    name: veto ? 'web-search-veto-request-test' : 'web-search-default-request-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    ...(veto ? { webSearch: 'off' as const } : {}),
  });

/** The tool entries of a body, whatever the provider calls the field (Anthropic/OpenAI/xAI `tools`, Google `tools` too). */
const tools = (body: RequestBody): unknown[] => (Array.isArray(body.tools) ? body.tools : []);

describe('the veto on the wire — a conversation built with webSearch: "off"', () => {
  describe.each(PROVIDERS)('$provider', ({ provider, model }) => {
    test.each([
      ['asked', true],
      ['not asked', undefined],
      ['asked off', false],
    ])(
      'the request the provider receives carries no web-search tool and no choice naming one (%s)',
      async (label, asked) => {
        const captured = await capture(newConversation(true), model, asked as boolean | undefined);
        record(`${provider}-vetoed-${label.replace(/ /g, '-')}`, captured);
        for (const mark of SEARCH_MARKS) {
          expect(captured.raw).not.toContain(mark);
        }
        expect(tools(captured.body)).toEqual([]);
        expect(captured.body.tool_choice).toBeUndefined();
        expect(captured.body.toolConfig).toBeUndefined();
      }
    );
  });
});

describe('the default on the wire — a conversation built without the option', () => {
  test('Anthropic: the search tool rides every request; the ask forces it', async () => {
    const plain = await capture(newConversation(false), 'claude-opus-4-8');
    record('anthropic-default-not-asked', plain);
    expect(tools(plain.body)).toEqual([expect.objectContaining({ type: 'web_search_20250305', name: 'web_search' })]);
    expect(plain.body.tool_choice).toEqual({ type: 'auto' });

    const asked = await capture(newConversation(false), 'claude-opus-4-8', true);
    record('anthropic-default-asked', asked);
    expect(tools(asked.body)).toEqual([expect.objectContaining({ type: 'web_search_20250305', name: 'web_search' })]);
    expect(asked.body.tool_choice).toEqual(expect.objectContaining({ type: 'tool', name: 'web_search' }));
  });

  test('OpenAI: the search tool rides every request; the ask forces it', async () => {
    const plain = await capture(newConversation(false), 'gpt-5.5');
    record('openai-default-not-asked', plain);
    expect(tools(plain.body)).toEqual([expect.objectContaining({ type: 'web_search' })]);
    expect(plain.body.tool_choice).toEqual('auto');

    const asked = await capture(newConversation(false), 'gpt-5.5', true);
    record('openai-default-asked', asked);
    expect(tools(asked.body)).toEqual([expect.objectContaining({ type: 'web_search' })]);
    expect(asked.body.tool_choice).toEqual(expect.objectContaining({ type: 'web_search' }));
  });

  test('xAI: the search tool rides every request (the ask reaches the wire as the tool alone — the provider package posts no tool_choice for a forced provider tool)', async () => {
    const plain = await capture(newConversation(false), 'grok-4.3');
    record('xai-default-not-asked', plain);
    expect(tools(plain.body)).toEqual([expect.objectContaining({ type: 'web_search' })]);
    expect(plain.body.tool_choice).toEqual('auto');

    const asked = await capture(newConversation(false), 'grok-4.3', true);
    record('xai-default-asked', asked);
    expect(tools(asked.body)).toEqual([expect.objectContaining({ type: 'web_search' })]);
  });

  test('Google: the grounding tool rides the ask only', async () => {
    const plain = await capture(newConversation(false), 'gemini-3.5-flash');
    record('google-default-not-asked', plain);
    expect(plain.raw).not.toContain('googleSearch');
    expect(tools(plain.body)).toEqual([]);

    const asked = await capture(newConversation(false), 'gemini-3.5-flash', true);
    record('google-default-asked', asked);
    expect(tools(asked.body)).toEqual([expect.objectContaining({ googleSearch: expect.any(Object) })]);
  });
});
