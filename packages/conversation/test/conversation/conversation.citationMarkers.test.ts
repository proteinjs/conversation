import { Conversation } from '../../src/Conversation';
import { OpenAiResponses } from '../../src/OpenAiResponses';

/**
 * OpenAI Responses web-search output embeds in-band citation-marker runs in
 * streamed text using Unicode private-use-area delimiters (U+E200 opens a run,
 * U+E202 separates fields, U+E201 closes it) around `turnXsearchY` /
 * `turnXviewY` / `turnXnewsY` ids. Rendered raw, they show as tofu glyph
 * blocks in chat (prod tickets b2c01570 / 52494d00).
 *
 * These tests drive the streaming read path (`mapFullStream`, which every
 * fullStream consumer reads) with AI-SDK-shaped text-delta parts and assert
 * the OpenAI egress emerges clean — including a marker run split across two
 * deltas — while other providers' text passes through untouched.
 */

const MARKER_CHAR_RE = /[\uE200-\uE2FF]/;
const MARKER_ID_RE = /turn\d+(?:search|view|news)\d+/;

type EmittedPart = { type: string; textDelta?: string; source?: { url?: string; title?: string } };

type ConversationInternals = {
  mapFullStream: (aiSdkFullStream: AsyncIterable<unknown>, provider: string) => AsyncIterable<EmittedPart>;
};

function internals(): ConversationInternals {
  return new Conversation({ name: 'test-citationMarkers' }) as unknown as ConversationInternals;
}

async function* fakeSdkStream(parts: unknown[]): AsyncIterable<unknown> {
  for (const part of parts) {
    yield part;
  }
}

async function collect(stream: AsyncIterable<EmittedPart>): Promise<EmittedPart[]> {
  const out: EmittedPart[] = [];
  for await (const part of stream) {
    out.push(part);
  }
  return out;
}

const textDeltas = (parts: EmittedPart[]) => parts.filter((p) => p.type === 'text-delta');
const joinedText = (parts: EmittedPart[]) =>
  textDeltas(parts)
    .map((p) => p.textDelta ?? '')
    .join('');

describe('Conversation citation markers (streaming read path)', () => {
  test('openai text-deltas emerge clean, including a marker run split across two deltas', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([
          // Marker run STARTS in this delta and closes in the next one.
          { type: 'text-delta', delta: 'The Eiffel Tower is 330 m tall.\uE200cite\uE202turn0' },
          { type: 'text-delta', delta: 'search1\uE201 It was completed in 1889.' },
          { type: 'text-delta', delta: ' Visit at night.\uE200cite\uE202turn0view0\uE201' },
        ]),
        'openai'
      )
    );

    const joined = joinedText(parts);
    expect(joined).toBe('The Eiffel Tower is 330 m tall. It was completed in 1889. Visit at night.');
    for (const part of textDeltas(parts)) {
      expect(part.textDelta).not.toMatch(MARKER_CHAR_RE);
      expect(part.textDelta).not.toMatch(MARKER_ID_RE);
    }
  });

  test('a delta that is entirely marker payload emits no text-delta part', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([
          { type: 'text-delta', delta: 'Prose before.' },
          { type: 'text-delta', delta: '\uE200navlist\uE202Sights\uE202turn0news2\uE201' },
          { type: 'text-delta', delta: ' Prose after.' },
        ]),
        'openai'
      )
    );

    expect(joinedText(parts)).toBe('Prose before. Prose after.');
    expect(textDeltas(parts)).toHaveLength(2);
  });

  test('source parts pass through beside sanitized text', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([
          { type: 'text-delta', delta: 'See the official site.\uE200cite\uE202turn0search3\uE201' },
          { type: 'source', sourceType: 'url', url: 'https://example.com/site', title: 'Official site' },
        ]),
        'openai'
      )
    );

    expect(joinedText(parts)).toBe('See the official site.');
    const sources = parts.filter((p) => p.type === 'source');
    expect(sources).toHaveLength(1);
    expect(sources[0].source).toEqual({ url: 'https://example.com/site', title: 'Official site' });
  });

  test('non-openai providers pass text through untouched', async () => {
    const markedText = 'Anthropic prose with a stray marker \uE200cite\uE202turn0search0\uE201 kept as-is.';
    const parts = await collect(
      internals().mapFullStream(fakeSdkStream([{ type: 'text-delta', delta: markedText }]), 'anthropic')
    );

    expect(joinedText(parts)).toBe(markedText);
  });
});

/**
 * The stripped runs' information must not be dropped: url_citation annotations surface as
 * house source entries. On the live AI-SDK streaming path the OpenAI adapter already mints
 * one `source` part per url_citation annotation (out-of-band from the text deltas), so the
 * streaming egress's job is dedupe — a url cited at several claims arrives as several parts
 * and must emerge once. On the background-polling path (`generateStreamViaPolling`) nothing
 * arrived at all before this: the annotations OpenAiResponses collects must surface through
 * both sources channels of the fabricated StreamResult.
 */
describe('Conversation citation sources', () => {
  test('openai: repeated same-url source parts collapse to one, straddled marker run and all', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([
          // Marker run STARTS in this delta; the SDK mints the annotation's source part
          // mid-run, before the closing delimiter arrives in the next delta.
          { type: 'text-delta', delta: 'The tower is 330 m tall.\uE200cite\uE202turn0' },
          { type: 'source', sourceType: 'url', url: 'https://example.com/tower', title: 'Tower facts' },
          { type: 'text-delta', delta: 'search1\uE201 It opened in 1889.' },
          // The same page cited at a second claim — the SDK mints a SECOND part for the same url.
          { type: 'source', sourceType: 'url', url: 'https://example.com/tower', title: 'Tower facts' },
        ]),
        'openai'
      )
    );

    expect(joinedText(parts)).toBe('The tower is 330 m tall. It opened in 1889.');
    const sourceParts = parts.filter((p) => p.type === 'source');
    expect(sourceParts).toHaveLength(1);
    expect(sourceParts[0].source).toEqual({ url: 'https://example.com/tower', title: 'Tower facts' });
  });

  test('openai: distinct urls each emerge once', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([
          { type: 'source', sourceType: 'url', url: 'https://example.com/a', title: 'A' },
          { type: 'source', sourceType: 'url', url: 'https://example.com/b', title: 'B' },
        ]),
        'openai'
      )
    );

    expect(parts.filter((p) => p.type === 'source').map((p) => p.source)).toEqual([
      { url: 'https://example.com/a', title: 'A' },
      { url: 'https://example.com/b', title: 'B' },
    ]);
  });

  test('marker runs without annotations invent no source parts', async () => {
    const parts = await collect(
      internals().mapFullStream(
        fakeSdkStream([{ type: 'text-delta', delta: 'Claim.\uE200cite\uE202turn0search0\uE201' }]),
        'openai'
      )
    );

    expect(joinedText(parts)).toBe('Claim.');
    expect(parts.filter((p) => p.type === 'source')).toHaveLength(0);
  });

  test('non-openai source parts pass through without url dedupe', async () => {
    const src = { type: 'source', sourceType: 'url', url: 'https://example.com/a', title: 'A' };
    const parts = await collect(internals().mapFullStream(fakeSdkStream([src, { ...src }]), 'anthropic'));

    expect(parts.filter((p) => p.type === 'source')).toHaveLength(2);
  });
});

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// Background-polling path (generateStreamViaPolling)
// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

const PROSE_1 = 'Paris is lovely in spring.';
const PROSE_2 = 'The Louvre is busiest on weekends.';

const POLLING_MARKED_TEXT =
  `${PROSE_1}\uE200cite\uE202turn1search0\uE201 ` + `${PROSE_2}\uE200cite\uE202turn1view0\uE201`;

/** What the fixture's annotations must emerge as: one entry per cited url, first title wins. */
const POLLING_EXPECTED_SOURCES = [
  { url: 'https://example.com/paris-guide', title: 'Paris travel guide' },
  { url: 'https://example.com/louvre-hours', title: 'Louvre visiting hours' },
];

/** A completed Responses payload: marker runs in the text, url_citation annotations out-of-band. */
function createPollingResponseFixture() {
  return {
    id: 'resp_polling_citation_fixture',
    status: 'completed',
    output: [
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'output_text',
            text: POLLING_MARKED_TEXT,
            annotations: [
              {
                type: 'url_citation',
                url: 'https://example.com/paris-guide',
                title: 'Paris travel guide',
                start_index: PROSE_1.length,
                end_index: PROSE_1.length + 1,
              },
              {
                type: 'url_citation',
                url: 'https://example.com/louvre-hours',
                title: 'Louvre visiting hours',
                start_index: POLLING_MARKED_TEXT.length,
                end_index: POLLING_MARKED_TEXT.length,
              },
              // Duplicate url at a second claim — must dedupe.
              {
                type: 'url_citation',
                url: 'https://example.com/paris-guide',
                title: 'Paris travel guide (mirror)',
                start_index: POLLING_MARKED_TEXT.length,
                end_index: POLLING_MARKED_TEXT.length,
              },
            ],
          },
        ],
      },
    ],
  };
}

/**
 * A real OpenAiResponses whose OpenAI client is swapped for a fixture-returning fake — the
 * constructor requires an api-key env var to exist; it is restored immediately so live-gated
 * suites in the same worker never see a bogus key.
 */
function createAdapterWithFakeClient(fixture: unknown): OpenAiResponses {
  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = 'test-key-never-used';
  try {
    const adapter = new OpenAiResponses();
    (adapter as unknown as { client: unknown }).client = {
      responses: {
        create: async () => fixture,
        // backgroundMode polls retrieve until a terminal status; the fixture is already completed.
        retrieve: async () => fixture,
      },
    };
    return adapter;
  } finally {
    if (prevKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = prevKey;
    }
  }
}

describe('Conversation citation sources (background-polling path)', () => {
  test('url_citation annotations surface as source parts and sources, deduped by url, beside clean text', async () => {
    const conversation = new Conversation({ name: 'test-citationSources' });
    (conversation as unknown as { createOpenAiResponses: () => OpenAiResponses }).createOpenAiResponses = () =>
      createAdapterWithFakeClient(createPollingResponseFixture());

    const stream = await conversation.generateStream({
      messages: ['Tell me about Paris.'],
      model: 'gpt-5.2',
      backgroundMode: true,
    });

    const parts = (await collect(stream.fullStream as AsyncIterable<EmittedPart>)) as EmittedPart[];
    const text = await stream.text;
    const sources = await stream.sources;

    // Text egress stays clean (the strip behavior, unchanged).
    expect(text).not.toMatch(MARKER_CHAR_RE);
    expect(text).not.toMatch(MARKER_ID_RE);
    expect(text).toContain(PROSE_1);
    expect(text).toContain(PROSE_2);

    // The fabricated fullStream carries the citations as house source parts (what
    // per-message source lists downstream consume) …
    const sourceParts = parts.filter((p) => p.type === 'source');
    expect(sourceParts.map((p) => p.source)).toEqual(POLLING_EXPECTED_SOURCES);

    // … and the sources promise resolves the same deduped entries.
    expect(sources).toEqual(POLLING_EXPECTED_SOURCES);
  });

  test('a response without annotations yields zero source parts and empty sources', async () => {
    const fixture = createPollingResponseFixture();
    (fixture.output[0].content[0] as { annotations: unknown[] }).annotations = [];
    const conversation = new Conversation({ name: 'test-citationSources' });
    (conversation as unknown as { createOpenAiResponses: () => OpenAiResponses }).createOpenAiResponses = () =>
      createAdapterWithFakeClient(fixture);

    const stream = await conversation.generateStream({
      messages: ['Tell me about Paris.'],
      model: 'gpt-5.2',
      backgroundMode: true,
    });

    const parts = (await collect(stream.fullStream as AsyncIterable<EmittedPart>)) as EmittedPart[];

    expect(parts.filter((p) => p.type === 'source')).toHaveLength(0);
    expect(await stream.sources).toEqual([]);
  });
});
