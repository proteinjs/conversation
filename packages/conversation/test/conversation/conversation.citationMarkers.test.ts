import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

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
  return new Conversation({
    modelData: fixtureModelData,
    name: 'test-citationMarkers',
  }) as unknown as ConversationInternals;
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
 * and must emerge once. (The background-polling path that once needed its own sources channel
 * is gone — every OpenAI call streams; the adapter's own citation handling is pinned in
 * test/openai/openAiResponses.citationMarkers.test.ts.)
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
