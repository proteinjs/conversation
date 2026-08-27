import { Conversation } from '../../src/Conversation';

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
