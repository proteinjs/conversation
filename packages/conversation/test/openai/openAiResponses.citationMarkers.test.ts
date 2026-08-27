import { OpenAiResponses } from '../../src/OpenAiResponses';

/**
 * OpenAI Responses web-search output embeds in-band citation-marker runs in
 * `output_text` using Unicode private-use-area delimiters (U+E200 opens a run,
 * U+E202 separates fields, U+E201 closes it), referencing `turnXsearchY` /
 * `turnXviewY` / `turnXnewsY` ids; matching `url_citation` annotations ride
 * out-of-band on the content part. Rendered raw, the delimiters show as tofu
 * glyph blocks around the ids (prod tickets b2c01570 / 52494d00).
 *
 * These tests drive the buffered read path (generateText -> extractAssistantText)
 * with a marker-bearing Responses payload and assert the text emerges clean:
 * no U+E200-U+E2FF chars, no marker ids, all surrounding prose intact.
 */

const RUN_OPEN = '\uE200';
const RUN_CLOSE = '\uE201';
const FIELD_SEP = '\uE202';

const MARKER_CHAR_RE = /[\uE200-\uE2FF]/;
const MARKER_ID_RE = /turn\d+(?:search|view|news)\d+/;

const PROSE_1 = 'Paris is lovely in spring.';
const PROSE_2 = 'The Louvre is busiest on weekends.';

/** Text as the Responses API produces it after a web search: prose with marker runs appended to claims. */
const MARKED_TEXT =
  `${PROSE_1}${RUN_OPEN}cite${FIELD_SEP}turn1search0${FIELD_SEP}turn1search2${RUN_CLOSE} ` +
  `${PROSE_2}${RUN_OPEN}cite${FIELD_SEP}turn1view0${RUN_CLOSE}` +
  `${RUN_OPEN}navlist${FIELD_SEP}Paris sights${FIELD_SEP}turn1news1${RUN_CLOSE}`;

/** A completed Responses payload whose output_text part carries the marker runs + url_citation annotations. */
function createMarkedResponseFixture() {
  return {
    id: 'resp_citation_fixture',
    status: 'completed',
    output: [
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'output_text',
            text: MARKED_TEXT,
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
                start_index: MARKED_TEXT.indexOf(PROSE_2) + PROSE_2.length,
                end_index: MARKED_TEXT.indexOf(PROSE_2) + PROSE_2.length + 1,
              },
              // The same page cited at a second claim — the API mints a SECOND annotation for
              // the same url (different title text is possible); sources must dedupe by url
              // with the first entry winning.
              {
                type: 'url_citation',
                url: 'https://example.com/paris-guide',
                title: 'Paris travel guide (mirror)',
                start_index: MARKED_TEXT.length,
                end_index: MARKED_TEXT.length,
              },
            ],
          },
        ],
      },
    ],
  };
}

/** What the fixture's annotations must emerge as: one entry per cited url, first title wins. */
const EXPECTED_SOURCES = [
  { url: 'https://example.com/paris-guide', title: 'Paris travel guide' },
  { url: 'https://example.com/louvre-hours', title: 'Louvre visiting hours' },
];

/**
 * The OpenAI SDK client is constructed in the OpenAiResponses constructor and
 * requires an api-key env var to exist; the fake client swapped in below means
 * no request is ever made. The env var is restored immediately so live-gated
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

describe('OpenAiResponses citation markers (buffered read path)', () => {
  test('generateText strips PUA-delimited marker runs while preserving all prose', async () => {
    const adapter = createAdapterWithFakeClient(createMarkedResponseFixture());

    const result = await adapter.generateText({ messages: ['Tell me about Paris.'] });

    expect(result.message).not.toMatch(MARKER_CHAR_RE);
    expect(result.message).not.toMatch(MARKER_ID_RE);
    expect(result.message).toContain(PROSE_1);
    expect(result.message).toContain(PROSE_2);
  });

  test('generateText strips markers on the direct output_text fallback path', async () => {
    const adapter = createAdapterWithFakeClient({
      id: 'resp_citation_fixture_direct',
      status: 'completed',
      output: [],
      output_text: MARKED_TEXT,
    });

    const result = await adapter.generateText({ messages: ['Tell me about Paris.'] });

    expect(result.message).not.toMatch(MARKER_CHAR_RE);
    expect(result.message).not.toMatch(MARKER_ID_RE);
    expect(result.message).toContain(PROSE_1);
    expect(result.message).toContain(PROSE_2);
  });
});

/**
 * The stripped runs' information must not be dropped: each url_citation annotation riding
 * out-of-band on the content part surfaces as a house source entry (url + title), deduped
 * by url — the buffered counterpart of the AI-SDK streaming path's source parts.
 */
describe('OpenAiResponses citation sources (buffered read path)', () => {
  test('generateText surfaces url_citation annotations as source entries deduped by url', async () => {
    const adapter = createAdapterWithFakeClient(createMarkedResponseFixture());

    const result = await adapter.generateText({ messages: ['Tell me about Paris.'] });

    expect(result.sources).toEqual(EXPECTED_SOURCES);
  });

  test('generateText yields zero sources when the response carries no annotations', async () => {
    const fixture = createMarkedResponseFixture();
    (fixture.output[0].content[0] as { annotations: unknown[] }).annotations = [];
    const adapter = createAdapterWithFakeClient(fixture);

    const result = await adapter.generateText({ messages: ['Tell me about Paris.'] });

    expect(result.sources).toEqual([]);
  });

  test('generateText yields zero sources on the direct output_text fallback path (no parts, no annotations)', async () => {
    const adapter = createAdapterWithFakeClient({
      id: 'resp_citation_fixture_direct',
      status: 'completed',
      output: [],
      output_text: MARKED_TEXT,
    });

    const result = await adapter.generateText({ messages: ['Tell me about Paris.'] });

    expect(result.sources).toEqual([]);
  });
});
