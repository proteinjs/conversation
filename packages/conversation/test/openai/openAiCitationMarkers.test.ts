import { OpenAiCitationMarkers } from '../../src/OpenAiCitationMarkers';

/**
 * Grammar tests for the one sanitizer owner: a marker run spans U+E200
 * through the next U+E201 (or end of text when unterminated) and is removed
 * whole; any other U+E200-U+E2FF char outside a run is dropped alone;
 * surrounding prose is preserved exactly.
 */

describe('OpenAiCitationMarkers.strip', () => {
  test('removes a terminated marker run whole, preserving surrounding prose exactly', () => {
    expect(OpenAiCitationMarkers.strip('Before.\uE200cite\uE202turn0search1\uE201 After.')).toBe('Before. After.');
  });

  test('removes multiple runs, including runs at the start and end of the text', () => {
    expect(
      OpenAiCitationMarkers.strip('\uE200cite\uE202turn0search0\uE201Middle.\uE200navlist\uE202turn0news1\uE201')
    ).toBe('Middle.');
  });

  test('an unterminated run drops to the end of the text', () => {
    expect(OpenAiCitationMarkers.strip('Prose stays.\uE200cite\uE202turn0view0')).toBe('Prose stays.');
  });

  test('a stray marker-alphabet char outside a run is dropped alone', () => {
    expect(OpenAiCitationMarkers.strip('One\uE202two\uE2FFthree')).toBe('Onetwothree');
  });

  test('text without marker chars passes through unchanged', () => {
    const text = 'Plain prose with unicode: café, éèê, and turn 1 of 3.';
    expect(OpenAiCitationMarkers.strip(text)).toBe(text);
  });
});

describe('OpenAiCitationMarkers push (streaming state)', () => {
  test('a run split across chunk boundaries is recognized and dropped', () => {
    const markers = new OpenAiCitationMarkers();
    expect(markers.push('Tall tower.\uE200cite\uE202turn0')).toBe('Tall tower.');
    expect(markers.push('search1\uE201 Built 1889.')).toBe(' Built 1889.');
  });

  test('a chunk entirely inside a run returns empty', () => {
    const markers = new OpenAiCitationMarkers();
    expect(markers.push('Start.\uE200cite')).toBe('Start.');
    expect(markers.push('\uE202turn0search2\uE202turn0search3')).toBe('');
    expect(markers.push('\uE201End.')).toBe('End.');
  });

  test('run state resets after close so later prose flows normally', () => {
    const markers = new OpenAiCitationMarkers();
    expect(markers.push('A\uE200x\uE201B')).toBe('AB');
    expect(markers.push('C')).toBe('C');
  });
});

describe('OpenAiCitationMarkers.stripStream', () => {
  test('elides chunks left empty by stripping and cleans the rest', async () => {
    async function* chunks() {
      yield 'Hello.';
      yield '\uE200cite\uE202turn0search0\uE201';
      yield ' World.';
    }
    const out: string[] = [];
    for await (const chunk of OpenAiCitationMarkers.stripStream(chunks())) {
      out.push(chunk);
    }
    expect(out).toEqual(['Hello.', ' World.']);
  });
});

describe('OpenAiCitationMarkers.sourcesFromUrlCitations', () => {
  test('mints url+title entries from url_citation annotations, deduped by url with the first winning', () => {
    expect(
      OpenAiCitationMarkers.sourcesFromUrlCitations([
        { type: 'url_citation', url: 'https://a.example', title: 'A', start_index: 0, end_index: 1 },
        { type: 'url_citation', url: 'https://b.example', title: 'B', start_index: 2, end_index: 3 },
        { type: 'url_citation', url: 'https://a.example', title: 'A (again)', start_index: 4, end_index: 5 },
      ])
    ).toEqual([
      { url: 'https://a.example', title: 'A' },
      { url: 'https://b.example', title: 'B' },
    ]);
  });

  test('ignores non-url annotation types and entries without a url; empty title is omitted', () => {
    expect(
      OpenAiCitationMarkers.sourcesFromUrlCitations([
        { type: 'file_citation', file_id: 'f1', filename: 'notes.txt', index: 0 },
        { type: 'url_citation', title: 'No url' },
        { type: 'url_citation', url: 'https://c.example', title: '' },
        null,
        'not-an-annotation',
      ])
    ).toEqual([{ url: 'https://c.example' }]);
  });

  test('returns [] for an empty annotations list', () => {
    expect(OpenAiCitationMarkers.sourcesFromUrlCitations([])).toEqual([]);
  });
});

describe('OpenAiCitationMarkers.dedupeSourcesByUrl', () => {
  test('first occurrence per url wins; url-less entries pass through', () => {
    expect(
      OpenAiCitationMarkers.dedupeSourcesByUrl([
        { url: 'https://a.example', title: 'A' },
        { title: 'no url 1' },
        { url: 'https://a.example', title: 'A later' },
        { title: 'no url 2' },
        { url: 'https://b.example' },
      ])
    ).toEqual([
      { url: 'https://a.example', title: 'A' },
      { title: 'no url 1' },
      { title: 'no url 2' },
      { url: 'https://b.example' },
    ]);
  });
});

describe('OpenAiCitationMarkers admitSource (streaming state)', () => {
  test('admits a url once per stream instance, independent of marker-run state', () => {
    const markers = new OpenAiCitationMarkers();
    // Mid-run: the SDK mints source parts while a marker run straddles chunks.
    expect(markers.push('Claim.\uE200cite\uE202turn0')).toBe('Claim.');
    expect(markers.admitSource('https://a.example')).toBe(true);
    expect(markers.push('search1\uE201 Done.')).toBe(' Done.');
    expect(markers.admitSource('https://a.example')).toBe(false);
    expect(markers.admitSource('https://b.example')).toBe(true);
  });
});
