/**
 * Strips OpenAI Responses in-band citation-marker runs from assistant text.
 *
 * When web search runs, the Responses API embeds marker runs directly in
 * `output_text` using Unicode private-use-area delimiters — U+E200 opens a
 * run, U+E202 separates its fields, U+E201 closes it — e.g.
 * `"\uE200cite\uE202turn1view0\uE201"`, with navlist/news/search variants
 * using the same framing around `turnXsearchY` / `turnXviewY` / `turnXnewsY`
 * ids. The matching `url_citation` annotations arrive out-of-band on the
 * content part's `annotations`, so the in-band run carries no prose; rendered
 * raw it shows as tofu glyph blocks around the ids (prod tickets b2c01570 /
 * 52494d00).
 *
 * One owner for every Responses text egress:
 * - `OpenAiResponses.extractAssistantText` (buffered / background-polling
 *   path) uses `strip`;
 * - `Conversation`'s OpenAI streaming egress uses an instance per stream
 *   (`push` carries run state across chunk boundaries) and `stripStream` for
 *   the plain text stream.
 *
 * Grammar: a run spans U+E200 through the next U+E201 — or the end of the
 * text when unterminated — and is removed whole; everything inside is marker
 * payload, never prose. Any other U+E200–U+E2FF char outside a run is a stray
 * marker glyph and is dropped alone. Surrounding prose is preserved exactly.
 *
 * The stripped runs' information is not dropped: the out-of-band `url_citation`
 * annotations they reference surface as house source entries (url + title),
 * deduped by url. This class owns that side too — `sourcesFromUrlCitations`
 * mints the entries on the buffered path (`OpenAiResponses`), and the same
 * per-stream instance that carries marker-run state carries the admitted-source
 * set (`admitSource`) so the streaming egress collapses the one-part-per-annotation
 * repeats the AI-SDK OpenAI adapter emits for a url cited at several claims.
 *
 * AN OPENAI TURN'S SOURCES ARE THE PAGES ITS SEARCHES RETURNED (ask 842 — the founder: "for astra,
 * source icons didn't show up in real time as searches happened"), exactly as Anthropic's are: its
 * adapter mints a source part per search result the moment a search lands. OpenAI's adapter hands a
 * search's pages over only inside the provider-executed tool's RESULT (`{ action, sources }` — it
 * asks for `web_search_call.action.sources` whenever the web search tool rides the request) and
 * mints source parts only for the answer's citations, which arrive with the final text. So the
 * stream's sources are, in arrival order: each search's pages as it settles
 * (`searchResultSources`), then the citations — one admission (`admitSource`) over both, keyed by
 * the page (`sourceKey`: a cited url is the returned one plus OpenAI's own `utm_source=openai`).
 * `sourcesOfRound` is the same rule over a round's buffered content, so the buffered read of a
 * round's sources is the list the stream carried.
 */
export class OpenAiCitationMarkers {
  /** Strip all marker runs and stray marker glyphs from a complete text. */
  static strip(text: string): string {
    return new OpenAiCitationMarkers().push(text);
  }

  /** Wrap a text stream so every emitted chunk is marker-free (chunks left empty by stripping are elided). */
  static async *stripStream(stream: AsyncIterable<string>): AsyncIterable<string> {
    const markers = new OpenAiCitationMarkers();
    for await (const chunk of stream) {
      const cleaned = markers.push(chunk);
      if (cleaned) {
        yield cleaned;
      }
    }
  }

  /**
   * Mint house source entries from a Responses content part's `annotations`: every
   * `url_citation` (the SDK's `ResponseOutputText.URLCitation` — url + title ride the
   * annotation itself) becomes `{ url, title }`, deduped by url with the first entry
   * winning. Non-url annotation types (`file_citation`, `container_file_citation`,
   * `file_path`) carry no web source and are ignored.
   */
  static sourcesFromUrlCitations(annotations: readonly unknown[]): CitationSource[] {
    const sources: CitationSource[] = [];
    for (const annotation of annotations) {
      if (!annotation || typeof annotation !== 'object') {
        continue;
      }
      const rec = annotation as Record<string, unknown>;
      if (rec.type !== 'url_citation' || typeof rec.url !== 'string' || rec.url.length === 0) {
        continue;
      }
      sources.push({
        url: rec.url,
        ...(typeof rec.title === 'string' && rec.title.length > 0 ? { title: rec.title } : {}),
      });
    }
    return OpenAiCitationMarkers.dedupeSourcesByUrl(sources);
  }

  /**
   * Dedupe a sources list by url — the first occurrence wins; entries without a url pass
   * through untouched. The house contract for citation-derived source lists: one entry per
   * cited web page.
   */
  static dedupeSourcesByUrl<T extends { url?: string }>(sources: readonly T[]): T[] {
    const seen = new Set<string>();
    const out: T[] = [];
    for (const source of sources) {
      if (typeof source.url === 'string' && source.url.length > 0) {
        if (seen.has(source.url)) {
          continue;
        }
        seen.add(source.url);
      }
      out.push(source);
    }
    return out;
  }

  /**
   * The pages a web search RETURNED, in the order it returned them — read from the output the
   * AI-SDK OpenAI adapter hands over as the provider-executed web search tool's result
   * (`{ action: { type: 'search', query }, sources: [{ type: 'url', url } | { type: 'api', name }] }`).
   * An `api` source (a provider data feed) is no page and is skipped; so is every output that is
   * not a search's (an `openPage` / `findInPage` action, another tool's result): [].
   */
  static searchResultSources(output: unknown): CitationSource[] {
    if (!output || typeof output !== 'object') {
      return [];
    }
    const rec = output as { action?: { type?: unknown }; sources?: unknown };
    if (rec.action?.type !== 'search' || !Array.isArray(rec.sources)) {
      return [];
    }
    const pages: CitationSource[] = [];
    for (const source of rec.sources) {
      const entry = source as { type?: unknown; url?: unknown } | null;
      if (entry?.type === 'url' && typeof entry.url === 'string' && entry.url.length > 0) {
        pages.push({ url: entry.url });
      }
    }
    return pages;
  }

  /**
   * A round's sources from its buffered content (the AI SDK's step `content`s, every step of the
   * round in order — a round is one model call and may run several steps when a house tool is
   * called between a search and the answer) — the stream's rule applied after the fact: each
   * search result's pages (`searchResultSources`) and each url source part, through one admission
   * (`admitSource`), so the list equals the source parts the streaming egress yielded for the round.
   */
  static sourcesOfRound(content: readonly unknown[]): CitationSource[] {
    const admission = new OpenAiCitationMarkers();
    const sources: CitationSource[] = [];
    const admit = (source: CitationSource) => {
      const admitted = admission.admitSource(source);
      if (admitted) {
        sources.push(admitted);
      }
    };
    for (const part of content) {
      const rec = part as { type?: unknown; output?: unknown; sourceType?: unknown; url?: unknown; title?: unknown };
      if (rec?.type === 'tool-result') {
        OpenAiCitationMarkers.searchResultSources(rec.output).forEach(admit);
      } else if (
        rec?.type === 'source' &&
        rec.sourceType === 'url' &&
        typeof rec.url === 'string' &&
        rec.url.length > 0
      ) {
        admit({ url: rec.url, ...(typeof rec.title === 'string' && rec.title.length > 0 ? { title: rec.title } : {}) });
      }
    }
    return sources;
  }

  /**
   * Sanitize one streamed chunk, carrying marker-run state across calls: a
   * run split across chunk boundaries stays recognized, and its payload is
   * dropped as it arrives — an unterminated run never leaks its delimiters or
   * ids. All delimiters are single UTF-16 code units, so chunk boundaries
   * cannot split a delimiter itself.
   */
  push(chunk: string): string {
    if (!this.inMarkerRun && !OpenAiCitationMarkers.MARKER_CHAR.test(chunk)) {
      return chunk;
    }

    let out = '';
    for (let i = 0; i < chunk.length; i++) {
      const code = chunk.charCodeAt(i);
      if (this.inMarkerRun) {
        if (code === OpenAiCitationMarkers.RUN_CLOSE) {
          this.inMarkerRun = false;
        }
        continue;
      }
      if (code === OpenAiCitationMarkers.RUN_OPEN) {
        this.inMarkerRun = true;
        continue;
      }
      if (code >= OpenAiCitationMarkers.BLOCK_START && code <= OpenAiCitationMarkers.BLOCK_END) {
        continue;
      }
      out += chunk[i];
    }
    return out;
  }

  /**
   * Per-stream source admission — the entry to emit for this source, or `undefined` when it says
   * nothing new. A page (`sourceKey`) is admitted once: the first time it arrives, as it arrived. A
   * later arrival of the same page is emitted once more only when it carries the title the first
   * lacked (a search returns bare urls; the answer's citation of the same page names it), under the
   * FIRST arrival's url string — so every consumer that keys a sources list by url holds one entry
   * for the page and can take its title. Everything else repeats nothing: the AI-SDK OpenAI adapter
   * mints one `source` part per `url_citation` annotation, so a page cited at several claims arrives
   * several times. The same stateful instance that carries marker-run state across chunk
   * boundaries carries the admitted pages across parts.
   */
  admitSource(source: CitationSource): CitationSource | undefined {
    const key = OpenAiCitationMarkers.sourceKey(source.url);
    const admitted = this.admittedSources.get(key);
    if (!admitted) {
      this.admittedSources.set(key, { url: source.url, titled: !!source.title });
      return { url: source.url, ...(source.title ? { title: source.title } : {}) };
    }
    if (!admitted.titled && source.title) {
      admitted.titled = true;
      return { url: admitted.url, title: source.title };
    }
    return undefined;
  }

  /**
   * The page a source url names: the url without the `utm_source=openai` parameter OpenAI appends
   * to every url it cites (the search that returned the page returned it without one). Exactly that
   * pair is set aside, textually — every other parameter rides as the page spelled it (its own
   * `utm_source`, a `%20`, a bare `&b`), so the key of a cited page is the key of the page its search
   * returned. A url that does not parse is its own key.
   */
  private static sourceKey(url: string): string {
    try {
      const parsed = new URL(url);
      const query = parsed.search.startsWith('?') ? parsed.search.slice(1) : parsed.search;
      parsed.search = query
        .split('&')
        .filter((pair) => pair.length > 0 && pair !== OpenAiCitationMarkers.CITATION_TAG)
        .join('&');
      return parsed.toString();
    } catch {
      return url;
    }
  }

  /** The query parameter OpenAI appends to every url its answers cite. */
  private static readonly CITATION_TAG = 'utm_source=openai';

  /** U+E200 — opens a marker run. */
  private static readonly RUN_OPEN = 0xe200;
  /** U+E201 — closes a marker run. */
  private static readonly RUN_CLOSE = 0xe201;
  /** OpenAI's marker alphabet: the U+E200–U+E2FF private-use block. */
  private static readonly BLOCK_START = 0xe200;
  private static readonly BLOCK_END = 0xe2ff;
  /** Fast path: text without any marker-alphabet char passes through untouched. */
  private static readonly MARKER_CHAR = /[\uE200-\uE2FF]/;

  private inMarkerRun = false;
  /** The pages admitted on this stream (`sourceKey` → the url first emitted, and whether it carried a title). */
  private readonly admittedSources = new Map<string, { url: string; titled: boolean }>();
}

/**
 * A source citation minted from a `url_citation` annotation — the house sources-entry shape
 * (what `StreamSource` consumers and per-message source lists render as pills).
 */
export type CitationSource = {
  url: string;
  title?: string;
};
