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
}
