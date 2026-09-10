/**
 * The kinds of place a streaming answer may be cut at for a mid-turn note (plans/FREE_AGENT.md
 * §M.16), coarsest first: a paragraph break (`\n\n`), a sentence end (`.` `?` `!` — closers such
 * as quotes, brackets and emphasis marks allowed — followed by whitespace, and not a numbered-list
 * marker or a decimal), a line end (a heading's, a list item's, a table row's), a code fence's
 * closing line, and — the last resort — a word boundary. Nothing inside a code fence is a
 * boundary of any kind: a fence is left only through its close.
 */
export type CutBoundaryKind = 'paragraph' | 'sentence' | 'line' | 'fence-close' | 'word';

/**
 * THE CUT BOUNDARY (plans/FREE_AGENT.md §M.16; founder ruling 2026-09-10: a cut is only acceptable
 * when "the net result is a cohesive answer") — where the round loop may cut a streaming answer
 * to take a mid-turn note in. The live proof cut at the N + 2 s deadline wherever the stream
 * happened to be, and the acknowledgment landed inside a sentence, a word and a heading in three
 * runs of three; this class is the rule that it lands only where the text already rests.
 *
 * One scanner per step: the loop feeds it EVERY text delta (the fence state is line-grained and
 * has to see every line) with the kinds the moment allows — nothing before N, a paragraph break
 * past N, any boundary past the deadline, a word boundary past the window — and it answers the
 * split index into that delta: the earliest position at which the text so far ends at an allowed
 * boundary. The text up to the split is shown; the delta's tail past it is dropped for the
 * continuation to say again (the transcript the continuation runs on ends at the split), so
 * nothing on screen is ever retracted and nothing shown is ever half a word. A split of 0 means
 * the text before this delta already rested on a boundary — the loop asks that with an empty
 * delta when a phase edge (N, the deadline, the window) passes with no part in flight.
 *
 * A space boundary is BEFORE the whitespace (`sentence.` ▮ ` next` — the joiner the cut adds
 * supplies the paragraph break); a newline boundary is AFTER the newline run (`sentence.\n\n` ▮).
 */
export class CutBoundary {
  /** No kind: the cut is not armed (before N). */
  static readonly NONE: ReadonlySet<CutBoundaryKind> = new Set();
  /** Past N: a paragraph break only (the grace — the generation's own chance to finish). */
  static readonly PARAGRAPH: ReadonlySet<CutBoundaryKind> = new Set<CutBoundaryKind>(['paragraph']);
  /** Past the deadline: any boundary but a bare word's. */
  static readonly ANY_BOUNDARY: ReadonlySet<CutBoundaryKind> = new Set<CutBoundaryKind>([
    'paragraph',
    'sentence',
    'line',
    'fence-close',
  ]);
  /** Past the window: a word boundary too — never mid-word, whatever the stream offers. */
  static readonly ANY_WORD: ReadonlySet<CutBoundaryKind> = new Set<CutBoundaryKind>([
    'paragraph',
    'sentence',
    'line',
    'fence-close',
    'word',
  ]);

  private static readonly FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/;
  private static readonly FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})\s*$/;
  private static readonly CLOSERS = new Set(['"', "'", '’', '”', ')', ']', '}', '*', '_']);
  private static readonly SENTENCE_END = new Set(['.', '?', '!']);

  private text = '';
  /** Where the line the scan is inside began (the fence state is judged line by line). */
  private lineStart = 0;
  /** The open fence's marker (``` or ~~~, at least three), or undefined outside a fence. */
  private fenceMark: string | undefined;
  /** The position just past the newline that closed the last fence — a boundary of its own kind. */
  private fenceClosedAt = -1;

  /**
   * Appends `delta` to the step's text and answers the split index INTO IT — the earliest position
   * at which the text so far ends at a boundary of an `allowed` kind (0 = the text before this delta
   * already did) — or -1 when there is none. Called for every delta whatever `allowed` says.
   */
  split(delta: string, allowed: ReadonlySet<CutBoundaryKind>): number {
    const from = this.text.length;
    this.text += delta;
    const end = this.text.length;
    let found = -1;
    for (let p = from; p <= end; p++) {
      if (p > from && this.text[p - 1] === '\n') {
        this.judgeLine(p);
      }
      if (found < 0 && allowed.size > 0) {
        const kind = this.kindAt(p);
        if (kind !== undefined && allowed.has(kind)) {
          found = p - from;
        }
      }
    }
    return found;
  }

  // ─── helpers ───────────────────────────────────────────────────────────────

  /** The kind of boundary the text's first `p` characters end at, or undefined. */
  private kindAt(p: number): CutBoundaryKind | undefined {
    if (p === 0) {
      return undefined;
    }
    if (this.fenceMark !== undefined) {
      return undefined;
    }
    const last = this.text[p - 1];
    if (last === '\n') {
      if (this.fenceClosedAt === p) {
        return 'fence-close';
      }
      let newlines = 0;
      let q = p - 1;
      while (q >= 0 && CutBoundary.isWhitespace(this.text[q])) {
        if (this.text[q] === '\n') {
          newlines++;
        }
        q--;
      }
      if (q < 0) {
        return undefined;
      }
      if (newlines >= 2) {
        return 'paragraph';
      }
      return this.endsSentence(q) ? 'sentence' : 'line';
    }
    if (CutBoundary.isWhitespace(last)) {
      return undefined;
    }
    const next = this.text[p];
    if (next !== ' ' && next !== '\t') {
      return undefined;
    }
    return this.endsSentence(p - 1) ? 'sentence' : 'word';
  }

  /**
   * Whether the character at `i` (the last non-whitespace one before a boundary) ends a sentence:
   * closers skipped, then `.` `?` `!` (a run of them), preceded by a character that is neither
   * whitespace nor a digit (`1.` is a list marker, `3.5` never reaches here).
   */
  private endsSentence(i: number): boolean {
    let j = i;
    while (j >= 0 && CutBoundary.CLOSERS.has(this.text[j])) {
      j--;
    }
    if (j < 0 || !CutBoundary.SENTENCE_END.has(this.text[j])) {
      return false;
    }
    while (j >= 0 && CutBoundary.SENTENCE_END.has(this.text[j])) {
      j--;
    }
    if (j < 0 || CutBoundary.isWhitespace(this.text[j])) {
      return false;
    }
    if (!/[0-9]/.test(this.text[j])) {
      return true;
    }
    // A number may end a sentence ("…in 2024."); a numbered-list marker ("2. Training") may not —
    // its digits open the line.
    let k = j;
    while (k >= 0 && /[0-9]/.test(this.text[k])) {
      k--;
    }
    while (k >= 0 && (this.text[k] === ' ' || this.text[k] === '\t')) {
      k--;
    }
    return k >= 0 && this.text[k] !== '\n';
  }

  /** The line ending just before `p` is complete: a fence opens or closes on it. */
  private judgeLine(p: number): void {
    const line = this.text.slice(this.lineStart, p - 1);
    this.lineStart = p;
    if (this.fenceMark === undefined) {
      const open = CutBoundary.FENCE_OPEN.exec(line);
      if (open) {
        this.fenceMark = open[1];
      }
      return;
    }
    const close = CutBoundary.FENCE_CLOSE.exec(line);
    if (close && close[1][0] === this.fenceMark[0] && close[1].length >= this.fenceMark.length) {
      this.fenceMark = undefined;
      this.fenceClosedAt = p;
    }
  }

  private static isWhitespace(c: string | undefined): boolean {
    return c === ' ' || c === '\t' || c === '\n' || c === '\r';
  }
}
