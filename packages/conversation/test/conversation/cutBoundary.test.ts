import { CutBoundary } from '../../src/CutBoundary';

/**
 * THE CUT BOUNDARY (plans/FREE_AGENT.md §M.16): where a streaming answer may be cut for a mid-turn
 * note — the scanner's rules, one delta at a time. A split index is INTO the delta; 0 = the text
 * before it already rested on a boundary; -1 = none of the allowed kinds. Space boundaries sit
 * BEFORE the whitespace, newline boundaries AFTER the newline run.
 */
describe('CutBoundary — where the cut may land (FREE_AGENT §M.16)', () => {
  const feed = (deltas: string[], allowed: ReadonlySet<import('../../src/CutBoundary').CutBoundaryKind>) => {
    const boundary = new CutBoundary();
    return deltas.map((delta) => boundary.split(delta, allowed));
  };

  test('nothing is a boundary with no kind allowed — the scanner still takes every delta', () => {
    expect(feed(['One.\n\n', 'Two. ', 'three'], CutBoundary.NONE)).toEqual([-1, -1, -1]);
  });

  test('a paragraph break is after its newlines; a lone newline is a sentence end or a line end, never a paragraph', () => {
    expect(feed(['Sentence one.\n\nTwo'], CutBoundary.PARAGRAPH)).toEqual([15]);
    expect(feed(['Sentence one.\nTwo'], CutBoundary.PARAGRAPH)).toEqual([-1]);
    expect(feed(['Sentence one.\nTwo'], CutBoundary.ANY_BOUNDARY)).toEqual([14]);
    expect(feed(['- an item\n- another'], CutBoundary.ANY_BOUNDARY)).toEqual([10]);
    // The break split across deltas: the second newline completes it.
    expect(feed(['Sentence one.\n', '\nTwo'], CutBoundary.PARAGRAPH)).toEqual([-1, 1]);
  });

  test('a sentence end is before the space that follows the period — also across deltas, and with closers', () => {
    expect(feed(['Sentence one. Two'], CutBoundary.ANY_BOUNDARY)).toEqual([13]);
    expect(feed(['Sentence one.', ' Two'], CutBoundary.ANY_BOUNDARY)).toEqual([-1, 0]);
    expect(feed(['He asked "why?" and left'], CutBoundary.ANY_BOUNDARY)).toEqual([15]);
    expect(feed(['**Bold.** Next'], CutBoundary.ANY_BOUNDARY)).toEqual([9]);
    expect(feed(['Really?! Yes'], CutBoundary.ANY_BOUNDARY)).toEqual([8]);
  });

  test('a number may end a sentence; a numbered-list marker, a decimal and a version never do', () => {
    expect(feed(['Shipped in 2024. Then'], CutBoundary.ANY_BOUNDARY)).toEqual([16]);
    expect(feed(['2. Training is next'], CutBoundary.ANY_BOUNDARY)).toEqual([-1]);
    expect(feed(['Intro\n\n12. Training is next'], CutBoundary.PARAGRAPH)).toEqual([7]);
    expect(feed(['Take 3.5 grams'], CutBoundary.ANY_BOUNDARY)).toEqual([-1]);
    expect(feed(['At v1.28.0 the gate reds'], CutBoundary.ANY_BOUNDARY)).toEqual([-1]);
  });

  test('a word boundary is the last resort — before the space, only when allowed', () => {
    expect(feed(['alpha beta gamma'], CutBoundary.ANY_BOUNDARY)).toEqual([-1]);
    expect(feed(['alpha beta gamma'], CutBoundary.ANY_WORD)).toEqual([5]);
    expect(feed(['alpha', ' beta'], CutBoundary.ANY_WORD)).toEqual([-1, 0]);
  });

  test('inside a code fence nothing is a boundary; the fence closes on its own line and that close is one', () => {
    const boundary = new CutBoundary();
    expect(boundary.split('Here is code:\n\n```ts\n', CutBoundary.NONE)).toBe(-1);
    expect(boundary.split('const a = 1. done;\n', CutBoundary.ANY_WORD)).toBe(-1);
    expect(boundary.split('\n\nconst b = 2;\n', CutBoundary.ANY_WORD)).toBe(-1);
    expect(boundary.split('```\n\nAfter.', CutBoundary.ANY_BOUNDARY)).toBe(4);
    // A tilde fence closes only with tildes; a shorter marker does not close a longer one.
    const tilde = new CutBoundary();
    expect(tilde.split('~~~~\nx. y\n```\nz. w\n', CutBoundary.ANY_BOUNDARY)).toBe(-1);
    expect(tilde.split('~~~\nstill inside. yes\n', CutBoundary.ANY_BOUNDARY)).toBe(-1);
    expect(tilde.split('~~~~\nout. now', CutBoundary.ANY_BOUNDARY)).toBe(5);
  });

  test('an empty delta asks whether the text already rests on a boundary (the phase edge with no part in flight)', () => {
    const boundary = new CutBoundary();
    expect(boundary.split('First sentence.\n', CutBoundary.PARAGRAPH)).toBe(-1);
    expect(boundary.split('', CutBoundary.PARAGRAPH)).toBe(-1);
    expect(boundary.split('', CutBoundary.ANY_BOUNDARY)).toBe(0);
    const rest = new CutBoundary();
    expect(rest.split('Ends mid-word', CutBoundary.ANY_WORD)).toBe(4);
    const trailing = new CutBoundary();
    expect(trailing.split('Ends on a period.', CutBoundary.ANY_WORD)).toBe(4);
    expect(trailing.split('', CutBoundary.ANY_WORD)).toBe(-1);
  });
});
