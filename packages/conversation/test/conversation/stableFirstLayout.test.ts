import { createAnthropic } from '@ai-sdk/anthropic';
import { Conversation, type ConversationMessage } from '../../src/Conversation';
import type { ConversationSkill, SystemMessageSegment } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';
import { PromptCacheScriptedTransport, type WireBody } from './PromptCacheScriptedTransport';

/**
 * THE STABLE-FIRST LAYOUT, AT THE WIRE. A skill may hand the conversation its system message as
 * segments by stability ({@link ConversationSkill.getSystemMessageSegments}); the conversation
 * lays the prompt out stable-first — the tools, then every skill's stable block, then a cache
 * breakpoint; then the volatile tail (the caller's own system messages, every skill's volatile
 * blocks, in the order they were added), then the transcript — with the tools tier marked as its
 * own cache entry.
 *
 * THE LAW this suite pins, shape by shape: the model sees the SAME BYTES as the one-message
 * rendering — nothing added, nothing dropped, nothing reworded — regrouped by stability, and the
 * breakpoints land where the layout says. Proven on the request body the real Anthropic provider
 * sends (a scripted transport under it), never on an intermediate.
 */

const TIMEOUT = 30_000;

/** Varied prose that resists compression — a block of the given size. */
const prose = (seed: string, chars: number): string => {
  const words = ['plan', 'review', 'the', 'budget', 'before', 'travel', 'keep', 'notes', 'short', 'and', 'clear', seed];
  let out = '';
  let i = 0;
  while (out.length < chars) {
    out += `${words[(i * 7 + seed.length) % words.length]} `;
    i++;
  }
  return out.slice(0, chars).trimEnd() + '.';
};

const fn = (name: string, description: string): Function => ({
  definition: {
    name,
    description,
    parameters: { type: 'object', properties: { note: { type: 'string', description: prose(name, 160) } } },
  },
  call: async () => ({ ok: true }),
});

/** A skill of the one-message kind: its whole message reads as stable. */
const plainSkill = (id: string, name: string, text: string | string[], functions: Function[] = []): ConversationSkill =>
  ({
    getId: () => id,
    getName: () => name,
    getSystemMessages: () => text,
    getMessageModerators: () => [],
    getFunctions: () => functions,
  }) as unknown as ConversationSkill;

/** A segmented skill: the one-message rendering IS the segments joined in order. */
const segmentedSkill = (
  id: string,
  name: string,
  segments: SystemMessageSegment[],
  functions: Function[] = []
): ConversationSkill =>
  ({
    getId: () => id,
    getName: () => name,
    getSystemMessages: () => segments.map((segment) => segment.text).join(''),
    getSystemMessageSegments: async () => segments,
    getMessageModerators: () => [],
    getFunctions: () => functions,
  }) as unknown as ConversationSkill;

const heading = (skill: ConversationSkill) => `The following are instructions from the ${skill.getName()} skill:\n`;

async function drain(fullStream: AsyncIterable<unknown>): Promise<void> {
  for await (const _part of fullStream) {
    // consumed
  }
}

/** One request over the given skills and history (added BEFORE the turn, as a consumer's history is). */
async function request(
  skills: ConversationSkill[],
  history: ConversationMessage[],
  message = 'tidy the plan'
): Promise<WireBody> {
  const transport = new PromptCacheScriptedTransport();
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'stable-first-layout',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills,
  });
  if (history.length > 0) {
    conversation.addMessagesToHistory(history);
  }
  const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })('claude-opus-5');
  const result = await conversation.generateStream({ messages: [message], model: model as never });
  await drain(result.fullStream);
  await result.usage;
  expect(transport.requests).toHaveLength(1);
  return transport.requests[0];
}

const systemTexts = (body: WireBody): string[] => (body.system ?? []).map((block) => block.text ?? '');
const systemMarks = (body: WireBody): string =>
  (body.system ?? []).map((block) => (block.cache_control ? 'M' : '-')).join('');
const toolMarks = (body: WireBody): string =>
  (body.tools ?? []).map((tool) => (tool.cache_control ? 'M' : '-')).join('');
const messageMarks = (body: WireBody): string =>
  body.messages
    .map((message) =>
      typeof message.content !== 'string' && message.content.some((part) => part.cache_control) ? 'M' : '-'
    )
    .join('');

/**
 * THE BYTE LAW for one segmented skill against the request's system blocks: its stable block is
 * the heading plus its stable segments joined; each volatile segment is a block of its own; the
 * blocks together carry exactly the bytes of the one-message rendering (the segments joined and
 * trimmed at the ends, as a one-message skill's is) — every segment verbatim in one block, and
 * not a byte more.
 */
function expectByteLaw(body: WireBody, skill: ConversationSkill, segments: SystemMessageSegment[]) {
  const texts = systemTexts(body);
  const whole = segments.map((segment) => segment.text).join('');
  const lead = whole.length - whole.trimStart().length;
  const trail = whole.trimEnd().length;
  // The segments as the end trim leaves them.
  let offset = 0;
  const trimmed = segments.map((segment) => {
    const start = Math.max(offset, lead);
    const end = Math.min(offset + segment.text.length, trail);
    offset += segment.text.length;
    return { text: end > start ? whole.slice(start, end) : '', stable: segment.stable };
  });
  const stable = trimmed
    .filter((segment) => segment.stable)
    .map((segment) => segment.text)
    .join('');
  const volatile = trimmed.filter((segment) => !segment.stable && segment.text.trim()).map((segment) => segment.text);
  const blocks: string[] = [];
  if (stable) {
    const head = texts.find((text) => text === `${heading(skill)}${stable}`);
    expect(head).toBeDefined();
    blocks.push(head!);
  }
  volatile.forEach((text, index) => {
    const expected = `${!stable && index === 0 ? heading(skill) : ''}${text}`;
    const block = texts.find((candidate) => candidate === expected);
    expect(block).toBeDefined();
    blocks.push(block!);
  });
  // Nothing added, nothing dropped: the blocks' bytes, less the one heading, are the message's.
  const bytes = blocks.reduce((sum, text) => sum + text.length, 0) - heading(skill).length;
  expect(bytes).toBe(whole.trim().length);
  for (const segment of trimmed) {
    if (segment.text.trim()) {
      expect(blocks.some((block) => block.includes(segment.text))).toBe(true);
    }
  }
}

// ─── the fixtures ────────────────────────────────────────────────────────────

const INSTRUCTIONS = prose('instructions', 4000);
const CONDUCT = prose('conduct', 1500);
const HOW_TO = prose('howto', 900);
const DOCUMENT_V1 = `# Trip\n\n${prose('document', 2500)}`;
const TREES = `\n\n# Memory\n## Preferences\n- ${prose('memory', 400)}`;
const NAME_LINE = 'The user you are working with is named Sabrina (account id: u-7).';
const SUMMARY_TIER = `# Earlier topics\n\n- **Kickoff**: ${prose('summary', 300)}`;
const TOOLS = [fn('editDocument', prose('edit', 300)), fn('readDocument', prose('read', 200))];

const editorSegments = (document: string): SystemMessageSegment[] => [
  { text: INSTRUCTIONS, stable: true },
  { text: `# Open document\n\n\`\`\`\n${document}\n\`\`\``, stable: false },
];
const contextSegments: SystemMessageSegment[] = [
  // The volatile index FIRST in the one-message rendering, the how-to after it.
  { text: 'Documents linked here:\n- **Budget** (id: d-1) — the numbers\n', stable: false },
  { text: `\n${HOW_TO}\nThe conversation id is: c-1\n`, stable: false },
];
const memorySegments: SystemMessageSegment[] = [
  { text: CONDUCT, stable: true },
  { text: TREES, stable: false },
];
/** The per-user line amid stable conduct: stable, volatile, stable. */
const conductSegments: SystemMessageSegment[] = [
  { text: `${prose('before', 600)}\n\n`, stable: true },
  { text: NAME_LINE, stable: false },
  { text: `\n\n${prose('after', 500)}\n`, stable: true },
];

describe('the stable-first layout at the wire', () => {
  test(
    'A PLAIN CONVERSATION (one-message skills, a transcript): every block stable, the head mark on the last system block, the tools tier marked',
    async () => {
      const about = plainSkill('about', 'About', ['Who you are.', 'How you talk.']);
      const editor = plainSkill('editor', 'Editor', INSTRUCTIONS, TOOLS);
      const body = await request(
        [about, editor],
        [
          { role: 'user', content: 'earlier question' },
          { role: 'assistant', content: 'earlier answer' },
        ]
      );
      // The one-message skills render as before: the heading, then the texts joined with '. '.
      expect(systemTexts(body)).toEqual([
        `${heading(about)}Who you are.. How you talk.`,
        `${heading(editor)}${INSTRUCTIONS}`,
      ]);
      expect(systemMarks(body)).toBe('-M');
      // The tools tier: the LAST function tool carries the mark; the provider's search tool rides after it, unmarked.
      const tools = body.tools ?? [];
      expect(tools.map((tool) => tool.name)).toEqual(['editDocument', 'readDocument', 'web_search']);
      expect(toolMarks(body)).toBe('-M-');
      // The rolling pair on the last two messages; four breakpoints in all.
      expect(messageMarks(body)).toBe('-MM');
    },
    TIMEOUT
  );

  test(
    'AN EDIT TURN WITH A DOCUMENT: the instructions ride the head, the document the tail, the mark between them; the bytes are the one-message rendering',
    async () => {
      const about = plainSkill('about', 'About', 'Who you are.');
      const segments = editorSegments(DOCUMENT_V1);
      const editor = segmentedSkill('editor', 'Editor', segments, TOOLS);
      const body = await request(
        [about, editor],
        [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'hello' },
        ]
      );
      expect(systemTexts(body)).toEqual([
        `${heading(about)}Who you are.`,
        `${heading(editor)}${INSTRUCTIONS}`,
        `# Open document\n\n\`\`\`\n${DOCUMENT_V1}\n\`\`\``,
      ]);
      expect(systemMarks(body)).toBe('-M-');
      expect(toolMarks(body)).toBe('-M-');
      expect(messageMarks(body)).toBe('-MM');
      expectByteLaw(body, editor, segments);
    },
    TIMEOUT
  );

  test(
    'A CONVERSATION WITH MEMORY TREES AND A PER-USER LINE: the trees, the index and the name leave the head; a volatile-first skill keeps its heading on its index; stable text around a volatile line joins as one block',
    async () => {
      const conduct = segmentedSkill('conduct', 'Conduct', conductSegments);
      const context = segmentedSkill('context', 'Context', contextSegments);
      const memory = segmentedSkill('memory', 'Memory', memorySegments);
      const body = await request([conduct, context, memory], []);
      const texts = systemTexts(body);
      expect(texts).toEqual([
        // The head, in skill order: the conduct's stable segments joined under its heading — the
        // per-user line gone from it — and the memory conduct. The context skill has no stable text.
        `${heading(conduct)}${prose('before', 600)}\n\n\n\n${prose('after', 500)}`,
        `${heading(memory)}${CONDUCT}`,
        // The tail, in skill order: the name line; the index with the context skill's heading on it
        // (its first volatile block — the skill has no stable text); the how-to; the trees.
        NAME_LINE,
        `${heading(context)}Documents linked here:\n- **Budget** (id: d-1) — the numbers\n`,
        `\n${HOW_TO}\nThe conversation id is: c-1`,
        TREES,
      ]);
      expect(texts[0]).not.toContain('Sabrina');
      expect(systemMarks(body)).toBe('-M----');
      expectByteLaw(body, conduct, conductSegments);
      expectByteLaw(body, context, contextSegments);
      expectByteLaw(body, memory, memorySegments);
    },
    TIMEOUT
  );

  test(
    "A MATURE CONVERSATION WITH SUMMARY TIERS: the caller's own system blocks sit behind the head in the order they were added, ahead of the skills' volatile blocks",
    async () => {
      const editor = segmentedSkill('editor', 'Editor', editorSegments(DOCUMENT_V1), TOOLS);
      const memory = segmentedSkill('memory', 'Memory', memorySegments);
      const body = await request(
        [editor, memory],
        [
          { role: 'system', content: SUMMARY_TIER },
          { role: 'system', content: '# Topic: Planning\n\nWe planned.' },
          { role: 'user', content: 'what did we decide?' },
          { role: 'assistant', content: 'Colter Bay.' },
        ]
      );
      expect(systemTexts(body)).toEqual([
        `${heading(editor)}${INSTRUCTIONS}`,
        `${heading(memory)}${CONDUCT}`,
        SUMMARY_TIER,
        '# Topic: Planning\n\nWe planned.',
        `# Open document\n\n\`\`\`\n${DOCUMENT_V1}\n\`\`\``,
        TREES,
      ]);
      expect(systemMarks(body)).toBe('-M----');
      expect(messageMarks(body)).toBe('-MM');
    },
    TIMEOUT
  );

  test(
    'NO STABLE BLOCK: a conversation of caller system messages alone keeps the mark on its last system message, as before; no system message, no system mark',
    async () => {
      const withSystem = await request(
        [],
        [
          { role: 'system', content: 'You are terse.' },
          { role: 'system', content: 'Answer in one line.' },
        ]
      );
      expect(systemMarks(withSystem)).toBe('-M');
      expect(messageMarks(withSystem)).toBe('M');
      const bare = await request([], []);
      expect(bare.system ?? []).toHaveLength(0);
      expect(messageMarks(bare)).toBe('M');
      expect(bare.tools?.map((tool) => tool.name)).toEqual(['web_search']);
      // A provider-defined tool alone carries no breakpoint in this SDK: nothing to mark.
      expect(toolMarks(bare)).toBe('-');
    },
    TIMEOUT
  );

  test(
    'THE SAME BYTES: a segmented skill and its one-message twin put the same bytes in front of the model',
    async () => {
      const segments = conductSegments;
      const twin = plainSkill('conduct', 'Conduct', segments.map((segment) => segment.text).join(''));
      const legacy = await request([twin], []);
      const segmented = await request([segmentedSkill('conduct', 'Conduct', segments)], []);
      const legacyBlock = systemTexts(legacy)[0];
      expect(legacyBlock).toBe(
        `${heading(twin)}${segments
          .map((segment) => segment.text)
          .join('')
          .trim()}`
      );
      const blocks = systemTexts(segmented);
      expect(blocks).toHaveLength(2);
      // Not a byte more, not a byte less than the one-message rendering …
      const bytes = blocks.reduce((sum, text) => sum + text.length, 0) - heading(twin).length;
      expect(bytes).toBe(legacyBlock.length - heading(twin).length);
      // … and every segment a verbatim run of it, carried whole in one block — regrouped, never reworded.
      const whole = segments.map((segment) => segment.text).join('');
      const runs = [segments[0].text.trimStart(), segments[1].text, segments[2].text.trimEnd()];
      expect(runs.join('')).toBe(whole.trim());
      for (const run of runs) {
        expect(legacyBlock.includes(run)).toBe(true);
        expect(blocks.some((block) => block.includes(run))).toBe(true);
      }
    },
    TIMEOUT
  );
});
