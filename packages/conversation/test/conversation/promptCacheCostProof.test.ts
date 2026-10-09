import { createAnthropic } from '@ai-sdk/anthropic';
import { Conversation, type ConversationMessage } from '../../src/Conversation';
import type { ConversationSkill, SystemMessageSegment } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';
import { PromptCacheScriptedTransport, type PromptCacheBilling } from './PromptCacheScriptedTransport';

/**
 * THE COST PROOF of the stable-first layout: what a turn's first request READS from the prompt
 * cache and what it WRITES, billed the way the provider's cache bills (the scripted transport
 * under the real Anthropic provider), over the turn shapes that cost the most before:
 *
 *  - an EDIT turn after an edit turn (the open document changed between them): the head — the
 *    tools and the stable instructions — is READ; the write is the tail (the document, the
 *    trees, the transcript) only;
 *  - a turn after a MEMORY WRITE (the trees changed): the head is read;
 *  - a SECOND USER on the same stable head (another name, another document): the head is read,
 *    the second user's tail written;
 *  - an INSTRUCTION change over the same roster: the tools tier is read;
 *  - the unchanged turn (the control): everything up to the previous request's last breakpoint
 *    is read, as before the layout.
 *
 * Before the layout — the head's breakpoint on the LAST system block, the document inside the
 * system region, no breakpoint on the tools — the changed-document turn read NOTHING and wrote
 * the whole prefix again. Each case prints its per-step billing so a run at the old code shows
 * the numbers it replaces.
 */

const TIMEOUT = 30_000;

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
    parameters: {
      type: 'object',
      properties: {
        note: { type: 'string', description: prose(name, 400) },
        reason: { type: 'string', description: prose(`${name}-reason`, 200) },
      },
    },
  },
  call: async () => 'applied',
});

/** The roster: six tools, as big as a real editing roster's definitions. */
const ROSTER = ['editDocument', 'readDocument', 'searchNotes', 'listFolders', 'saveMemory', 'readMemory'].map((name) =>
  fn(name, prose(name, 600))
);
const INSTRUCTIONS = prose('instructions', 12_000);
const CONDUCT = prose('conduct', 3_000);
const MEMORY_CONDUCT = prose('memory-conduct', 2_000);
const document = (version: number) => `# Trip plan v${version}\n\n${prose(`document-${version}`, 8_000)}`;
const trees = (version: number) => `\n\n# Memory\n## Preferences\n- ${prose(`memory-${version}`, 1_200)}`;
const nameLine = (name: string) => `The user you are working with is named ${name} (account id: u-${name.length}).`;

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

/** One user's turn shape: the stable head is the same for everyone; the volatile parts are theirs. */
type Shape = { instructions?: string; name: string; document: string; trees: string };

const skillsFor = (shape: Shape): ConversationSkill[] => [
  segmentedSkill('conduct', 'Conduct', [
    { text: CONDUCT, stable: true },
    { text: `\n\n${nameLine(shape.name)}`, stable: false },
  ]),
  segmentedSkill(
    'editor',
    'Editor',
    [
      { text: shape.instructions ?? INSTRUCTIONS, stable: true },
      { text: `\n\n# Open document\n\n\`\`\`\n${shape.document}\n\`\`\``, stable: false },
    ],
    ROSTER
  ),
  segmentedSkill('memory', 'Memory', [
    { text: MEMORY_CONDUCT, stable: true },
    { text: shape.trees, stable: false },
  ]),
];

async function drain(fullStream: AsyncIterable<unknown>): Promise<void> {
  for await (const _part of fullStream) {
    // consumed
  }
}

type StepUsage = { cachedInputTokens: number; cacheWriteTokens: number; inputTokens: number };

/**
 * One turn as a consumer runs it: a fresh Conversation over the shape's skills, the earlier
 * exchanges added as history, the message sent; the transport bills every request of the turn.
 */
async function turn(
  transport: PromptCacheScriptedTransport,
  shape: Shape,
  history: ConversationMessage[],
  message: string
): Promise<{ steps: StepUsage[]; billings: PromptCacheBilling[] }> {
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'prompt-cache-cost-proof',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: skillsFor(shape),
  });
  if (history.length > 0) {
    conversation.addMessagesToHistory(history);
  }
  const model = createAnthropic({ apiKey: 'scripted', fetch: transport.fetch })('claude-opus-5');
  const first = transport.billings.length;
  const result = await conversation.generateStream({ messages: [message], model: model as never });
  await drain(result.fullStream);
  const usage = await result.usage;
  const steps = (usage.steps ?? []).map((step) => ({
    cachedInputTokens: step.cachedInputTokens,
    cacheWriteTokens: step.cacheWriteTokens,
    inputTokens: step.inputTokens,
  }));
  return { steps, billings: transport.billings.slice(first) };
}

/**
 * The sizes of the request's tiers, as the transport bills them: the function tools (the tools
 * tier's entry), every tool (the provider's search tool rides after the function tools, in the
 * head's entry), the head, the tail, the messages.
 */
function tiers(transport: PromptCacheScriptedTransport, requestIndex: number, stableBlocks: number) {
  const blocks = PromptCacheScriptedTransport.prefixBlocks(transport.requests[requestIndex]);
  const sum = (values: number[]) => values.reduce((total, n) => total + n, 0);
  return {
    functionTools: sum(blocks.tools.slice(0, ROSTER.length)),
    tools: sum(blocks.tools),
    head: sum(blocks.system.slice(0, stableBlocks)),
    tail: sum(blocks.system.slice(stableBlocks)),
    messages: sum(blocks.messages),
  };
}

const EDIT = 'tighten the Saturday section [edit]';
const exchange = (message: string): ConversationMessage[] => [
  { role: 'user', content: message },
  { role: 'assistant', content: 'Done.' },
];

function print(label: string, turnResult: { steps: StepUsage[]; billings: PromptCacheBilling[] }) {
  const rows = turnResult.steps.map((step, i) => {
    const billing = turnResult.billings[i];
    return `  step ${i + 1}: read ${step.cachedInputTokens} · write ${step.cacheWriteTokens} · fresh ${step.inputTokens} · marks ${billing?.marks.join(',') ?? '?'}`;
  });
  // eslint-disable-next-line no-console
  console.log([`${label}`, ...rows].join('\n'));
}

describe('the cost proof: what the first request of a turn reads and writes', () => {
  const sabrina: Shape = { name: 'Sabrina', document: document(1), trees: trees(1) };
  // The head: three stable blocks (the conduct, the instructions, the memory conduct).
  const STABLE_BLOCKS = 3;

  test(
    'AN EDIT TURN AFTER AN EDIT TURN: the second turn reads the head (tools + instructions) and writes only the tail',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      transport.editTool = 'editDocument';
      // Turn 1: cold. The model edits the document (a tool call, then its closing line).
      const first = await turn(transport, sabrina, [], EDIT);
      print('edit → edit, turn 1 (cold)', first);
      expect(first.steps).toHaveLength(2);
      expect(first.steps[0].cachedInputTokens).toBe(0);
      // Turn 2: the document changed under the edit; the first turn's exchange is now history.
      const second = await turn(transport, { ...sabrina, document: document(2) }, exchange(EDIT), EDIT);
      print('edit → edit, turn 2 (the document changed)', second);
      const sizes = tiers(transport, first.billings.length, STABLE_BLOCKS);
      const step1 = second.steps[0];
      // THE HEAD IS READ: the tools and every stable block.
      expect(step1.cachedInputTokens).toBeGreaterThanOrEqual(sizes.tools + sizes.head);
      // THE WRITE IS THE TAIL: the volatile blocks, the transcript and the message — never the head.
      expect(step1.cacheWriteTokens).toBeLessThanOrEqual(sizes.tail + sizes.messages);
      expect(step1.cacheWriteTokens).toBeGreaterThan(0);
      expect(step1.cacheWriteTokens).toBeLessThan(sizes.head);
      // The hit fell exactly at the head's breakpoint: the ROSTER + the stable blocks.
      expect(second.billings[0].hitAt).toBe(ROSTER.length + 1 + STABLE_BLOCKS);
      // The second step of the turn reads what the first wrote (the rolling pair, as before).
      expect(second.steps[1].cachedInputTokens).toBeGreaterThanOrEqual(
        step1.cachedInputTokens + step1.cacheWriteTokens
      );
    },
    TIMEOUT
  );

  test(
    'A TURN AFTER A MEMORY WRITE: the trees changed; the head is read, the tail written',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      const first = await turn(transport, sabrina, [], 'remember that I prefer window seats');
      print('memory write, turn 1 (cold)', first);
      const second = await turn(
        transport,
        { ...sabrina, trees: trees(2) },
        exchange('remember that I prefer window seats'),
        'what did I say about seats?'
      );
      print('memory write, turn 2 (the trees changed)', second);
      const sizes = tiers(transport, first.billings.length, STABLE_BLOCKS);
      expect(second.steps[0].cachedInputTokens).toBeGreaterThanOrEqual(sizes.tools + sizes.head);
      expect(second.steps[0].cacheWriteTokens).toBeLessThanOrEqual(sizes.tail + sizes.messages);
      expect(second.billings[0].hitAt).toBe(ROSTER.length + 1 + STABLE_BLOCKS);
    },
    TIMEOUT
  );

  test(
    'TWO USERS ON ONE HEAD: the second user’s first request reads the head the first user wrote',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      const first = await turn(transport, sabrina, [], 'what is in my plan?');
      print('two users, Sabrina (cold)', first);
      const kevin: Shape = { name: 'Kevin', document: `# Grocery list\n\n${prose('kevin', 3_000)}`, trees: trees(3) };
      const second = await turn(transport, kevin, [], 'what is on my list?');
      print('two users, Kevin (the same head)', second);
      const sizes = tiers(transport, first.billings.length, STABLE_BLOCKS);
      expect(second.steps[0].cachedInputTokens).toBeGreaterThanOrEqual(sizes.tools + sizes.head);
      expect(second.steps[0].cacheWriteTokens).toBeLessThanOrEqual(sizes.tail + sizes.messages);
      expect(second.billings[0].hitAt).toBe(ROSTER.length + 1 + STABLE_BLOCKS);
      // The head bytes were the same for both: the first user's system head equals the second's.
      const headOf = (index: number) => (transport.requests[index].system ?? []).slice(0, STABLE_BLOCKS);
      expect(JSON.stringify(headOf(0))).toBe(JSON.stringify(headOf(first.billings.length)));
    },
    TIMEOUT
  );

  test(
    'AN INSTRUCTION CHANGE OVER THE SAME ROSTER: the tools tier is read, the head and tail written',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      const first = await turn(transport, sabrina, [], 'what is in my plan?');
      print('instruction change, turn 1 (cold)', first);
      const second = await turn(
        transport,
        { ...sabrina, instructions: prose('instructions-v2', 12_000) },
        [],
        'what is in my plan?'
      );
      print('instruction change, turn 2 (the instructions changed)', second);
      const sizes = tiers(transport, first.billings.length, STABLE_BLOCKS);
      expect(second.steps[0].cachedInputTokens).toBe(sizes.functionTools);
      expect(second.billings[0].hitAt).toBe(ROSTER.length);
      expect(second.steps[0].cacheWriteTokens).toBeGreaterThanOrEqual(sizes.tools - sizes.functionTools + sizes.head);
    },
    TIMEOUT
  );

  test(
    'THE CONTROL — nothing changed: the next turn reads everything up to the previous request’s last breakpoint, as before the layout',
    async () => {
      const transport = new PromptCacheScriptedTransport();
      const first = await turn(transport, sabrina, [], 'what is in my plan?');
      print('control, turn 1 (cold)', first);
      const second = await turn(transport, sabrina, exchange('what is in my plan?'), 'and the budget?');
      print('control, turn 2 (unchanged)', second);
      const previous = first.billings[first.billings.length - 1];
      // The previous request wrote up to its last breakpoint — the next one reads at least the
      // whole system region and the first message of the transcript.
      const sizes = tiers(transport, first.billings.length, STABLE_BLOCKS);
      expect(second.steps[0].cachedInputTokens).toBeGreaterThanOrEqual(sizes.tools + sizes.head + sizes.tail);
      expect(second.steps[0].cachedInputTokens).toBeGreaterThanOrEqual(previous.read + previous.write - sizes.messages);
      expect(second.billings[0].hitAt).toBeGreaterThan(ROSTER.length + 1 + STABLE_BLOCKS);
    },
    TIMEOUT
  );
});
