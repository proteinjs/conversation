import { Conversation } from '../../src/Conversation';

type AnyMessage = { role: string; content: unknown };
type ConversationStatics = {
  pruneToolResultsOverBudget(
    messages: AnyMessage[],
    budget: number,
    evictionFloorRatio?: number,
    countTokens?: (text: string) => number
  ): AnyMessage[];
  TOOL_RESULT_PRUNED_PLACEHOLDER: string;
};
const statics = Conversation as unknown as ConversationStatics;
// Deterministic counter: 1 token per character.
const charCount = (text: string) => text.length;
const prune = (messages: AnyMessage[], budget: number, countTokens: (text: string) => number = charCount) =>
  statics.pruneToolResultsOverBudget(messages, budget, 0.75, countTokens);

const PLACEHOLDER = '[tool result pruned to fit the context budget — re-run the tool if you still need it]';

const textToolMessage = (id: string, tokens: number) => ({
  role: 'tool',
  content: [
    {
      type: 'tool-result',
      toolCallId: id,
      toolName: 'getThoughtContent',
      output: { type: 'text', value: 'x'.repeat(tokens) },
    },
  ],
});

const jsonToolMessage = (id: string, value: unknown) => ({
  role: 'tool',
  content: [{ type: 'tool-result', toolCallId: id, toolName: 'getTopics', output: { type: 'json', value } }],
});

const imageToolMessage = (id: string) => ({
  role: 'tool',
  content: [
    {
      type: 'tool-result',
      toolCallId: id,
      toolName: 'computer',
      output: { type: 'content', value: [{ type: 'media', data: `png-${id}`, mediaType: 'image/png' }] },
    },
  ],
});

const assistant = () => ({ role: 'assistant', content: 'calling a tool…' });

/**
 * A realistic accumulated tool loop: assistant/tool pairs, ending with the
 * trailing tool message the model has not yet responded to.
 */
const loop = (...toolMessages: AnyMessage[]): AnyMessage[] => {
  const out: AnyMessage[] = [{ role: 'user', content: 'go' }];
  for (const msg of toolMessages) {
    out.push(assistant(), msg);
  }
  return out;
};

const outputOf = (msg: AnyMessage, part = 0) => (msg.content as any[])[part].output;
// Tool messages sit at odd offsets after the user message: 2, 4, 6, …
const toolMsg = (messages: AnyMessage[], n: number) => messages[2 + n * 2];

describe('Conversation.pruneToolResultsOverBudget', () => {
  it('evicts the oldest results past the budget, down to the hysteresis floor', () => {
    // 4 × 100 tokens, budget 300 (floor 225). The 4th result crosses the budget →
    // evict r0 (live 300 > 225) and r1 (live 200 ≤ 225); r2 + trailing r3 stay.
    const messages = loop(
      textToolMessage('r0', 100),
      textToolMessage('r1', 100),
      textToolMessage('r2', 100),
      textToolMessage('r3', 100)
    );
    const result = prune(messages, 300);
    expect(outputOf(toolMsg(result, 0))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(result, 1))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(result, 2)).value).toBe('x'.repeat(100));
    expect(outputOf(toolMsg(result, 3)).value).toBe('x'.repeat(100));
  });

  it('uses the exact placeholder text', () => {
    expect(statics.TOOL_RESULT_PRUNED_PLACEHOLDER).toBe(PLACEHOLDER);
  });

  it('returns the input untouched while under budget', () => {
    const messages = loop(textToolMessage('r0', 100), textToolMessage('r1', 100));
    expect(prune(messages, 300)).toBe(messages);
  });

  it('hysteresis: holds byte-stable between epochs, then evicts a batch at the next crossing', () => {
    const results = [
      textToolMessage('r0', 100),
      textToolMessage('r1', 100),
      textToolMessage('r2', 100),
      textToolMessage('r3', 100),
    ];
    // Epoch 1 (as in the first test): r0 + r1 evicted, live 200.
    const stepN = prune(loop(...results), 300);

    // Next step appends r4 (live 300, ≤ budget): NO new eviction — the projection
    // of the shared prefix is identical to the previous step's (cache-stable).
    const stepN1 = prune(loop(...results, textToolMessage('r4', 100)), 300);
    for (let i = 0; i < 4; i++) {
      expect(outputOf(toolMsg(stepN1, i))).toEqual(outputOf(toolMsg(stepN, i)));
    }
    expect(outputOf(toolMsg(stepN1, 4)).value).toBe('x'.repeat(100));

    // r5 crosses again (live 400 > 300): evict r2 and r3 in one batch (live 200).
    const stepN2 = prune(loop(...results, textToolMessage('r4', 100), textToolMessage('r5', 100)), 300);
    expect(outputOf(toolMsg(stepN2, 2))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(stepN2, 3))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(stepN2, 4)).value).toBe('x'.repeat(100));
    expect(outputOf(toolMsg(stepN2, 5)).value).toBe('x'.repeat(100));
  });

  it('never evicts the trailing results the model has not responded to yet', () => {
    // A single huge trailing result far over budget: counted, but protected.
    const messages = loop(textToolMessage('huge', 1000));
    expect(prune(messages, 300)).toBe(messages);

    // With prior results present, only the prior ones are evicted.
    const withHistory = loop(textToolMessage('r0', 100), textToolMessage('huge', 1000));
    const result = prune(withHistory, 300);
    expect(outputOf(toolMsg(result, 0))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(result, 1)).value).toBe('x'.repeat(1000));
  });

  it('counts json outputs and evicts them like text', () => {
    const bigJson = jsonToolMessage('r0', { rows: 'y'.repeat(200) });
    const messages = loop(bigJson, textToolMessage('r1', 100), textToolMessage('r2', 100));
    // JSON.stringify length ≈ 200 + wrapper; budget 250 → crossing at r1 evicts r0.
    const result = prune(messages, 250);
    expect(outputOf(toolMsg(result, 0))).toEqual({ type: 'text', value: PLACEHOLDER });
    expect(outputOf(toolMsg(result, 1)).value).toBe('x'.repeat(100));
  });

  it('ignores image (content-type) outputs and non-tool messages', () => {
    const image = imageToolMessage('shot');
    const user = { role: 'user', content: 'x'.repeat(5000) };
    const messages = [
      user,
      assistant(),
      image,
      assistant(),
      textToolMessage('r0', 100),
      assistant(),
      textToolMessage('r1', 100),
    ];
    // Image bytes and user bytes don't count toward the budget; text results are under it.
    expect(prune(messages, 300)).toBe(messages);
  });

  it('does not mutate the original messages when pruning', () => {
    const original = textToolMessage('r0', 100);
    const messages = loop(original, textToolMessage('r1', 300));
    const result = prune(messages, 300);
    expect(outputOf(original).value).toBe('x'.repeat(100));
    expect(outputOf(toolMsg(result, 0))).toEqual({ type: 'text', value: PLACEHOLDER });
  });

  it('zero or negative budget is a no-op', () => {
    const messages = loop(textToolMessage('r0', 100), textToolMessage('r1', 100));
    expect(prune(messages, 0)).toBe(messages);
    expect(prune(messages, -1)).toBe(messages);
  });

  it('memoizes token counts per part object (each result encoded once across steps)', () => {
    const counter = jest.fn(charCount);
    const results = [textToolMessage('r0', 100), textToolMessage('r1', 100), textToolMessage('r2', 100)];
    prune(loop(...results), 1000, counter);
    expect(counter).toHaveBeenCalledTimes(3);
    // Next step re-projects the same part objects plus one new result: only the
    // new result is encoded.
    prune(loop(...results, textToolMessage('r3', 100)), 1000, counter);
    expect(counter).toHaveBeenCalledTimes(4);
  });
});
