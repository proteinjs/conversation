import { Conversation } from '../../src/Conversation';

type AnyMessage = { role: string; content: unknown };
type ConversationStatics = {
  pruneStaleToolImages(messages: AnyMessage[], keepLast: number, evictionBatch?: number): AnyMessage[];
};
// Batch of 1 = exact window semantics; the hysteresis behavior has its own test.
const prune = (messages: AnyMessage[], keepLast: number, evictionBatch = 1) =>
  (Conversation as unknown as ConversationStatics).pruneStaleToolImages(messages, keepLast, evictionBatch);

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

const textToolMessage = (id: string) => ({
  role: 'tool',
  content: [{ type: 'tool-result', toolCallId: id, toolName: 'bash', output: { type: 'text', value: 'ok' } }],
});

const outputOf = (msg: AnyMessage, part = 0) => (msg.content as any[])[part].output;

describe('Conversation.pruneStaleToolImages', () => {
  it('replaces the oldest image outputs past the keep window with a text placeholder', () => {
    const messages = [imageToolMessage('a'), imageToolMessage('b'), imageToolMessage('c'), imageToolMessage('d')];
    const result = prune(messages, 2);
    expect(outputOf(result[0]).type).toBe('text');
    expect(outputOf(result[0]).value).toContain('stale screenshot removed');
    expect(outputOf(result[1]).type).toBe('text');
    expect(outputOf(result[2]).type).toBe('content');
    expect(outputOf(result[3]).type).toBe('content');
  });

  it('returns the input untouched when within the keep window', () => {
    const messages = [imageToolMessage('a'), imageToolMessage('b')];
    expect(prune(messages, 2)).toBe(messages);
    expect(prune(messages, 5)).toBe(messages);
  });

  it('never touches text tool results or user/assistant messages', () => {
    const user = { role: 'user', content: [{ type: 'image', image: 'user-uploaded' }] };
    const assistant = { role: 'assistant', content: 'looking…' };
    const textMsg = textToolMessage('t1');
    const messages = [user, textMsg, imageToolMessage('a'), assistant, imageToolMessage('b'), imageToolMessage('c')];
    const result = prune(messages, 1);
    expect(result[0]).toBe(user);
    expect(result[1]).toBe(textMsg);
    expect(outputOf(result[1])).toEqual({ type: 'text', value: 'ok' });
    expect(result[3]).toBe(assistant);
    // Oldest two screenshots pruned, newest kept.
    expect(outputOf(result[2]).type).toBe('text');
    expect(outputOf(result[4]).type).toBe('text');
    expect(outputOf(result[5]).type).toBe('content');
  });

  it('does not mutate the original messages when pruning', () => {
    const original = imageToolMessage('a');
    const result = prune([original, imageToolMessage('b')], 1);
    expect(outputOf(original).type).toBe('content');
    expect(outputOf(result[0]).type).toBe('text');
  });

  it('keepLast 0 prunes every image', () => {
    const result = prune([imageToolMessage('a'), imageToolMessage('b')], 0);
    expect(outputOf(result[0]).type).toBe('text');
    expect(outputOf(result[1]).type).toBe('text');
  });

  it('hysteresis: holds the window stable until a full eviction batch accumulates', () => {
    const make = (count: number) => Array.from({ length: count }, (_, i) => imageToolMessage(`f${i}`));
    // keep 2, batch 4: untouched (identity) until count reaches 6…
    const atFive = make(5);
    expect(prune(atFive, 2, 4)).toBe(atFive);
    // …then evicts down to the keep target in one batch.
    const atSix = prune(make(6), 2, 4);
    expect(atSix.filter((m) => outputOf(m).type === 'content')).toHaveLength(2);
    expect(atSix.filter((m) => outputOf(m).type === 'text')).toHaveLength(4);
  });
});
