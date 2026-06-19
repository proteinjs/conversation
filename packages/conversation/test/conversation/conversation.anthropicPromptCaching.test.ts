import { Conversation } from '../../src/Conversation';

type AnyMessage = { role: string; content: unknown; providerOptions?: Record<string, Record<string, unknown>> };
type ConversationStatics = {
  applyAnthropicPromptCaching(messages: AnyMessage[]): AnyMessage[];
};
const applyCaching = (messages: AnyMessage[]) =>
  (Conversation as unknown as ConversationStatics).applyAnthropicPromptCaching(messages);

const system = (text: string): AnyMessage => ({ role: 'system', content: text });
const user = (text: string): AnyMessage => ({ role: 'user', content: text });
const assistant = (text: string): AnyMessage => ({ role: 'assistant', content: text });

const cacheControlOf = (msg: AnyMessage) =>
  (msg.providerOptions?.anthropic as { cacheControl?: { type: string } } | undefined)?.cacheControl;
const markedCount = (messages: AnyMessage[]) => messages.filter((m) => cacheControlOf(m)).length;

describe('Conversation.applyAnthropicPromptCaching', () => {
  it('marks the last system message and the last two non-system messages', () => {
    const messages = [system('s'), user('u1'), assistant('a1'), user('u2')];
    const result = applyCaching(messages);

    expect(cacheControlOf(result[0])).toEqual({ type: 'ephemeral' });
    expect(cacheControlOf(result[1])).toBeUndefined();
    expect(cacheControlOf(result[2])).toEqual({ type: 'ephemeral' });
    expect(cacheControlOf(result[3])).toEqual({ type: 'ephemeral' });
    expect(markedCount(result)).toBe(3);
  });

  it('rolls the marks forward and strips stale breakpoints from earlier steps (Anthropic caps at 4 total)', () => {
    const step1 = applyCaching([system('s'), user('u1'), assistant('a1')]);
    // The loop appends a step: previous marks persist on the carried-over messages.
    const step2 = applyCaching([...step1, assistant('a2'), user('u2')]);

    // Exactly three marks: system + the new last two; u1/a1's stale marks are stripped.
    expect(markedCount(step2)).toBe(3);
    expect(cacheControlOf(step2[0])).toEqual({ type: 'ephemeral' });
    expect(cacheControlOf(step2[1])).toBeUndefined();
    expect(cacheControlOf(step2[2])).toBeUndefined();
    expect(cacheControlOf(step2[3])).toEqual({ type: 'ephemeral' });
    expect(cacheControlOf(step2[4])).toEqual({ type: 'ephemeral' });
  });

  it('preserves unrelated providerOptions when marking and when stripping', () => {
    const marked: AnyMessage = {
      ...user('old'),
      providerOptions: { anthropic: { cacheControl: { type: 'ephemeral' }, other: 'keep' }, openai: { x: 1 } },
    };
    const result = applyCaching([system('s'), marked, assistant('a'), user('u')]);

    // Stripped of the stale breakpoint but other options survive.
    expect(cacheControlOf(result[1])).toBeUndefined();
    expect(result[1].providerOptions?.anthropic).toEqual({ other: 'keep' });
    expect(result[1].providerOptions?.openai).toEqual({ x: 1 });
  });

  it('marks what exists when the conversation is shorter than the rolling window', () => {
    const result = applyCaching([system('s'), user('u')]);
    expect(markedCount(result)).toBe(2);
    expect(cacheControlOf(result[0])).toEqual({ type: 'ephemeral' });
    expect(cacheControlOf(result[1])).toEqual({ type: 'ephemeral' });
  });

  it('never mutates the input messages', () => {
    const messages = [system('s'), user('u1'), user('u2')];
    applyCaching(messages);
    for (const msg of messages) {
      expect(msg.providerOptions).toBeUndefined();
    }
  });
});
