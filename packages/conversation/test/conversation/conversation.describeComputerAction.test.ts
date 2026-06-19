import { Conversation } from '../../src/Conversation';

type ConversationStatics = {
  describeComputerAction(input: Record<string, unknown>): { suffix: string; detail?: string } | undefined;
};
const describe_ = (input: Record<string, unknown>) =>
  (Conversation as unknown as ConversationStatics).describeComputerAction(input);

/**
 * Timeline display mapping for the provider `computer` tool: every action gets a
 * category suffix + a human detail, so long browser sessions read as distinct
 * steps in the thinking timeline.
 */
describe('Conversation.describeComputerAction', () => {
  it('maps click variants to one category with coordinates', () => {
    expect(describe_({ action: 'left_click', coordinate: [200, 125] })).toEqual({
      suffix: 'click',
      detail: '(200,125)',
    });
    expect(describe_({ action: 'double_click', coordinate: [5, 9] })).toEqual({ suffix: 'click', detail: '(5,9)' });
  });

  it('quotes and truncates typed text', () => {
    expect(describe_({ action: 'type', text: 'hello' })).toEqual({ suffix: 'type', detail: '"hello"' });
    const long = 'x'.repeat(60);
    expect(describe_({ action: 'type', text: long })!.detail).toBe(`"${'x'.repeat(40)}…"`);
  });

  it('describes keys, scrolls, drags, waits, and screenshots', () => {
    expect(describe_({ action: 'key', text: 'Return' })).toEqual({ suffix: 'key', detail: 'Return' });
    expect(describe_({ action: 'scroll', scroll_direction: 'down', scroll_amount: 3 })).toEqual({
      suffix: 'scroll',
      detail: 'down ×3',
    });
    expect(describe_({ action: 'left_click_drag', start_coordinate: [1, 2], coordinate: [3, 4] })).toEqual({
      suffix: 'drag',
      detail: '(1,2) → (3,4)',
    });
    expect(describe_({ action: 'wait', duration: 2 })).toEqual({ suffix: 'wait', detail: '2s' });
    expect(describe_({ action: 'screenshot' })).toEqual({ suffix: 'screenshot' });
  });

  it('falls back to a hyphenated suffix for unknown actions and undefined for non-actions', () => {
    expect(describe_({ action: 'left_mouse_down', coordinate: [1, 1] })).toEqual({
      suffix: 'left-mouse-down',
      detail: '(1,1)',
    });
    expect(describe_({ command: 'view', path: '/a' })).toBeUndefined();
    expect(describe_({})).toBeUndefined();
  });
});
