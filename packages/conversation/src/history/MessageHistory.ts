import { ChatCompletionMessageParam } from 'openai/resources/chat';
import { Logger } from '@proteinjs/logger';

export interface MessageHistoryParams {
  /**
   * When true, `prune()` drops non-system messages beyond `maxMessages` (FIFO).
   * Defaults to `false`: the in-memory `MessageHistory` is a pass-through cache;
   * callers that need pruning (older tests, OpenAI legacy code path) opt in
   * explicitly. Production thought flow persists messages separately and feeds
   * them in via `generateStream({ messages })` — it should never silently lose
   * history here.
   */
  enforceMessageLimit?: boolean;
  maxMessages: number; // max number of non-system messages to retain, fifo
}

export class MessageHistory {
  private logger = new Logger({ name: this.constructor.name });
  private messages: ChatCompletionMessageParam[] = [];
  private params: MessageHistoryParams;

  constructor(params?: Partial<MessageHistoryParams>) {
    this.params = Object.assign({ maxMessages: 20, enforceMessageLimit: false }, params);
  }

  getMessages() {
    return this.messages;
  }

  toString() {
    return this.messages.map((message) => message.content).join('. ');
  }

  setMessages(messages: ChatCompletionMessageParam[]): MessageHistory {
    this.messages = messages;
    this.prune();
    return this;
  }

  push(messages: ChatCompletionMessageParam[]): MessageHistory {
    this.messages.push(...messages);
    this.prune();
    return this;
  }

  prune() {
    if (!this.params.enforceMessageLimit) {
      return;
    }

    let messageCount = 0;
    const messagesToRemoveIndexes: number[] = [];
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const message = this.messages[i];
      if (message.role == 'system') {
        continue;
      }

      messageCount++;
      if (messageCount > this.params.maxMessages) {
        messagesToRemoveIndexes.push(i);
      }
    }

    this.messages = this.messages.filter((message, i) => !messagesToRemoveIndexes.includes(i));
    this.logger.debug({ message: `Pruned ${messagesToRemoveIndexes.length} messages` });
  }
}
