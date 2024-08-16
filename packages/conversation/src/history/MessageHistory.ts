import { ChatCompletionMessageParam } from 'openai/resources/chat';
import { Logger } from '@proteinjs/logger';

export interface MessageHistoryParams {
  enforceMessageLimit?: boolean;
  maxMessages: number; // max number of non-system messages to retain, fifo
}

export class MessageHistory {
  private logger = new Logger({ name: this.constructor.name });
  private messages: ChatCompletionMessageParam[] = [];
  private params: MessageHistoryParams;

  constructor(params?: Partial<MessageHistoryParams>) {
    this.params = Object.assign({ maxMessages: 20 }, params);
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
    if (this.params.enforceMessageLimit === false) {
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
