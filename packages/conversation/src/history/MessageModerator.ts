import { ChatCompletionMessageParam } from 'openai/resources/chat';

export interface MessageModerator {
  /** Given a set of messages, modify and return. Only compatible with message content of type string. */
  observe(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[];
}
