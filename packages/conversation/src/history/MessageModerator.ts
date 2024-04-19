import { ChatCompletionMessageParam } from 'openai/resources/chat';

export interface MessageModerator {
  // given a set of messages, modify and return
  observe(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[];
}