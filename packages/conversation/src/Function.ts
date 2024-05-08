import { ChatCompletionCreateParams } from 'openai/resources/chat';

export interface Function {
  definition: ChatCompletionCreateParams.Function;
  call(obj: any): Promise<any>;
  instructions?: string[];
}
