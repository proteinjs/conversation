import { ChatCompletionTool } from 'openai/resources/chat';

export interface Function {
  definition: ChatCompletionTool['function'];
  call(obj: any): Promise<any>;
  instructions?: string[];
}
