import { ChatCompletionFunctionTool } from 'openai/resources/chat';

export interface Function {
  definition: ChatCompletionFunctionTool['function'];
  call(obj: any): Promise<any>;
  instructions?: string[];
}
