import { ChatCompletionFunctionTool } from 'openai/resources/chat';

export interface Function {
  definition: ChatCompletionFunctionTool['function'];
  call(obj: any): Promise<any>;
  instructions?: string[];
  /**
   * Optional: produce a short, human-meaningful subject for a call to this
   * tool — typically the acted-on entity's name/title — to personalize the
   * call's node in the chat thinking timeline. May do a lookup. Best-effort:
   * the framework swallows errors and falls back to a generic detail.
   */
  getTimelineDetail?(args: any): string | undefined | Promise<string | undefined>;
}
