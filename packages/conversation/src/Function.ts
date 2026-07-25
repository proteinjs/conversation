import { ChatCompletionFunctionTool } from 'openai/resources/chat';

/**
 * A tool call's timeline subject: display text plus an optional app deep-link href that lets the
 * rendering layer make the timeline detail clickable (e.g. `thought://nav?...` opens the edited
 * document). Plain-string details remain valid — most tools have nothing to link to.
 */
export type ToolTimelineDetail = { text: string; href?: string };

export interface Function {
  definition: ChatCompletionFunctionTool['function'];
  call(obj: any): Promise<any>;
  instructions?: string[];
  /**
   * Optional: produce a short, human-meaningful subject for a call to this
   * tool — typically the acted-on entity's name/title — to personalize the
   * call's node in the chat thinking timeline. May do a lookup. Best-effort:
   * the framework swallows errors and falls back to a generic detail.
   * Return a `ToolTimelineDetail` to also carry a deep-link href.
   */
  getTimelineDetail?(
    args: any
  ): string | ToolTimelineDetail | undefined | Promise<string | ToolTimelineDetail | undefined>;
}
