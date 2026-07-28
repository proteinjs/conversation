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
  /**
   * Optional: outcome-aware re-labeling of the just-finished call's timeline node. The
   * call-time node is named from the INPUT alone, so a call that ends up applying nothing
   * (e.g. an edit refused by a freshness/lock fence) would still render as done — this hook
   * inspects the RESULT and relabels. Best-effort: the framework swallows errors and keeps
   * the call-time labeling. `name` REPLACES the node's tool name — use the `tool:variant`
   * suffix convention (e.g. `editThoughts:deferred`) so presentation maps can target it;
   * `detail` replaces the node's detail (omit to keep the call-time one). Return undefined
   * (or an empty object) for no relabel.
   */
  getTimelineOutcome?(
    args: any,
    result: any
  ):
    | { name?: string; detail?: string | ToolTimelineDetail }
    | undefined
    | Promise<{ name?: string; detail?: string | ToolTimelineDetail } | undefined>;
}
