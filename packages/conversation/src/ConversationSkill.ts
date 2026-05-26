import type { ToolSet } from 'ai';
import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';

export interface ConversationSkill {
  getName(): string;
  /** Return array of strings that will be formatted with periods in between or return a preformatted string */
  getSystemMessages(): string[] | string | Promise<string[] | string>;
  getFunctions(): Function[];
  getMessageModerators(): MessageModerator[];
  /**
   * Optional provider-defined tools (e.g. Anthropic's native `text_editor` /
   * `bash`) that cannot be expressed as plain `Function`s. These are injected
   * directly into the AI SDK tool set — bypassing `buildAiSdkTools` — the same
   * way `getWebSearchTools` injects provider-executed web search.
   *
   * `provider` is the resolved provider of the active model (e.g. `anthropic`,
   * `openai`), so a skill can return only the tools that provider natively
   * supports and an empty set otherwise.
   */
  getProviderDefinedTools?(provider: string): ToolSet;
}

export interface ConversationSkillFactory {
  createSkill(repoPath: string): Promise<ConversationSkill>;
}
