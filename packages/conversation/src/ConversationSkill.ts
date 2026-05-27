import type { ToolSet } from 'ai';
import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';

export interface ConversationSkill {
  /**
   * Stable, kebab-case identifier for this skill. Must be unique across all
   * skills loaded into a single `Conversation` and durable across renames —
   * consumers (pin lists, catalogs, persisted "active skills" sets, telemetry)
   * key off this. Pick once and don't change it; rename `getName()` freely
   * but leave `getId()` alone.
   */
  getId(): string;
  getName(): string;
  /**
   * One-line, model-facing summary of what this skill is and roughly when to
   * reach for it. Surfaced by `SkillDispatcherSkill` in its catalog so an
   * unpinned skill can be discovered. Keep it short (a single sentence).
   */
  getSummary?(): string;
  /**
   * Optional usage hint — extra detail on when to reach for this skill, what
   * it's best at, and when *not* to use it. Surfaced by
   * `SkillDispatcherSkill` alongside the summary when the model drills in
   * with `describeSkill`.
   */
  getWhenToUse?(): string;
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
