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
   * reach for it — the line the model routes on: `SkillDispatcherSkill` renders
   * `name — summary` in its catalog so an unpinned skill can be discovered.
   * Keep it short (a single sentence). A person never reads it; their line is
   * `getDescription()`.
   */
  getSummary?(): string;
  /**
   * One-line, person-facing description of what this skill does for the
   * person, in their everyday words — the line a picker or a catalog shows
   * under `getName()`. The model never reads it: `SkillDispatcherSkill` routes
   * on `getSummary()` alone and renders this line nowhere, so it is written for
   * people without moving what the model matches on. One declaration, two
   * renderings: the summary for the model, the description for the person.
   * Optional — a skill with no surface for people omits it.
   */
  getDescription?(): string;
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
   * `provider` is the provider the active model's calls are routed to (e.g.
   * `anthropic`, `openai`), so a skill can return only the tools that provider
   * natively supports and an empty set otherwise.
   *
   * A skill may also return ordinary FUNCTION tools here (a portable stand-in
   * for a native tool on the other providers). Those are told what every other
   * function tool is told about strict mode (see `ToolStrictness`): non-strict
   * unless the tool itself sets `strict`.
   */
  getProviderDefinedTools?(provider: string): ToolSet;
}

export interface ConversationSkillFactory {
  createSkill(repoPath: string): Promise<ConversationSkill>;
}
