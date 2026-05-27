import { ConversationSkill } from './ConversationSkill';
import { Function as ConvFunction } from './Function';
import { MessageModerator } from './history/MessageModerator';

export interface SkillDispatcherOptions {
  /**
   * Fired when the model invokes `useSkill` to dispatch into one of the
   * unpinned skills. The callback's argument is the skill's `getId()`.
   * Implementers typically use this to auto-pin the skill for subsequent
   * turns so the model can call its tools directly without dispatcher
   * indirection. Best-effort: errors thrown by the callback are swallowed
   * so they cannot fail the dispatch itself.
   */
  onSkillUsed?: (skillId: string) => void | Promise<void>;
}

/**
 * Cross-provider lazy-loading layer for `ConversationSkill`s.
 *
 * In the chat-agent's "pin list / dispatcher" model, a small set of skills
 * load eagerly each turn (their tools are emitted natively, their
 * instructions injected). The remaining skills are handed to this
 * `SkillDispatcherSkill`, which surfaces them through three function tools
 * the model uses to discover and invoke them mid-turn:
 *
 *   - `listAvailableSkills()` — index of unpinned skills (id + summary)
 *   - `describeSkill({ skill })` — full instructions + per-tool JSON schemas
 *   - `useSkill({ skill, tool, args })` — dispatch into the resolved tool
 *
 * The pattern mirrors `ProgressiveContextSkill`'s drill-down (catalog →
 * detail → dispatch) and deliberately avoids provider-specific
 * mid-turn-loading features (Anthropic's `defer_loading`/`tool_search`,
 * OpenAI's equivalent on the Responses API): one code path, works on any
 * provider that supports function calling. When provider-native tool search
 * becomes valuable for catalog sizes well beyond what `describeSkill` can
 * comfortably render, the dispatcher can be swapped for a provider-aware
 * implementation without changing the rest of the system — the
 * pinned/unpinned partition stays the same.
 *
 * Only function tools (from `getFunctions()`) are dispatcher-reachable.
 * Provider-defined tools (Anthropic's native `text_editor` / `bash`, etc.)
 * are only available when the owning skill is pinned and emitted directly
 * to the model. Skills whose primary surface is provider-defined should
 * provide function-tool fallbacks if mid-turn dispatcher access matters
 * before auto-pin kicks in on the next turn.
 *
 * Duplicate-id detection runs at construction time: passing two skills
 * with the same `getId()` throws immediately, so misconfigured catalogs
 * fail at boot rather than silently overwriting one entry with the other.
 */
export class SkillDispatcherSkill implements ConversationSkill {
  private readonly skills = new Map<string, ConversationSkill>();
  private readonly onSkillUsed?: SkillDispatcherOptions['onSkillUsed'];

  constructor(skills: ConversationSkill[], options: SkillDispatcherOptions = {}) {
    for (const skill of skills) {
      const id = skill.getId();
      const existing = this.skills.get(id);
      if (existing) {
        throw new Error(
          `Duplicate skill id "${id}": both ${existing.getName()} and ${skill.getName()} ` +
            `claim it. Skill ids must be unique within a single Conversation.`
        );
      }
      this.skills.set(id, skill);
    }
    this.onSkillUsed = options.onSkillUsed;
  }

  // ─── ConversationSkill surface ─────────────────────────────────────────────

  getId(): string {
    return 'skill-dispatcher';
  }

  getName(): string {
    return 'Skill Dispatcher';
  }

  getSummary(): string {
    return 'Discover and invoke skills that are available but not loaded directly this turn.';
  }

  getSystemMessages(): string {
    if (this.skills.size === 0) {
      return '';
    }
    const ids = Array.from(this.skills.keys()).sort();
    return (
      `You have additional skills available that are not eagerly loaded this turn. ` +
      `Use \`listAvailableSkills\` to see them, \`describeSkill\` to learn how to ` +
      `call a specific skill's tools, and \`useSkill\` to invoke a tool from an ` +
      `unpinned skill. After the first \`useSkill\` call, that skill is pinned ` +
      `for subsequent turns and its tools become directly callable. ` +
      `Skills available now: ${ids.join(', ')}.`
    );
  }

  getFunctions(): ConvFunction[] {
    if (this.skills.size === 0) {
      return [];
    }
    return [this.listAvailableSkillsFunction(), this.describeSkillFunction(), this.useSkillFunction()];
  }

  getMessageModerators(): MessageModerator[] {
    return [];
  }

  // ─── public read access (handy for callers/telemetry) ──────────────────────

  has(skillId: string): boolean {
    return this.skills.has(skillId);
  }

  size(): number {
    return this.skills.size;
  }

  // ─── tools ─────────────────────────────────────────────────────────────────

  private listAvailableSkillsFunction(): ConvFunction {
    return {
      definition: {
        name: 'listAvailableSkills',
        description:
          'List the skills that are available this turn but not directly loaded. ' +
          'Returns one entry per skill with its id and a short summary. Use this when ' +
          "you don't already know which skill can do what you need; then call " +
          '`describeSkill` to drill in.',
        parameters: { type: 'object', properties: {}, additionalProperties: false },
      },
      call: async () => this.renderCatalog(),
    };
  }

  private describeSkillFunction(): ConvFunction {
    return {
      definition: {
        name: 'describeSkill',
        description:
          'Return the full instructions and tool catalog (with JSON schemas) for an ' +
          'unpinned skill, so you know what its tools do and how to call them via ' +
          '`useSkill`.',
        parameters: {
          type: 'object',
          properties: {
            skill: {
              type: 'string',
              description: 'The skill id (as returned by `listAvailableSkills`).',
            },
          },
          required: ['skill'],
          additionalProperties: false,
        },
      },
      call: async ({ skill }: { skill?: string }) => this.renderDetail(skill),
    };
  }

  private useSkillFunction(): ConvFunction {
    return {
      definition: {
        name: 'useSkill',
        description:
          'Invoke a specific tool on an unpinned skill. Call `describeSkill` first if ' +
          "you don't already know the tool's parameter shape. After this call the " +
          'skill is pinned for subsequent turns and its tools become directly callable.',
        parameters: {
          type: 'object',
          properties: {
            skill: { type: 'string', description: 'The skill id.' },
            tool: { type: 'string', description: 'The name of a tool exposed by the skill.' },
            args: {
              type: 'object',
              description: 'Arguments for the tool, matching its declared parameters.',
              additionalProperties: true,
            },
          },
          required: ['skill', 'tool', 'args'],
          additionalProperties: false,
        },
      },
      call: async ({ skill, tool, args }: { skill?: string; tool?: string; args?: unknown }) =>
        this.dispatch(skill, tool, args),
    };
  }

  // ─── rendering ─────────────────────────────────────────────────────────────

  private renderCatalog(): string {
    if (this.skills.size === 0) {
      return 'No additional skills are available this turn.';
    }
    const lines: string[] = ['# Available skills', ''];
    for (const skill of this.sortedSkills()) {
      const id = skill.getId();
      const name = skill.getName();
      const summary = skill.getSummary?.() ?? '';
      lines.push(`- **${id}** (${name})${summary ? ` — ${summary}` : ''}`);
    }
    lines.push('');
    lines.push('Call `describeSkill({ skill: "<id>" })` to see a skill\'s tools and how to use them.');
    return lines.join('\n');
  }

  private async renderDetail(skillId?: string): Promise<string> {
    if (!skillId) {
      return this.errorMissing('skill');
    }
    const skill = this.skills.get(skillId);
    if (!skill) {
      return this.errorUnknownSkill(skillId);
    }
    const lines: string[] = [`# ${skill.getName()}`, '', `**id:** \`${skill.getId()}\``];

    const summary = skill.getSummary?.();
    if (summary) {
      lines.push('', `**Summary:** ${summary}`);
    }

    const whenToUse = skill.getWhenToUse?.();
    if (whenToUse) {
      lines.push('', '## When to use', '', whenToUse);
    }

    const instructions = await Promise.resolve(skill.getSystemMessages());
    const instructionsText = Array.isArray(instructions) ? instructions.join('\n\n') : instructions;
    if (instructionsText && instructionsText.trim()) {
      lines.push('', '## Instructions', '', instructionsText.trim());
    }

    const tools = skill.getFunctions();
    if (tools.length === 0) {
      lines.push('', '## Tools', '', '_This skill has no dispatcher-reachable tools._');
    } else {
      lines.push('', '## Tools', '');
      lines.push(
        `Invoke any of these with \`useSkill({ skill: "${skill.getId()}", tool: "<name>", args: { ... } })\`.`,
        ''
      );
      for (const tool of tools) {
        const def = tool.definition;
        lines.push(`### ${def.name}`, '');
        if (def.description) {
          lines.push(def.description, '');
        }
        const schema = def.parameters ? JSON.stringify(def.parameters, null, 2) : '_(no parameters)_';
        lines.push('**Parameters:**', '', '```json', schema, '```', '');
      }
    }

    return lines.join('\n');
  }

  // ─── dispatch ──────────────────────────────────────────────────────────────

  private async dispatch(skillId: string | undefined, toolName: string | undefined, args: unknown): Promise<string> {
    if (!skillId) {
      return this.errorMissing('skill');
    }
    if (!toolName) {
      return this.errorMissing('tool');
    }
    const skill = this.skills.get(skillId);
    if (!skill) {
      return this.errorUnknownSkill(skillId);
    }
    const tool = skill.getFunctions().find((f) => f.definition.name === toolName);
    if (!tool) {
      const available = skill.getFunctions().map((f) => f.definition.name);
      return (
        `Skill "${skillId}" has no dispatcher-reachable tool named "${toolName}". ` +
        `Available tools: ${available.length === 0 ? '(none)' : available.join(', ')}.`
      );
    }

    let result: unknown;
    try {
      result = await tool.call(args ?? {});
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      return `Error invoking ${skillId}.${toolName}: ${msg}`;
    }

    if (this.onSkillUsed) {
      try {
        await this.onSkillUsed(skillId);
      } catch {
        // Auto-pin notification is best-effort; never fail the dispatch.
      }
    }

    if (typeof result === 'string') {
      return result;
    }
    try {
      return JSON.stringify(result, null, 2);
    } catch {
      return String(result);
    }
  }

  // ─── helpers ───────────────────────────────────────────────────────────────

  private sortedSkills(): ConversationSkill[] {
    return Array.from(this.skills.values()).sort((a, b) => a.getId().localeCompare(b.getId()));
  }

  private errorMissing(field: string): string {
    return `Missing required argument: \`${field}\`.`;
  }

  private errorUnknownSkill(skillId: string): string {
    const known = Array.from(this.skills.keys()).sort().join(', ');
    return `No skill with id "${skillId}" is available. Known skills: ${known || '(none)'}.`;
  }
}
