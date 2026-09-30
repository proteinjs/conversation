import { randomUUID } from 'crypto';
import type { ToolSet } from 'ai';
import type { Logger } from '@proteinjs/logger';
import type { Function } from './Function';

/** A tool result set aside WHOLE — what the consumer's store keeps and what the read tool opens. */
export type ToolResultRecord = {
  /** The tool call's id — the pointer the transcript carries and the read tool takes. */
  id: string;
  toolName: string;
  /** The call's arguments as the model gave them (a store may bound its own copy). */
  input: unknown;
  /** The whole result as text: a string result verbatim, any other value as its JSON. */
  text: string;
  chars: number;
  lines: number;
  /** The count of `text` in the encoder every budgeting layer shares (o200k). */
  tokens: number;
  /**
   * Why it was set aside: larger than the per-result ceiling when the tool returned it, or
   * evicted from the outgoing request by the per-turn budget (`toolResultTokenBudget`).
   */
  reason: 'over-ceiling' | 'over-budget';
};

/** The consumer's durable keep of a set-aside result (an artifact, a file — its own record). */
export type ToolResultStore = {
  keep(record: ToolResultRecord): Promise<void> | void;
};

/**
 * The per-result ceiling on what a tool result may carry INTO the conversation whole, and the
 * store that keeps the whole when one is set aside (see {@link ToolResultOverflow}).
 */
export type ToolResultCeiling = {
  /** The largest result that enters the transcript whole, in tokens of the shared (o200k) encoder. */
  tokensPerResult: number;
  /**
   * The durable keep. Optional: without one the whole lives in the conversation's own registry
   * for the conversation's lifetime, which is all the model's read door needs.
   */
  store?: ToolResultStore;
};

/** The arguments of the read tool ({@link ToolResultOverflow.READ_TOOL_NAME}). */
export type ReadToolResultArgs = {
  id: string;
  /** 1-based first line of the range (default 1). */
  from?: number;
  /** Lines in the range, or matches for a search (default 200, at most 1,000). */
  lines?: number;
  /** Literal text to search for (case-insensitive) — the matching lines with their numbers, instead of a range. */
  find?: string;
};

/** The symbol `buildAiSdkTools` stashes resolved multimodal content parts under (a registry symbol, shared by key). */
export const MULTIMODAL_TOOL_RESULT: unique symbol = Symbol.for('conversation.tool.multimodal');

type TextPartLike = { type?: string; text?: string };

/**
 * AN OVERSIZED TOOL RESULT NEVER ENTERS THE CONVERSATION WHOLE — the categorical law for every
 * tool the loop executes in-process (function tools and provider-defined tools alike), never a
 * special case for one tool. A loop resends every prior step's results on every step, and one
 * result can be larger than a model's whole window (a listing of forty thousand paths counted
 * over a million tokens by itself, 2026-09-29): nothing downstream can bound what has already
 * been appended, so the bound sits where the result is produced.
 *
 * Over the ceiling, the result is SET ASIDE: the whole is kept (the conversation's registry for
 * the model's read door; the consumer's store for its durable record — nothing is lost, nothing
 * is trimmed away) and the transcript gets its head and a pointer. The model opens the whole
 * through {@link READ_TOOL_NAME}: a range of lines, or a search for literal text. The per-turn
 * budget's eviction ({@link evicted}) hands out the same pointer, so a result evicted from the
 * request is re-openable instead of gone.
 */
export class ToolResultOverflow {
  static readonly READ_TOOL_NAME = 'read_tool_result';

  private static readonly HEAD_LINES = 40;
  private static readonly HEAD_CHARS = 4_000;
  private static readonly READ_LINES_DEFAULT = 200;
  private static readonly READ_LINES_MAX = 1_000;

  constructor(
    private readonly args: {
      ceiling: ToolResultCeiling;
      /** The conversation's registry of set-aside results — shared across its calls so a later call's read finds an earlier set-aside. */
      registry: Map<string, ToolResultRecord>;
      countTokens: (text: string) => number;
      logger: Logger;
    }
  ) {}

  /** Every tool that executes in-process gets its output bounded — a tool without `execute` (a server tool) rides untouched. */
  wrapTools(tools: ToolSet): ToolSet {
    const wrapped: ToolSet = {};
    for (const [name, tool] of Object.entries(tools)) {
      const execute = (tool as { execute?: unknown }).execute;
      if (typeof execute !== 'function') {
        wrapped[name] = tool;
        continue;
      }
      wrapped[name] = {
        ...tool,
        execute: async (input: unknown, options?: { toolCallId?: string }) => {
          const output = await (execute as (input: unknown, options?: unknown) => Promise<unknown>)(input, options);
          return this.bound(output, { toolCallId: options?.toolCallId ?? randomUUID(), toolName: name, input });
        },
      } as ToolSet[string];
    }
    return wrapped;
  }

  /** The tool the model opens a set-aside result with. */
  readFunction(): Function {
    return {
      definition: {
        name: ToolResultOverflow.READ_TOOL_NAME,
        description:
          'Open a tool result that was set aside because it was too large to enter the conversation whole. ' +
          'Read a range of its lines (from, lines) or search it for literal text (find). ' +
          'The message that replaced the result names its id.',
        parameters: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'The set-aside result’s id, as the message that replaced it names it.' },
            from: { type: 'number', description: '1-based first line to read (default 1).' },
            lines: {
              type: 'number',
              description: 'How many lines to read, or how many matches to return (default 200, at most 1000).',
            },
            find: {
              type: 'string',
              description:
                'Literal text to search for (case-insensitive): returns the matching lines with their numbers instead of a range.',
            },
          },
          required: ['id'],
          additionalProperties: false,
        },
      },
      call: async (args: ReadToolResultArgs) => this.read(args),
    };
  }

  /**
   * What replaces a result the per-turn budget evicts from the outgoing request: the whole is
   * registered (and kept by the store, detached — a projection never waits on I/O) and the
   * pointer comes back as the placeholder's text. Idempotent per tool call.
   */
  evicted(part: { toolCallId?: string; toolName?: string; input?: unknown }, text: string): string {
    const id = part.toolCallId ?? randomUUID();
    let record = this.args.registry.get(id);
    if (!record) {
      record = this.record({ id, toolName: part.toolName ?? 'tool', input: part.input, text, reason: 'over-budget' });
      this.args.registry.set(id, record);
      const store = this.args.ceiling.store;
      if (store) {
        void Promise.resolve()
          .then(() => store.keep(record!))
          .catch((error: unknown) => {
            this.args.logger.warn({
              message: 'The store did not keep an evicted tool result',
              obj: { id, error: error instanceof Error ? error.message : String(error) },
            });
          });
      }
    }
    return (
      `[Tool result set aside to fit the context budget: ${record.toolName}, ${ToolResultOverflow.n(record.tokens)} tokens ` +
      `over ${ToolResultOverflow.n(record.lines)} lines. The whole is kept as tool result "${id}" — read it with ` +
      `${ToolResultOverflow.READ_TOOL_NAME} (id, from, lines) or search it with ${ToolResultOverflow.READ_TOOL_NAME} (id, find).]`
    );
  }

  /** The record the registry holds and the store keeps. */
  private record(args: {
    id: string;
    toolName: string;
    input: unknown;
    text: string;
    reason: ToolResultRecord['reason'];
  }): ToolResultRecord {
    return {
      id: args.id,
      toolName: args.toolName,
      input: args.input,
      text: args.text,
      chars: args.text.length,
      lines: ToolResultOverflow.lineCount(args.text),
      tokens: this.args.countTokens(args.text),
      reason: args.reason,
    };
  }

  /**
   * The bound at the tool's return: a result whose text is over the ceiling is set aside and its
   * head + pointer go back in its place. A multimodal result keeps its image parts and has its
   * text parts replaced; a result with no text form (a streaming tool's iterable) rides untouched.
   */
  private async bound(
    output: unknown,
    call: { toolCallId: string; toolName: string; input: unknown }
  ): Promise<unknown> {
    const text = ToolResultOverflow.textOf(output);
    if (text === undefined) {
      return output;
    }
    const tokens = this.args.countTokens(text);
    if (tokens <= this.args.ceiling.tokensPerResult) {
      return output;
    }
    const record: ToolResultRecord = {
      ...this.record({ id: call.toolCallId, toolName: call.toolName, input: call.input, text, reason: 'over-ceiling' }),
      tokens,
    };
    this.args.registry.set(record.id, record);
    await this.args.ceiling.store?.keep(record);
    this.args.logger.info({
      message: 'Tool result set aside: over the per-result ceiling',
      obj: {
        id: record.id,
        toolName: record.toolName,
        tokens,
        lines: record.lines,
        ceiling: this.args.ceiling.tokensPerResult,
      },
    });
    const placeholder = this.setAsideText(record);
    if (ToolResultOverflow.isMultimodal(output)) {
      const parts = (output as Record<symbol, TextPartLike[]>)[MULTIMODAL_TOOL_RESULT];
      return {
        [MULTIMODAL_TOOL_RESULT]: [...parts.filter((p) => p.type !== 'text'), { type: 'text', text: placeholder }],
      };
    }
    return placeholder;
  }

  private setAsideText(record: ToolResultRecord): string {
    return (
      `[Tool result set aside: ${ToolResultOverflow.n(record.tokens)} tokens over ${ToolResultOverflow.n(record.lines)} lines is ` +
      `past the ${ToolResultOverflow.n(this.args.ceiling.tokensPerResult)}-token limit for one result. The whole is kept as ` +
      `tool result "${record.id}" — read it with ${ToolResultOverflow.READ_TOOL_NAME} (id, from, lines) or search it ` +
      `with ${ToolResultOverflow.READ_TOOL_NAME} (id, find). It begins:\n${ToolResultOverflow.head(record.text)}]`
    );
  }

  /** The read door: a numbered range, or the matching lines of a literal search — never more than the ceiling. */
  private read(args: ReadToolResultArgs): string {
    const record = this.args.registry.get(String(args.id ?? ''));
    if (!record) {
      return `No set-aside tool result has id "${String(args.id ?? '')}".`;
    }
    const all = record.text.split('\n');
    const total = all.length;
    const limit = ToolResultOverflow.clamp(
      args.lines ?? ToolResultOverflow.READ_LINES_DEFAULT,
      1,
      ToolResultOverflow.READ_LINES_MAX
    );
    let header: string;
    let body: string[];
    if (typeof args.find === 'string' && args.find.length > 0) {
      const needle = args.find.toLowerCase();
      const matches: string[] = [];
      let found = 0;
      for (let i = 0; i < all.length; i++) {
        if (all[i].toLowerCase().includes(needle)) {
          found++;
          if (matches.length < limit) {
            matches.push(`${i + 1}: ${all[i]}`);
          }
        }
      }
      header = `Matches for "${args.find}" in tool result "${record.id}" (${record.toolName}, ${ToolResultOverflow.n(total)} lines): ${ToolResultOverflow.n(matches.length)} of ${ToolResultOverflow.n(found)} shown.`;
      body = matches;
    } else {
      const from = ToolResultOverflow.clamp(Math.floor(args.from ?? 1), 1, total);
      const to = Math.min(total, from + limit - 1);
      header = `Lines ${ToolResultOverflow.n(from)}–${ToolResultOverflow.n(to)} of ${ToolResultOverflow.n(total)} (tool result "${record.id}", ${record.toolName}).`;
      body = all.slice(from - 1, to).map((line, i) => `${from + i}: ${line}`);
    }
    return this.withinCeiling(header, body);
  }

  /** A read is itself a tool result: it stays under the ceiling, cut from the end with the cut named. */
  private withinCeiling(header: string, body: string[]): string {
    let lines = body;
    let text = [header, ...lines].join('\n');
    while (this.args.countTokens(text) > this.args.ceiling.tokensPerResult && lines.length > 1) {
      lines = lines.slice(0, Math.max(1, Math.floor(lines.length / 2)));
      text = [
        header,
        ...lines,
        `[cut at the ${ToolResultOverflow.n(this.args.ceiling.tokensPerResult)}-token limit after ${ToolResultOverflow.n(lines.length)} lines — ask for fewer lines]`,
      ].join('\n');
    }
    return text;
  }

  /** A result's text form: a string verbatim, a multimodal result's text parts, anything else as JSON; none for a value with no text form. */
  private static textOf(output: unknown): string | undefined {
    if (typeof output === 'string') {
      return output;
    }
    if (output === undefined || output === null) {
      return undefined;
    }
    if (ToolResultOverflow.isMultimodal(output)) {
      const parts = (output as Record<symbol, TextPartLike[]>)[MULTIMODAL_TOOL_RESULT];
      const texts = parts.filter((p) => p.type === 'text' && typeof p.text === 'string').map((p) => p.text as string);
      return texts.length > 0 ? texts.join('\n') : undefined;
    }
    if (typeof output === 'object' && Symbol.asyncIterator in (output as object)) {
      return undefined;
    }
    try {
      return JSON.stringify(output) ?? String(output);
    } catch {
      return String(output);
    }
  }

  private static isMultimodal(output: unknown): boolean {
    return (
      typeof output === 'object' &&
      output !== null &&
      MULTIMODAL_TOOL_RESULT in (output as object) &&
      Array.isArray((output as Record<symbol, unknown>)[MULTIMODAL_TOOL_RESULT])
    );
  }

  private static head(text: string): string {
    const lines = text.split('\n').slice(0, ToolResultOverflow.HEAD_LINES).join('\n');
    return lines.length > ToolResultOverflow.HEAD_CHARS ? `${lines.slice(0, ToolResultOverflow.HEAD_CHARS)}…` : lines;
  }

  private static lineCount(text: string): number {
    return text.length === 0 ? 0 : text.split('\n').length;
  }

  private static clamp(value: number, min: number, max: number): number {
    return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min;
  }

  private static n(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }
}
