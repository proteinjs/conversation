/**
 * The real Anthropic provider over a scripted transport that bills as the provider's prompt cache
 * does — the documented mechanics mirrored on the wire body. A request's prefix is its tools, then
 * its system blocks, then its messages, block by block; cache entries are created at every explicit
 * breakpoint (`cache_control`); a request READS the longest prefix a previous request's breakpoint
 * created, WRITES from the end of that read to its own last breakpoint, and pays everything after
 * its last breakpoint at the fresh price. Tokens are sized by bytes (four a token); entries never
 * expire here (the TTL is a clock, not a layout). The answers are scripted from the request alone:
 * a user message asking for `[edit]` draws a call to the named tool, anything else a text answer.
 *
 * Beside the usage the provider reports (what a step's `UsageData` row carries), every request's
 * billing is kept with the cut it hit and the marks it carried, so a suite can say which tier a
 * request read and which it wrote.
 */

export type WireContentBlock = { type: string; text?: string; cache_control?: unknown } & Record<string, unknown>;
export type WireMessage = { role: string; content: string | WireContentBlock[] };
export type WireTool = { name?: string; type?: string; cache_control?: unknown } & Record<string, unknown>;
export type WireBody = {
  model: string;
  tools?: WireTool[];
  system?: WireContentBlock[];
  messages: WireMessage[];
};

/** One request's billing: the tokens by tier, and where the cache hit fell in the request's blocks. */
export type PromptCacheBilling = {
  read: number;
  write: number;
  fresh: number;
  /** The block boundary the read reached (0 = nothing read); the last breakpoint's boundary. */
  hitAt: number;
  lastMark: number;
  /** The breakpoints, as `tool:<i>`, `system:<i>`, `message:<i>`. */
  marks: string[];
};

/** The prefix's blocks of one request, sized one by one: the tools, the system blocks, the messages. */
export type PrefixBlocks = {
  tools: number[];
  system: number[];
  messages: number[];
};

const sse = (payload: Record<string, unknown>): string =>
  `event: ${String(payload.type)}\ndata: ${JSON.stringify(payload)}\n\n`;

export class PromptCacheScriptedTransport {
  readonly requests: WireBody[] = [];
  readonly billings: PromptCacheBilling[] = [];
  /** The tool a `[edit]` ask draws a call to; a text answer when absent. */
  editTool?: string;
  private readonly entries = new Set<string>();

  readonly fetch = async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const body = JSON.parse(String(init?.body)) as WireBody;
    this.requests.push(body);
    const billing = this.bill(body);
    this.billings.push(billing);
    const asksForEdit = this.editTool && PromptCacheScriptedTransport.lastUserText(body).includes('[edit]');
    const answer = asksForEdit
      ? PromptCacheScriptedTransport.toolCallAnswer(body.model, this.editTool!, billing)
      : PromptCacheScriptedTransport.textAnswer(body.model, 'Done.', billing);
    return new Response(answer, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };

  /** The sizes of a request's blocks, the way this transport bills them. */
  static prefixBlocks(body: WireBody): PrefixBlocks {
    const blocks = PromptCacheScriptedTransport.blocks(body);
    const toolCount = (body.tools ?? []).length;
    const systemCount = (body.system ?? []).length;
    const sized = blocks.map((block) => PromptCacheScriptedTransport.tokensOf(block));
    return {
      tools: sized.slice(0, toolCount),
      system: sized.slice(toolCount, toolCount + systemCount),
      messages: sized.slice(toolCount + systemCount),
    };
  }

  /** The text of the request's last user message. */
  static lastUserText(body: WireBody): string {
    const last = body.messages[body.messages.length - 1];
    if (!last || last.role !== 'user') {
      return '';
    }
    return typeof last.content === 'string'
      ? last.content
      : last.content.map((part) => (part.type === 'text' ? part.text ?? '' : '')).join('');
  }

  private bill(body: WireBody): PromptCacheBilling {
    const blocks = PromptCacheScriptedTransport.blocks(body);
    const sized = blocks.map((block) => PromptCacheScriptedTransport.tokensOf(block));
    const marks = PromptCacheScriptedTransport.marks(body);
    const markCuts = marks.map((mark) => mark.cut);
    const lastMark = markCuts.length > 0 ? Math.max(...markCuts) : 0;
    const total = (from: number, to: number) => sized.slice(from, to).reduce((sum, n) => sum + n, 0);
    // The longest prefix a previous breakpoint created, checked at every block boundary up to this
    // request's last breakpoint.
    let hitAt = 0;
    for (let cut = lastMark; cut > 0; cut--) {
      if (this.entries.has(PromptCacheScriptedTransport.key(blocks, cut))) {
        hitAt = cut;
        break;
      }
    }
    for (const cut of markCuts) {
      if (cut > hitAt) {
        this.entries.add(PromptCacheScriptedTransport.key(blocks, cut));
      }
    }
    return {
      read: total(0, hitAt),
      write: total(hitAt, lastMark),
      fresh: total(lastMark, blocks.length),
      hitAt,
      lastMark,
      marks: marks.map((mark) => mark.name),
    };
  }

  /** The request's blocks in prefix order, each serialized without its breakpoint. */
  private static blocks(body: WireBody): string[] {
    const strip = (value: unknown): unknown => {
      if (Array.isArray(value)) {
        return value.map(strip);
      }
      if (value && typeof value === 'object') {
        const out: Record<string, unknown> = {};
        for (const [key, inner] of Object.entries(value as Record<string, unknown>)) {
          if (key !== 'cache_control') {
            out[key] = strip(inner);
          }
        }
        return out;
      }
      return value;
    };
    return [
      ...(body.tools ?? []).map((tool) => `tool:${JSON.stringify(strip(tool))}`),
      ...(body.system ?? []).map((block) => `system:${JSON.stringify(strip(block))}`),
      ...body.messages.map((message) => `message:${JSON.stringify(strip(message))}`),
    ];
  }

  /** The request's breakpoints as the boundary each one cuts the blocks at. */
  private static marks(body: WireBody): Array<{ name: string; cut: number }> {
    const marks: Array<{ name: string; cut: number }> = [];
    const tools = body.tools ?? [];
    const system = body.system ?? [];
    tools.forEach((tool, i) => {
      if (tool.cache_control) {
        marks.push({ name: `tool:${i}`, cut: i + 1 });
      }
    });
    system.forEach((block, i) => {
      if (block.cache_control) {
        marks.push({ name: `system:${i}`, cut: tools.length + i + 1 });
      }
    });
    body.messages.forEach((message, i) => {
      const marked =
        typeof message.content !== 'string' && message.content.some((part) => part.cache_control !== undefined);
      if (marked) {
        marks.push({ name: `message:${i}`, cut: tools.length + system.length + i + 1 });
      }
    });
    return marks;
  }

  private static key(blocks: string[], cut: number): string {
    return blocks.slice(0, cut).join('\u0000');
  }

  private static tokensOf(block: string): number {
    return Math.ceil(Buffer.byteLength(block, 'utf8') / 4);
  }

  private static messageStart(model: string, billing: PromptCacheBilling): string {
    return sse({
      type: 'message_start',
      message: {
        id: 'msg_scripted',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        usage: {
          input_tokens: billing.fresh,
          cache_creation_input_tokens: billing.write,
          cache_read_input_tokens: billing.read,
          output_tokens: 1,
        },
      },
    });
  }

  private static textAnswer(model: string, text: string, billing: PromptCacheBilling): string {
    return [
      PromptCacheScriptedTransport.messageStart(model, billing),
      sse({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
      sse({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }),
      sse({ type: 'content_block_stop', index: 0 }),
      sse({
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: 2 },
      }),
      sse({ type: 'message_stop' }),
    ].join('');
  }

  private static toolCallAnswer(model: string, toolName: string, billing: PromptCacheBilling): string {
    return [
      PromptCacheScriptedTransport.messageStart(model, billing),
      sse({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'tool_use', id: `toolu_${toolName}`, name: toolName, input: {} },
      }),
      sse({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{}' } }),
      sse({ type: 'content_block_stop', index: 0 }),
      sse({
        type: 'message_delta',
        delta: { stop_reason: 'tool_use', stop_sequence: null },
        usage: { output_tokens: 3 },
      }),
      sse({ type: 'message_stop' }),
    ].join('');
  }
}
