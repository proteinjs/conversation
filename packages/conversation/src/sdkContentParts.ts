/**
 * Adapters from OpenAI-shape content parts (`ChatCompletionContentPart[]` — what
 * `@n3xah/chat-server` produces for UI uploads and what `getFiles`-style tool
 * factories return) to the Vercel AI SDK shapes used on the two seams where
 * content-part data must reach the model intact:
 *
 *   - User / assistant message content → `UserContent` shape
 *     (`Array<TextPart | ImagePart | FilePart>`).  Used by
 *     `Conversation.buildAiSdkMessages` when a message has array content.
 *
 *   - Tool result content → `ToolResultOutput.type: 'content'`'s `value` shape
 *     (`Array<{type: 'text'|'image-data'|'image-url'|...}>`).  Used by
 *     `Conversation.buildAiSdkTools` when a tool returns a
 *     `ChatCompletionMessageParamFactory` (e.g. `getFiles`) or a raw array of
 *     content parts.
 *
 * The two output shapes differ — messages use `{type:'image', image, mediaType}`
 * while tool-result content uses `{type:'image-data', data, mediaType}` — so the
 * helpers are siblings rather than one function with a mode switch.
 *
 * Without these adapters, images get stripped before reaching the provider
 * adapters (the legacy path collapsed `ChatCompletionContentPart[]` to a plain
 * string, and the default tool-result path JSON-stringified factory objects to
 * useless metadata blobs).  Providers like Anthropic then emit "I don't see an
 * image attached" even though the file is present a few layers upstream.
 */

import type { TextPart, ImagePart, FilePart } from '@ai-sdk/provider-utils';
import type { ChatCompletionContentPart, ChatCompletionMessageParam } from 'openai/resources/chat';
import { ChatCompletionMessageParamFactory } from './ChatCompletionMessageParamFactory';

/**
 * One entry in a `ToolResultOutput`'s `type: 'content'` value array. We support
 * the subset actually produced today: text, inlined image bytes (data URI), and
 * external image URL. Extend here when new inbound shapes appear.
 */
export type SdkToolResultContentPart =
  | { type: 'text'; text: string }
  | { type: 'image-data'; data: string; mediaType: string }
  | { type: 'image-url'; url: string };

type Target = 'user' | 'tool-result';

/**
 * Stateless adapters between OpenAI-shape `ChatCompletionContentPart[]` and the
 * Vercel AI SDK's two distinct multimodal shapes (user-message parts vs.
 * tool-result content parts). All methods are static by design — there is no
 * per-conversation state — but they live on a class per this codebase's
 * "classes over modules of functions" convention.
 */
export class SdkContentParts {
  // ────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────

  /**
   * Convert OpenAI-shape content parts into Vercel AI SDK user-message parts.
   * Drops parts it can't interpret. Empty input → empty output (the caller
   * decides whether to coerce back to a string for empty messages).
   */
  static toUserContentParts(parts: ReadonlyArray<ChatCompletionContentPart>): Array<TextPart | ImagePart | FilePart> {
    const out: Array<TextPart | ImagePart | FilePart> = [];
    for (const part of parts) {
      const mapped = SdkContentParts.mapPart(part, 'user');
      if (mapped) {
        out.push(mapped as unknown as TextPart | ImagePart | FilePart);
      }
    }
    return out;
  }

  /**
   * Convert OpenAI-shape content parts into the value array for a
   * `ToolResultOutput.type: 'content'` payload.
   */
  static toToolResultContentParts(parts: ReadonlyArray<ChatCompletionContentPart>): SdkToolResultContentPart[] {
    const out: SdkToolResultContentPart[] = [];
    for (const part of parts) {
      const mapped = SdkContentParts.mapPart(part, 'tool-result');
      if (mapped) {
        out.push(mapped as SdkToolResultContentPart);
      }
    }
    return out;
  }

  /**
   * Runtime shape-detection for tool return values: recognize both the legacy
   * `ChatCompletionMessageParamFactory` (async `create()` → messages with
   * content parts) and bare `ChatCompletionContentPart[]` arrays. Returns the
   * extracted content-parts array, or `undefined` when the tool returned
   * something that shouldn't be treated as structured content (primitives,
   * plain JSON objects, etc. — those flow through unchanged to preserve
   * backward-compat for tools that return strings/objects today).
   */
  static async extractContentPartsFromToolReturn(result: unknown): Promise<ChatCompletionContentPart[] | undefined> {
    if (result == null) {
      return undefined;
    }

    // Direct array-of-parts: `[{type:'text',...}, {type:'image_url',...}]`.
    if (Array.isArray(result) && result.length > 0 && SdkContentParts.looksLikeContentPart(result[0])) {
      return result as ChatCompletionContentPart[];
    }

    // Legacy factory shape — includes `getFiles`'s FileContentPartFactory.
    if (result instanceof ChatCompletionMessageParamFactory) {
      const messages = await result.create();
      return SdkContentParts.flattenMessageContentParts(messages);
    }

    // Some factories inherit via structural typing (`extends` across package
    // boundaries) and instanceof misses them. Duck-type fallback: async
    // `create()` that yields message-like objects.
    if (typeof (result as { create?: unknown }).create === 'function') {
      try {
        const produced = await (result as { create: () => unknown | Promise<unknown> }).create();
        if (Array.isArray(produced)) {
          // Could be an array of messages OR an array of parts.
          if (produced.length > 0 && SdkContentParts.looksLikeContentPart(produced[0])) {
            return produced as ChatCompletionContentPart[];
          }
          return SdkContentParts.flattenMessageContentParts(produced as ChatCompletionMessageParam[]);
        }
      } catch {
        // Not a factory after all — fall through to `undefined`.
      }
    }

    return undefined;
  }

  // ────────────────────────────────────────────────────────────
  // Private helpers
  // ────────────────────────────────────────────────────────────

  private static mapPart(part: unknown, target: Target): Record<string, unknown> | undefined {
    if (typeof part !== 'object' || part == null) {
      return undefined;
    }
    const p = part as Record<string, unknown>;

    if (p.type === 'text' && typeof p.text === 'string') {
      return { type: 'text', text: p.text };
    }

    // OpenAI's `image_url` part:
    //   { type: 'image_url', image_url: { url, detail? } }
    // `url` is either a data URI (`data:image/png;base64,...`) or a plain https URL.
    if (p.type === 'image_url') {
      const imageUrlField = p.image_url as { url?: unknown } | undefined;
      const url = typeof imageUrlField?.url === 'string' ? imageUrlField.url : undefined;
      if (!url) {
        return undefined;
      }
      const parsed = SdkContentParts.parseDataUri(url);
      if (parsed) {
        return target === 'user'
          ? { type: 'image', image: parsed.data, mediaType: parsed.mediaType }
          : { type: 'image-data', data: parsed.data, mediaType: parsed.mediaType };
      }
      return target === 'user' ? { type: 'image', image: url } : { type: 'image-url', url };
    }

    return undefined;
  }

  private static parseDataUri(url: string): { mediaType: string; data: string } | undefined {
    if (!url.startsWith('data:')) {
      return undefined;
    }
    const comma = url.indexOf(',');
    if (comma < 0) {
      return undefined;
    }
    // head: `image/png;base64` or `image/png`
    const head = url.slice(5, comma);
    const data = url.slice(comma + 1);
    const mediaType = head.split(';')[0] || 'application/octet-stream';
    return { mediaType, data };
  }

  private static looksLikeContentPart(p: unknown): boolean {
    if (typeof p !== 'object' || p == null) {
      return false;
    }
    const t = (p as { type?: unknown }).type;
    return t === 'text' || t === 'image_url' || t === 'image' || t === 'file' || t === 'input_audio';
  }

  private static flattenMessageContentParts(
    messages: ChatCompletionMessageParam[]
  ): ChatCompletionContentPart[] | undefined {
    const out: ChatCompletionContentPart[] = [];
    for (const msg of messages) {
      if (Array.isArray(msg?.content)) {
        for (const part of msg.content as ChatCompletionContentPart[]) {
          out.push(part);
        }
      } else if (typeof msg?.content === 'string' && msg.content.length > 0) {
        out.push({ type: 'text', text: msg.content });
      }
    }
    return out.length > 0 ? out : undefined;
  }
}
