import type {
  LanguageModelV3,
  LanguageModelV3Middleware,
  LanguageModelV3StreamPart,
  SharedV3ProviderMetadata,
} from '@ai-sdk/provider';
import { wrapLanguageModel } from 'ai';

/**
 * THE METADATA A PROVIDER STREAMS ON AN EMPTY TEXT DELTA RIDES TO THE BLOCK'S END.
 *
 * A Gemini thinking model returns its thought signature on the final part of a text answer, and a
 * streamed answer may carry that part with empty text ("When streaming, this signature might be
 * returned in a part with empty text content, so be sure to parse all parts until finish_reason is
 * returned by the model" — Google's thought-signatures page). The provider package forwards it as
 * a `text-delta` whose delta is '' with the signature in `providerMetadata`; the SDK core discards
 * every empty text delta before it builds the step's content — metadata included — and the
 * provider's own `text-end` carries none. So the signature never reaches the step's text part, and
 * a request that re-sends that part (this library's own continuation rounds: the output-limit
 * continuation, the mid-turn note's cut) goes back without it. The API does not reject a text part
 * without its signature; it recommends the signature "for best performance" — the model's
 * reasoning continuity. A `functionCall` part's signature rides on the part itself and is never
 * touched by this.
 *
 * This middleware sits on every resolved model, under the transport-retry wrapper. Per text block
 * it holds the metadata of an empty delta and puts it on the block's `text-end` when that part
 * carries none — the core keeps a `text-end`'s metadata on the part. A non-empty delta is never
 * touched, a block whose end already carries metadata keeps its own, and nothing is held across
 * blocks or requests. The fix at the origin is the core keeping an empty delta's metadata, or the
 * provider carrying the signature on `text-end`; this stands until either lands.
 */
export class EmptyDeltaMetadata {
  /** Wrap a resolved model so the metadata of its empty text deltas reaches the step's text parts. */
  static follow(model: LanguageModelV3): LanguageModelV3 {
    return wrapLanguageModel({ model, middleware: EmptyDeltaMetadata.middleware() });
  }

  /** The stream with every empty text delta's metadata carried onto its block's end. Exported for tests. */
  static carried(): TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart> {
    const held = new Map<string, SharedV3ProviderMetadata>();
    return new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
      transform(part, controller) {
        if (part.type === 'text-delta' && part.delta.length === 0 && part.providerMetadata) {
          held.set(part.id, EmptyDeltaMetadata.merged(held.get(part.id), part.providerMetadata));
        } else if (part.type === 'text-end') {
          const metadata = held.get(part.id);
          held.delete(part.id);
          if (metadata && !part.providerMetadata) {
            controller.enqueue({ ...part, providerMetadata: metadata });
            return;
          }
        }
        controller.enqueue(part);
      },
    });
  }

  private static middleware(): LanguageModelV3Middleware {
    return {
      specificationVersion: 'v3',
      wrapStream: async ({ doStream }) => {
        const result = await doStream();
        return { ...result, stream: result.stream.pipeThrough(EmptyDeltaMetadata.carried()) };
      },
    };
  }

  /** Two empty deltas of one block: the later one's keys win within each provider's object. */
  private static merged(
    earlier: SharedV3ProviderMetadata | undefined,
    later: SharedV3ProviderMetadata
  ): SharedV3ProviderMetadata {
    if (!earlier) {
      return later;
    }
    const out: SharedV3ProviderMetadata = { ...earlier };
    for (const [provider, fields] of Object.entries(later)) {
      out[provider] = { ...(earlier[provider] ?? {}), ...fields };
    }
    return out;
  }
}
