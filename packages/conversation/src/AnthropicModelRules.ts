/** The thinking configuration a Claude model takes for OFF (`providerOptions.anthropic.thinking`). */
export type AnthropicThinkingOff = { type: 'disabled' } | { type: 'between_tools' };

/** The tier word of a Claude model id. */
export type AnthropicTier = 'opus' | 'sonnet' | 'haiku' | 'fable' | 'mythos';

/**
 * THE PER-MODEL RULES FOR ANTHROPIC, read from the vendor's own versioning grammar — the tier word
 * and the generation of the id (`claude-sonnet-5-5` → sonnet 5.5, `claude-opus-4-8` → opus 4.8,
 * `claude-haiku-4-5-20251001` → haiku 4.5, `claude-3-5-haiku-20241022` → haiku 3.5) — never a list
 * of ids, so a rule does not trail the next release (the premise of `OpenAiModelRules`).
 *
 * THE OFF SWITCH. A request with no `thinking` block is NOT a request without thinking from the
 * fifth generation on: Sonnet 5, Opus 5, Haiku 5.5, Sonnet 5.5, Opus 5.5 and Fable run ADAPTIVE
 * thinking by default — the model decides, and does think (a consumer measured Sonnet 5 reasoning
 * on 19 of 50 short questions sent with no thinking block, 80–427 reasoning tokens a row,
 * 2026-11-04). Only the fourth generation and earlier (Opus 4.8/4.7/4.6, Sonnet 4.x, Haiku 4.5)
 * read a missing block as off. So effort 'none' is the EXPLICIT off, in the vendor's word for the
 * family (the vendor's model documentation and `@ai-sdk/anthropic`'s own model catalog, read
 * 2026-11-04):
 * - `{ type: 'disabled' }` — Sonnet through 5, Opus through 5 (accepted at effort high or below),
 *   Haiku in every generation (5.5: at effort high or below), and the fourth generation and earlier;
 * - `{ type: 'between_tools' }` — Sonnet from 5.5 on: `disabled` is refused there (HTTP 400) and
 *   `between_tools` is its lowest thinking setting (no upfront thinking; accepted at effort high
 *   or below, no other field beside it);
 * - no off switch at all — Fable and Mythos in every generation, and Opus from 5.5 on: thinking is
 *   always on, and `disabled` and `between_tools` alike are refused (HTTP 400). 'none' is not a
 *   level those models offer; the least they take is effort 'low', and the caller names it —
 *   exactly as on Gemini, which has no off switch either.
 */
export class AnthropicModelRules {
  /** `claude-<tier>-<major>[-<minor>][-<date>]` — the grammar from the fourth generation on. */
  private static readonly TIER_FIRST = /^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d)(?!\d))?/;
  /** `claude-<major>[-<minor>]-<tier>[-<date>]` — the third generation's grammar. */
  private static readonly GENERATION_FIRST = /^claude-(\d+)(?:-(\d)(?!\d))?-(opus|sonnet|haiku)/;
  /** `claude-2.1`, `claude-instant-1.2` — the earliest ids, no tier word. */
  private static readonly GENERATION_ONLY = /^claude-(?:instant-)?(\d+)(?:[.-](\d)(?!\d))?/;

  /**
   * The explicit OFF this model takes for effort 'none' — `disabled`, or `between_tools` on Sonnet
   * from 5.5 on — or undefined when the model has no off switch at all (Fable, Mythos, Opus from
   * 5.5 on): then 'none' is not a level it offers, and its least is effort 'low'.
   */
  static thinkingOff(modelId: string): AnthropicThinkingOff | undefined {
    const tier = AnthropicModelRules.tier(modelId);
    const generation = AnthropicModelRules.generation(modelId) ?? 0;
    if (tier === 'fable' || tier === 'mythos') {
      return undefined;
    }
    if (tier === 'opus' && generation >= 5.5) {
      return undefined;
    }
    if (tier === 'sonnet' && generation >= 5.5) {
      return { type: 'between_tools' };
    }
    return { type: 'disabled' };
  }

  /** The tier word of a model id, in either of the vendor's grammars; undefined when the id names none. */
  static tier(modelId: string): AnthropicTier | undefined {
    return AnthropicModelRules.parse(modelId)?.tier;
  }

  /**
   * The generation of a model id: `claude-sonnet-5-5` → 5.5, `claude-opus-4-8` → 4.8,
   * `claude-sonnet-4-20250514` → 4 (a date is never a minor), `claude-3-5-haiku-20241022` → 3.5;
   * undefined for an id that is not a Claude model's.
   */
  static generation(modelId: string): number | undefined {
    return AnthropicModelRules.parse(modelId)?.generation;
  }

  private static parse(modelId: string): { tier?: AnthropicTier; generation?: number } | undefined {
    const bare = AnthropicModelRules.bare(modelId);
    const tierFirst = AnthropicModelRules.TIER_FIRST.exec(bare);
    if (tierFirst) {
      return {
        tier: tierFirst[1] as AnthropicTier,
        generation: AnthropicModelRules.number(tierFirst[2], tierFirst[3]),
      };
    }
    const generationFirst = AnthropicModelRules.GENERATION_FIRST.exec(bare);
    if (generationFirst) {
      return {
        tier: generationFirst[3] as AnthropicTier,
        generation: AnthropicModelRules.number(generationFirst[1], generationFirst[2]),
      };
    }
    const generationOnly = AnthropicModelRules.GENERATION_ONLY.exec(bare);
    if (generationOnly) {
      return { generation: AnthropicModelRules.number(generationOnly[1], generationOnly[2]) };
    }
    return undefined;
  }

  private static number(major: string, minor: string | undefined): number {
    return minor === undefined ? Number(major) : Number(`${major}.${minor}`);
  }

  /** The id after any `provider:` prefix, lower-cased. */
  private static bare(modelId: string): string {
    const raw = String(modelId ?? '')
      .trim()
      .toLowerCase();
    const colon = raw.indexOf(':');
    return colon > 0 ? raw.slice(colon + 1) : raw;
  }
}
