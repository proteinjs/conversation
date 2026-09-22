import type { ReasoningEffort } from './Conversation';

/** The effort levels OpenAI's Responses API accepts (`reasoning.effort`). */
export type OpenAiReasoningEffort = 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/**
 * THE PER-MODEL RULES for OpenAI, read from the vendor's own versioning grammar — never from a
 * list of ids. `@ai-sdk/openai` decides "is this a reasoning model" from a prefix list it grows
 * by hand (`gpt-5`, `o1`, `o3`, `o4-mini`), so on 2026-09-22 it read `gpt-6-sol` as a
 * NON-reasoning model: it dropped `reasoning.effort` and the reasoning summary from every
 * request with a warning, and each call "passed" with the effort never sent. A rule that reads
 * the GENERATION (`gpt-6-…` → 6) does not trail the next release.
 */
export class OpenAiModelRules {
  /** The GPT generation of a model id (`gpt-6-sol` → 6, `gpt-5.6-terra` → 5, `gpt-4o` → 4); undefined for the o-series and non-GPT ids. */
  static generation(modelId: string): number | undefined {
    const match = /^gpt-(\d+)/i.exec(OpenAiModelRules.bare(modelId));
    return match ? Number(match[1]) : undefined;
  }

  /**
   * Whether the model reasons — the o-series, and every GPT from the 5th generation on except the
   * `-chat` variants (OpenAI's non-reasoning chat builds). What the SDK's `forceReasoning` is told,
   * so the reasoning parameters reach the request whether or not the SDK's list knows the id.
   */
  static reasons(modelId: string): boolean {
    const bare = OpenAiModelRules.bare(modelId).toLowerCase();
    if (/^o[134](-|$)/.test(bare)) {
      return true;
    }
    const generation = OpenAiModelRules.generation(bare);
    return generation !== undefined && generation >= 5 && !/^gpt-\d+(\.\d+)?-chat/.test(bare);
  }

  /**
   * Our effort as the provider names it for THIS model. `auto` omits the parameter (the model's
   * default); `max` is a level of its own from the 6th generation on, and the highest level the
   * earlier generations have — `xhigh` — before that.
   */
  static reasoningEffort(modelId: string, effort?: ReasoningEffort): OpenAiReasoningEffort | undefined {
    if (!effort || effort === 'auto') {
      return undefined;
    }
    if (effort === 'max') {
      const generation = OpenAiModelRules.generation(modelId);
      return generation !== undefined && generation >= 6 ? 'max' : 'xhigh';
    }
    return effort;
  }

  /** The id after any `provider:` prefix. */
  private static bare(modelId: string): string {
    const raw = String(modelId ?? '').trim();
    const colon = raw.indexOf(':');
    return colon > 0 ? raw.slice(colon + 1) : raw;
  }
}
