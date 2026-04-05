import type { LanguageModel } from 'ai';

/**
 * Known provider prefixes and their model factory functions.
 *
 * Each factory lazily imports the provider package and creates a model instance.
 * This keeps imports optional: if a provider package is not installed, the
 * factory simply throws a helpful error at call time.
 */
const PROVIDER_FACTORIES: Record<string, (modelId: string) => LanguageModel> = {
  openai: (modelId) => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { openai } = require('@ai-sdk/openai');
    return openai(modelId);
  },
  anthropic: (modelId) => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { anthropic } = require('@ai-sdk/anthropic');
    return anthropic(modelId);
  },
  google: (modelId) => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { google } = require('@ai-sdk/google');
    return google(modelId);
  },
  xai: (modelId) => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { xai } = require('@ai-sdk/xai');
    // Multi-agent models require the Responses API, not Chat Completions.
    if (/multi-agent/i.test(modelId)) {
      return xai.responses(modelId);
    }
    return xai(modelId);
  },
};

/**
 * Model name patterns that map to a provider.
 *
 * Order matters: first match wins. Patterns are tested against the raw model
 * string (case-insensitive).
 */
const MODEL_PROVIDER_PATTERNS: Array<{ test: RegExp; provider: string }> = [
  // OpenAI models
  { test: /^(gpt-|o[134]-|o[134]$|chatgpt|dall-e|computer-use|codex)/i, provider: 'openai' },

  // Anthropic models
  { test: /^claude/i, provider: 'anthropic' },

  // Google models
  { test: /^(gemini|gemma)/i, provider: 'google' },

  // xAI models
  { test: /^grok/i, provider: 'xai' },
];

/**
 * Resolve a model identifier to a concrete `LanguageModel` instance.
 *
 * Accepted inputs:
 * - A `LanguageModel` instance (returned as-is)
 * - A prefixed string like `"openai:gpt-5"` or `"anthropic:claude-sonnet-4-20250514"`
 * - A bare model name like `"gpt-5"`, `"claude-sonnet-4-20250514"`, `"gemini-2.5-pro"`, `"grok-3"`
 *   (provider inferred from name patterns)
 */
export function resolveModel(model: LanguageModel | string): LanguageModel {
  // Already a model instance
  if (typeof model !== 'string') {
    return model;
  }

  const raw = model.trim();
  if (!raw) {
    throw new Error('resolveModel: empty model string');
  }

  // Explicit provider prefix: "provider:model-id"
  const colonIdx = raw.indexOf(':');
  if (colonIdx > 0) {
    const prefix = raw.slice(0, colonIdx).toLowerCase();
    const modelId = raw.slice(colonIdx + 1);
    const factory = PROVIDER_FACTORIES[prefix];
    if (!factory) {
      throw new Error(
        `resolveModel: unknown provider prefix "${prefix}" in "${raw}". ` +
          `Known providers: ${Object.keys(PROVIDER_FACTORIES).join(', ')}`
      );
    }
    return factory(modelId);
  }

  // Infer provider from model name patterns
  for (const { test, provider } of MODEL_PROVIDER_PATTERNS) {
    if (test.test(raw)) {
      return PROVIDER_FACTORIES[provider](raw);
    }
  }

  // Default to OpenAI for unrecognized model names
  // (OpenAI has the most model aliases and is the most common provider)
  return PROVIDER_FACTORIES.openai(raw);
}

/**
 * Extract the provider name from a model identifier string.
 * Returns 'openai', 'anthropic', 'google', 'xai', or 'unknown'.
 */
export function inferProvider(model: LanguageModel | string): string {
  if (typeof model !== 'string') {
    // Try to extract from the model's provider property if available
    const modelId = (model as any).modelId ?? '';
    return inferProvider(modelId);
  }

  const raw = model.trim();

  // Explicit prefix
  const colonIdx = raw.indexOf(':');
  if (colonIdx > 0) {
    const prefix = raw.slice(0, colonIdx).toLowerCase();
    if (PROVIDER_FACTORIES[prefix]) {
      return prefix;
    }
  }

  // Pattern match
  for (const { test, provider } of MODEL_PROVIDER_PATTERNS) {
    if (test.test(raw)) {
      return provider;
    }
  }

  return 'unknown';
}
