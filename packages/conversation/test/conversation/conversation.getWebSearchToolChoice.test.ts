import { Conversation } from '../../src/Conversation';

/**
 * Unit tests for the toolChoice helper used by the webSearch toggle.
 *
 * Contract: when the user toggles search on, the toggle's meaning is
 * "guarantee a search this turn." For model-called search tools
 * (OpenAI / Anthropic / xAI) this is delivered by forcing toolChoice
 * to the search tool on step 1. For grounding-based search (Google),
 * attaching the tool already forces grounding, so toolChoice is
 * irrelevant and we return undefined.
 *
 * When the toggle is off, the model decides; we never force.
 *
 * Model exception: claude-fable-5-1 REJECTS forced tool_choice — the API 400s
 * ('tool_choice: type "tool" and "any" are not supported for this model',
 * observed live 2026-09-01; Fable 5 accepted it). For that model the search
 * tool still attaches and the toggle softens to "search strongly available" —
 * forcing would kill the whole turn.
 */

const conv = new Conversation({ name: 'test-getWebSearchToolChoice' });

type SearchToolChoice = { type: 'tool'; toolName: string } | undefined;

const callGetWebSearchToolChoice = (
  provider: string,
  modelString: string,
  webSearchTools: Record<string, unknown>,
  webSearchRequested?: boolean
): SearchToolChoice => {
  return (conv as any).getWebSearchToolChoice(provider, modelString, webSearchTools, webSearchRequested);
};

describe('Conversation.getWebSearchToolChoice', () => {
  describe('webSearchRequested = true', () => {
    test('forces web_search for OpenAI', () => {
      expect(callGetWebSearchToolChoice('openai', 'gpt-5.6-sol', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('forces web_search for Anthropic', () => {
      expect(callGetWebSearchToolChoice('anthropic', 'claude-opus-5', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
      // Fable 5 accepts forcing (verified live) — the 5.1 exception must not widen.
      expect(callGetWebSearchToolChoice('anthropic', 'claude-fable-5', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('forces web_search for xAI', () => {
      expect(callGetWebSearchToolChoice('xai', 'grok-4.5', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('returns undefined for claude-fable-5-1 — the model 400s on forced tool_choice', () => {
      // The tool is attached; only the FORCING is dropped. A `provider:` prefix on the
      // model string must not defeat the gate (the id is matched after any prefix).
      expect(callGetWebSearchToolChoice('anthropic', 'claude-fable-5-1', { web_search: {} }, true)).toBeUndefined();
      expect(
        callGetWebSearchToolChoice('anthropic', 'anthropic:claude-fable-5-1', { web_search: {} }, true)
      ).toBeUndefined();
    });

    test('returns undefined for Google (grounding auto-invokes; no model choice involved)', () => {
      // Even though googleSearch is in the toolset, we don't force it via
      // toolChoice — attaching the grounding tool already forces grounding.
      expect(
        callGetWebSearchToolChoice('google', 'gemini-3.1-pro-preview', { google_search: {} }, true)
      ).toBeUndefined();
    });

    test('returns undefined when no search tool is available (e.g. Haiku/nano)', () => {
      // Even with toggle on, if the model class excludes search tools, we
      // can't force what's not there. Falls back to no toolChoice.
      expect(callGetWebSearchToolChoice('anthropic', 'claude-haiku-4-5', {}, true)).toBeUndefined();
      expect(callGetWebSearchToolChoice('openai', 'gpt-5-nano', {}, true)).toBeUndefined();
    });
  });

  describe('webSearchRequested = false', () => {
    test('returns undefined across all providers (toggle off = model decides)', () => {
      expect(callGetWebSearchToolChoice('openai', 'gpt-5.6-sol', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('anthropic', 'claude-opus-5', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('xai', 'grok-4.5', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('google', 'gemini-3.1-pro-preview', {}, false)).toBeUndefined();
    });
  });

  describe('webSearchRequested = undefined', () => {
    test('returns undefined (default state)', () => {
      expect(callGetWebSearchToolChoice('openai', 'gpt-5.6-sol', { web_search: {} })).toBeUndefined();
      expect(callGetWebSearchToolChoice('anthropic', 'claude-opus-5', { web_search: {} })).toBeUndefined();
      expect(callGetWebSearchToolChoice('xai', 'grok-4.5', { web_search: {} })).toBeUndefined();
    });
  });
});
