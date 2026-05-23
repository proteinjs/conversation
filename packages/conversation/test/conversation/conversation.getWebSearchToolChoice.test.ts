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
 */

const conv = new Conversation({ name: 'test-getWebSearchToolChoice' });

type SearchToolChoice = { type: 'tool'; toolName: string } | undefined;

const callGetWebSearchToolChoice = (
  provider: string,
  webSearchTools: Record<string, unknown>,
  webSearchRequested?: boolean
): SearchToolChoice => {
  return (conv as any).getWebSearchToolChoice(provider, webSearchTools, webSearchRequested);
};

describe('Conversation.getWebSearchToolChoice', () => {
  describe('webSearchRequested = true', () => {
    test('forces web_search for OpenAI', () => {
      expect(callGetWebSearchToolChoice('openai', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('forces web_search for Anthropic', () => {
      expect(callGetWebSearchToolChoice('anthropic', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('forces web_search for xAI', () => {
      expect(callGetWebSearchToolChoice('xai', { web_search: {} }, true)).toEqual({
        type: 'tool',
        toolName: 'web_search',
      });
    });

    test('returns undefined for Google (grounding auto-invokes; no model choice involved)', () => {
      // Even though googleSearch is in the toolset, we don't force it via
      // toolChoice — attaching the grounding tool already forces grounding.
      expect(callGetWebSearchToolChoice('google', { google_search: {} }, true)).toBeUndefined();
    });

    test('returns undefined when no search tool is available (e.g. Haiku/nano)', () => {
      // Even with toggle on, if the model class excludes search tools, we
      // can't force what's not there. Falls back to no toolChoice.
      expect(callGetWebSearchToolChoice('anthropic', {}, true)).toBeUndefined();
      expect(callGetWebSearchToolChoice('openai', {}, true)).toBeUndefined();
    });
  });

  describe('webSearchRequested = false', () => {
    test('returns undefined across all providers (toggle off = model decides)', () => {
      expect(callGetWebSearchToolChoice('openai', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('anthropic', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('xai', { web_search: {} }, false)).toBeUndefined();
      expect(callGetWebSearchToolChoice('google', {}, false)).toBeUndefined();
    });
  });

  describe('webSearchRequested = undefined', () => {
    test('returns undefined (default state)', () => {
      expect(callGetWebSearchToolChoice('openai', { web_search: {} })).toBeUndefined();
      expect(callGetWebSearchToolChoice('anthropic', { web_search: {} })).toBeUndefined();
      expect(callGetWebSearchToolChoice('xai', { web_search: {} })).toBeUndefined();
    });
  });
});
