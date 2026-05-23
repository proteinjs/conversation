import { Conversation } from '../../src/Conversation';

/**
 * Unit tests for the web-search tool dispatcher.
 *
 * Pure mapping from (provider, modelString, webSearchRequested) to the
 * tool set passed to streamText. No API calls.
 *
 * Provider-specific notes:
 * - OpenAI / Anthropic: tool-use search. Always attached (when the model
 *   supports tool calling); model decides when to call.
 * - Google: grounding-based — attaching the tool forces grounding on
 *   every turn. Only attach when the user explicitly requests search.
 * - Haiku 4.5 + GPT nano models don't support tool-call search reliably,
 *   so search tools are dropped for them.
 */

const conv = new Conversation({ name: 'test-getWebSearchTools' });

const callGetWebSearchTools = (provider: string, modelString: string, webSearchRequested?: boolean) => {
  // Private method; cast keeps prod surface narrow.
  return (conv as any).getWebSearchTools(provider, modelString, webSearchRequested) as Record<string, unknown>;
};

describe('Conversation.getWebSearchTools', () => {
  describe('openai', () => {
    test('attaches web_search for tool-capable models', () => {
      const tools = callGetWebSearchTools('openai', 'gpt-5.5');
      expect(tools).toHaveProperty('web_search');
      expect(tools.web_search).toBeDefined();
    });

    test('drops web_search for nano-class models', () => {
      const tools = callGetWebSearchTools('openai', 'gpt-5.4-nano');
      expect(tools).toEqual({});
    });
  });

  describe('anthropic', () => {
    test('attaches web_search for Opus and Sonnet', () => {
      expect(callGetWebSearchTools('anthropic', 'claude-opus-4-7')).toHaveProperty('web_search');
      expect(callGetWebSearchTools('anthropic', 'claude-sonnet-4-6')).toHaveProperty('web_search');
    });

    test('drops web_search for Haiku', () => {
      const tools = callGetWebSearchTools('anthropic', 'claude-haiku-4-5');
      expect(tools).toEqual({});
    });
  });

  describe('google', () => {
    // Unlike OpenAI/Anthropic tool-call search, Google's googleSearch is
    // grounding-based — attaching it forces grounding on every response in
    // the turn. So we only attach when the user explicitly toggles search on
    // (Gated on params.webSearch via the third arg).

    test('omits search when webSearchRequested is false', () => {
      expect(callGetWebSearchTools('google', 'gemini-3.5-flash', false)).toEqual({});
    });

    test('omits search when webSearchRequested is undefined', () => {
      expect(callGetWebSearchTools('google', 'gemini-3.5-flash')).toEqual({});
    });

    test('attaches google_search when webSearchRequested is true', () => {
      const tools = callGetWebSearchTools('google', 'gemini-3.5-flash', true);
      expect(tools).toHaveProperty('google_search');
      expect(tools.google_search).toBeDefined();
    });

    test('also attaches for Gemini Pro', () => {
      const tools = callGetWebSearchTools('google', 'gemini-3.1-pro-preview', true);
      expect(tools).toHaveProperty('google_search');
    });
  });

  describe('xai', () => {
    // xAI Live Search behaves like Google's grounding — attaching the tool
    // changes how the model produces its answer, not whether it calls a
    // separate "search this" tool. Gate on webSearchRequested for parity.

    test('omits search when webSearchRequested is false', () => {
      expect(callGetWebSearchTools('xai', 'grok-4.3', false)).toEqual({});
    });

    test('omits search when webSearchRequested is undefined', () => {
      expect(callGetWebSearchTools('xai', 'grok-4.3')).toEqual({});
    });

    test('attaches web_search for Grok 4.3 when webSearchRequested is true', () => {
      const tools = callGetWebSearchTools('xai', 'grok-4.3', true);
      expect(tools).toHaveProperty('web_search');
      expect(tools.web_search).toBeDefined();
    });

    test('also attaches for Grok 4.1 Fast', () => {
      const tools = callGetWebSearchTools('xai', 'grok-4-1-fast-reasoning', true);
      expect(tools).toHaveProperty('web_search');
    });
  });

  describe('unknown providers', () => {
    test('returns no tools for unrecognized provider', () => {
      const tools = callGetWebSearchTools('made-up', 'whatever-model');
      expect(tools).toEqual({});
    });
  });
});
