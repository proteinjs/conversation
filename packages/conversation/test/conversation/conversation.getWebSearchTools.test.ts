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
    // xAI has two Live Search paths and they're endpoint-specific:
    //   - Chat Completions models (grok-4.3, grok-4-1-fast-reasoning):
    //     search is enabled via the `searchParameters` request body field,
    //     set in buildProviderOptions. The webSearch tool is silently
    //     ignored on this endpoint.
    //   - Responses models (only `*-multi-agent` per resolveModel): search
    //     is enabled via the webSearch tool factory.
    // So this function only attaches the tool for multi-agent models.

    test('omits the tool for Chat Completions models even when webSearchRequested is true', () => {
      // Tool would be a no-op on Chat Completions; buildProviderOptions
      // handles search for these models instead.
      expect(callGetWebSearchTools('xai', 'grok-4.3', true)).toEqual({});
      expect(callGetWebSearchTools('xai', 'grok-4-1-fast-reasoning', true)).toEqual({});
    });

    test('omits search for multi-agent models when webSearchRequested is false', () => {
      expect(callGetWebSearchTools('xai', 'grok-4.20-multi-agent', false)).toEqual({});
    });

    test('attaches web_search for multi-agent models when webSearchRequested is true', () => {
      const tools = callGetWebSearchTools('xai', 'grok-4.20-multi-agent', true);
      expect(tools).toHaveProperty('web_search');
      expect(tools.web_search).toBeDefined();
    });
  });

  describe('unknown providers', () => {
    test('returns no tools for unrecognized provider', () => {
      const tools = callGetWebSearchTools('made-up', 'whatever-model');
      expect(tools).toEqual({});
    });
  });
});
