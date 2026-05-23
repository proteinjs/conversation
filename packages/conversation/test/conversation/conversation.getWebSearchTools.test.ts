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

  describe('unknown providers', () => {
    test('returns no tools for unrecognized provider', () => {
      const tools = callGetWebSearchTools('made-up', 'whatever-model');
      expect(tools).toEqual({});
    });
  });
});
