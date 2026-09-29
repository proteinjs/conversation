import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/**
 * THE CONSUMER'S VETO (`ConversationParams.webSearch: 'off'`): a conversation built with it attaches NO
 * provider web-search tool on any request — Anthropic, OpenAI, xAI and Google alike — and names none in
 * toolChoice, even when the request asks for a search (`webSearch: true`).
 *
 * RED before the option existed: the tool-use providers attached the search UNCONDITIONALLY (the request's
 * flag only forced toolChoice; only Google obeyed it), so no consumer seat could withhold it — a skill whose
 * Web door was off still searched on its own turns. The default is unchanged: a conversation built without
 * the option attaches exactly what it always did (the last case).
 */
type Tools = Record<string, unknown>;
type ToolChoice = { type: 'tool'; toolName: string } | undefined;
type Internals = {
  getWebSearchTools(provider: string, modelString: string, webSearchRequested?: boolean): Tools;
  getWebSearchToolChoice(provider: string, tools: Tools, webSearchRequested?: boolean): ToolChoice;
};
const internals = (conversation: Conversation) => conversation as unknown as Internals;

const vetoed = new Conversation({ modelData: fixtureModelData, name: 'test-webSearchVeto', webSearch: 'off' });
const plain = new Conversation({ modelData: fixtureModelData, name: 'test-webSearchDefault' });

/** What the request would carry: the search tools and the toolChoice, as generateStream assembles them. */
const request = (conversation: Conversation, provider: string, model: string, asked?: boolean) => {
  const tools = internals(conversation).getWebSearchTools(provider, model, asked);
  return { tools, toolChoice: internals(conversation).getWebSearchToolChoice(provider, tools, asked) };
};

const TOOL_USE_PROVIDERS: Array<[string, string]> = [
  ['anthropic', 'claude-opus-4-8'],
  ['openai', 'gpt-5.5'],
  ['xai', 'grok-4.3'],
];
const GOOGLE: [string, string] = ['google', 'gemini-3.5-flash'];

describe('the consumer’s veto — a conversation built with webSearch: "off"', () => {
  describe.each([...TOOL_USE_PROVIDERS, GOOGLE])('%s', (provider, model) => {
    test('the request carries no web-search tool and toolChoice names none, asked or not', () => {
      expect(request(vetoed, provider, model)).toEqual({ tools: {}, toolChoice: undefined });
      expect(request(vetoed, provider, model, true)).toEqual({ tools: {}, toolChoice: undefined });
      expect(request(vetoed, provider, model, false)).toEqual({ tools: {}, toolChoice: undefined });
    });
  });

  test('the default is unchanged: without the option the tool-use providers carry the search (forced on the ask), Google on the ask only', () => {
    for (const [provider, model] of TOOL_USE_PROVIDERS) {
      expect(request(plain, provider, model).tools).toHaveProperty('web_search');
      expect(request(plain, provider, model).toolChoice).toBeUndefined();
      expect(request(plain, provider, model, true).toolChoice).toEqual({ type: 'tool', toolName: 'web_search' });
    }
    expect(request(plain, ...GOOGLE).tools).toEqual({});
    expect(request(plain, ...GOOGLE, true).tools).toHaveProperty('google_search');
  });
});
