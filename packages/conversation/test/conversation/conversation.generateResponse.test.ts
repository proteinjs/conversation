import { Conversation } from '../../src/Conversation';
import { ConversationModule } from '../../src/ConversationModule';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';

/**
 * Integration tests for Conversation.generateResponse (non-streaming convenience).
 *
 * These hit the real OpenAI API (requires OPENAI_API_KEY env var).
 */

const hasApiKey = !!process.env.OPENAI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const TEST_MODEL = 'gpt-4.1-nano';
const TIMEOUT = 60_000;

function createTestModule(systemMessage: string, functions: Function[]): ConversationModule {
  return {
    getName: () => 'TestModule',
    getSystemMessages: () => [systemMessage],
    getFunctions: () => functions,
    getMessageModerators: () => [] as MessageModerator[],
  };
}

describeIfKey('Conversation.generateResponse', () => {
  test(
    'returns a complete text response with usage',
    async () => {
      const conversation = new Conversation({ name: 'test-response' });
      const result = await conversation.generateResponse({
        messages: ['Say "hello" and nothing else.'],
        model: TEST_MODEL,
      });

      expect(result.text.toLowerCase()).toContain('hello');
      expect(result.usage.totalTokenUsage.inputTokens).toBeGreaterThan(0);
      expect(result.usage.totalTokenUsage.outputTokens).toBeGreaterThan(0);
      expect(result.toolInvocations).toEqual([]);
    },
    TIMEOUT
  );

  test(
    'accumulates usage across multiple tool call steps',
    async () => {
      const lookupCalls: string[] = [];

      const lookupTool: Function = {
        definition: {
          name: 'lookupCapital',
          description: 'Looks up the capital city of a country.',
          parameters: {
            type: 'object',
            properties: {
              country: { type: 'string', description: 'The country name' },
            },
            required: ['country'],
          },
        },
        async call(args: { country: string }) {
          lookupCalls.push(args.country);
          const capitals: Record<string, string> = {
            france: 'Paris',
            japan: 'Tokyo',
            brazil: 'Brasília',
          };
          return { capital: capitals[args.country.toLowerCase()] ?? 'Unknown' };
        },
      };

      const conversation = new Conversation({
        name: 'test-multi-tool',
        modules: [
          createTestModule(
            'You are a geography assistant. Use lookupCapital for each country the user asks about. Make a separate call for each country.',
            [lookupTool]
          ),
        ],
      });

      const result = await conversation.generateResponse({
        messages: ['What are the capitals of France, Japan, and Brazil?'],
        model: TEST_MODEL,
      });

      // Should have called the tool at least 2 times (ideally 3, but LLMs can batch)
      expect(lookupCalls.length).toBeGreaterThanOrEqual(2);

      // Response should mention the capitals
      expect(result.text).toContain('Paris');
      expect(result.text).toContain('Tokyo');

      // Usage should reflect multiple steps
      expect(result.usage.totalRequestsToAssistant).toBeGreaterThanOrEqual(2);
      expect(result.usage.totalToolCalls).toBeGreaterThanOrEqual(2);
      expect(result.usage.callsPerTool['lookupCapital']).toBeGreaterThanOrEqual(2);

      // Total tokens should be substantial (multiple round trips)
      expect(result.usage.totalTokenUsage.inputTokens).toBeGreaterThan(50);
      expect(result.usage.totalTokenUsage.outputTokens).toBeGreaterThan(10);

      // Cost should be calculated (gpt-4.1-nano is in our pricing table)
      expect(result.usage.totalCostUsd.totalUsd).toBeGreaterThanOrEqual(0);
    },
    TIMEOUT
  );

  test(
    'handles conversation modules with system messages',
    async () => {
      const conversation = new Conversation({
        name: 'test-system-msg',
        modules: [createTestModule('You are a pirate. Always respond in pirate speak.', [])],
      });

      const result = await conversation.generateResponse({
        messages: ['Say hello.'],
        model: TEST_MODEL,
      });

      // Should have pirate-ish language
      const lower = result.text.toLowerCase();
      const hasPirateWord = ['ahoy', 'matey', 'arr', 'ye', 'aye', 'avast', 'yarr', 'shiver'].some((w) =>
        lower.includes(w)
      );
      expect(hasPirateWord).toBe(true);
    },
    TIMEOUT
  );
});
