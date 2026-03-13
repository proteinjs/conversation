import { Conversation } from '../../src/Conversation';
import { ConversationModule } from '../../src/ConversationModule';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';

/**
 * Integration tests for Conversation.generateStream.
 *
 * These hit the real OpenAI API (requires OPENAI_API_KEY env var).
 * They verify the full path through the AI SDK: message building,
 * tool schema wiring, streaming, usage extraction, and tool invocation reporting.
 */

const hasApiKey = !!process.env.OPENAI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const TEST_MODEL = 'gpt-4.1-nano';
const TIMEOUT = 60_000;

/** A simple tool that adds two numbers. */
function createAddTool(): { fn: Function; calls: Array<{ a: number; b: number }> } {
  const calls: Array<{ a: number; b: number }> = [];
  const fn: Function = {
    definition: {
      name: 'addNumbers',
      description: 'Adds two numbers together and returns the sum.',
      parameters: {
        type: 'object',
        properties: {
          a: { type: 'number', description: 'First number' },
          b: { type: 'number', description: 'Second number' },
        },
        required: ['a', 'b'],
      },
    },
    async call(args: { a: number; b: number }) {
      calls.push(args);
      return { sum: args.a + args.b };
    },
  };
  return { fn, calls };
}

/** A tool with no parameters (the case that caused the type: "None" bug). */
function createNoParamTool(): { fn: Function; callCount: number[] } {
  const callCount = [0];
  const fn: Function = {
    definition: {
      name: 'getServerTime',
      description: 'Returns the current server time. Takes no parameters.',
    },
    async call() {
      callCount[0]++;
      return { time: new Date().toISOString() };
    },
  };
  return { fn, callCount };
}

/** A simple module that provides a system message and a tool. */
function createTestModule(systemMessage: string, functions: Function[]): ConversationModule {
  return {
    getName: () => 'TestModule',
    getSystemMessages: () => [systemMessage],
    getFunctions: () => functions,
    getMessageModerators: () => [] as MessageModerator[],
  };
}

describeIfKey('Conversation.generateStream', () => {
  test(
    'streams a text response and resolves usage data',
    async () => {
      const conversation = new Conversation({ name: 'test-stream' });

      const result = await conversation.generateStream({
        messages: ['What is 2+2? Reply with just the number.'],
        model: TEST_MODEL,
      });

      // Collect streamed text
      const chunks: string[] = [];
      for await (const chunk of result.textStream) {
        chunks.push(chunk);
      }

      // Should have streamed something
      expect(chunks.length).toBeGreaterThan(0);

      // Full text should resolve and contain "4"
      const text = await result.text;
      expect(text).toContain('4');

      // Usage data should be populated with non-zero tokens
      const usage = await result.usage;
      expect(usage.totalTokenUsage.inputTokens).toBeGreaterThan(0);
      expect(usage.totalTokenUsage.outputTokens).toBeGreaterThan(0);
      expect(usage.totalTokenUsage.totalTokens).toBeGreaterThan(0);
      expect(usage.model).toBeTruthy();
    },
    TIMEOUT
  );

  test(
    'invokes a tool with parameters and reports tool invocations',
    async () => {
      const { fn: addTool, calls } = createAddTool();

      const conversation = new Conversation({
        name: 'test-tool-call',
        modules: [createTestModule('You are a calculator. Use the addNumbers tool to compute sums.', [addTool])],
      });

      const result = await conversation.generateStream({
        messages: ['What is 7 + 13?'],
        model: TEST_MODEL,
      });

      // Consume the stream
      const text = await result.text;
      const usage = await result.usage;
      const toolInvocations = await result.toolInvocations;

      // The tool should have been called
      expect(calls.length).toBeGreaterThan(0);
      expect(calls[0].a + calls[0].b).toBe(20);

      // The response should contain "20"
      expect(text).toContain('20');

      // Tool invocations should be reported
      expect(toolInvocations.length).toBeGreaterThan(0);
      expect(toolInvocations[0].name).toBe('addNumbers');

      // Usage should reflect multiple steps (at least one tool call step + final response)
      expect(usage.totalRequestsToAssistant).toBeGreaterThanOrEqual(2);
      expect(usage.totalToolCalls).toBeGreaterThan(0);
      expect(usage.callsPerTool['addNumbers']).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'handles a tool with no parameters (the type: "None" regression)',
    async () => {
      const { fn: noParamTool, callCount } = createNoParamTool();

      const conversation = new Conversation({
        name: 'test-no-param-tool',
        modules: [
          createTestModule('You have access to a getServerTime tool. When the user asks for the time, call it.', [
            noParamTool,
          ]),
        ],
      });

      const result = await conversation.generateStream({
        messages: ['What time is it on the server?'],
        model: TEST_MODEL,
      });

      // This should not throw — the old bug caused an API error here
      const text = await result.text;
      const toolInvocations = await result.toolInvocations;

      // The tool should have been called
      expect(callCount[0]).toBeGreaterThan(0);
      expect(toolInvocations.length).toBeGreaterThan(0);
      expect(toolInvocations[0].name).toBe('getServerTime');
    },
    TIMEOUT
  );
});
