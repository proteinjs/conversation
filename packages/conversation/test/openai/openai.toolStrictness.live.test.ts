import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from '../conversation/fixtureModelData';

/**
 * LIVE (real OpenAI API — skipped without OPENAI_API_KEY, like the other live suites): a tool's
 * optional properties come back OMITTED when the request gives no reason to set them.
 *
 * A one-off reminder is asked for; the tool offers an optional `note` and an optional `repeat`
 * object. At the pre-fix library the model's input carried both, filled by the provider's strict
 * default (`note: ""`, `repeat: { every: "day" }` — a one-off reminder became a daily one). The
 * mechanical pin is `conversation.toolStrictness.test.ts`; this suite shows the provider itself.
 *
 *   OPENAI_API_KEY=… npx jest test/openai/openai.toolStrictness.live
 *   TOOL_STRICTNESS_LIVE_MODEL=<model id> to run another OpenAI model (default gpt-5.6-sol).
 */
const hasApiKey = !!process.env.OPENAI_API_KEY;
const testIfKey = hasApiKey ? test : test.skip;
const LIVE_MODEL = process.env.TOOL_STRICTNESS_LIVE_MODEL || 'gpt-5.6-sol';
const LIVE_CALL_TIMEOUT_MS = 120_000;

testIfKey(
  'an OpenAI model omits the optional properties of a tool when nothing calls for them',
  async () => {
    let received: Record<string, unknown> | undefined;
    const setReminder: Function = {
      definition: {
        name: 'setReminder',
        description: 'Set a reminder.',
        parameters: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            note: { type: 'string', description: 'Optional note. Omit when the user gave none.' },
            repeat: {
              type: 'object',
              description: 'Optional. Set ONLY for a repeating reminder.',
              properties: { every: { type: 'string', enum: ['day', 'week', 'month'] } },
              required: ['every'],
              additionalProperties: false,
            },
          },
          required: ['title'],
          additionalProperties: false,
        },
      },
      call: async (input: Record<string, unknown>) => {
        received = input;
        return 'Reminder set.';
      },
    };
    const skill: ConversationSkill = {
      getId: () => 'tool-strictness-live-skill',
      getName: () => 'ToolStrictnessLiveSkill',
      getSystemMessages: () => [],
      getFunctions: () => [setReminder],
      getMessageModerators: () => [] as MessageModerator[],
    };
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'tool-strictness-live',
      logLevel: 'error',
      limits: { enforceLimits: false },
      skills: [skill],
    });
    const result = await conversation.generateStream({
      messages: ['Set a one-off reminder titled "call mom". Nothing else. Use the setReminder tool.'],
      model: LIVE_MODEL,
    });
    for await (const part of result.fullStream) {
      void part;
    }

    expect(received).toBeDefined();
    expect(Object.keys(received ?? {}).sort()).toEqual(['title']);
  },
  LIVE_CALL_TIMEOUT_MS
);
