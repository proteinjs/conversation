import { OpenAiResponses } from '../../src/OpenAiResponses';
import type { ConversationSkill } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import type { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from '../conversation/fixtureModelData';

/**
 * Every Responses request the buffered adapter makes is STATELESS: `store: false` on the body,
 * so OpenAI retains no response past the request (the create reference's `store`: "Defaults to
 * true when omitted. If set to true, response data will be stored for at least 30 days").
 *
 * A stateless request has no server-side response to chain to, so the tool loop carries its own
 * transcript: each step's output items — the reasoning items with their `encrypted_content`, the
 * function calls — are replayed verbatim in the next request's input ahead of the function
 * outputs (the reasoning guide, "Preserve reasoning without stored responses", and its
 * function-calling passage: "pass back all reasoning items, function call items, and function
 * call output items, since the last `user` message").
 *
 * RED at the pre-fix adapter: the plain request carries no `store` at all (the provider's
 * default, true, applies), the background request says `store: true`, and the loop's second
 * request chains by `previous_response_id` with an input of the function outputs alone — no
 * reasoning item, no function call.
 */

const USAGE = {
  input_tokens: 10,
  input_tokens_details: { cached_tokens: 0 },
  output_tokens: 5,
  output_tokens_details: { reasoning_tokens: 3 },
  total_tokens: 15,
};

/** The first step's reasoning item, as the API returns it in stateless mode: encrypted content aboard. */
const REASONING_ITEM = {
  type: 'reasoning',
  id: 'rs_1',
  summary: [{ type: 'summary_text', text: 'Need the weather first.' }],
  encrypted_content: 'enc-rs-1',
};

const FUNCTION_CALL = {
  type: 'function_call',
  id: 'fc_1',
  call_id: 'call_1',
  name: 'getWeather',
  arguments: '{"city":"Paris"}',
  status: 'completed',
};

const ANSWER = 'It is sunny in Paris.';

const answerMessage = () => ({
  type: 'message',
  id: 'msg_1',
  role: 'assistant',
  status: 'completed',
  content: [{ type: 'output_text', text: ANSWER, annotations: [] }],
});

/** A completed response that asks for one function call. */
const toolStep = () => ({ id: 'resp_1', status: 'completed', output: [REASONING_ITEM, FUNCTION_CALL], usage: USAGE });

/** A completed response that answers. */
const answerStep = (id = 'resp_2') => ({ id, status: 'completed', output: [answerMessage()], usage: USAGE });

type RequestBody = Record<string, any>;

/**
 * The OpenAI SDK client is constructed in the adapter's constructor and needs an api-key env
 * var to EXIST; the fake client swapped in below means no request is ever made. The env var is
 * restored at once so live-gated suites in the same worker never see a bogus key.
 */
const adapterWithFakeClient = (args: {
  responses: unknown[];
  retrieved?: unknown;
  skills?: ConversationSkill[];
}): { adapter: OpenAiResponses; requests: RequestBody[] } => {
  const requests: RequestBody[] = [];
  const queue = [...args.responses];
  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = 'test-key-never-used';
  try {
    const adapter = new OpenAiResponses({ modelData: fixtureModelData, skills: args.skills, logLevel: 'error' });
    (adapter as unknown as { client: unknown }).client = {
      responses: {
        create: async (body: RequestBody) => {
          requests.push(body);
          const next = queue.shift();
          if (!next) {
            throw new Error(`the fake client has no response for request ${requests.length}`);
          }
          return next;
        },
        retrieve: async () => args.retrieved,
      },
    };
    return { adapter, requests };
  } finally {
    if (prevKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = prevKey;
    }
  }
};

const weatherSkill = (calls: unknown[]): ConversationSkill => {
  const getWeather: Function = {
    definition: {
      name: 'getWeather',
      description: 'The weather in a city.',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
        additionalProperties: false,
      },
    },
    call: async (args: unknown) => {
      calls.push(args);
      return { sky: 'sunny' };
    },
  };
  return {
    getId: () => 'stateless-test-skill',
    getName: () => 'StatelessTestSkill',
    getSystemMessages: () => [],
    getFunctions: () => [getWeather],
    getMessageModerators: () => [] as MessageModerator[],
  };
};

const expectStateless = (body: RequestBody) => {
  expect(body.store).toBe(false);
  expect(body.include).toContain('reasoning.encrypted_content');
  expect(body.previous_response_id).toBeUndefined();
};

describe('OpenAiResponses — every request is stateless', () => {
  test('a plain text request says store: false and asks for the encrypted reasoning', async () => {
    const { adapter, requests } = adapterWithFakeClient({ responses: [answerStep()] });

    const result = await adapter.generateText({ messages: ['Weather?'], model: 'gpt-5.2' as never });

    expect(result.message).toBe(ANSWER);
    expect(requests).toHaveLength(1);
    expectStateless(requests[0]);
  });

  test('a background request says store: false (the run is polled, never retained)', async () => {
    const { adapter, requests } = adapterWithFakeClient({
      responses: [{ id: 'resp_bg', status: 'queued', output: [] }],
      retrieved: answerStep('resp_bg'),
    });

    const result = await adapter.generateText({
      messages: ['Weather?'],
      model: 'gpt-5.2' as never,
      backgroundMode: true,
    });

    expect(result.message).toBe(ANSWER);
    expect(requests).toHaveLength(1);
    expect(requests[0].background).toBe(true);
    expectStateless(requests[0]);
  });

  test('a structured-output request says store: false', async () => {
    const { adapter, requests } = adapterWithFakeClient({
      responses: [
        {
          id: 'resp_obj',
          status: 'completed',
          output: [
            {
              ...answerMessage(),
              content: [{ type: 'output_text', text: JSON.stringify({ sky: 'sunny' }), annotations: [] }],
            },
          ],
          usage: USAGE,
        },
      ],
    });

    const { object } = await adapter.generateObject<{ sky: string }>({
      messages: ['Weather?'],
      model: 'gpt-5.2' as never,
      schema: {
        type: 'object',
        properties: { sky: { type: 'string' } },
        required: ['sky'],
        additionalProperties: false,
      },
    });

    expect(object).toEqual({ sky: 'sunny' });
    expect(requests).toHaveLength(1);
    expectStateless(requests[0]);
  });
});

describe('OpenAiResponses — the tool loop carries its own transcript', () => {
  test('the second request replays the first response’s reasoning item, with its encrypted content, ahead of the function output', async () => {
    const calls: unknown[] = [];
    const { adapter, requests } = adapterWithFakeClient({
      responses: [toolStep(), answerStep()],
      skills: [weatherSkill(calls)],
    });

    const result = await adapter.generateText({
      messages: ['What is the weather in Paris?'],
      model: 'gpt-5.2' as never,
    });

    // The loop completed: the tool ran once with the model's arguments and the answer came back.
    expect(calls).toEqual([{ city: 'Paris' }]);
    expect(result.message).toBe(ANSWER);
    expect(result.toolInvocations).toHaveLength(1);
    expect(requests).toHaveLength(2);
    requests.forEach(expectStateless);

    // The second request's input is the whole exchange: the caller's messages, then the first
    // response's output items verbatim (the reasoning item — encrypted content included — and the
    // function call), then the function's output.
    const [first, second] = requests;
    expect(second.input).toEqual([
      ...first.input,
      REASONING_ITEM,
      FUNCTION_CALL,
      { type: 'function_call_output', call_id: 'call_1', output: JSON.stringify({ sky: 'sunny' }) },
    ]);
  });
});
