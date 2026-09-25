import { Conversation } from '../../src/Conversation';
import type { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';

/**
 * Every Responses request the streaming path makes is STATELESS: `store: false` on the body, so
 * OpenAI retains no response past the request (the create reference's `store`: "Defaults to true
 * when omitted. If set to true, response data will be stored for at least 30 days").
 *
 * A stateless request has no server-side response to chain to, so a tool loop carries its own
 * transcript: the reasoning item the first step returned — `encrypted_content` aboard, as the
 * API populates it in stateless mode — is replayed in the second request's input beside the
 * function call and its output, and the reasoning summary the first step streamed still reaches
 * the caller (the thinking timeline the host application shows).
 *
 * The exchange is a recorded two-step Responses stream served by a fake `fetch`; the model is
 * the real `@ai-sdk/openai` responses model the library routes OpenAI through, so the request
 * bodies asserted here are the bodies OpenAI would receive.
 *
 * RED at the pre-fix library: no request carries `store` at all (the provider's default, true,
 * applies), and the second request refers to the reasoning item by id (`item_reference`) instead
 * of carrying it — a reference into a stored response the library no longer asks OpenAI to keep.
 */

const REASONING_SUMMARY = 'Need the weather first.';
const ENCRYPTED_REASONING = 'enc-rs-1';
const ANSWER = 'It is sunny in Paris.';

const USAGE = {
  input_tokens: 10,
  input_tokens_details: { cached_tokens: 0 },
  output_tokens: 5,
  output_tokens_details: { reasoning_tokens: 3 },
};

/** Step 1 as the API streams it: a reasoning item (summary streamed, encrypted content on its done event), then a function call. */
const toolStepEvents = [
  { type: 'response.created', response: { id: 'resp_1', created_at: 1, model: 'gpt-5.5' } },
  {
    type: 'response.output_item.added',
    output_index: 0,
    item: { type: 'reasoning', id: 'rs_1', encrypted_content: null },
  },
  { type: 'response.reasoning_summary_part.added', item_id: 'rs_1', summary_index: 0 },
  { type: 'response.reasoning_summary_text.delta', item_id: 'rs_1', summary_index: 0, delta: REASONING_SUMMARY },
  { type: 'response.reasoning_summary_part.done', item_id: 'rs_1', summary_index: 0 },
  {
    type: 'response.output_item.done',
    output_index: 0,
    item: {
      type: 'reasoning',
      id: 'rs_1',
      encrypted_content: ENCRYPTED_REASONING,
      summary: [{ type: 'summary_text', text: REASONING_SUMMARY }],
    },
  },
  {
    type: 'response.output_item.added',
    output_index: 1,
    item: { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'getWeather', arguments: '' },
  },
  { type: 'response.function_call_arguments.delta', item_id: 'fc_1', output_index: 1, delta: '{"city":"Paris"}' },
  {
    type: 'response.output_item.done',
    output_index: 1,
    item: {
      type: 'function_call',
      id: 'fc_1',
      call_id: 'call_1',
      name: 'getWeather',
      arguments: '{"city":"Paris"}',
      status: 'completed',
    },
  },
  { type: 'response.completed', response: { id: 'resp_1', usage: USAGE } },
];

/** Step 2: the answer. */
const answerStepEvents = [
  { type: 'response.created', response: { id: 'resp_2', created_at: 2, model: 'gpt-5.5' } },
  { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } },
  { type: 'response.output_text.delta', item_id: 'msg_1', delta: ANSWER },
  { type: 'response.output_item.done', output_index: 0, item: { type: 'message', id: 'msg_1' } },
  { type: 'response.completed', response: { id: 'resp_2', usage: USAGE } },
];

const sse = (events: object[]): string => events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');

type RequestBody = Record<string, any>;

/** A fake `fetch` that serves the recorded steps in order and keeps every request body. */
const recordedExchange = (steps: object[][]): { requests: RequestBody[]; fetch: typeof fetch } => {
  const requests: RequestBody[] = [];
  const fake = async (_url: unknown, init?: RequestInit): Promise<Response> => {
    requests.push(JSON.parse(String(init?.body)));
    const events = steps[requests.length - 1];
    if (!events) {
      throw new Error(`the recorded exchange has no step for request ${requests.length}`);
    }
    return new Response(sse(events), { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };
  return { requests, fetch: fake as typeof fetch };
};

const getWeather = (calls: unknown[]): Function => ({
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
});

const originalFetch = globalThis.fetch;
const prevKey = process.env.OPENAI_API_KEY;

beforeAll(() => {
  // The provider reads the key at request time; the fake fetch means no request leaves the process.
  process.env.OPENAI_API_KEY = 'test-key-never-used';
});

afterAll(() => {
  globalThis.fetch = originalFetch;
  if (prevKey === undefined) {
    delete process.env.OPENAI_API_KEY;
  } else {
    process.env.OPENAI_API_KEY = prevKey;
  }
});

describe('Conversation.generateStream — OpenAI requests are stateless', () => {
  test('a two-step tool loop says store: false on every request and replays the encrypted reasoning; the summary still streams', async () => {
    const exchange = recordedExchange([toolStepEvents, answerStepEvents]);
    globalThis.fetch = exchange.fetch;
    const calls: unknown[] = [];
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'openai-stateless-test',
      logLevel: 'error',
      limits: { enforceLimits: false },
    });

    const result = await conversation.generateStream({
      messages: ['What is the weather in Paris?'],
      model: 'gpt-5.5',
      tools: [getWeather(calls)],
    });
    const reasoning: string[] = [];
    for await (const part of result.fullStream) {
      if (part.type === 'reasoning-delta') {
        reasoning.push(part.textDelta);
      }
    }
    const text = await result.text;

    // The loop completed: the tool ran once with the model's arguments, the answer came back,
    // and the first step's reasoning summary reached the stream.
    expect(calls).toEqual([{ city: 'Paris' }]);
    expect(text).toBe(ANSWER);
    expect(reasoning.join('')).toContain(REASONING_SUMMARY);

    // Both requests are stateless.
    expect(exchange.requests).toHaveLength(2);
    for (const body of exchange.requests) {
      expect(body.store).toBe(false);
      expect(body.previous_response_id).toBeUndefined();
    }

    // The second request carries the reasoning item itself — encrypted content aboard — ahead of
    // the function call and its output; nothing refers to a stored item by id.
    const input: RequestBody[] = exchange.requests[1].input;
    expect(input.some((item) => item.type === 'item_reference')).toBe(false);
    const reasoningAt = input.findIndex((item) => item.type === 'reasoning');
    const callAt = input.findIndex((item) => item.type === 'function_call');
    const outputAt = input.findIndex((item) => item.type === 'function_call_output');
    expect(reasoningAt).toBeGreaterThanOrEqual(0);
    expect(input[reasoningAt]).toEqual(expect.objectContaining({ encrypted_content: ENCRYPTED_REASONING }));
    expect(callAt).toBeGreaterThan(reasoningAt);
    expect(outputAt).toBeGreaterThan(callAt);
    expect(input[callAt]).toEqual(expect.objectContaining({ call_id: 'call_1', name: 'getWeather' }));
    expect(input[outputAt]).toEqual(expect.objectContaining({ call_id: 'call_1' }));
  }, 30_000);
});
