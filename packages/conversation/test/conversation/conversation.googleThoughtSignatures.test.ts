import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { Conversation } from '../../src/Conversation';
import type { ConversationSkill } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';

/**
 * Google's thought signatures ride every step of a turn — proven on the wire.
 *
 * Gemini thinking models return an encrypted `thoughtSignature` on a response part (a
 * `functionCall` part; on a parallel call only the FIRST one; the final part of a text answer when
 * no call was made), and the request that continues the turn must carry each signature back
 * inside its original part: on `functionCall` parts the API validates this strictly across every
 * step of the current turn (a missing one is a 400 on Gemini 3+); on text parts it is recommended,
 * not enforced. Two more rules from the same page: never merge a part that carries a signature with
 * one that does not, and never fabricate one for a call the model did not make.
 *
 * The real `@ai-sdk/google` provider runs over a scripted `fetch` that answers the SSE wire the way
 * the API does, so the proof reads the next request's BODY — what the API would receive — through
 * every seam this library owns on the way: the tool loop's per-step projection (tool-result
 * pruning, tool-call input coercion, the request dump) and the library's own continuation rounds
 * (the output-limit continuation re-sends the cut assistant message itself). No network, no key.
 *
 * The request shape is pinned on the same wire: `generationConfig.thinkingConfig` with
 * `includeThoughts: true` on every thinking request, `thinkingLevel` as the level's lowercase word
 * when a level is asked, absent at `auto` (the model's own default), and `thoughtsTokenCount`
 * read back as the step's reasoning tokens.
 */

const TIMEOUT = 30_000;

type WirePart = Record<string, unknown> & {
  text?: string;
  thought?: boolean;
  thoughtSignature?: string;
  functionCall?: { name: string; args?: unknown; id?: string };
  functionResponse?: { name: string; response?: unknown; id?: string };
};
type WireContent = { role: string; parts: WirePart[] };
type WireBody = {
  contents: WireContent[];
  generationConfig?: Record<string, unknown>;
  tools?: unknown;
};

const usage = (over: { prompt?: number; candidates?: number; thoughts?: number } = {}) => ({
  promptTokenCount: over.prompt ?? 10,
  candidatesTokenCount: over.candidates ?? 5,
  thoughtsTokenCount: over.thoughts ?? 0,
  totalTokenCount: (over.prompt ?? 10) + (over.candidates ?? 5) + (over.thoughts ?? 0),
});

/** One streamed response: each entry is one SSE chunk (the parts it carries); the last carries the finish. */
const sse = (chunks: WirePart[][], finishReason: string, usageMetadata = usage()): string =>
  chunks
    .map((parts, i) => {
      const last = i === chunks.length - 1;
      const candidate: Record<string, unknown> = { content: { parts, role: 'model' }, index: 0 };
      if (last) {
        candidate.finishReason = finishReason;
      }
      const payload: Record<string, unknown> = { candidates: [candidate] };
      if (last) {
        payload.usageMetadata = usageMetadata;
      }
      return `data: ${JSON.stringify(payload)}\n\n`;
    })
    .join('');

/** The provider over a scripted wire: every request body kept, answers served in order. */
class ScriptedGoogleWire {
  readonly requests: WireBody[] = [];
  private answers: string[] = [];

  constructor(answers: string[]) {
    this.answers = [...answers];
  }

  readonly fetch = async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    this.requests.push(JSON.parse(String(init?.body)) as WireBody);
    const answer = this.answers.shift();
    if (answer === undefined) {
      throw new Error(`the wire has no answer for request ${this.requests.length}`);
    }
    return new Response(answer, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };

  model(modelId: string) {
    return createGoogleGenerativeAI({ apiKey: 'scripted', fetch: this.fetch as never })(modelId);
  }
}

const lookup: Function = {
  definition: {
    name: 'lookup',
    description: 'Look a thing up',
    parameters: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] },
  },
  call: async (args: { q: string }) => ({ found: `${args.q}: 42` }),
};

const skill: ConversationSkill = {
  getId: () => 'lookup-skill',
  getName: () => 'Lookup',
  getSystemMessages: () => [],
  getFunctions: () => [lookup],
  getMessageModerators: () => [],
};

const conversationOver = () =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'google-thought-signatures',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [skill],
  });

/** Drains the stream; returns the reasoning text it streamed (every step's — `result.reasoning` is the last step's). */
const drain = async (fullStream: AsyncIterable<unknown>): Promise<string> => {
  let reasoning = '';
  for await (const part of fullStream) {
    const p = part as { type?: string; textDelta?: string };
    if (p.type === 'reasoning-delta') {
      reasoning += p.textDelta ?? '';
    }
  }
  return reasoning;
};

const modelTurns = (body: WireBody) => body.contents.filter((c) => c.role === 'model');
const callParts = (content: WireContent) => content.parts.filter((p) => p.functionCall);

describe('Google thought signatures — every step of a turn carries them back on the wire', () => {
  test(
    'the request shape: generationConfig.thinkingConfig carries includeThoughts and the level as its lowercase word; auto sends no level',
    async () => {
      const wire = new ScriptedGoogleWire([
        sse([[{ text: 'Four.', thoughtSignature: 'sig-final' }]], 'STOP', usage({ thoughts: 3 })),
        sse([[{ text: 'Four.' }]], 'STOP'),
      ]);
      const conversation = conversationOver();
      const atLow = await conversation.generateStream({
        messages: ['What is two plus two?'],
        model: wire.model('gemini-3.5-flash'),
        reasoningEffort: 'low',
      });
      await drain(atLow.fullStream);
      const atAuto = await conversation.generateStream({
        messages: ['And again?'],
        model: wire.model('gemini-3.5-flash'),
        reasoningEffort: 'auto',
      });
      await drain(atAuto.fullStream);

      expect(wire.requests).toHaveLength(2);
      expect(wire.requests[0].generationConfig?.thinkingConfig).toEqual({
        includeThoughts: true,
        thinkingLevel: 'low',
      });
      expect(wire.requests[1].generationConfig?.thinkingConfig).toEqual({ includeThoughts: true });
      // The usage read back: the API's thoughtsTokenCount is the step's reasoning tokens.
      const lowUsage = await atLow.usage;
      expect(lowUsage.totalTokenUsage.reasoningTokens).toBe(3);
    },
    TIMEOUT
  );

  test(
    'a tool call step: the functionCall part goes back on the next step with its signature, inside its own part, ahead of the function response',
    async () => {
      const wire = new ScriptedGoogleWire([
        sse(
          [
            [{ text: 'Let me look that up.', thought: true }],
            [{ functionCall: { name: 'lookup', args: { q: 'the answer' } }, thoughtSignature: 'sig-call-1' }],
          ],
          'STOP',
          usage({ thoughts: 7 })
        ),
        sse([[{ text: 'It is 42.', thoughtSignature: 'sig-final-2' }]], 'STOP'),
      ]);
      const conversation = conversationOver();
      const result = await conversation.generateStream({
        messages: ['Look up the answer.'],
        model: wire.model('gemini-3.5-flash'),
        reasoningEffort: 'medium',
      });
      const reasoning = await drain(result.fullStream);

      expect(await result.text).toBe('It is 42.');
      expect(reasoning).toBe('Let me look that up.');
      expect(wire.requests).toHaveLength(2);
      const step2 = wire.requests[1];
      const [modelTurn] = modelTurns(step2);
      expect(modelTurn).toBeDefined();
      const [call] = callParts(modelTurn);
      expect(call.functionCall).toMatchObject({ name: 'lookup', args: { q: 'the answer' } });
      // THE LAW: the signature rides back inside the functionCall's own part.
      expect(call.thoughtSignature).toBe('sig-call-1');
      // The function response follows the model turn as the API requires.
      const responseTurn = step2.contents[step2.contents.indexOf(modelTurn) + 1];
      expect(responseTurn.role).toBe('user');
      expect(responseTurn.parts.some((p) => p.functionResponse?.name === 'lookup')).toBe(true);
      // The step's reasoning tokens come from thoughtsTokenCount.
      const total = await result.usage;
      expect(total.totalTokenUsage.reasoningTokens).toBe(7);
    },
    TIMEOUT
  );

  test(
    'parallel calls: the first functionCall part carries the signature back and the second carries none — never merged, never fabricated',
    async () => {
      const wire = new ScriptedGoogleWire([
        sse(
          [
            [
              { functionCall: { name: 'lookup', args: { q: 'alpha' } }, thoughtSignature: 'sig-parallel' },
              { functionCall: { name: 'lookup', args: { q: 'beta' } } },
            ],
          ],
          'STOP',
          usage({ thoughts: 5 })
        ),
        sse([[{ text: 'Both found.' }]], 'STOP'),
      ]);
      const conversation = conversationOver();
      const result = await conversation.generateStream({
        messages: ['Look up alpha and beta.'],
        model: wire.model('gemini-3.5-flash'),
        reasoningEffort: 'high',
      });
      await drain(result.fullStream);

      expect(await result.text).toBe('Both found.');
      const step2 = wire.requests[1];
      const [modelTurn] = modelTurns(step2);
      const calls = callParts(modelTurn);
      expect(calls.map((c) => c.functionCall?.args)).toEqual([{ q: 'alpha' }, { q: 'beta' }]);
      expect(calls[0].thoughtSignature).toBe('sig-parallel');
      expect(calls[1].thoughtSignature).toBeUndefined();
      // All calls, then all responses — the order the API requires.
      const responseTurn = step2.contents[step2.contents.indexOf(modelTurn) + 1];
      expect(responseTurn.parts.filter((p) => p.functionResponse).length).toBe(2);
    },
    TIMEOUT
  );

  test(
    "the library's own continuation round: a text answer cut at the output limit is re-sent with the signature on its final part",
    async () => {
      const wire = new ScriptedGoogleWire([
        sse([[{ text: 'The first half of the answer' }], [{ text: '', thoughtSignature: 'sig-cut' }]], 'MAX_TOKENS'),
        sse([[{ text: ' and the second half.' }]], 'STOP'),
      ]);
      const conversation = conversationOver();
      const result = await conversation.generateStream({
        messages: ['Write the whole answer.'],
        model: wire.model('gemini-3.5-flash'),
        reasoningEffort: 'auto',
      });
      await drain(result.fullStream);

      expect(wire.requests).toHaveLength(2);
      const continuation = wire.requests[1];
      const [modelTurn] = modelTurns(continuation);
      expect(modelTurn).toBeDefined();
      // The cut text goes back as ONE part with the signature the wire put on it — the streamed
      // signature arrived on an empty-text part, which belongs to the text block it closed.
      const textParts = modelTurn.parts.filter((p) => typeof p.text === 'string');
      expect(textParts).toHaveLength(1);
      expect(textParts[0].text).toBe('The first half of the answer');
      expect(textParts[0].thoughtSignature).toBe('sig-cut');
    },
    TIMEOUT
  );
});
