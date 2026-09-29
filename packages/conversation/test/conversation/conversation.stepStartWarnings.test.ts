import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { fixtureModelData } from './fixtureModelData';

/** The stream's parts read structurally — the suite is RED against a library whose union lacks the part. */
type Part = { type: string; warnings?: unknown[]; finishReason?: string };

/**
 * THE STEP'S WARNINGS RIDE THE STREAM: every model request the loop makes arrives on
 * `fullStream` as a `step-start` part carrying the SDK's warnings for that request — a feature
 * the adapter dropped (`unsupported`), a substitution a middleware made (`compatibility` —
 * RequestedEffort's omitted effort rides with `feature: 'reasoningEffort'`), a note (`other`) —
 * so a consumer READS what the transport changed instead of inferring it from the answer. A
 * catalog's checks fail a claimed level on that warning; a chat's timeline names the omission.
 *
 * RED before this part existed: the library's stream mapped text, reasoning, tools, sources and
 * step finishes, and NO part carried warnings — the SDK's `start-step` was dropped on the floor.
 */

const TIMEOUT = 30_000;
const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const REFUSAL_WARNING = {
  type: 'compatibility',
  feature: 'reasoningEffort',
  details: "gpt-6-astra does not accept reasoning effort 'none' — re-issued with the effort omitted.",
};
const DROPPED_WARNING = { type: 'unsupported', feature: 'seed', details: 'the provider does not take a seed' };

const stepWith = (warnings: unknown[]) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: warnings as never },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: 'THE ANSWER' },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const modelWhoseStreamWarns = (warnings: unknown[]) =>
  new MockLanguageModelV3({
    provider: 'openai.responses',
    modelId: 'gpt-6-astra',
    doStream: async () => ({ stream: stepWith(warnings) }),
  });

async function parts(model: MockLanguageModelV3): Promise<Part[]> {
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'step-start-warnings',
    logLevel: 'error',
    limits: { enforceLimits: false },
  });
  const result = await conversation.generateStream({ messages: ['2+2?'], model: model as never });
  const seen: Part[] = [];
  for await (const part of result.fullStream as AsyncIterable<Part>) {
    seen.push(part);
  }
  return seen;
}

const stepStarts = (seen: Part[]) => seen.filter((part) => part.type === 'step-start');

describe('the step-start part forwards the SDK’s warnings for each request', () => {
  test(
    'a request the provider’s stream-start warned about surfaces every warning, as data, before the step’s first output',
    async () => {
      const seen = await parts(modelWhoseStreamWarns([REFUSAL_WARNING, DROPPED_WARNING]));
      const starts = stepStarts(seen);
      expect(starts).toHaveLength(1);
      expect(starts[0].warnings).toEqual([REFUSAL_WARNING, DROPPED_WARNING]);
      // Before the step's text, after nothing: the part opens the step.
      expect(seen.findIndex((part) => part.type === 'step-start')).toBeLessThan(
        seen.findIndex((part) => part.type === 'text-delta')
      );
      expect(seen[seen.length - 1]).toMatchObject({ type: 'step-finish', finishReason: 'stop' });
    },
    TIMEOUT
  );

  test(
    'a request with nothing to warn about still opens its step — an empty list, so a consumer can count steps by it',
    async () => {
      const seen = await parts(modelWhoseStreamWarns([]));
      expect(stepStarts(seen).map((part) => part.warnings)).toEqual([[]]);
    },
    TIMEOUT
  );

  test(
    'a warning in a shape this library does not know still rides, typed by its `type` alone — never dropped, never a throw',
    async () => {
      const seen = await parts(
        modelWhoseStreamWarns([
          { type: 'other', message: 'an aside' },
          { type: 'odd', extra: 1 },
        ])
      );
      expect(stepStarts(seen)[0].warnings).toEqual([{ type: 'other', message: 'an aside' }, { type: 'odd' }]);
    },
    TIMEOUT
  );
});
