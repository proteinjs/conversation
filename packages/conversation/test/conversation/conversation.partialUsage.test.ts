import { Conversation } from '../../src/Conversation';
import type { StreamPart } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import type { UsageData } from '../../src/UsageData';

/**
 * Per-step live-usage contract, per provider. `onPartialUsageData` fires after
 * each step (tool-call round) with the cumulative usage so far — the data behind
 * the live, in-flight token/cost climb on a flow row. We verify, for every
 * provider we ship, that it (a) fires, (b) is monotonically non-decreasing, and
 * (c) the LAST partial reconciles EXACTLY to the final usage (since both are the
 * sum across steps). The console line documents each provider's step CADENCE —
 * the UX story degrades gracefully whether that's many updates or one, but this
 * test makes each model's behavior a known, verified contract rather than a hope.
 *
 * Gated per provider key (degrades to skip), same as the reasoning suite. A
 * tool the prompt induces the model to call forces multiple steps.
 */

// Live-provider suite: transient API flake must not gate releases — deterministic failures still fail all 3 attempts.
jest.retryTimes(2, { logErrorsBeforeRetry: true });

type UsageCase = { provider: string; model: string; keyEnv: string };

const MODELS: UsageCase[] = [
  { provider: 'anthropic', model: 'claude-opus-4-8', keyEnv: 'ANTHROPIC_API_KEY' },
  { provider: 'anthropic', model: 'claude-sonnet-4-6', keyEnv: 'ANTHROPIC_API_KEY' },
  { provider: 'openai', model: 'gpt-5.5', keyEnv: 'OPENAI_API_KEY' },
  { provider: 'google', model: 'gemini-3.5-flash', keyEnv: 'GOOGLE_GENERATIVE_AI_API_KEY' },
  { provider: 'xai', model: 'grok-4-1-fast-reasoning', keyEnv: 'XAI_API_KEY' },
];

const TIMEOUT = 120_000;

const lookupTool: Function = {
  definition: {
    name: 'lookupCapital',
    description: 'Looks up the capital city of a country.',
    parameters: {
      type: 'object',
      properties: { country: { type: 'string', description: 'The country name' } },
      required: ['country'],
    },
  },
  async call(args: { country: string }) {
    const capitals: Record<string, string> = { france: 'Paris', japan: 'Tokyo', brazil: 'Brasília' };
    return { capital: capitals[args.country.toLowerCase()] ?? 'Unknown' };
  },
};

function geoSkill(): ConversationSkill {
  return {
    getId: () => 'geo-skill',
    getName: () => 'GeoSkill',
    getSystemMessages: () => [
      'You are a geography assistant. Use lookupCapital for each country the user asks about, making a SEPARATE call per country.',
    ],
    getFunctions: () => [lookupTool],
    getMessageModerators: () => [] as MessageModerator[],
  };
}

describe('Conversation.generateStream — onPartialUsageData fires per step and reconciles for every model', () => {
  for (const { provider, model, keyEnv } of MODELS) {
    const testIfKey = process.env[keyEnv] ? test : test.skip;

    testIfKey(
      `${provider}/${model}: per-step partial usage climbs and reconciles to the final`,
      async () => {
        const conversation = new Conversation({ name: `test-partial-${model}`, skills: [geoSkill()] });

        const partials: UsageData[] = [];
        const result = await conversation.generateStream({
          messages: ['What are the capitals of France, Japan, and Brazil? Look up each one.'],
          model,
          onPartialUsageData: async (u) => {
            partials.push(u);
          },
        });

        // Drain the stream so all steps complete and onStepFinish fires.
        for await (const _part of result.fullStream as AsyncIterable<StreamPart>) {
          // consume
        }
        const finalUsage = await result.usage;

        const outs = partials.map((p) => p.totalTokenUsage.outputTokens);
        // eslint-disable-next-line no-console
        console.log(
          `[partial] ${provider}/${model}: steps=${partials.length} cumulativeOut=[${outs.join(',')}] ` +
            `finalOut=${finalUsage.totalTokenUsage.outputTokens} finalTotal=${finalUsage.totalTokenUsage.totalTokens}`
        );

        // (a) it fires
        expect(partials.length).toBeGreaterThan(0);

        // (b) cumulative is monotonically non-decreasing
        for (let i = 1; i < partials.length; i++) {
          expect(partials[i].totalTokenUsage.outputTokens).toBeGreaterThanOrEqual(
            partials[i - 1].totalTokenUsage.outputTokens
          );
          expect(partials[i].totalTokenUsage.totalTokens).toBeGreaterThanOrEqual(
            partials[i - 1].totalTokenUsage.totalTokens
          );
        }

        // (c) the last partial reconciles EXACTLY to the final (both = sum across steps)
        const last = partials[partials.length - 1];
        expect(last.totalTokenUsage.totalTokens).toBe(finalUsage.totalTokenUsage.totalTokens);
        expect(last.totalTokenUsage.outputTokens).toBe(finalUsage.totalTokenUsage.outputTokens);
        expect(last.totalTokenUsage.inputTokens).toBe(finalUsage.totalTokenUsage.inputTokens);
      },
      TIMEOUT
    );
  }
});
