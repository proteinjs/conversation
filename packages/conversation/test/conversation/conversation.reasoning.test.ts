import { Conversation } from '../../src/Conversation';
import type { StreamPart } from '../../src/Conversation';
import type { ReasoningEffort } from '../../src/Conversation';

/**
 * Provider-agnostic reasoning contract: every reasoning-capable model we ship
 * must stream thinking TEXT — `reasoning-delta` parts that carry content — and
 * aggregate it into `result.reasoning`. That's the contract the chat + flow
 * timelines render. We loop one representative model per provider (plus both
 * Anthropic tiers, since the flow uses Opus and the chat uses Sonnet); each
 * case is gated on its provider's API key so the suite degrades gracefully
 * where a key is absent.
 *
 * Why "text" and not just "a reasoning block": Anthropic streams thinking as
 * `thinking_delta` (text) AND `signature_delta` (an empty `reasoning-delta`,
 * signature only). A model that returns SIGNATURE-ONLY thinking yields an empty
 * `result.reasoning` — there's nothing to show in the UI even though a thinking
 * block "happened". `claude-opus-4-7` did exactly this (its `display:summarized`
 * mitigation is stripped by the installed @ai-sdk/anthropic schema), which is
 * why the flow's plan turns showed no reasoning until we moved to `4-8`. This
 * test fails loudly on any model that regresses to signature-only.
 */

// Live-provider suite: transient API flake must not gate releases — deterministic failures still fail all 3 attempts.
jest.retryTimes(2, { logErrorsBeforeRetry: true });

type ReasoningCase = { provider: string; model: string; keyEnv: string; effort?: ReasoningEffort; prompt?: string };

const REASONING_MODELS: ReasoningCase[] = [
  // claude-opus-5, not 4-8: in 2026-08 the API migrated the opus tier to ADAPTIVE-only thinking
  // (legacy `thinking.type: enabled` is rejected for 4-8) and the shipped flow tier tracks
  // getLatestModelInFamily('opus'). ADAPTIVE MEANS THE MODEL DECIDES: at effort auto on an easy
  // prompt, opus-tier chooses ~zero thinking and its summarized block streams EMPTY (probed
  // directly against the API — 4-8 stays empty even at high effort, the 4-7 regression one
  // generation later). So the opus case must ELICIT thinking to test the streaming contract:
  // explicit high effort + a proof-grade prompt (probed: 293 chars of summarized thinking).
  // STREAMING additionally requires @ai-sdk/anthropic >= 3.0.107: 3.0.104 accepted the adaptive
  // config but dropped the summarized thinking deltas in stream translation (0 deltas through
  // this stack while the raw API returned text) — the bump alone flipped this case red -> green.
  // Sonnet stays on auto + the easy prompt — its adaptive temperament thinks anyway, which keeps
  // the default chat path's auto contract covered.
  {
    provider: 'anthropic',
    model: 'claude-opus-5',
    keyEnv: 'ANTHROPIC_API_KEY',
    effort: 'high',
    prompt: 'Prove that the sum of the first n odd numbers is n squared. Reason it out, then state the result in one sentence.',
  },
  { provider: 'anthropic', model: 'claude-sonnet-4-6', keyEnv: 'ANTHROPIC_API_KEY' },
  { provider: 'openai', model: 'gpt-5.5', keyEnv: 'OPENAI_API_KEY' },
  { provider: 'google', model: 'gemini-3.5-flash', keyEnv: 'GOOGLE_GENERATIVE_AI_API_KEY' },
  { provider: 'xai', model: 'grok-4-1-fast-reasoning', keyEnv: 'XAI_API_KEY' },
];

const TIMEOUT = 120_000;

// A prompt that benefits from a few steps of thinking but asks for a short
// answer, so reasoning — not output length — is what we exercise.
const THINKY_PROMPT =
  'A farmer has 17 sheep; all but 9 run away. Then he buys twice as many as he has left. ' +
  'How many sheep does he have? Think it through, then give the number in one sentence.';

describe('Conversation.generateStream — reasoning text streams for every shipped reasoning model', () => {
  for (const { provider, model, keyEnv, effort, prompt } of REASONING_MODELS) {
    const testIfKey = process.env[keyEnv] ? test : test.skip;

    testIfKey(
      `${provider}/${model}: streams thinking text into result.reasoning`,
      async () => {
        const conversation = new Conversation({ name: `test-reasoning-${model}` });

        const result = await conversation.generateStream({
          messages: [prompt ?? THINKY_PROMPT],
          model,
          reasoningEffort: effort ?? 'auto',
        });

        // Drain the interleaved stream, counting reasoning-delta parts that
        // actually carried text (mapFullStream drops empty signature-only ones).
        let reasoningTextDeltas = 0;
        for await (const part of result.fullStream as AsyncIterable<StreamPart>) {
          if (part.type === 'reasoning-delta' && part.textDelta) {
            reasoningTextDeltas += 1;
          }
        }

        const reasoning = await result.reasoning;
        const text = await result.text;

        // Diagnostic — shows the text-vs-signature split per model when run.
        // eslint-disable-next-line no-console
        console.log(
          `[reasoning] ${provider}/${model}: reasoningTextDeltas=${reasoningTextDeltas} reasoningLen=${reasoning.length} textLen=${text.length}`
        );

        expect(reasoningTextDeltas).toBeGreaterThan(0);
        expect(reasoning.length).toBeGreaterThan(0);
        expect(text.length).toBeGreaterThan(0);
      },
      TIMEOUT
    );
  }
});
