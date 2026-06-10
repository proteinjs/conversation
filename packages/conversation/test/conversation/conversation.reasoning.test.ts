import { Conversation } from '../../src/Conversation';
import type { StreamPart } from '../../src/Conversation';

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

type ReasoningCase = { provider: string; model: string; keyEnv: string };

const REASONING_MODELS: ReasoningCase[] = [
  { provider: 'anthropic', model: 'claude-opus-4-8', keyEnv: 'ANTHROPIC_API_KEY' },
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
  for (const { provider, model, keyEnv } of REASONING_MODELS) {
    const testIfKey = process.env[keyEnv] ? test : test.skip;

    testIfKey(
      `${provider}/${model}: streams thinking text into result.reasoning`,
      async () => {
        const conversation = new Conversation({ name: `test-reasoning-${model}` });

        const result = await conversation.generateStream({
          messages: [THINKY_PROMPT],
          model,
          reasoningEffort: 'auto',
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
