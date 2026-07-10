import { Conversation } from '../../src/Conversation';

/**
 * Integration tests for Google Gemini reasoning text + web search grounding.
 *
 * Verifies the contract: when a user picks a Gemini model and asks a
 * reasoning-friendly question, the assistant's reasoning summary streams
 * via `result.reasoning`. When `webSearch: true` is set, googleSearch is
 * attached and the response is grounded.
 *
 * Hits the real Gemini API (requires GOOGLE_GENERATIVE_AI_API_KEY env var).
 *
 * Notes on Gemini specifics:
 * - Reasoning text only flows when `thinkingConfig.includeThoughts: true`
 *   is set on providerOptions (handled in buildProviderOptions).
 * - Google's search is grounding-based, not a model-called tool. The
 *   model doesn't emit a `tool-call` chunk for it; instead the response
 *   text becomes grounded and sources surface via the sources stream.
 */

// Live-provider suite: transient API flake must not gate releases — deterministic failures still fail all 3 attempts.
jest.retryTimes(2, { logErrorsBeforeRetry: true });

const hasApiKey = !!process.env.GOOGLE_GENERATIVE_AI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const REASONING_MODEL = 'gemini-3.5-flash';
const TIMEOUT = 120_000;

describeIfKey('Conversation.generateStream — Gemini reasoning + grounded search', () => {
  test(
    'streams reasoning text for gemini-3.5-flash when reasoningEffort is set',
    async () => {
      const conversation = new Conversation({ name: 'test-gemini-reasoning' });

      const result = await conversation.generateStream({
        messages: [
          'If a train leaves Chicago at 9am going 60mph and another leaves NYC at 11am going 80mph, do they meet before 4pm? Answer yes or no with one sentence of reasoning.',
        ],
        model: REASONING_MODEL,
        reasoningEffort: 'low',
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const reasoning = await result.reasoning;
      const text = await result.text;

      // The fix being tested: with includeThoughts: true, reasoning summary
      // text should be non-empty.
      expect(reasoning).toBeTruthy();
      expect(reasoning.length).toBeGreaterThan(0);

      // Sanity: we also got a text answer.
      expect(text.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'grounds responses with sources when webSearch: true',
    async () => {
      const conversation = new Conversation({ name: 'test-gemini-grounded' });

      // A recency question — should force the grounding tool to consult
      // current web content.
      const result = await conversation.generateStream({
        messages: ['What is the latest news headline today? One sentence, name the publisher.'],
        model: REASONING_MODEL,
        reasoningEffort: 'low',
        webSearch: true,
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const sources = await result.sources;

      // Grounding should produce at least one source URL on a recency
      // question. If this becomes flaky, soften to `expect(sources).toBeDefined()`.
      expect(sources.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'does NOT ground responses when webSearch: false (the gate works)',
    async () => {
      const conversation = new Conversation({ name: 'test-gemini-no-grounding' });

      // A pure-reasoning question that should not pull in web sources when
      // grounding is gated off.
      const result = await conversation.generateStream({
        messages: ['What is 12 * 8? Answer with just the number.'],
        model: REASONING_MODEL,
        reasoningEffort: 'low',
        webSearch: false,
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const sources = await result.sources;
      const text = await result.text;

      // No sources should be present without the grounding tool.
      expect(sources.length).toBe(0);
      // Still got the answer.
      expect(text).toMatch(/96/);
    },
    TIMEOUT
  );
});
