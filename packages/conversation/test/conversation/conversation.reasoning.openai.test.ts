import { Conversation } from '../../src/Conversation';

/**
 * Integration tests for OpenAI reasoning text + web search.
 *
 * Verifies the contract we promise the UI: when a user picks a reasoning
 * model (gpt-5.5) and asks a question, the assistant's reasoning summary
 * streams in `result.reasoning`. When `webSearch: true` is set, the model
 * has the OpenAI web_search tool available.
 *
 * These hit the real OpenAI API (requires OPENAI_API_KEY env var) and
 * specifically exercise the Responses API path that resolveModel routes
 * OpenAI through.
 */

// Live-provider suite: transient API flake must not gate releases — deterministic failures still fail all 3 attempts.
jest.retryTimes(2, { logErrorsBeforeRetry: true });

const hasApiKey = !!process.env.OPENAI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const REASONING_MODEL = 'gpt-5.5';
const TIMEOUT = 120_000;

describeIfKey('Conversation.generateStream — OpenAI reasoning + web search', () => {
  test(
    'streams reasoning text for gpt-5.5 when reasoningEffort is set',
    async () => {
      const conversation = new Conversation({ name: 'test-openai-reasoning' });

      // A genuinely multi-step problem at 'medium' effort: at 'low' on a
      // simpler prompt, gpt-5.5 occasionally emits reasoning-start/end with
      // ZERO summary deltas (observed as CI release flake) — the model just
      // doesn't have enough to summarize. The assertion stays strict (it
      // guards the reasoningSummary-not-requested regression), so make the
      // reasoning reliably non-empty instead of weakening the check. NOTE:
      // 'high'/'xhigh'/'max' would reroute to the background/polling path
      // (shouldUseBackgroundMode) — 'medium' is the max effort that stays on
      // the streaming Responses path this test pins.
      const result = await conversation.generateStream({
        messages: [
          'A farmer has chickens and rabbits: 35 heads and 94 legs in total. How many chickens and how many rabbits? ' +
            'Then check your answer: if the chicken and rabbit counts were swapped, how many legs would there be? ' +
            'Answer with the two counts and the swapped-legs number.',
        ],
        model: REASONING_MODEL,
        reasoningEffort: 'medium',
      });

      // Drain the full stream so reasoning is captured.
      for await (const _ of result.fullStream) {
        // no-op
      }

      const reasoning = await result.reasoning;
      const text = await result.text;

      // The fix being tested: with reasoningSummary: 'auto' on the Responses
      // API, reasoning text should be non-empty.
      expect(reasoning).toBeTruthy();
      expect(reasoning.length).toBeGreaterThan(0);

      // Sanity: we also got a text answer.
      expect(text.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'attaches the OpenAI web_search tool when webSearch: true',
    async () => {
      const conversation = new Conversation({ name: 'test-openai-web-search' });

      // A question that requires fresh web info — gives the model a strong
      // signal to call the search tool.
      const result = await conversation.generateStream({
        messages: ['What is the latest news headline today? One sentence, cite the publisher.'],
        model: REASONING_MODEL,
        reasoningEffort: 'low',
        webSearch: true,
      });

      for await (const _ of result.fullStream) {
        // no-op
      }

      const toolInvocations = await result.toolInvocations;

      // We don't strictly assert the model decided to call search (it's
      // model-dependent), but we do assert the tool was wired so it *could*
      // be called. If it was called, the invocation should be the web search
      // tool.
      const searchCalls = toolInvocations.filter((t) => t.name === 'web_search');
      // Soft expectation: a recency question should usually trigger search.
      // If this becomes flaky, drop to `expect(searchCalls).toBeDefined()`.
      expect(searchCalls.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );
});
