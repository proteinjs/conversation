import { Conversation } from '../../src/Conversation';

/**
 * Integration tests for xAI Grok reasoning text + Live Search.
 *
 * The primary purpose of this suite is to **empirically verify Grok 4.3
 * works through @ai-sdk/xai's SDK passthrough**. Grok 4.3 (May 2026) is
 * not yet in the SDK's known-model union type — the SDK accepts any
 * `(string & {})` model id and forwards it to the API. If the passthrough
 * is solid, these tests pass and we ship 4.3. If they fail, we fall back
 * to keeping a known model in that catalog slot until the xai SDK bumps.
 *
 * Hits the real xAI API (requires XAI_API_KEY env var).
 *
 * Notes on xAI specifics:
 * - Grok 4 / 4.20 / 4.3 do *not* accept the `reasoningEffort` parameter
 *   (the model decides effort internally). The `/fast/i` gate in
 *   buildProviderOptions reflects this.
 * - Live Search via `xai.tools.webSearch()` is grounding-style — when
 *   attached, the model consults web/news sources rather than emitting
 *   a tool call. Sources surface via result.sources.
 */

const hasApiKey = !!process.env.XAI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const FLAGSHIP_MODEL = 'grok-4.3';
const FAST_MODEL = 'grok-4-1-fast-reasoning';
const TIMEOUT = 120_000;

describeIfKey('Conversation.generateStream — xAI Grok reasoning + Live Search', () => {
  test(
    'grok-4.3 passes through the SDK and returns a usable response',
    async () => {
      const conversation = new Conversation({ name: 'test-grok-4.3-passthrough' });

      const result = await conversation.generateStream({
        messages: ['Reply with just the number: what is 2 + 2?'],
        model: FLAGSHIP_MODEL,
        reasoningEffort: 'auto',
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const text = await result.text;
      const usage = await result.usage;

      // Critical check: the SDK passthrough didn't blow up and we got an
      // answer back. Pins the "Grok 4.3 works via @ai-sdk/xai 3.0.92
      // passthrough" assumption that the catalog change relies on.
      expect(text).toContain('4');
      expect(usage.totalTokenUsage.inputTokens).toBeGreaterThan(0);
      expect(usage.totalTokenUsage.outputTokens).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'grok-4.3 streams reasoning text on a non-trivial prompt',
    async () => {
      const conversation = new Conversation({ name: 'test-grok-4.3-reasoning' });

      const result = await conversation.generateStream({
        messages: ['A train leaves Chicago at 9am at 60mph; another leaves NYC at 11am at 80mph (distance ~790 miles, traveling toward each other). Do they meet before 4pm? Answer yes or no with one sentence of reasoning.'],
        model: FLAGSHIP_MODEL,
        reasoningEffort: 'auto',
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const reasoning = await result.reasoning;
      const text = await result.text;

      // Grok 4.3 is a built-in reasoning model — reasoning should be
      // non-empty (even though we don't pass reasoningEffort).
      expect(reasoning).toBeTruthy();
      expect(reasoning.length).toBeGreaterThan(0);
      expect(text.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'grok-4-1-fast-reasoning accepts reasoningEffort and streams reasoning',
    async () => {
      const conversation = new Conversation({ name: 'test-grok-fast-reasoning' });

      const result = await conversation.generateStream({
        messages: ['What is 12 × 8? Briefly explain.'],
        model: FAST_MODEL,
        reasoningEffort: 'low',
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const reasoning = await result.reasoning;
      const text = await result.text;

      expect(reasoning.length).toBeGreaterThan(0);
      expect(text).toMatch(/96/);
    },
    TIMEOUT
  );

  test(
    'webSearch: true grounds the response with sources',
    async () => {
      const conversation = new Conversation({ name: 'test-grok-search' });

      const result = await conversation.generateStream({
        messages: ['What is the latest news headline today? One sentence, name the publisher.'],
        model: FLAGSHIP_MODEL,
        webSearch: true,
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const sources = await result.sources;
      // Live Search should surface at least one source on a recency
      // question. If flaky, soften to `expect(sources).toBeDefined()`.
      expect(sources.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'webSearch: false with a non-search prompt leaves the model un-grounded (mode=auto)',
    async () => {
      // With webSearch off, we send `mode: 'auto'` so Grok can decide. On a
      // pure-knowledge question that doesn't need fresh web data, it should
      // not search — letting us pin that the 'auto' default isn't forcing
      // grounding on every turn.
      const conversation = new Conversation({ name: 'test-grok-no-search' });

      const result = await conversation.generateStream({
        messages: ['What is the capital of France? One word.'],
        model: FAST_MODEL,
        webSearch: false,
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const sources = await result.sources;
      const text = await result.text;

      expect(sources.length).toBe(0);
      expect(text).toMatch(/Paris/i);
    },
    TIMEOUT
  );

  test(
    'webSearch: false with an explicit search request still grounds (mode=auto)',
    async () => {
      // The point of `mode: 'auto'` (default) is letting the model search
      // when the user asks for it in plain text — without having to toggle.
      const conversation = new Conversation({ name: 'test-grok-text-asks-search' });

      const result = await conversation.generateStream({
        messages: ['Please search the web and tell me the latest news headline today. One sentence, name the publisher.'],
        model: FAST_MODEL,
        webSearch: false,
      });

      for await (const _ of result.fullStream) {
        // drain
      }

      const sources = await result.sources;
      // Soft expectation: usually 1+ source. If this becomes flaky with
      // certain Grok versions, downgrade to `>= 0` and rely on the unit
      // test for the mode wiring.
      expect(sources.length).toBeGreaterThan(0);
    },
    TIMEOUT
  );
});
