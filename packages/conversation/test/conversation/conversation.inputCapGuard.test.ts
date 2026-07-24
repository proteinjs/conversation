import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { Function } from '../../src/Function';

/**
 * Hard input-cap guard tests — no network, no API keys. The guard runs BEFORE dispatch at
 * every transport seam (generateStream, generateObject, and the object tool loop), so an
 * over-cap claude-haiku-4-5 call must reject with the descriptive routing-bug error without
 * the transport ever being invoked, while the same content on a 1M-window model dispatches
 * normally.
 */

const TIMEOUT = 60_000;

// ~10 o200k tokens per repeat; 25k repeats lands well past the 190k guard threshold
// (95% of haiku's 200k cap).
const OVER_CAP_TEXT = 'the quick brown fox jumps over the lazy dog. '.repeat(25_000);

const GUARD_MESSAGE = /route this call to a 1M-context model/;

const newConversation = () =>
  new Conversation({ name: 'input-cap-guard-test', logLevel: 'error', limits: { enforceLimits: false } });

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const goodStream = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

const objectResult = (json: string) => ({
  content: [{ type: 'text' as const, text: json }],
  finishReason: { unified: 'stop' as const, raw: 'stop' },
  usage,
  warnings: [],
});

describe('Conversation input-cap guard', () => {
  test(
    'generateResponse on claude-haiku-4-5 (string route) rejects with the descriptive guard error',
    async () => {
      const err = await newConversation()
        .generateResponse({ messages: [OVER_CAP_TEXT], model: 'claude-haiku-4-5' })
        .then(
          () => null,
          (e: Error) => e
        );

      expect(err).toBeInstanceOf(Error);
      expect(err!.message).toContain('claude-haiku-4-5');
      expect(err!.message).toContain('200000-token input cap');
      expect(err!.message).toContain('[conversation: input-cap-guard-test]');
      expect(err!.message).toMatch(GUARD_MESSAGE);
    },
    TIMEOUT
  );

  test(
    'generateStream: the guard fires BEFORE dispatch — the transport is never invoked',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        modelId: 'claude-haiku-4-5',
        doStream: async () => {
          calls++;
          return { stream: goodStream('should never happen') };
        },
      });

      await expect(
        newConversation().generateResponse({ messages: [OVER_CAP_TEXT], model: model as never })
      ).rejects.toThrow(GUARD_MESSAGE);
      expect(calls).toBe(0);
    },
    TIMEOUT
  );

  test(
    'the same content on claude-fable-5 dispatches normally (no guard error)',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        modelId: 'claude-fable-5',
        doStream: async () => {
          calls++;
          return { stream: goodStream('ok') };
        },
      });

      const result = await newConversation().generateResponse({ messages: [OVER_CAP_TEXT], model: model as never });

      expect(calls).toBe(1);
      expect(result.text).toBe('ok');
    },
    TIMEOUT
  );

  test(
    'generateObject on claude-haiku-4-5 rejects at the guard — the transport is never invoked',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        modelId: 'claude-haiku-4-5',
        doGenerate: async () => {
          calls++;
          return objectResult('{"answer":"ok"}');
        },
      });

      await expect(
        newConversation().generateObject<{ answer: string }>({
          messages: [OVER_CAP_TEXT],
          model: model as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
        })
      ).rejects.toThrow(GUARD_MESSAGE);
      expect(calls).toBe(0);
    },
    TIMEOUT
  );

  test(
    'the object tool loop on claude-haiku-4-5 rejects at the guard — the transport is never invoked',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        modelId: 'claude-haiku-4-5',
        doGenerate: async () => {
          calls++;
          return objectResult('{"answer":"ok"}');
        },
      });
      const noopTool: Function = {
        definition: { name: 'noop', description: 'does nothing', parameters: { type: 'object', properties: {} } },
        call: async () => 'ok',
      };

      await expect(
        newConversation().generateObject<{ answer: string }>({
          messages: [OVER_CAP_TEXT],
          model: model as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
          tools: [noopTool],
          maxToolCalls: 5,
        })
      ).rejects.toThrow(GUARD_MESSAGE);
      expect(calls).toBe(0);
    },
    TIMEOUT
  );
});
