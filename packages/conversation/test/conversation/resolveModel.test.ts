import { MockLanguageModelV3 } from 'ai/test';
import { inferProvider, resolveModel, routedProvider } from '../../src/resolveModel';

/**
 * One routing decision for a model string: `resolveModel` builds the model from it and
 * `routedProvider` reports it. Whatever a conversation says to "the provider" is said to
 * `routedProvider(model)`, so the two must agree for every string form — including a bare name no
 * pattern recognizes, which is routed to OpenAI (the case a name-only guess answered "unknown"
 * for, so nothing provider-specific was said to a model that was in fact OpenAI's).
 */
describe('resolveModel / routedProvider — one routing decision', () => {
  /** The family of the model instance `resolveModel` built (`"openai.responses"` → `openai`). */
  const builtFor = (model: string): string =>
    String((resolveModel(model) as { provider?: string }).provider ?? '').split('.')[0];

  test('routedProvider names the provider of the model resolveModel builds, for every string form', () => {
    const cases: Array<[string, string]> = [
      ['gpt-5.6-sol', 'openai'],
      ['o3', 'openai'],
      ['claude-sonnet-5', 'anthropic'],
      ['gemini-3.1-pro-preview', 'google'],
      ['grok-4.5', 'xai'],
      ['openai:any-deployment-name', 'openai'],
      ['anthropic:some-model', 'anthropic'],
      ['  gpt-5.6-sol  ', 'openai'],
      // No pattern recognizes these: they are ROUTED to OpenAI, and must be reported as OpenAI.
      ['acme-house-model-v2', 'openai'],
      ['my-fine-tune', 'openai'],
    ];
    for (const [model, provider] of cases) {
      expect([model, routedProvider(model)]).toEqual([model, provider]);
      expect([model, builtFor(model)]).toEqual([model, provider]);
    }
  });

  test('a prefixed model is built under the id after the prefix', () => {
    expect((resolveModel('openai:any-deployment-name') as { modelId?: string }).modelId).toBe('any-deployment-name');
  });

  test('an unknown provider prefix and an empty string are errors for both — never a silent route', () => {
    for (const ask of [resolveModel, routedProvider]) {
      expect(() => ask('acme:some-model')).toThrow(/unknown provider prefix "acme"/);
      expect(() => ask('   ')).toThrow(/empty model string/);
    }
  });

  test('a model INSTANCE is not routed: it is read for who it says it is — its provider family, else its id’s name pattern, else unknown', () => {
    const instance = (provider: string, modelId: string) => new MockLanguageModelV3({ provider, modelId }) as never;
    expect(routedProvider(instance('openai.responses', 'a-deployment-name'))).toBe('openai');
    expect(routedProvider(instance('anthropic.messages', 'a-deployment-name'))).toBe('anthropic');
    expect(routedProvider(instance('some-gateway', 'gpt-5.6-sol'))).toBe('openai');
    expect(routedProvider(instance('some-gateway', 'a-deployment-name'))).toBe('unknown');
    const passedThrough = instance('some-gateway', 'a-deployment-name');
    expect(resolveModel(passedThrough)).toBe(passedThrough);
  });

  test('inferProvider still classifies a NAME (unknown when nothing recognizes it) and never throws', () => {
    expect(inferProvider('gpt-5.6-sol')).toBe('openai');
    expect(inferProvider('anthropic:some-model')).toBe('anthropic');
    expect(inferProvider('acme-house-model-v2')).toBe('unknown');
    expect(inferProvider('acme:some-model')).toBe('unknown');
    expect(inferProvider('')).toBe('unknown');
  });
});
