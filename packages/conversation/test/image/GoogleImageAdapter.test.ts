import { ImageGenerator } from '../../src/image/ImageGenerator';
import { GoogleImageAdapter } from '../../src/image/GoogleImageAdapter';
import type { ImageInput } from '../../src/image/ImageGenerationRequest';
import type {
  ImageTransport,
  ImageTransportRequest,
  ImageTransportResponse,
} from '../../src/image/ImageProviderAdapter';
import { modelDataFromRows } from '../conversation/fixtureModelData';
import { RecordingImageTransport, TINY_PNG_BYTES } from './openAiImageFixtures';
import {
  GOOGLE_FIXTURE_ROW,
  googleInteraction,
  googleInteractionInSteps,
  googleNoPicture,
  googlePngRefused,
  googleRateLimited,
  googleSafetyBlocked,
  googleUsageWithoutSplit,
} from './googleRecraftFixtures';

/**
 * The Google picture adapter (the Interactions API), driven through `ImageGenerator` against a
 * transport double replaying documented bodies with the probe's recorded token counts. Every case
 * reads an OUTCOME — what was on the wire, what came back, what it cost.
 */

const MODEL = 'gemini-3.1-flash-image';
const TEST_KEY = 'test-google-key-never-a-real-one';
const modelData = modelDataFromRows({ standard: { [MODEL]: GOOGLE_FIXTURE_ROW } });

const generatorOver = (transport: ImageTransport) =>
  new ImageGenerator({
    modelData,
    transport,
    adapters: [new GoogleImageAdapter({ apiKey: TEST_KEY })],
    logLevel: 'error',
  });

const NOTHING = { textInputUsd: 0, imageInputUsd: 0, outputUsd: 0, totalUsd: 0 };

const jsonOf = (request: ImageTransportRequest): Record<string, unknown> => {
  if (request.body.kind !== 'json') {
    throw new Error(`expected a JSON body, saw ${request.body.kind}`);
  }
  return request.body.json;
};

const photo = (input: Partial<ImageInput> = {}): ImageInput => ({
  bytes: TINY_PNG_BYTES,
  mimeType: 'image/png',
  ...input,
});

describe('a room ask with the person’s photo', () => {
  test('goes to /interactions as JSON: the words, then the photo, JPEG asked for, the shape from the size', async () => {
    const transport = new RecordingImageTransport(googleInteraction);
    const outcome = await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'Scandinavian, keep the fireplace.',
      inputs: [photo({ role: 'structure', fidelity: 'high' })],
      size: '1536x1024',
    });

    const sent = transport.only();
    expect(sent.url).toBe('https://generativelanguage.googleapis.com/v1beta/interactions');
    expect(sent.headers).toEqual({ 'x-goog-api-key': TEST_KEY });
    const json = jsonOf(sent);
    expect(json.model).toBe(MODEL);
    const input = json.input as Record<string, unknown>[];
    expect(input[0].type).toBe('text');
    expect(String(input[0].text)).toContain('Scandinavian, keep the fireplace.');
    expect(String(input[0].text)).toContain('1. The layout - keep its geometry, walls, windows and light exactly.');
    expect(input[1]).toEqual({ type: 'image', mime_type: 'image/png', data: TINY_PNG_BYTES.toString('base64') });
    expect(json.response_format).toEqual({
      type: 'image',
      mime_type: 'image/jpeg',
      aspect_ratio: '3:2',
      image_size: '2K',
    });
    expect(json.previous_interaction_id).toBeUndefined();

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(1);
    expect(outcome.images[0].mimeType).toBe('image/jpeg');
    expect(outcome.continuationId).toBe('interaction_fixture_1');
    expect(outcome.sent).toBe(true);
  });

  test('never asks for a PNG — the vendor answers 400 to it (measured 2026-09-16)', async () => {
    const transport = new RecordingImageTransport(googleInteraction);
    await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
      outputFormat: 'png',
    });
    const format = jsonOf(transport.only()).response_format as Record<string, unknown>;
    expect(format.mime_type).toBe('image/jpeg');
    expect(format.mime_type).not.toBe('image/png');
  });

  test('prices from the vendor’s by-modality usage: the recorded room ask is $0.0686', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(googleInteraction)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'Scandinavian, keep the fireplace.',
      inputs: [photo()],
    });
    expect(outcome.kind).toBe('ok');
    expect(outcome.usage).toEqual({
      textInputTokens: 36,
      imageInputTokens: 320,
      imageOutputTokens: 1120,
      textOutputTokens: 407,
      totalTokens: 1883,
    });
    expect(outcome.cost?.totalUsd).toBeCloseTo(0.0686, 4);
    expect(outcome.cost?.outputUsd).toBeCloseTo(1120 * 60e-6 + 407 * 3e-6, 8);
  });

  test('a total with no by-modality split leaves the ask unpriced — never a guess', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(googleUsageWithoutSplit)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
    });
    expect(outcome.kind).toBe('ok');
    expect(outcome.cost).toBeUndefined();
  });

  test('a follow-up carries the earlier ask’s handle as previous_interaction_id', async () => {
    const transport = new RecordingImageTransport(googleInteraction);
    await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'Now the sofa in green.',
      previousInteractionId: 'interaction_fixture_1',
    });
    expect(jsonOf(transport.only()).previous_interaction_id).toBe('interaction_fixture_1');
  });

  test('reads the picture from a step’s content when output_image is absent', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(googleInteractionInSteps)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
    });
    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(1);
    expect(outcome.continuationId).toBe('interaction_fixture_steps');
  });
});

describe('more than one picture', () => {
  test('is one call per picture, the usage summed and every picture kept', async () => {
    let calls = 0;
    const transport = new RecordingImageTransport(() => googleInteraction(`interaction_${++calls}`));
    const outcome = await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
      count: 2,
    });
    expect(transport.requests).toHaveLength(2);
    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(2);
    expect(outcome.usage?.imageOutputTokens).toBe(2240);
    expect(outcome.cost?.totalUsd).toBeCloseTo(2 * 0.0686, 4);
    expect(outcome.continuationId).toBe('interaction_2');
  });

  test('a picture already made stands when a later one fails', async () => {
    let calls = 0;
    const transport = new RecordingImageTransport(() => (++calls === 1 ? googleInteraction() : googleRateLimited()));
    const outcome = await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
      count: 3,
    });
    expect(transport.requests).toHaveLength(2);
    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(1);
  });
});

describe('before the wire', () => {
  test.each([
    ['a transparent background', { background: 'transparent' as const }, 'transparent'],
    ['a fifth picture', { count: 5 }, 'count must be'],
    [
      'a utility this vendor does not offer',
      { operation: 'vectorize' as const, inputs: [photo()] },
      'no "vectorize" utility',
    ],
    ['a malformed size', { size: 'big' }, 'size must be'],
  ])('%s is refused with nothing sent and a known zero cost', async (_name, fields, words) => {
    const transport = new RecordingImageTransport(googleInteraction);
    const outcome = await generatorOver(transport).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
      ...fields,
    });
    expect(transport.requests).toHaveLength(0);
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe('invalid_request');
    expect(outcome.message).toContain(words);
    expect(outcome.sent).toBe(false);
    expect(outcome.cost).toEqual(NOTHING);
  });

  test('no key is an auth failure with nothing sent', async () => {
    const saved = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    delete process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    try {
      const transport = new RecordingImageTransport(googleInteraction);
      const generator = new ImageGenerator({
        modelData,
        transport,
        adapters: [new GoogleImageAdapter()],
        logLevel: 'error',
      });
      const outcome = await generator.generate({ provider: 'google', model: MODEL, prompt: 'A lamp.' });
      expect(transport.requests).toHaveLength(0);
      expect(outcome.kind).toBe('failed');
      if (outcome.kind !== 'failed') {
        return;
      }
      expect(outcome.errorKind).toBe('auth');
    } finally {
      if (saved !== undefined) {
        process.env.GOOGLE_GENERATIVE_AI_API_KEY = saved;
      }
    }
  });
});

describe('the vendor’s answers that are not pictures', () => {
  test('a safety block is a refusal that cost a known zero', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(googleSafetyBlocked)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'Something the vendor declines.',
    });
    expect(outcome.kind).toBe('refused');
    if (outcome.kind !== 'refused') {
      return;
    }
    expect(outcome.reason).toBe('INVALID_ARGUMENT');
    expect(outcome.message).toContain('SAFETY');
    expect(outcome.cost).toEqual(NOTHING);
  });

  test.each<[string, () => ImageTransportResponse, string, boolean]>([
    ['a rejected parameter', googlePngRefused, 'invalid_request', false],
    ['a rate limit', googleRateLimited, 'rate_limited', true],
  ])('%s is a failure with a known zero cost', async (_name, answer, errorKind, transient) => {
    const outcome = await generatorOver(new RecordingImageTransport(answer)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
    });
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe(errorKind);
    expect(outcome.transient).toBe(transient);
    expect(outcome.cost).toEqual(NOTHING);
  });

  test('a 2xx with no picture is malformed, and its cost is not known', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(googleNoPicture)).generate({
      provider: 'google',
      model: MODEL,
      prompt: 'A lamp.',
    });
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe('malformed_response');
    expect(outcome.cost).toBeUndefined();
  });
});
