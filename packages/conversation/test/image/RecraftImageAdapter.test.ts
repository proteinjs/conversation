import { ImageGenerator } from '../../src/image/ImageGenerator';
import { RECRAFT_SVG_MIME_TYPE, RecraftImageAdapter } from '../../src/image/RecraftImageAdapter';
import type { ImageInput } from '../../src/image/ImageGenerationRequest';
import type { ImageTransport, ImageTransportPart, ImageTransportRequest } from '../../src/image/ImageProviderAdapter';
import { modelDataFromRows } from '../conversation/fixtureModelData';
import { RecordingImageTransport, TINY_PNG_BYTES } from './openAiImageFixtures';
import {
  FIXTURE_SVG,
  RECRAFT_UTILITY_ROW,
  RECRAFT_VECTOR_ROW,
  recraftBadKey,
  recraftEmptyAnswer,
  recraftModerated,
  recraftNoUnits,
  recraftUtilityAnswer,
  recraftVectorGeneration,
} from './googleRecraftFixtures';

/**
 * The Recraft adapter — the vector vendor — driven through `ImageGenerator` against a transport
 * double replaying the DOCUMENTED bodies (no key exists yet; the live leg waits on it). Every case
 * reads an OUTCOME: what was on the wire, what came back, what it cost.
 */

const VECTOR_MODEL = 'recraftv4_1_vector';
const VECTORIZE = 'recraft-vectorize';
const TEST_KEY = 'test-recraft-key-never-a-real-one';
const modelData = modelDataFromRows({
  standard: { [VECTOR_MODEL]: RECRAFT_VECTOR_ROW, [VECTORIZE]: RECRAFT_UTILITY_ROW },
});

const generatorOver = (transport: ImageTransport) =>
  new ImageGenerator({
    modelData,
    transport,
    adapters: [new RecraftImageAdapter({ apiKey: TEST_KEY })],
    logLevel: 'error',
  });

const NOTHING = { textInputUsd: 0, imageInputUsd: 0, outputUsd: 0, totalUsd: 0 };

const jsonOf = (request: ImageTransportRequest): Record<string, unknown> => {
  if (request.body.kind !== 'json') {
    throw new Error(`expected a JSON body, saw ${request.body.kind}`);
  }
  return request.body.json;
};
const partsOf = (request: ImageTransportRequest): ImageTransportPart[] => {
  if (request.body.kind !== 'multipart') {
    throw new Error(`expected a multipart body, saw ${request.body.kind}`);
  }
  return request.body.parts;
};
const raster = (input: Partial<ImageInput> = {}): ImageInput => ({
  bytes: TINY_PNG_BYTES,
  mimeType: 'image/png',
  ...input,
});

describe('a mark from words on the vector model', () => {
  test('goes through the server-enforced vector door and comes back as SVG files, priced flat per picture', async () => {
    const transport = new RecordingImageTransport(() => recraftVectorGeneration(3));
    const outcome = await generatorOver(transport).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'Harbor Coffee: an anchor whose flukes are a coffee cup, two colours, no text.',
      count: 3,
      size: '1024x1024',
    });

    const sent = transport.only();
    expect(sent.url).toBe('https://external.api.recraft.ai/v1/images/generations/vector');
    expect(sent.headers).toEqual({ Authorization: `Bearer ${TEST_KEY}` });
    expect(jsonOf(sent)).toEqual({
      prompt: 'Harbor Coffee: an anchor whose flukes are a coffee cup, two colours, no text.',
      model: VECTOR_MODEL,
      n: 3,
      size: '1024x1024',
      response_format: 'b64_json',
    });
    expect(jsonOf(sent).store_info_for_deep_exploration).toBeUndefined();

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(3);
    expect(outcome.images.every((image) => image.mimeType === RECRAFT_SVG_MIME_TYPE)).toBe(true);
    expect(Buffer.from(outcome.images[0].bytes).toString('utf8')).toBe(FIXTURE_SVG);
    expect(outcome.cost).toEqual({ textInputUsd: 0, imageInputUsd: 0, outputUsd: 0.24, totalUsd: 0.24 });
  });

  test('a raster model goes through the plain generations door', async () => {
    const transport = new RecordingImageTransport(() => recraftVectorGeneration(1));
    await generatorOver(transport).generate({ provider: 'recraft', model: 'recraftv4_1', prompt: 'A mark.' });
    expect(transport.only().url).toBe('https://external.api.recraft.ai/v1/images/generations');
  });
});

describe('the utilities', () => {
  test('vectorize sends the one raster as multipart `file` and answers an SVG at the flat price', async () => {
    const transport = new RecordingImageTransport(recraftUtilityAnswer);
    const outcome = await generatorOver(transport).generate({
      provider: 'recraft',
      model: VECTORIZE,
      operation: 'vectorize',
      prompt: '',
      inputs: [raster({ name: 'mark-2.png' })],
    });
    const sent = transport.only();
    expect(sent.url).toBe('https://external.api.recraft.ai/v1/images/vectorize');
    const parts = partsOf(sent);
    expect(parts.find((part) => part.name === 'response_format')).toEqual({
      name: 'response_format',
      value: 'b64_json',
    });
    const file = parts.find((part) => part.name === 'file');
    expect(file && 'bytes' in file ? { mimeType: file.mimeType, filename: file.filename } : undefined).toEqual({
      mimeType: 'image/png',
      filename: 'mark-2.png',
    });

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images).toHaveLength(1);
    expect(outcome.images[0].mimeType).toBe(RECRAFT_SVG_MIME_TYPE);
    expect(outcome.cost?.totalUsd).toBe(0.01);
  });

  test('remove-background goes to its own door and answers the picture, not an SVG', async () => {
    const transport = new RecordingImageTransport(() => recraftUtilityAnswer(TINY_PNG_BYTES.toString('base64')));
    const outcome = await generatorOver(transport).generate({
      provider: 'recraft',
      model: VECTORIZE,
      operation: 'remove-background',
      prompt: '',
      inputs: [raster({ mimeType: 'image/jpeg' })],
    });
    expect(transport.only().url).toBe('https://external.api.recraft.ai/v1/images/removeBackground');
    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') {
      return;
    }
    expect(outcome.images[0].mimeType).toBe('image/png');
  });
});

describe('before the wire', () => {
  test.each([
    [
      'a utility with two pictures',
      { operation: 'vectorize' as const, inputs: [raster(), raster()] },
      'exactly one picture',
    ],
    [
      'a utility with a GIF',
      { operation: 'vectorize' as const, inputs: [raster({ mimeType: 'image/gif' })] },
      'PNG, JPEG or WEBP',
    ],
    ['a seventh picture', { count: 7 }, 'count must be'],
    ['references on a generation', { inputs: [raster()] }, 'no reference pictures'],
    ['an empty prompt', { prompt: '   ' }, 'prompt is empty'],
  ])('%s is refused with nothing sent and a known zero cost', async (_name, fields, words) => {
    const transport = new RecordingImageTransport(() => recraftVectorGeneration(1));
    const outcome = await generatorOver(transport).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'A mark.',
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

  test('no key is an auth failure with nothing sent — the leg that waits on a key', async () => {
    const saved = process.env.RECRAFT_API_KEY;
    delete process.env.RECRAFT_API_KEY;
    try {
      const transport = new RecordingImageTransport(() => recraftVectorGeneration(1));
      const generator = new ImageGenerator({
        modelData,
        transport,
        adapters: [new RecraftImageAdapter()],
        logLevel: 'error',
      });
      const outcome = await generator.generate({ provider: 'recraft', model: VECTOR_MODEL, prompt: 'A mark.' });
      expect(transport.requests).toHaveLength(0);
      expect(outcome.kind).toBe('failed');
      if (outcome.kind !== 'failed') {
        return;
      }
      expect(outcome.errorKind).toBe('auth');
      expect(outcome.cost).toEqual(NOTHING);
    } finally {
      if (saved !== undefined) {
        process.env.RECRAFT_API_KEY = saved;
      }
    }
  });
});

describe('the vendor’s answers that are not pictures', () => {
  test('an empty prepaid balance is a billing failure, never a rate limit', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(recraftNoUnits)).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'A mark.',
    });
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe('billing');
    expect(outcome.transient).toBe(false);
    expect(outcome.cost).toEqual(NOTHING);
  });

  test('a bad key is an auth failure', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(recraftBadKey)).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'A mark.',
    });
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe('auth');
  });

  test('a moderated prompt is a refusal that cost a known zero', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(recraftModerated)).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'A mark.',
    });
    expect(outcome.kind).toBe('refused');
    if (outcome.kind !== 'refused') {
      return;
    }
    expect(outcome.reason).toBe('moderation');
    expect(outcome.cost).toEqual(NOTHING);
  });

  test('a 2xx with no picture is malformed and its cost is not known', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(recraftEmptyAnswer)).generate({
      provider: 'recraft',
      model: VECTOR_MODEL,
      prompt: 'A mark.',
    });
    expect(outcome.kind).toBe('failed');
    if (outcome.kind !== 'failed') {
      return;
    }
    expect(outcome.errorKind).toBe('malformed_response');
    expect(outcome.cost).toBeUndefined();
  });
});
