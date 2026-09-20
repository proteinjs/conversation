import { ImageGenerator } from '../../src/image/ImageGenerator';
import { OpenAiImageAdapter } from '../../src/image/OpenAiImageAdapter';
import type { ImageInput } from '../../src/image/ImageGenerationRequest';
import type {
  ImageTransport,
  ImageTransportPart,
  ImageTransportRequest,
  ImageTransportResponse,
} from '../../src/image/ImageProviderAdapter';
import { modelDataFromRows } from '../conversation/fixtureModelData';
import {
  FIXTURE_TOKEN_PRICED_ROW,
  RecordingImageTransport,
  TINY_PNG_BYTES,
  documentedCreditExhausted,
  documentedInvalidAuthentication,
  documentedModerationBlocked,
  documentedOverloaded,
  documentedRateLimit,
  recordedEdit,
  recordedGeneration,
  recordedInputFidelityRejection,
} from './openAiImageFixtures';

/**
 * The OpenAI picture adapter, driven through `ImageGenerator` against a transport double that
 * replays recorded and documented vendor bodies (see `openAiImageFixtures.ts`). Every case reads
 * an OUTCOME — what was on the wire, what came back, what it cost — never which method ran.
 */

const MODEL = 'gpt-image-2.5-sunburst';
const TEST_KEY = 'test-key-never-a-real-one';
const modelData = modelDataFromRows({ standard: { [MODEL]: FIXTURE_TOKEN_PRICED_ROW } });

const generatorOver = (transport: ImageTransport, options: { timeoutMs?: number } = {}) =>
  new ImageGenerator({
    modelData,
    transport,
    adapters: [new OpenAiImageAdapter({ apiKey: TEST_KEY })],
    logLevel: 'error',
    ...options,
  });

const reference = (input: Partial<ImageInput> = {}): ImageInput => ({
  bytes: TINY_PNG_BYTES,
  mimeType: 'image/png',
  ...input,
});

const multipartParts = (request: ImageTransportRequest): ImageTransportPart[] => {
  if (request.body.kind !== 'multipart') {
    throw new Error(`expected a multipart body, saw ${request.body.kind}`);
  }
  return request.body.parts;
};

const fieldValue = (parts: ImageTransportPart[], name: string): string | undefined => {
  const part = parts.find((candidate) => candidate.name === name);
  return part && 'value' in part ? part.value : undefined;
};

describe('an ask with no reference pictures', () => {
  test('goes to /images/generations as JSON and comes back as a priced picture', async () => {
    const transport = new RecordingImageTransport(recordedGeneration);
    const outcome = await generatorOver(transport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'A ceramic coffee mug on a wooden table by a window.',
      size: '1024x1024',
      quality: 'high',
    });

    const sent = transport.only();
    expect(sent.url).toBe('https://api.openai.com/v1/images/generations');
    expect(sent.headers).toEqual({ Authorization: `Bearer ${TEST_KEY}` });
    expect(sent.body).toEqual({
      kind: 'json',
      json: {
        model: MODEL,
        prompt: 'A ceramic coffee mug on a wooden table by a window.',
        n: 1,
        size: '1024x1024',
        quality: 'high',
        output_format: 'png',
      },
    });

    if (outcome.kind !== 'ok') {
      throw new Error(`expected ok, saw ${outcome.kind}`);
    }
    expect(outcome.images).toHaveLength(1);
    expect(Buffer.from(outcome.images[0].bytes).equals(TINY_PNG_BYTES)).toBe(true);
    expect(outcome.images[0].mimeType).toBe('image/png');
    expect(outcome.usage).toEqual({
      textInputTokens: 28,
      imageInputTokens: 0,
      textOutputTokens: 0,
      imageOutputTokens: 1756,
      totalTokens: 1784,
    });
    // 1,756 picture tokens × $30 per 1M = $0.05268; 28 text tokens × $5 per 1M = $0.00014.
    expect(outcome.cost?.outputUsd).toBeCloseTo(0.05268, 10);
    expect(outcome.cost?.textInputUsd).toBeCloseTo(0.00014, 10);
    expect(outcome.cost?.totalUsd).toBeCloseTo(0.05282, 10);
    expect(outcome.provider).toBe('openai');
    expect(outcome.model).toBe(MODEL);
    expect(outcome.vendorRequestId).toBe('req_fixture_generation');
  });

  test('when the vendor reports no usage, the picture is returned and the price is not known', async () => {
    const withoutUsage = (): ImageTransportResponse => {
      const answer = recordedGeneration();
      delete (answer.json as { usage?: unknown }).usage;
      return answer;
    };
    const outcome = await generatorOver(new RecordingImageTransport(withoutUsage)).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'A mug.',
    });

    if (outcome.kind !== 'ok') {
      throw new Error(`expected ok, saw ${outcome.kind}`);
    }
    expect(outcome.images).toHaveLength(1);
    expect(outcome.usage).toBeUndefined();
    expect(outcome.cost).toBeUndefined();
  });

  test('a usage total with no text/picture split is not priced', async () => {
    const withoutSplit = (): ImageTransportResponse => {
      const answer = recordedGeneration();
      delete (answer.json as { usage: { input_tokens_details?: unknown } }).usage.input_tokens_details;
      return answer;
    };
    const outcome = await generatorOver(new RecordingImageTransport(withoutSplit)).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'A mug.',
    });

    if (outcome.kind !== 'ok') {
      throw new Error(`expected ok, saw ${outcome.kind}`);
    }
    expect(outcome.usage?.imageOutputTokens).toBe(1756);
    expect(outcome.cost).toBeUndefined();
  });

  test('a transparent background is sent with png, and is never sent with jpeg', async () => {
    const transport = new RecordingImageTransport(recordedGeneration);
    await generatorOver(transport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'A mark.',
      background: 'transparent',
    });
    const json = transport.only().body.kind === 'json' ? (transport.only().body as { json: object }).json : {};
    expect(json).toMatchObject({ background: 'transparent', output_format: 'png' });

    const jpegTransport = new RecordingImageTransport(recordedGeneration);
    const outcome = await generatorOver(jpegTransport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'A mark.',
      background: 'transparent',
      outputFormat: 'jpeg',
    });
    expect(outcome).toMatchObject({ kind: 'failed', errorKind: 'invalid_request', sent: false, transient: false });
    expect(jpegTransport.requests).toHaveLength(0);
  });
});

describe('an ask with reference pictures', () => {
  test('goes to /images/edits as multipart, says what each reference is for in words, and never sends input_fidelity or a mask to a 2.5 model', async () => {
    const transport = new RecordingImageTransport(recordedEdit);
    const inputs = [
      reference({ name: 'product.jpg', mimeType: 'image/jpeg', role: 'subject', fidelity: 'high' }),
      ...Array.from({ length: 15 }, () => reference({ role: 'style', fidelity: 'low' })),
    ];
    const outcome = await generatorOver(transport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'Place the headphones on a white marble counter.',
      inputs,
      size: '1536x1024',
      quality: 'high',
    });

    const sent = transport.only();
    expect(sent.url).toBe('https://api.openai.com/v1/images/edits');
    const parts = multipartParts(sent);
    expect(parts.map((part) => part.name)).not.toContain('input_fidelity');
    expect(parts.map((part) => part.name)).not.toContain('mask');
    expect(fieldValue(parts, 'model')).toBe(MODEL);
    expect(fieldValue(parts, 'n')).toBe('1');
    expect(fieldValue(parts, 'size')).toBe('1536x1024');
    expect(fieldValue(parts, 'quality')).toBe('high');
    expect(fieldValue(parts, 'output_format')).toBe('png');

    const pictures = parts.filter((part) => part.name === 'image[]');
    expect(pictures).toHaveLength(16);
    expect(pictures[0]).toMatchObject({ filename: 'product.jpg', mimeType: 'image/jpeg' });
    expect(pictures[1]).toMatchObject({ filename: 'reference-2.png', mimeType: 'image/png' });

    const prompt = fieldValue(parts, 'prompt') ?? '';
    expect(prompt.startsWith('Place the headphones on a white marble counter.')).toBe(true);
    expect(prompt).toContain('1. The subject - keep it exactly as it is');
    expect(prompt).toContain('16. A style reference - borrow only its look');

    if (outcome.kind !== 'ok') {
      throw new Error(`expected ok, saw ${outcome.kind}`);
    }
    // 240 text × $5 + 11,552 picture-in × $8 + 1,372 picture-out × $30, per 1M.
    expect(outcome.usage).toMatchObject({ textInputTokens: 240, imageInputTokens: 11552, imageOutputTokens: 1372 });
    expect(outcome.cost?.imageInputUsd).toBeCloseTo(0.092416, 10);
    expect(outcome.cost?.totalUsd).toBeCloseTo(0.134776, 10);
  });

  test('references with no role leave the prompt exactly as written', async () => {
    const transport = new RecordingImageTransport(recordedEdit);
    await generatorOver(transport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'Make it evening.',
      inputs: [reference()],
    });
    expect(fieldValue(multipartParts(transport.only()), 'prompt')).toBe('Make it evening.');
  });

  test('a 17th reference is refused before the wire', async () => {
    const transport = new RecordingImageTransport(recordedEdit);
    const outcome = await generatorOver(transport).generate({
      provider: 'openai',
      model: MODEL,
      prompt: 'Too many.',
      inputs: Array.from({ length: 17 }, () => reference()),
    });

    expect(outcome).toMatchObject({ kind: 'failed', errorKind: 'invalid_request', sent: false, transient: false });
    expect(transport.requests).toHaveLength(0);
  });
});

describe('what the vendor can answer instead of a picture', () => {
  const ask = { provider: 'openai', model: MODEL, prompt: 'A mug.' };

  test('moderation_blocked is a refusal carrying the vendor’s reason — no picture, no price', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(documentedModerationBlocked)).generate(ask);

    expect(outcome).toEqual({
      kind: 'refused',
      reason: 'moderation_blocked',
      stage: 'input',
      categories: ['harassment'],
      vendorRequestId: 'req_fixture_moderation',
      provider: 'openai',
      model: MODEL,
      latencyMs: expect.any(Number),
    });
  });

  test('a rate limit and an overloaded vendor are transient failures', async () => {
    const limited = await generatorOver(new RecordingImageTransport(documentedRateLimit)).generate(ask);
    expect(limited).toMatchObject({ kind: 'failed', errorKind: 'rate_limited', transient: true, statusCode: 429 });

    const overloaded = await generatorOver(new RecordingImageTransport(documentedOverloaded)).generate(ask);
    expect(overloaded).toMatchObject({
      kind: 'failed',
      errorKind: 'provider_error',
      transient: true,
      statusCode: 503,
      code: 'server_is_overloaded',
    });
  });

  test('an empty account answers 429 too, and is never read as a rate limit', async () => {
    const outcome = await generatorOver(new RecordingImageTransport(documentedCreditExhausted)).generate(ask);
    expect(outcome).toMatchObject({
      kind: 'failed',
      errorKind: 'billing',
      transient: false,
      statusCode: 429,
      code: 'credit_balance_exhausted',
    });
  });

  test('a rejected credential and a rejected parameter are not transient', async () => {
    const auth = await generatorOver(new RecordingImageTransport(documentedInvalidAuthentication)).generate(ask);
    expect(auth).toMatchObject({ kind: 'failed', errorKind: 'auth', transient: false, sent: true });

    const rejected = await generatorOver(new RecordingImageTransport(recordedInputFidelityRejection)).generate(ask);
    expect(rejected).toMatchObject({
      kind: 'failed',
      errorKind: 'invalid_request',
      transient: false,
      sent: true,
      statusCode: 400,
      code: 'invalid_input_fidelity_model',
    });
  });

  test('one unreadable picture never loses the others; none readable is a failure', async () => {
    const oneOfTwo = (): ImageTransportResponse => {
      const answer = recordedGeneration();
      const body = answer.json as { data: Array<Record<string, unknown>> };
      body.data = [{ b64_json: '' }, body.data[0]];
      return answer;
    };
    const partial = await generatorOver(new RecordingImageTransport(oneOfTwo)).generate({ ...ask, count: 2 });
    if (partial.kind !== 'ok') {
      throw new Error(`expected ok, saw ${partial.kind}`);
    }
    expect(partial.images).toHaveLength(1);

    const none = (): ImageTransportResponse => ({ status: 200, json: { created: 1, data: [] } });
    const empty = await generatorOver(new RecordingImageTransport(none)).generate(ask);
    expect(empty).toMatchObject({ kind: 'failed', errorKind: 'malformed_response', sent: true });
  });

  test('a dropped connection is a transient failure', async () => {
    const dropped: ImageTransport = {
      post: async () => {
        throw new TypeError('fetch failed');
      },
    };
    const outcome = await generatorOver(dropped).generate(ask);
    expect(outcome).toMatchObject({ kind: 'failed', errorKind: 'network', transient: true, message: 'fetch failed' });
  });
});

describe('the stop', () => {
  const ask = { provider: 'openai', model: MODEL, prompt: 'A mug.' };

  test('a stop mid-request rejects with the stop’s reason and surfaces no picture', async () => {
    const transport = new RecordingImageTransport(recordedGeneration, { holdUntilStopped: true });
    const controller = new AbortController();
    const stopped = new Error('stopped by the person');
    const pending = generatorOver(transport).generate({ ...ask, signal: controller.signal });
    const settled = pending.then(
      (outcome) => ({ outcome }),
      (error) => ({ error })
    );

    await new Promise((resolve) => setImmediate(resolve));
    expect(transport.requests).toHaveLength(1);
    controller.abort(stopped);

    expect(await settled).toEqual({ error: stopped });
  });

  test('a picture that arrives after the stop is dropped, not returned', async () => {
    let answer: (response: ImageTransportResponse) => void = () => undefined;
    const deafToTheSignal: ImageTransport = {
      post: () => new Promise<ImageTransportResponse>((resolve) => (answer = resolve)),
    };
    const controller = new AbortController();
    const stopped = new Error('stopped by the person');
    const settled = generatorOver(deafToTheSignal)
      .generate({ ...ask, signal: controller.signal })
      .then(
        (outcome) => ({ outcome }),
        (error) => ({ error })
      );

    await new Promise((resolve) => setImmediate(resolve));
    controller.abort(stopped);
    answer(recordedGeneration());

    expect(await settled).toEqual({ error: stopped });
  });

  test('an ask already stopped never reaches the wire', async () => {
    const transport = new RecordingImageTransport(recordedGeneration);
    const controller = new AbortController();
    controller.abort(new Error('stopped first'));

    await expect(generatorOver(transport).generate({ ...ask, signal: controller.signal })).rejects.toThrow(
      'stopped first'
    );
    expect(transport.requests).toHaveLength(0);
  });

  test('the generator’s own deadline is a transient failure, not a rejection', async () => {
    const transport = new RecordingImageTransport(recordedGeneration, { holdUntilStopped: true });
    const outcome = await generatorOver(transport, { timeoutMs: 20 }).generate(ask);
    expect(outcome).toMatchObject({ kind: 'failed', errorKind: 'timeout', transient: true, sent: true });
  });
});

describe('the credential', () => {
  const ask = { provider: 'openai', model: MODEL, prompt: 'A mug.' };

  test('rides the Authorization header only — never an outcome', async () => {
    const outcomes = [
      await generatorOver(new RecordingImageTransport(recordedGeneration)).generate(ask),
      await generatorOver(new RecordingImageTransport(documentedInvalidAuthentication)).generate(ask),
    ];
    for (const outcome of outcomes) {
      const readable = JSON.stringify({ ...outcome, images: undefined });
      expect(readable).not.toContain(TEST_KEY);
    }
  });

  test('with no key anywhere, nothing is sent', async () => {
    const saved = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      const transport = new RecordingImageTransport(recordedGeneration);
      const generator = new ImageGenerator({
        modelData,
        transport,
        adapters: [new OpenAiImageAdapter()],
        logLevel: 'error',
      });
      const outcome = await generator.generate(ask);
      expect(outcome).toMatchObject({ kind: 'failed', errorKind: 'auth', sent: false });
      expect(transport.requests).toHaveLength(0);
    } finally {
      if (saved !== undefined) {
        process.env.OPENAI_API_KEY = saved;
      }
    }
  });
});

test('an ask for a provider with no adapter is a wiring error, not an outcome', async () => {
  const generator = generatorOver(new RecordingImageTransport(recordedGeneration));
  await expect(generator.generate({ provider: 'nobody', model: 'x', prompt: 'A mug.' })).rejects.toThrow(
    'no adapter for provider "nobody"'
  );
});
