import { ImageGenerator } from '../../src/image/ImageGenerator';
import { modelDataFromRows } from '../conversation/fixtureModelData';
import { FIXTURE_TOKEN_PRICED_ROW } from './openAiImageFixtures';

/**
 * LIVE and PAID (the real OpenAI Images API) — run by hand, never by CI. An API key alone does
 * NOT turn it on (CI has one): it also needs `IMAGE_GENERATION_LIVE=1`.
 *
 *   IMAGE_GENERATION_LIVE=1 OPENAI_API_KEY=… npx jest test/image/openai.imageGeneration.live
 *   IMAGE_GENERATION_LIVE_MODEL=<model id> to run another model (default gpt-image-2.5-flare).
 *
 * Two `low` 1024×1024 asks, a few cents at most: one from words, then one that edits the first
 * picture — so both endpoints, the multipart body, the usage split and the pricing math are
 * seen against the vendor itself. The rates are fixture rates; the assertion is that a price
 * comes out of the vendor's own usage, not what the price is.
 */
const isLive = process.env.IMAGE_GENERATION_LIVE === '1' && !!process.env.OPENAI_API_KEY;
const testIfLive = isLive ? test : test.skip;
const LIVE_MODEL = process.env.IMAGE_GENERATION_LIVE_MODEL || 'gpt-image-2.5-flare';
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47]);

testIfLive(
  'the vendor makes a picture from words, then edits it, and both come back priced from its usage',
  async () => {
    const generator = new ImageGenerator({
      modelData: modelDataFromRows({ standard: { [LIVE_MODEL]: FIXTURE_TOKEN_PRICED_ROW } }),
    });

    const made = await generator.generate({
      provider: 'openai',
      model: LIVE_MODEL,
      prompt: 'A plain ceramic coffee mug on a wooden table, soft morning light.',
      size: '1024x1024',
      quality: 'low',
    });
    if (made.kind !== 'ok') {
      throw new Error(`the first ask did not make a picture: ${JSON.stringify(made)}`);
    }
    expect(made.images).toHaveLength(1);
    expect(Buffer.from(made.images[0].bytes).subarray(0, 4).equals(PNG_MAGIC)).toBe(true);
    expect(made.usage?.imageOutputTokens).toBeGreaterThan(0);
    expect(made.usage?.textInputTokens).toBeGreaterThan(0);
    expect(made.cost?.totalUsd).toBeGreaterThan(0);

    const edited = await generator.generate({
      provider: 'openai',
      model: LIVE_MODEL,
      prompt: 'Put the same mug on a white marble counter.',
      inputs: [{ bytes: made.images[0].bytes, mimeType: 'image/png', role: 'subject', fidelity: 'high' }],
      size: '1024x1024',
      quality: 'low',
    });
    if (edited.kind !== 'ok') {
      throw new Error(`the edit did not make a picture: ${JSON.stringify(edited)}`);
    }
    expect(Buffer.from(edited.images[0].bytes).subarray(0, 4).equals(PNG_MAGIC)).toBe(true);
    expect(edited.usage?.imageInputTokens).toBeGreaterThan(0);
    expect(edited.cost?.imageInputUsd).toBeGreaterThan(0);
  },
  10 * 60 * 1000
);
