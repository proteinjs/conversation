import * as fs from 'fs';
import * as path from 'path';
import { ImageGenerator } from '../../src/image/ImageGenerator';
import { RECRAFT_SVG_MIME_TYPE } from '../../src/image/RecraftImageAdapter';
import type { ModelApiCost } from '../../src/UsageData';
import { modelDataFromRows } from '../conversation/fixtureModelData';
import { RECRAFT_UTILITY_ROW, RECRAFT_VECTOR_ROW } from './googleRecraftFixtures';

/**
 * LIVE and PAID (the real Recraft API) — run by hand, never by CI. A key alone does NOT turn it
 * on: it also needs `IMAGE_GENERATION_LIVE=1`.
 *
 *   IMAGE_GENERATION_LIVE=1 RECRAFT_API_KEY=… npx jest test/image/recraft.imageGeneration.live
 *   IMAGE_GENERATION_LIVE_OUT=<dir> also writes what came back there (the pictures and a summary).
 *
 * About $0.11 of prepaid units: one vector mark ($0.08 list), one fast raster ($0.007), its
 * background removed ($0.01) and the raster traced ($0.01). The rates are fixture rates; the
 * assertions are about the wire: the vector door answers SVG path data, and every picture is
 * labelled as what its bytes are.
 */
const isLive = process.env.IMAGE_GENERATION_LIVE === '1' && !!process.env.RECRAFT_API_KEY;
const testIfLive = isLive ? test : test.skip;
const OUT = process.env.IMAGE_GENERATION_LIVE_OUT;

const FAST_RASTER = 'recraftv4_1_flash';
const FAST_RASTER_ROW: ModelApiCost = { inputUsdPer1M: 0, outputUsdPer1M: 0, perImageUsd: 0.007 };
const modelData = modelDataFromRows({
  standard: {
    recraftv4_1_vector: RECRAFT_VECTOR_ROW,
    [FAST_RASTER]: FAST_RASTER_ROW,
    'recraft-vectorize': RECRAFT_UTILITY_ROW,
    'recraft-remove-background': RECRAFT_UTILITY_ROW,
  },
});

/** What the bytes are, by their signature — never by what the answer was labelled. */
const signatureOf = (bytes: Uint8Array): string => {
  const head = Buffer.from(bytes.subarray(0, 16));
  if (head.subarray(0, 4).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47]))) {
    return 'image/png';
  }
  if (head.subarray(0, 4).toString('latin1') === 'RIFF' && head.subarray(8, 12).toString('latin1') === 'WEBP') {
    return 'image/webp';
  }
  if (head.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))) {
    return 'image/jpeg';
  }
  return /<svg/.test(Buffer.from(bytes.subarray(0, 512)).toString('utf8')) ? RECRAFT_SVG_MIME_TYPE : 'unknown';
};

const keep = (name: string, bytes: Uint8Array | undefined, summary: Record<string, unknown>) => {
  if (!OUT) {
    return;
  }
  fs.mkdirSync(OUT, { recursive: true });
  if (bytes) {
    fs.writeFileSync(path.join(OUT, name), bytes);
  }
  fs.writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(summary, null, 2));
};

testIfLive(
  'a mark from words on the vector model comes back as SVG path data, priced flat per picture',
  async () => {
    const generator = new ImageGenerator({ modelData });
    const made = await generator.generate({
      provider: 'recraft',
      model: 'recraftv4_1_vector',
      prompt:
        'A flat vector logo mark: a lighthouse whose beam is a single leaf. Two colours, deep teal and white. No text.',
      size: '1024x1024',
    });
    if (made.kind !== 'ok') {
      throw new Error(`the vector ask did not make a picture: ${JSON.stringify(made)}`);
    }
    const svg = Buffer.from(made.images[0].bytes).toString('utf8');
    const paths = svg.match(/<path\b[^>]*\sd="M[^"]+"/g) ?? [];
    keep('vector-mark.svg', made.images[0].bytes, {
      kind: made.kind,
      images: made.images.length,
      mimeType: made.images[0].mimeType,
      bytes: made.images[0].bytes.length,
      paths: paths.length,
      head: svg.slice(0, 120),
      cost: made.cost,
      latencyMs: made.latencyMs,
      vendorRequestId: made.vendorRequestId ?? null,
    });
    expect(made.images).toHaveLength(1);
    expect(made.images[0].mimeType).toBe(RECRAFT_SVG_MIME_TYPE);
    expect(signatureOf(made.images[0].bytes)).toBe(RECRAFT_SVG_MIME_TYPE);
    expect(paths.length).toBeGreaterThan(0);
    expect(made.cost?.totalUsd).toBeCloseTo(RECRAFT_VECTOR_ROW.perImageUsd ?? 0, 6);
    expect(made.vendorRequestId).toEqual(expect.any(String));
  },
  5 * 60 * 1000
);

testIfLive(
  'a raster, its background removed and the raster traced — each labelled as what its bytes are',
  async () => {
    const generator = new ImageGenerator({ modelData });
    const raster = await generator.generate({
      provider: 'recraft',
      model: FAST_RASTER,
      prompt: 'A ceramic coffee mug on a wooden table, soft morning light, product photograph.',
      size: '1024x1024',
    });
    if (raster.kind !== 'ok') {
      throw new Error(`the raster ask did not make a picture: ${JSON.stringify(raster)}`);
    }
    keep(`raster.${raster.images[0].mimeType.split('/')[1]}`, raster.images[0].bytes, {
      mimeType: raster.images[0].mimeType,
      signature: signatureOf(raster.images[0].bytes),
      bytes: raster.images[0].bytes.length,
      cost: raster.cost,
      vendorRequestId: raster.vendorRequestId ?? null,
    });
    expect(signatureOf(raster.images[0].bytes)).toBe(raster.images[0].mimeType);

    const cutout = await generator.generate({
      provider: 'recraft',
      model: 'recraft-remove-background',
      operation: 'remove-background',
      prompt: '',
      inputs: [{ bytes: raster.images[0].bytes, mimeType: raster.images[0].mimeType }],
    });
    if (cutout.kind !== 'ok') {
      throw new Error(`the background removal did not answer a picture: ${JSON.stringify(cutout)}`);
    }
    keep(`cutout.${cutout.images[0].mimeType.split('/')[1]}`, cutout.images[0].bytes, {
      mimeType: cutout.images[0].mimeType,
      signature: signatureOf(cutout.images[0].bytes),
      bytes: cutout.images[0].bytes.length,
      cost: cutout.cost,
    });
    expect(signatureOf(cutout.images[0].bytes)).toBe(cutout.images[0].mimeType);

    const traced = await generator.generate({
      provider: 'recraft',
      model: 'recraft-vectorize',
      operation: 'vectorize',
      prompt: '',
      inputs: [{ bytes: raster.images[0].bytes, mimeType: raster.images[0].mimeType }],
    });
    if (traced.kind !== 'ok') {
      throw new Error(`the tracing did not answer an SVG: ${JSON.stringify(traced)}`);
    }
    keep('traced.svg', traced.images[0].bytes, {
      mimeType: traced.images[0].mimeType,
      signature: signatureOf(traced.images[0].bytes),
      bytes: traced.images[0].bytes.length,
      cost: traced.cost,
    });
    expect(traced.images[0].mimeType).toBe(RECRAFT_SVG_MIME_TYPE);
    expect(signatureOf(traced.images[0].bytes)).toBe(RECRAFT_SVG_MIME_TYPE);
  },
  5 * 60 * 1000
);
