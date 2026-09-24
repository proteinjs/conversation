import type { ModelApiCost } from '../../src/UsageData';
import type { ImageTransportResponse } from '../../src/image/ImageProviderAdapter';
import { TINY_PNG_BASE64 } from './openAiImageFixtures';

/**
 * Documented and recorded bodies for the Google and Recraft picture suites — no suite here calls
 * a vendor.
 *
 * GOOGLE — the Interactions API as documented at
 * https://ai.google.dev/gemini-api/docs/image-generation (read 2026-09-22): the picture under
 * `interaction.output_image { mime_type, data }`, the follow-up handle under `interaction.id`, the
 * count of what was used under `usage` by modality. The token counts are the probe's RECORDED room
 * ask of 2026-09-16 on `gemini-3.1-flash-image` (as measured: 356 in, 1,120 picture
 * out, the rest of the output text — $0.0686 at the pricing page's rates); the bytes are a 1×1 PNG
 * stand-in and the id is made up.
 *
 * RECRAFT — https://www.recraft.ai/docs/api-reference/usage (read 2026-09-22): generation answers
 * `{ data: [{ b64_json }], created }`, a `_vector` model's `b64_json` being SVG text; the utilities
 * answer `{ image: { b64_json } }`. No key has been provisioned (the probe is deferred), so these
 * are DOCUMENTED bodies, never recorded ones.
 */

/** Rates shaped like the Google row: text and pictures in at one rate, words out, pictures out. */
export const GOOGLE_FIXTURE_ROW: ModelApiCost = {
  inputUsdPer1M: 0.5,
  outputUsdPer1M: 3,
  imageInputUsdPer1M: 0.5,
  imageOutputUsdPer1M: 60,
};

/** The flat Recraft rows: a vector picture, and the two utilities. */
export const RECRAFT_VECTOR_ROW: ModelApiCost = { inputUsdPer1M: 0, outputUsdPer1M: 0, perImageUsd: 0.08 };
export const RECRAFT_UTILITY_ROW: ModelApiCost = { inputUsdPer1M: 0, outputUsdPer1M: 0, perImageUsd: 0.01 };

export const GOOGLE_USAGE_RECORDED = {
  total_input_tokens: 356,
  input_tokens_by_modality: [
    { modality: 'TEXT', tokens: 36 },
    { modality: 'IMAGE', tokens: 320 },
  ],
  total_output_tokens: 1527,
  output_tokens_by_modality: [
    { modality: 'IMAGE', tokens: 1120 },
    { modality: 'TEXT', tokens: 407 },
  ],
  total_thought_tokens: 380,
};

/** The room ask's answer: one JPEG picture, the handle, the usage. */
export const googleInteraction = (id = 'interaction_fixture_1'): ImageTransportResponse => ({
  status: 200,
  requestId: 'req_google_fixture',
  json: {
    interaction: {
      id,
      model: 'gemini-3.1-flash-image',
      output_image: { mime_type: 'image/jpeg', data: TINY_PNG_BASE64 },
      steps: [
        { type: 'model_generation', content: [{ type: 'image', mime_type: 'image/jpeg', data: TINY_PNG_BASE64 }] },
      ],
    },
    usage: GOOGLE_USAGE_RECORDED,
  },
});

/** The same answer with the picture only inside a step's content. */
export const googleInteractionInSteps = (): ImageTransportResponse => ({
  status: 200,
  json: {
    interaction: {
      id: 'interaction_fixture_steps',
      steps: [
        { type: 'model_generation', content: [{ type: 'text', text: 'Here is the room.' }] },
        { type: 'model_generation', content: [{ type: 'image', mime_type: 'image/jpeg', data: TINY_PNG_BASE64 }] },
      ],
    },
    usage: GOOGLE_USAGE_RECORDED,
  },
});

/** A 2xx with no picture and no usage. */
export const googleNoPicture = (): ImageTransportResponse => ({
  status: 200,
  json: { interaction: { id: 'interaction_fixture_empty', steps: [] } },
});

/** The usage with the by-modality split missing — a total alone cannot price a picture. */
export const googleUsageWithoutSplit = (): ImageTransportResponse => ({
  status: 200,
  json: {
    interaction: { id: 'x', output_image: { mime_type: 'image/jpeg', data: TINY_PNG_BASE64 } },
    usage: { total_input_tokens: 356, total_output_tokens: 1527 },
  },
});

/** DOCUMENTED — the vendor's safety block, a 400 whose status names it. */
export const googleSafetyBlocked = (): ImageTransportResponse => ({
  status: 400,
  json: {
    error: {
      code: 400,
      message: 'The response was blocked: SAFETY. The prompt could not be completed.',
      status: 'INVALID_ARGUMENT',
    },
  },
});

/** DOCUMENTED — the 400 measured 2026-09-16 when a PNG was asked for. */
export const googlePngRefused = (): ImageTransportResponse => ({
  status: 400,
  json: {
    error: {
      code: 400,
      message: "Unsupported response mime type 'image/png'. Supported values: 'image/jpeg'",
      status: 'INVALID_ARGUMENT',
    },
  },
});

export const googleRateLimited = (): ImageTransportResponse => ({
  status: 429,
  json: {
    error: { code: 429, message: 'Resource has been exhausted (e.g. check quota).', status: 'RESOURCE_EXHAUSTED' },
  },
});

/** A small real SVG: a mark with two paths, the shape of a logo file. */
export const FIXTURE_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M32 4a28 28 0 1 0 0 56a28 28 0 1 0 0-56z" fill="#0b3d5c"/><path d="M20 30h24v8H20z" fill="#f4e8d1"/></svg>';
export const FIXTURE_SVG_BASE64 = Buffer.from(FIXTURE_SVG, 'utf8').toString('base64');

/** DOCUMENTED — `POST /images/generations/vector`, three vector marks. */
export const recraftVectorGeneration = (n = 3): ImageTransportResponse => ({
  status: 200,
  requestId: 'req_recraft_fixture_vector',
  json: { created: 1789609371, data: Array.from({ length: n }, () => ({ b64_json: FIXTURE_SVG_BASE64 })) },
});

/** DOCUMENTED — `POST /images/vectorize` and `/images/removeBackground` answer one picture. */
export const recraftUtilityAnswer = (b64 = FIXTURE_SVG_BASE64): ImageTransportResponse => ({
  status: 200,
  json: { image: { b64_json: b64 } },
});

export const recraftEmptyAnswer = (): ImageTransportResponse => ({ status: 200, json: { data: [] } });

/** DOCUMENTED — an empty prepaid balance answers 402 (units are bought up front). */
export const recraftNoUnits = (): ImageTransportResponse => ({
  status: 402,
  json: { code: 'insufficient_balance', message: 'Not enough API units. Top up your balance.' },
});

export const recraftBadKey = (): ImageTransportResponse => ({
  status: 401,
  json: { code: 'unauthorized', message: 'Invalid API token' },
});

/** RECORDED (2026-09-24) — past five requests in a second: 429 with no Retry-After, and nothing billed. */
export const recraftRateLimited = (): ImageTransportResponse => ({
  status: 429,
  requestId: 'req_recraft_fixture_429',
  json: { code: 'rate_limit_exceeded', message: 'Rate limit exceeded' },
});

/** RECORDED (2026-09-24) — a raster model at the vector door: refused before anything is made. */
export const recraftRasterAtVectorDoor = (): ImageTransportResponse => ({
  status: 400,
  json: { code: 'invalid_image_type', message: "Style 'any' is not a vector style" },
});

export const recraftModerated = (): ImageTransportResponse => ({
  status: 400,
  json: { code: 'moderation', message: 'The prompt was flagged by the content moderation system.' },
});
