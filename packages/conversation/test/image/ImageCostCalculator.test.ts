import { ImageCostCalculator } from '../../src/image/ImageCostCalculator';
import type { ModelApiCost } from '../../src/UsageData';
import { FIXTURE_STANDARD_ROW, modelDataFromRows } from '../conversation/fixtureModelData';
import { FIXTURE_TOKEN_PRICED_ROW } from './openAiImageFixtures';

/**
 * The picture price: the vendor's own usage × the row's rates, or `undefined` — never a guess.
 * Fixture rows only; what any real model costs is the embedding platform's data.
 */

const FLAT_ROW: ModelApiCost = { inputUsdPer1M: 0, outputUsdPer1M: 0, perImageUsd: 0.08 };

const calculator = new ImageCostCalculator(
  modelDataFromRows({
    standard: {
      'by-the-token': FIXTURE_TOKEN_PRICED_ROW,
      'by-the-token-no-picture-input-rate': { inputUsdPer1M: 5, outputUsdPer1M: 0, imageOutputUsdPer1M: 30 },
      'by-the-token-no-cached-rates': {
        inputUsdPer1M: 5,
        outputUsdPer1M: 0,
        imageInputUsdPer1M: 8,
        imageOutputUsdPer1M: 30,
      },
      'by-the-picture': FLAT_ROW,
      'a-text-model': FIXTURE_STANDARD_ROW,
    },
  })
);

describe('a row billed by the token', () => {
  test('prices text in, pictures in and pictures out at their own rates', () => {
    const cost = calculator.cost({
      model: 'by-the-token',
      usage: { textInputTokens: 240, imageInputTokens: 11552, imageOutputTokens: 1372 },
      imageCount: 1,
    });

    expect(cost?.textInputUsd).toBeCloseTo(0.0012, 10);
    expect(cost?.imageInputUsd).toBeCloseTo(0.092416, 10);
    expect(cost?.outputUsd).toBeCloseTo(0.04116, 10);
    expect(cost?.totalUsd).toBeCloseTo(0.134776, 10);
  });

  test('the price follows the usage, not the number of pictures', () => {
    const usage = { textInputTokens: 28, imageInputTokens: 0, imageOutputTokens: 1756 };
    const one = calculator.cost({ model: 'by-the-token', usage, imageCount: 1 });
    const three = calculator.cost({ model: 'by-the-token', usage, imageCount: 3 });
    expect(one?.totalUsd).toBeCloseTo(0.05282, 10);
    expect(three?.totalUsd).toBeCloseTo(0.05282, 10);
  });

  test('cached input bills at the cached rate of its own class', () => {
    const cost = calculator.cost({
      model: 'by-the-token',
      usage: {
        textInputTokens: 1000,
        cachedTextInputTokens: 400,
        imageInputTokens: 10000,
        cachedImageInputTokens: 2500,
        imageOutputTokens: 0,
      },
      imageCount: 1,
    });
    // text: 600 × $5 + 400 × $1.25 = 3,500; pictures: 7,500 × $8 + 2,500 × $2 = 65,000 (per 1M).
    expect(cost?.textInputUsd).toBeCloseTo(0.0035, 10);
    expect(cost?.imageInputUsd).toBeCloseTo(0.065, 10);
  });

  test('a cached count with no cached rate bills at the full rate of its class', () => {
    const cost = calculator.cost({
      model: 'by-the-token-no-cached-rates',
      usage: {
        textInputTokens: 1000,
        cachedTextInputTokens: 400,
        imageInputTokens: 10000,
        cachedImageInputTokens: 2500,
        imageOutputTokens: 0,
      },
      imageCount: 1,
    });
    expect(cost?.textInputUsd).toBeCloseTo(0.005, 10);
    expect(cost?.imageInputUsd).toBeCloseTo(0.08, 10);
  });

  test('no usage, or a usage without the split, has no price', () => {
    expect(calculator.cost({ model: 'by-the-token', imageCount: 1 })).toBeUndefined();
    expect(
      calculator.cost({ model: 'by-the-token', usage: { imageOutputTokens: 1756 }, imageCount: 1 })
    ).toBeUndefined();
    expect(
      calculator.cost({ model: 'by-the-token', usage: { textInputTokens: 28, imageInputTokens: 0 }, imageCount: 1 })
    ).toBeUndefined();
  });

  test('picture input with no rate on the row has no price; an ask with no picture input is still priced', () => {
    const usage = { textInputTokens: 28, imageInputTokens: 11552, imageOutputTokens: 1756 };
    expect(calculator.cost({ model: 'by-the-token-no-picture-input-rate', usage, imageCount: 1 })).toBeUndefined();

    const textOnly = { textInputTokens: 28, imageInputTokens: 0, imageOutputTokens: 1756 };
    const cost = calculator.cost({ model: 'by-the-token-no-picture-input-rate', usage: textOnly, imageCount: 1 });
    expect(cost?.totalUsd).toBeCloseTo(0.05282, 10);
  });
});

describe('a row billed by the picture', () => {
  test('prices the pictures actually made, whatever the usage says', () => {
    expect(calculator.cost({ model: 'by-the-picture', imageCount: 3 })?.totalUsd).toBeCloseTo(0.24, 10);
    expect(
      calculator.cost({ model: 'by-the-picture', usage: { imageOutputTokens: 9999 }, imageCount: 2 })?.totalUsd
    ).toBeCloseTo(0.16, 10);
    expect(calculator.cost({ model: 'by-the-picture', imageCount: 0 })?.totalUsd).toBe(0);
  });
});

describe('no price to give', () => {
  test('a model with no row has no price', () => {
    expect(
      calculator.cost({
        model: 'never-heard-of-it',
        usage: { textInputTokens: 28, imageInputTokens: 0, imageOutputTokens: 1756 },
        imageCount: 1,
      })
    ).toBeUndefined();
  });

  test('a row with no picture arm never prices a picture from its text rates', () => {
    expect(
      calculator.cost({
        model: 'a-text-model',
        usage: { textInputTokens: 28, imageInputTokens: 0, imageOutputTokens: 1756 },
        imageCount: 1,
      })
    ).toBeUndefined();
  });
});
