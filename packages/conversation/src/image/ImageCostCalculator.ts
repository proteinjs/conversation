import type { ModelDataResolver } from '../ModelData';
import type { ModelApiCost } from '../UsageData';
import type { ImageCostUsd, ImageUsage } from './ImageGenerationOutcome';

const TOKENS_PER_1M = 1_000_000;

/**
 * What one ask for pictures cost: the vendor's own usage × the model's rates. The rates arrive
 * through the same `ModelDataResolver` seam the text path prices from — this class owns only the
 * math. It answers `undefined` whenever the price is NOT KNOWN — no row for the model, no usage
 * from the vendor, a usage without the split the rates need, or usage in a class the row has no
 * rate for — and never fills the gap with an estimate, a list price or a zero.
 *
 * Two kinds of row:
 * - billed by the token (the row carries `imageOutputUsdPer1M`): text input at `inputUsdPer1M`,
 *   picture input at `imageInputUsdPer1M`, pictures made at `imageOutputUsdPer1M`, any text
 *   output at `outputUsdPer1M`; a cached count bills at its cached rate, or at the full rate of
 *   its class where the row has none;
 * - billed by the picture (the row carries `perImageUsd` and no token rate for pictures):
 *   `perImageUsd` × the pictures actually made.
 */
export class ImageCostCalculator {
  constructor(private readonly modelData: ModelDataResolver) {}

  cost(ask: { model: string; usage?: ImageUsage; imageCount: number }): ImageCostUsd | undefined {
    const rates = this.modelData.pricing(ask.model, 'standard');
    if (!rates) {
      return undefined;
    }
    if (typeof rates.imageOutputUsdPer1M === 'number') {
      return this.byToken(rates, rates.imageOutputUsdPer1M, ask.usage);
    }
    if (typeof rates.perImageUsd === 'number') {
      return this.byPicture(rates.perImageUsd, ask.imageCount);
    }
    return undefined;
  }

  private byToken(
    rates: ModelApiCost,
    imageOutputRate: number,
    usage: ImageUsage | undefined
  ): ImageCostUsd | undefined {
    const textInput = usage?.textInputTokens;
    const imageInput = usage?.imageInputTokens;
    const imageOutput = usage?.imageOutputTokens;
    // The vendor must have said how the input splits and what the pictures took; a bare total
    // cannot be priced, because text and pictures bill at different rates.
    if (textInput === undefined || imageInput === undefined || imageOutput === undefined) {
      return undefined;
    }
    if (imageInput > 0 && typeof rates.imageInputUsdPer1M !== 'number') {
      return undefined;
    }
    const imageInputRate = rates.imageInputUsdPer1M ?? 0;
    const cachedText = Math.min(usage?.cachedTextInputTokens ?? 0, textInput);
    const cachedImage = Math.min(usage?.cachedImageInputTokens ?? 0, imageInput);
    const cachedTextRate = rates.cachedInputUsdPer1M ?? rates.inputUsdPer1M;
    const cachedImageRate = rates.cachedImageInputUsdPer1M ?? imageInputRate;

    const textInputUsd = ((textInput - cachedText) * rates.inputUsdPer1M + cachedText * cachedTextRate) / TOKENS_PER_1M;
    const imageInputUsd = ((imageInput - cachedImage) * imageInputRate + cachedImage * cachedImageRate) / TOKENS_PER_1M;
    const outputUsd =
      (imageOutput * imageOutputRate + (usage?.textOutputTokens ?? 0) * rates.outputUsdPer1M) / TOKENS_PER_1M;
    return { textInputUsd, imageInputUsd, outputUsd, totalUsd: textInputUsd + imageInputUsd + outputUsd };
  }

  private byPicture(perImageUsd: number, imageCount: number): ImageCostUsd | undefined {
    if (!Number.isFinite(imageCount) || imageCount < 0) {
      return undefined;
    }
    const outputUsd = perImageUsd * imageCount;
    return { textInputUsd: 0, imageInputUsd: 0, outputUsd, totalUsd: outputUsd };
  }
}
