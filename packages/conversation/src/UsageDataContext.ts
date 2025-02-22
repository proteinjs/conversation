import { AsyncLocalStorage } from 'async_hooks';
import { UsageDataAccumulator } from './UsageData';
import { TiktokenModel } from 'tiktoken';

interface ContextData {
  data: Map<TiktokenModel, UsageDataAccumulator>;
}

export class UsageDataAccumulatorContext {
  private static readonly storage = new AsyncLocalStorage<ContextData>();

  /**
   * Returns the `UsageDataAccumulator` stored in context depending on the `model` param provided.
   * If no `UsageDataAccumulator` is stored, returns `undefined`.
   */
  getUsageDataAccumulator(model: TiktokenModel): UsageDataAccumulator | undefined {
    const context = UsageDataAccumulatorContext.storage.getStore();

    if (!context || !context.data.has(model)) {
      return;
    }

    return context.data.get(model);
  }

  runInContext<T>(usageDataAccumulator: UsageDataAccumulator, fn: () => Promise<T>): Promise<T> {
    const initialContext: ContextData = {
      data: new Map([[usageDataAccumulator.usageData.model, usageDataAccumulator]]),
    };

    return new Promise<T>((resolve, reject) => {
      UsageDataAccumulatorContext.storage.run(initialContext, async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    });
  }
}
