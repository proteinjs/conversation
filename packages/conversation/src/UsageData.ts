import { TiktokenModel } from 'tiktoken';

export type TokenUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
};

/**
 * Usage data accumulated throughout the lifecycle of a single call to
 * `OpenAi.generateResponse` or `OpenAi.generateStreamingResponse`.
 */
export type UsageData = {
  /** The model used by the assistant */
  model: TiktokenModel;
  /** The token usage of the initial request sent to the assistant */
  initialRequestTokenUsage: TokenUsage;
  /** The total token usage of all requests sent to the assistant (ie. initial request + all subsequent tool call requests) */
  totalTokenUsage: TokenUsage;
  /** The number of requests sent to the assistant */
  totalRequestsToAssistant: number;
  /** The number of times each tool was called by the assistant */
  callsPerTool: { [toolName: string]: number };
  /** The total number of tool calls made by the assistant */
  totalToolCalls: number;
};

type UsageDataAccumulatorParams = {
  model: TiktokenModel;
};

export class UsageDataAccumulator {
  private processedInitialRequest = false;
  public usageData: UsageData;

  constructor({ model }: UsageDataAccumulatorParams) {
    this.usageData = {
      model,
      initialRequestTokenUsage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
      totalTokenUsage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
      totalRequestsToAssistant: 0,
      callsPerTool: {},
      totalToolCalls: 0,
    };
  }

  addTokenUsage(tokenUsage: TokenUsage) {
    this.usageData.totalRequestsToAssistant++;
    if (!this.processedInitialRequest) {
      this.usageData.initialRequestTokenUsage = tokenUsage;
      this.processedInitialRequest = true;
    }
    this.usageData.totalTokenUsage = {
      promptTokens: this.usageData.totalTokenUsage.promptTokens + tokenUsage.promptTokens,
      completionTokens: this.usageData.totalTokenUsage.completionTokens + tokenUsage.completionTokens,
      totalTokens: this.usageData.totalTokenUsage.totalTokens + tokenUsage.totalTokens,
    };
  }

  recordToolCall(toolName: string) {
    if (!this.usageData.callsPerTool[toolName]) {
      this.usageData.callsPerTool[toolName] = 0;
    }

    this.usageData.callsPerTool[toolName]++;
    this.usageData.totalToolCalls++;
  }
}
