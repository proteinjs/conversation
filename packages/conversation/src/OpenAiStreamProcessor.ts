import { ChatCompletionMessageToolCall, ChatCompletionChunk } from 'openai/resources/chat';
import { LogLevel, Logger } from '@proteinjs/logger';
import { Stream } from 'openai/streaming';
import { Readable, Transform, TransformCallback, PassThrough } from 'stream';
import { UsageData, UsageDataAccumulator } from './UsageData';

export interface AssistantResponseStreamChunk {
  content?: string;
  finishReason?: string;
}

/**
 * Processes streaming responses from OpenAI's `ChatCompletions` api.
 *   - When a tool call is received, it delegates processing to `onToolCalls`; this can happen recursively
 *   - When a response to the user is received, it writes to `outputStream`
 */
export class OpenAiStreamProcessor {
  private logger: Logger;
  private accumulatedToolCalls: Partial<ChatCompletionMessageToolCall>[] = [];
  private toolCallsExecuted = 0;
  private currentToolCall: Partial<ChatCompletionMessageToolCall> | null = null;
  private inputStream: Readable;
  private controlStream: Transform;
  private outputStream: Readable;
  private outputStreamTerminated = false;

  constructor(
    inputStream: Stream<ChatCompletionChunk>,
    private onToolCalls: (
      toolCalls: ChatCompletionMessageToolCall[],
      currentFunctionCalls: number
    ) => Promise<Readable>,
    private usageDataAccumulator: UsageDataAccumulator,
    logLevel?: LogLevel,
    private abortSignal?: AbortSignal,
    private onUsageData?: (usageData: UsageData) => Promise<void>
  ) {
    this.logger = new Logger({ name: this.constructor.name, logLevel });
    this.inputStream = Readable.from(inputStream);
    this.controlStream = this.createControlStream();
    this.outputStream = new PassThrough({ objectMode: true });
    this.inputStream.pipe(this.controlStream);
  }

  /**
   * @returns a `Readable` stream, in object mode, that will receive the assistant's text response to the user.
   *          The object chunks written to the stream will be of type `AssistantResponseStreamChunk`.
   */
  getOutputStream(): Readable {
    return this.outputStream;
  }

  /**
   * @returns a `Transform` that parses the input stream and delegates to tool calls or writes a text response to the user
   */
  private createControlStream(): Transform {
    let finishedProcessingToolCallStream = false;
    return new Transform({
      objectMode: true,
      transform: (chunk: ChatCompletionChunk, encoding: string, callback: TransformCallback) => {
        try {
          if (this.outputStream.destroyed) {
            this.logger.warn({ message: `Destroying input and control streams since output stream is destroyed` });
            this.inputStream.destroy();
            this.controlStream.destroy();
            return;
          }

          if (!chunk || !chunk.choices) {
            throw new Error(`Received invalid chunk:\n${JSON.stringify(chunk, null, 2)}`);
          } else if (chunk.choices[0]?.delta?.content) {
            this.outputStream.push({ content: chunk.choices[0].delta.content } as AssistantResponseStreamChunk);
          } else if (chunk.choices[0]?.delta?.tool_calls) {
            this.handleToolCallDelta(chunk.choices[0].delta.tool_calls);
          } else if (chunk.choices[0]?.finish_reason === 'tool_calls') {
            finishedProcessingToolCallStream = true;
          } else if (chunk.choices[0]?.finish_reason === 'stop') {
            this.outputStream.push({ finishReason: 'stop' } as AssistantResponseStreamChunk);
            this.outputStream.push(null);
            this.outputStreamTerminated = true;
          } else if (chunk.choices[0]?.finish_reason === 'length') {
            this.logger.info({ message: `The maximum number of output tokens was reached` });
            this.outputStream.push({ finishReason: 'length' } as AssistantResponseStreamChunk);
            this.outputStream.push(null);
            this.outputStreamTerminated = true;
          } else if (chunk.choices[0]?.finish_reason === 'content_filter') {
            this.logger.warn({ message: `Content was omitted due to a flag from OpenAI's content filters` });
            this.outputStream.push({ finishReason: 'content_filter' } as AssistantResponseStreamChunk);
            this.outputStream.push(null);
            this.outputStreamTerminated = true;
          } else if (chunk.usage) {
            this.usageDataAccumulator.addTokenUsage({
              promptTokens: chunk.usage.prompt_tokens,
              cachedPromptTokens: chunk.usage.prompt_tokens_details?.cached_tokens ?? 0,
              completionTokens: chunk.usage.completion_tokens,
              totalTokens: chunk.usage.total_tokens,
            });
            if (finishedProcessingToolCallStream) {
              this.handleToolCalls();
            } else if (this.outputStreamTerminated) {
              if (this.onUsageData) {
                this.onUsageData(this.usageDataAccumulator.usageData);
              }
              this.destroyStreams();
            }
          }
          callback();
        } catch (error: any) {
          this.logger.error({ message: 'Error tranforming chunk', error });
          this.destroyStreams(error);
        }
      },
    });
  }

  /**
   * Destroy all streams used by `OpenAiStreamProcessor`
   */
  private destroyStreams(error?: Error) {
    this.inputStream.destroy();
    this.controlStream.destroy();
    if (error) {
      this.outputStream.emit('error', error);
    }
    this.outputStream.destroy();
  }

  /**
   * Accumulates tool call deltas into complete tool calls.
   * @param toolCallDeltas `ChatCompletionChunk.Choice.Delta.ToolCall` objects that contain part of a complete `ChatCompletionMessageToolCall`
   */
  private handleToolCallDelta(toolCallDeltas: ChatCompletionChunk.Choice.Delta.ToolCall[]) {
    for (const delta of toolCallDeltas) {
      if (delta.id) {
        // Start of a new tool call
        if (this.currentToolCall) {
          this.accumulatedToolCalls.push(this.currentToolCall);
        }
        this.currentToolCall = {
          id: delta.id,
          type: delta.type || 'function',
          function: {
            name: delta.function?.name || '',
            arguments: delta.function?.arguments || '',
          },
        };
      } else {
        // Continue building the current tool call
        if (delta.function?.name) {
          this.currentToolCall!.function!.name += delta.function.name;
        }
        if (delta.function?.arguments) {
          this.currentToolCall!.function!.arguments += delta.function.arguments;
        }
      }
    }
  }

  /**
   * Delegates `ChatCompletionMessageToolCall`s to `onToolCalls`.
   *   - Manages refreshing the `inputStream` and `controlStream`
   *   - Manages tool call state (such as keeping track of the number of tool calls made)
   */
  private async handleToolCalls() {
    if (this.currentToolCall) {
      this.accumulatedToolCalls.push(this.currentToolCall);
      this.currentToolCall = null;
    }

    const completedToolCalls = this.accumulatedToolCalls.filter(
      (tc): tc is ChatCompletionMessageToolCall =>
        tc.id !== undefined && tc.function !== undefined && tc.type !== undefined
    );

    this.accumulatedToolCalls = [];
    this.inputStream.destroy();
    this.controlStream.destroy();
    this.controlStream = this.createControlStream();

    try {
      this.inputStream = await this.onToolCalls(completedToolCalls, this.toolCallsExecuted);
      this.inputStream.on('error', (error) => this.destroyStreams(error));
      this.inputStream.pipe(this.controlStream);
      this.toolCallsExecuted += completedToolCalls.length;
    } catch (error: any) {
      if (!this.abortSignal?.aborted) {
        this.logger.error({ message: 'Error processing tool calls', error });
      }
      this.destroyStreams(error);
    }
  }
}
