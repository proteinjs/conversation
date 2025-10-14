import { OpenAI as OpenAIApi } from 'openai';
import {
  ChatCompletionMessageParam,
  ChatCompletion,
  ChatCompletionMessageToolCall,
  ChatCompletionChunk,
} from 'openai/resources/chat';
import { isInstanceOf } from '@proteinjs/util';
import { LogLevel, Logger } from '@proteinjs/logger';
import { MessageModerator } from './history/MessageModerator';
import { Function } from './Function';
import { MessageHistory } from './history/MessageHistory';
import { TiktokenModel } from 'tiktoken';
import { ChatCompletionMessageParamFactory } from './ChatCompletionMessageParamFactory';
import { Stream } from 'openai/streaming';
import { Readable } from 'stream';
import { OpenAiStreamProcessor } from './OpenAiStreamProcessor';
import { UsageData, UsageDataAccumulator } from './UsageData';
import { UsageDataAccumulatorContext } from './UsageDataContext';

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Structured capture of each tool call during a single generateResponse loop. */
export type ToolInvocationResult = {
  id: string; // tool_call_id from the model
  name: string; // function name invoked
  startedAt: Date;
  finishedAt: Date;
  input: unknown; // parsed JSON args (or raw string if parse failed)
  ok: boolean;
  data?: unknown; // tool return value (JSON-serializable)
  error?: { message: string; stack?: string };
};

/** Realtime progress hook for tool calls. */
export type ToolInvocationProgressEvent =
  | {
      type: 'started';
      id: string;
      name: string;
      startedAt: Date;
      input: unknown;
    }
  | {
      type: 'finished';
      result: ToolInvocationResult;
    };

export type GenerateResponseParams = {
  messages: (string | ChatCompletionMessageParam)[];
  model?: TiktokenModel;
  /** Optional realtime hook for tool-call lifecycle (started/finished). */
  onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
};

export type GenerateResponseReturn = {
  message: string;
  usagedata: UsageData;
  /** Structured ledger of tool calls executed while producing this message. */
  toolInvocations: ToolInvocationResult[];
};

export type GenerateStreamingResponseParams = GenerateResponseParams & {
  abortSignal?: AbortSignal;
  onUsageData?: (usageData: UsageData) => Promise<void>;
};

type GenerateResponseHelperParams = GenerateStreamingResponseParams & {
  model: TiktokenModel;
  stream: boolean;
  currentFunctionCalls?: number;
  usageDataAccumulator?: UsageDataAccumulator;
  /** Accumulated across recursive tool loops. */
  toolInvocations?: ToolInvocationResult[];
};

export type OpenAiParams = {
  model?: TiktokenModel;
  history?: MessageHistory;
  functions?: Omit<Function, 'instructions'>[];
  messageModerators?: MessageModerator[];
  maxFunctionCalls?: number;
  logLevel?: LogLevel;
};

export const DEFAULT_MODEL: TiktokenModel = 'gpt-4o';
export const DEFAULT_MAX_FUNCTION_CALLS = 50;

export class OpenAi {
  private model: TiktokenModel;
  private history: MessageHistory;
  private functions?: Omit<Function, 'instructions'>[];
  private messageModerators?: MessageModerator[];
  private maxFunctionCalls: number;
  private logLevel?: LogLevel;

  constructor({
    model = DEFAULT_MODEL,
    history = new MessageHistory(),
    functions,
    messageModerators,
    maxFunctionCalls = DEFAULT_MAX_FUNCTION_CALLS,
    logLevel,
  }: OpenAiParams = {}) {
    this.model = model;
    this.history = history;
    this.functions = functions;
    this.messageModerators = messageModerators;
    this.maxFunctionCalls = maxFunctionCalls;
    this.logLevel = (process.env.PROTEINJS_CONVERSATION_OPENAI_LOG_LEVEL as LogLevel | undefined) ?? logLevel;
  }

  async generateResponse({ model, ...rest }: GenerateResponseParams): Promise<GenerateResponseReturn> {
    return (await this.generateResponseHelper({
      model: model ?? this.model,
      stream: false,
      ...rest,
      toolInvocations: [],
    })) as GenerateResponseReturn;
  }

  async generateStreamingResponse({ model, ...rest }: GenerateStreamingResponseParams): Promise<Readable> {
    return (await this.generateResponseHelper({
      model: model ?? this.model,
      stream: true,
      ...rest,
      toolInvocations: [],
    })) as Readable;
  }

  private async generateResponseHelper({
    messages,
    model,
    stream,
    abortSignal,
    onUsageData,
    onToolInvocation,
    usageDataAccumulator,
    currentFunctionCalls = 0,
    toolInvocations = [],
  }: GenerateResponseHelperParams): Promise<GenerateResponseReturn | Readable> {
    const logger = new Logger({ name: 'OpenAi.generateResponseHelper', logLevel: this.logLevel });
    this.updateMessageHistory(messages);

    const context = new UsageDataAccumulatorContext();
    const contextDataUsageAccumulator = context.getUsageDataAccumulator(model);
    const resolvedUsageDataAccumulator =
      usageDataAccumulator ?? contextDataUsageAccumulator ?? new UsageDataAccumulator({ model });
    const executeInContext = !contextDataUsageAccumulator;

    const execute = async () => {
      const response = await this.executeRequest(model, stream, resolvedUsageDataAccumulator, abortSignal);
      if (stream) {
        logger.info({ message: `Processing response stream` });
        const inputStream = response as Stream<ChatCompletionChunk>;

        // For subsequent tool calls, return the raw OpenAI stream to `OpenAiStreamProcessor`
        if (currentFunctionCalls > 0) {
          return Readable.from(inputStream);
        }

        // For the initial call to `generateResponseHelper`, return the `OpenAiStreamProcessor` output stream
        const onToolCalls = ((toolCalls, currentFunctionCalls) =>
          this.handleToolCalls(
            toolCalls,
            model,
            stream,
            currentFunctionCalls,
            resolvedUsageDataAccumulator,
            abortSignal,
            onUsageData,
            toolInvocations,
            onToolInvocation
          )) as (toolCalls: ChatCompletionMessageToolCall[], currentFunctionCalls: number) => Promise<Readable>;
        const streamProcessor = new OpenAiStreamProcessor(
          inputStream,
          onToolCalls,
          resolvedUsageDataAccumulator,
          this.logLevel,
          abortSignal,
          onUsageData
        );
        return streamProcessor.getOutputStream();
      }

      const responseMessage = (response as ChatCompletion).choices[0].message;
      if (responseMessage.tool_calls) {
        return await this.handleToolCalls(
          responseMessage.tool_calls,
          model,
          stream,
          currentFunctionCalls,
          resolvedUsageDataAccumulator,
          abortSignal,
          onUsageData,
          toolInvocations,
          onToolInvocation
        );
      }

      const responseText = responseMessage.content;
      if (!responseText) {
        throw new Error(`Response was empty for messages: ${messages.join('\n')}`);
      }

      this.history.push([responseMessage]);
      return { message: responseText, usagedata: resolvedUsageDataAccumulator.usageData, toolInvocations };
    };

    // Only wrap in context if this is the first call
    if (executeInContext) {
      return context.runInContext(resolvedUsageDataAccumulator, execute);
    }

    return execute();
  }

  private updateMessageHistory(messages: (string | ChatCompletionMessageParam)[]) {
    const messageParams: ChatCompletionMessageParam[] = messages.map((message) => {
      if (typeof message === 'string') {
        return { role: 'user', content: message };
      }

      return message;
    });
    this.history.push(messageParams);
    if (this.messageModerators) {
      this.moderateHistory(this.history, this.messageModerators);
    }
  }

  private moderateHistory(history: MessageHistory, messageModerators: MessageModerator[]) {
    for (const messageModerator of messageModerators) {
      history.setMessages(messageModerator.observe(history.getMessages()));
    }
  }

  private async executeRequest(
    model: TiktokenModel,
    stream: boolean,
    usageDataAccumulator: UsageDataAccumulator,
    abortSignal?: AbortSignal
  ): Promise<ChatCompletion | Stream<ChatCompletionChunk>> {
    const logger = new Logger({ name: 'OpenAi.executeRequest', logLevel: this.logLevel });
    const openaiApi = new OpenAIApi();
    try {
      const latestMessage = this.history.getMessages()[this.history.getMessages().length - 1];
      this.logRequestDetails(logger, latestMessage);

      const response = await openaiApi.chat.completions.create(
        {
          model,
          messages: this.history.getMessages(),
          ...(this.functions &&
            this.functions.length > 0 && {
              tools: this.functions.map((f) => ({
                type: 'function',
                function: f.definition,
              })),
            }),
          stream: stream,
          ...(stream && { stream_options: { include_usage: true } }),
        },
        { signal: abortSignal }
      );

      if (!stream) {
        this.logResponseDetails(logger, response as ChatCompletion, usageDataAccumulator);
      }

      return response;
    } catch (error: any) {
      return this.handleRequestError(model, logger, error, stream, usageDataAccumulator, abortSignal);
    }
  }

  private logRequestDetails(logger: Logger, latestMessage: ChatCompletionMessageParam) {
    if (latestMessage.role == 'tool') {
      logger.info({ message: `Sending request: returning output of tool call (${latestMessage.tool_call_id})` });
    } else if (latestMessage.content) {
      const requestContent =
        typeof latestMessage.content === 'string'
          ? latestMessage.content
          : latestMessage.content[0].type === 'text'
            ? latestMessage.content[0].text
            : 'image';
      logger.info({ message: `Sending request`, obj: { requestContent } });
    } else {
      logger.info({ message: `Sending request` });
    }

    logger.debug({ message: `Sending messages:`, obj: { messages: this.history.getMessages() } });
  }

  private logResponseDetails(logger: Logger, response: ChatCompletion, usageDataAccumulator: UsageDataAccumulator) {
    const responseMessage = response.choices[0].message;
    if (responseMessage.content) {
      logger.info({ message: `Received response`, obj: { response: responseMessage.content } });
    } else if (responseMessage.tool_calls) {
      logger.info({
        message: `Received response: call functions`,
        obj: { functions: responseMessage.tool_calls.map((toolCall) => toolCall.function.name) },
      });
    } else {
      logger.info({ message: `Received response` });
    }
    if (response.usage) {
      logger.info({ message: `Usage data`, obj: { usageData: response.usage } });
      usageDataAccumulator.addTokenUsage({
        promptTokens: response.usage.prompt_tokens,
        cachedPromptTokens: response.usage.prompt_tokens_details?.cached_tokens ?? 0,
        completionTokens: response.usage.completion_tokens,
        totalTokens: response.usage.total_tokens,
      });
    } else {
      logger.info({ message: `Usage data missing` });
    }
  }

  private async handleRequestError(
    model: TiktokenModel,
    logger: Logger,
    error: any,
    stream: boolean,
    usageDataAccumulator: UsageDataAccumulator,
    abortSignal?: AbortSignal
  ): Promise<ChatCompletion | Stream<ChatCompletionChunk>> {
    if (error.type) {
      logger.info({ message: `Received error response, error type: ${error.type}` });
    }
    if (typeof error.status !== 'undefined' && error.status == 429) {
      if (error.type == 'tokens' && typeof error.headers['x-ratelimit-reset-tokens'] === 'string') {
        const waitTime = parseInt(error.headers['x-ratelimit-reset-tokens']);
        const remainingTokens = error.headers['x-ratelimit-remaining-tokens'];
        const delayMs = 15000;
        logger.warn({
          message: `Waiting to retry due to throttling`,
          obj: { retryDelay: `${delayMs / 1000}s`, tokenResetWaitTime: `${waitTime}s`, remainingTokens },
        });
        await delay(delayMs);
        return await this.executeRequest(model, stream, usageDataAccumulator, abortSignal);
      }
    }
    throw error;
  }

  private async handleToolCalls(
    toolCalls: ChatCompletionMessageToolCall[],
    model: TiktokenModel,
    stream: boolean,
    currentFunctionCalls: number,
    usageDataAccumulator: UsageDataAccumulator,
    abortSignal?: AbortSignal,
    onUsageData?: (usageData: UsageData) => Promise<void>,
    toolInvocations: ToolInvocationResult[] = [],
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void
  ): Promise<GenerateResponseReturn | Readable> {
    if (currentFunctionCalls >= this.maxFunctionCalls) {
      throw new Error(`Max function calls (${this.maxFunctionCalls}) reached. Stopping execution.`);
    }

    // Create a message for the tool calls
    const toolCallMessage: ChatCompletionMessageParam = {
      role: 'assistant',
      content: null,
      tool_calls: toolCalls,
    };

    // Add the tool call message to the history
    this.history.push([toolCallMessage]);

    // Call the tools and get the responses
    const toolMessageParams = await this.callTools(toolCalls, usageDataAccumulator, toolInvocations, onToolInvocation);

    // Add the tool responses to the history
    this.history.push(toolMessageParams);

    // Generate the next response
    return this.generateResponseHelper({
      messages: [],
      model,
      stream,
      abortSignal,
      onUsageData,
      onToolInvocation,
      usageDataAccumulator,
      currentFunctionCalls: currentFunctionCalls + toolCalls.length,
      toolInvocations,
    });
  }

  private async callTools(
    toolCalls: ChatCompletionMessageToolCall[],
    usageDataAccumulator: UsageDataAccumulator,
    toolInvocations: ToolInvocationResult[],
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void
  ): Promise<ChatCompletionMessageParam[]> {
    const toolMessageParams: ChatCompletionMessageParam[] = (
      await Promise.all(
        toolCalls.map(
          async (toolCall) =>
            await this.callFunction(
              toolCall.function,
              toolCall.id,
              usageDataAccumulator,
              toolInvocations,
              onToolInvocation
            )
        )
      )
    ).reduce((acc, val) => acc.concat(val), []);

    return toolMessageParams;
  }

  private async callFunction(
    functionCall: ChatCompletionMessageToolCall.Function,
    toolCallId: string,
    usageDataAccumulator: UsageDataAccumulator,
    toolInvocations: ToolInvocationResult[],
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void
  ): Promise<ChatCompletionMessageParam[]> {
    const logger = new Logger({ name: 'OpenAi.callFunction', logLevel: this.logLevel });
    if (!this.functions) {
      const errorMessage = `Assistant attempted to call a function when no functions were provided`;
      logger.error({ message: errorMessage });
      return [{ role: 'tool', tool_call_id: toolCallId, content: JSON.stringify({ error: errorMessage }) }];
    }

    functionCall.name = functionCall.name.split('.').pop() as string;
    const f = this.functions.find((fx) => fx.definition.name === functionCall.name);
    if (!f) {
      const errorMessage = `Assistant attempted to call nonexistent function`;
      logger.error({ message: errorMessage, obj: { functionName: functionCall.name } });
      return [
        {
          role: 'tool',
          tool_call_id: toolCallId,
          content: JSON.stringify({ error: errorMessage, functionName: functionCall.name }),
        },
      ];
    }

    try {
      const parsedArguments = JSON.parse(functionCall.arguments);
      logger.info({
        message: `Assistant calling function: (${toolCallId}) ${f.definition.name}`,
        obj: { toolCallId, functionName: f.definition.name, args: parsedArguments },
      });
      usageDataAccumulator.recordToolCall(f.definition.name);

      const startedAt = new Date();

      onToolInvocation?.({
        type: 'started',
        id: toolCallId,
        name: f.definition.name,
        startedAt,
        input: parsedArguments,
      });

      const returnObject = await f.call(parsedArguments);
      const finishedAt = new Date();

      // Record success
      const rec: ToolInvocationResult = {
        id: toolCallId,
        name: f.definition.name,
        startedAt,
        finishedAt,
        input: parsedArguments,
        ok: true,
        data: returnObject,
      };
      toolInvocations.push(rec);

      onToolInvocation?.({ type: 'finished', result: rec });

      const returnObjectCompletionParams: ChatCompletionMessageParam[] = [];
      if (isInstanceOf(returnObject, ChatCompletionMessageParamFactory)) {
        // handle functions that return a ChatCompletionMessageParamFactory
        const chatCompletionMessageParamFactory = returnObject as ChatCompletionMessageParamFactory;
        const messageParams = await chatCompletionMessageParamFactory.create();
        const instructionMessageParam: ChatCompletionMessageParam = {
          role: 'tool',
          tool_call_id: toolCallId,
          content: `The the return data from this function is provided in the following messages`,
        };
        returnObjectCompletionParams.push(instructionMessageParam, ...messageParams);
        logger.info({
          message: `Assistant called function: (${toolCallId}) ${f.definition.name}`,
          obj: { toolCallId, functionName: f.definition.name, return: messageParams },
        });
      } else {
        // handle all other functions
        const serializedReturnObject = JSON.stringify(returnObject);
        returnObjectCompletionParams.push({
          role: 'tool',
          tool_call_id: toolCallId,
          content: serializedReturnObject,
        });
        logger.info({
          message: `Assistant called function: (${toolCallId}) ${f.definition.name}`,
          obj: { toolCallId, functionName: f.definition.name, return: returnObject },
        });
      }

      if (typeof returnObject === 'undefined') {
        return [
          {
            role: 'tool',
            tool_call_id: toolCallId,
            content: JSON.stringify({ result: 'Function with no return value executed successfully' }),
          },
        ];
      }

      return returnObjectCompletionParams;
    } catch (error: any) {
      const now = new Date();
      const attemptedArgs = (() => {
        try {
          return JSON.parse(functionCall.arguments);
        } catch {
          return functionCall.arguments;
        }
      })();

      // Record failure
      const rec: ToolInvocationResult = {
        id: toolCallId,
        name: functionCall.name,
        startedAt: now,
        finishedAt: now,
        input: attemptedArgs,
        ok: false,
        error: { message: String(error?.message ?? error), stack: (error as any)?.stack },
      };
      toolInvocations.push(rec);

      onToolInvocation?.({ type: 'finished', result: rec });

      logger.error({
        message: `An error occurred while executing function`,
        error,
        obj: { toolCallId, functionName: f.definition.name },
      });

      throw error;
    }
  }

  async generateCode({
    messages,
    model,
    includeSystemMessages = true,
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
    includeSystemMessages?: boolean;
  }) {
    const systemMessages: ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: 'Return only the code and exclude example usage, markdown, explanations, comments and notes.',
      },
      { role: 'system', content: `Write code in typescript.` },
      { role: 'system', content: `Declare explicit types for all function parameters.` },
      { role: 'system', content: 'Export all functions and objects generated.' },
      { role: 'system', content: 'Do not omit function implementations.' },
    ];
    if (includeSystemMessages) {
      this.history.push(systemMessages);
    }
    const { message } = await this.generateResponse({ messages, model });
    return OpenAi.parseCodeFromMarkdown(message);
  }

  async updateCode({
    code,
    description,
    model,
    includeSystemMessages = true,
  }: {
    code: string;
    description: string;
    model?: TiktokenModel;
    includeSystemMessages?: boolean;
  }) {
    return await this.generateCode({
      messages: [OpenAi.updateCodeDescription(code, description)],
      model,
      includeSystemMessages,
    });
  }

  static updateCodeDescription(code: string, description: string) {
    return `Update this code:\n\n${code}\n\n${description}`;
  }

  static parseCodeFromMarkdown(code: string) {
    if (!code.match(/```([\s\S]+?)```/g)) {
      return code;
    }

    const filteredLines: string[] = [];
    let inCodeBlock = false;
    for (const line of code.split('\n')) {
      if (line.startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        if (!inCodeBlock) {
          filteredLines.push('');
        }

        continue;
      }

      if (inCodeBlock) {
        filteredLines.push(line);
      }
    }

    // remove the last '' that will become a \n
    // we only want spaces between code blocks
    filteredLines.pop();

    return filteredLines.join('\n');
  }

  async generateList({
    messages,
    model,
    includeSystemMessages = true,
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
    includeSystemMessages?: boolean;
  }): Promise<string[]> {
    const systemMessages: ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: 'Return only the list and exclude example usage, markdown and all explanations, comments and notes.',
      },
      { role: 'system', content: 'Separate each item in the list by a ;' },
    ];
    if (includeSystemMessages) {
      this.history.push(systemMessages);
    }
    const { message } = await this.generateResponse({ messages, model });
    return message.split(';').map((item) => item.trim());
  }

  /**
   * Generates a concise response based on the given context and request.
   * This method is designed to produce brief, focused answers without any
   * conversational elements or additional explanations.
   *
   * @param context - The background information or context for the request.
   * @param request - The specific question or task to be addressed.
   * @param model - Optional. The specific model to use for generating the response. If not provided, the default model set for the class instance will be used.
   *
   */
  async generateConciseAnswer({
    context,
    request,
    model,
  }: {
    context: string;
    request: string;
    model?: TiktokenModel;
  }): Promise<string> {
    const systemMessages: ChatCompletionMessageParam[] = [
      { role: 'system', content: `Context: ${context}\n\nRequest: ${request}` },
      {
        role: 'system',
        content:
          'Provide only the requested information without any additional explanation or conversational elements.',
      },
    ];

    const { message } = await this.generateResponse({
      messages: systemMessages,
      model: model || this.model,
    });

    return message.trim();
  }
}
