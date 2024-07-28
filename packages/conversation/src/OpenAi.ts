import { OpenAI as OpenAIApi } from 'openai';
import {
  ChatCompletionMessageParam,
  ChatCompletion,
  ChatCompletionMessageToolCall,
  ChatCompletionChunk,
} from 'openai/resources/chat';
import { LogLevel, Logger, isInstanceOf } from '@proteinjs/util';
import { MessageModerator } from './history/MessageModerator';
import { Function } from './Function';
import { MessageHistory } from './history/MessageHistory';
import { TiktokenModel } from 'tiktoken';
import { ChatCompletionMessageParamFactory } from './ChatCompletionMessageParamFactory';
import { Stream } from 'openai/streaming';
import { Readable } from 'stream';
import { OpenAiStreamProcessor } from './OpenAiStreamProcessor';

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const DEFAULT_MODEL: TiktokenModel = 'gpt-3.5-turbo';
export class OpenAi {
  static async generateResponse(
    messages: (string | ChatCompletionMessageParam)[],
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    logLevel: LogLevel = 'info',
    maxFunctionCalls: number = 50
  ): Promise<string> {
    return (await this.generateResponseHelper(
      messages,
      false,
      0,
      model,
      history,
      functions,
      messageModerators,
      logLevel,
      maxFunctionCalls
    )) as string;
  }

  static async generateStreamingResponse(
    messages: (string | ChatCompletionMessageParam)[],
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    logLevel: LogLevel = 'info',
    maxFunctionCalls: number = 50
  ): Promise<Readable> {
    return (await this.generateResponseHelper(
      messages,
      true,
      0,
      model,
      history,
      functions,
      messageModerators,
      logLevel,
      maxFunctionCalls
    )) as Readable;
  }

  static async generateResponseHelper(
    messages: (string | ChatCompletionMessageParam)[],
    stream: boolean,
    currentFunctionCalls: number,
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    logLevel: LogLevel = 'info',
    maxFunctionCalls: number = 50
  ): Promise<string | Readable> {
    const logger = new Logger('OpenAi.generateResponseHelper', logLevel);
    const updatedHistory = OpenAi.getUpdatedMessageHistory(messages, history, messageModerators);
    const response = await OpenAi.executeRequest(updatedHistory, stream, logLevel, functions, model);
    if (stream) {
      logger.info(`Processing response stream`);
      const inputStream = response as Stream<ChatCompletionChunk>;

      // For subsequent tool calls, return the raw OpenAI stream to `OpenAiStreamProcessor`
      if (currentFunctionCalls > 0) {
        return Readable.from(inputStream);
      }

      // For the initial call to `generateResponseHelper`, return the `OpenAiStreamProcessor` output stream
      const onToolCalls = ((toolCalls, currentFunctionCalls) =>
        OpenAi.handleToolCalls(
          toolCalls,
          true,
          currentFunctionCalls,
          updatedHistory,
          model,
          functions,
          messageModerators,
          logLevel,
          maxFunctionCalls
        )) as (toolCalls: ChatCompletionMessageToolCall[], currentFunctionCalls: number) => Promise<Readable>;
      const streamProcessor = new OpenAiStreamProcessor(inputStream, onToolCalls, logLevel);
      return streamProcessor.getOutputStream();
    }

    const responseMessage = (response as ChatCompletion).choices[0].message;
    if (responseMessage.tool_calls) {
      return await OpenAi.handleToolCalls(
        responseMessage.tool_calls,
        stream,
        currentFunctionCalls,
        updatedHistory,
        model,
        functions,
        messageModerators,
        logLevel,
        maxFunctionCalls
      );
    }

    const responseText = responseMessage.content;
    if (!responseText) {
      throw new Error(`Response was empty for messages: ${messages.join('\n')}`);
    }

    updatedHistory.push([responseMessage]);
    return responseText;
  }

  private static getUpdatedMessageHistory(
    messages: (string | ChatCompletionMessageParam)[],
    history?: MessageHistory,
    messageModerators?: MessageModerator[]
  ) {
    const messageParams: ChatCompletionMessageParam[] = messages.map((message) => {
      if (typeof message === 'string') {
        return { role: 'user', content: message };
      }

      return message;
    });
    if (history) {
      history.push(messageParams);
    }
    let messageParamsWithHistory = history ? history : new MessageHistory().push(messageParams);
    if (messageModerators) {
      messageParamsWithHistory = OpenAi.moderateHistory(messageParamsWithHistory, messageModerators);
    }

    return messageParamsWithHistory;
  }

  private static moderateHistory(history: MessageHistory, messageModerators: MessageModerator[]) {
    for (const messageModerator of messageModerators) {
      history.setMessages(messageModerator.observe(history.getMessages()));
    }

    return history;
  }

  private static async executeRequest(
    messageParamsWithHistory: MessageHistory,
    stream: boolean,
    logLevel: LogLevel,
    functions?: Omit<Function, 'instructions'>[],
    model?: string
  ): Promise<ChatCompletion | Stream<ChatCompletionChunk>> {
    const logger = new Logger('OpenAi.executeRequest', logLevel);
    const openaiApi = new OpenAIApi();
    try {
      const latestMessage = messageParamsWithHistory.getMessages()[messageParamsWithHistory.getMessages().length - 1];
      this.logRequestDetails(logger, logLevel, latestMessage, messageParamsWithHistory);

      const response = await openaiApi.chat.completions.create({
        model: model ? model : DEFAULT_MODEL,
        temperature: 0,
        messages: messageParamsWithHistory.getMessages(),
        tools: functions?.map((f) => ({
          type: 'function',
          function: f.definition,
        })),
        stream: stream,
      });

      if (!stream) {
        this.logResponseDetails(logger, response as ChatCompletion);
      }

      return response;
    } catch (error: any) {
      return this.handleRequestError(logger, error, messageParamsWithHistory, stream, logLevel, functions, model);
    }
  }

  private static logRequestDetails(
    logger: Logger,
    logLevel: LogLevel,
    latestMessage: ChatCompletionMessageParam,
    messageParamsWithHistory: MessageHistory
  ) {
    if (latestMessage.role == 'tool') {
      logger.info(`Sending request: returning output of tool call (${latestMessage.tool_call_id})`);
    } else if (latestMessage.content) {
      const requestContent =
        typeof latestMessage.content === 'string'
          ? latestMessage.content
          : latestMessage.content[0].type === 'text'
            ? latestMessage.content[0].text
            : 'image';
      logger.info(`Sending request: ${requestContent}`);
    } else {
      logger.info(`Sending request`);
    }

    if (logLevel === 'debug') {
      logger.debug(`Sending messages: ${JSON.stringify(messageParamsWithHistory.getMessages(), null, 2)}`, true);
    }
  }

  private static logResponseDetails(logger: Logger, response: ChatCompletion) {
    const responseMessage = response.choices[0].message;
    if (responseMessage.content) {
      logger.info(`Received response: ${responseMessage.content}`);
    } else if (responseMessage.tool_calls) {
      logger.info(
        `Received response: call functions: ${JSON.stringify(responseMessage.tool_calls.map((toolCall) => toolCall.function.name))}`
      );
    } else {
      logger.info(`Received response`);
    }
    if (response.usage) {
      logger.info(JSON.stringify(response.usage));
    } else {
      logger.info(JSON.stringify(`Usage data missing`));
    }
  }

  private static async handleRequestError(
    logger: Logger,
    error: any,
    messageParamsWithHistory: MessageHistory,
    stream: boolean,
    logLevel: LogLevel,
    functions?: Omit<Function, 'instructions'>[],
    model?: string
  ): Promise<ChatCompletion | Stream<ChatCompletionChunk>> {
    logger.info(`Received error response, error type: ${error.type}`);
    if (typeof error.status !== 'undefined' && error.status == 429) {
      if (error.type == 'tokens' && typeof error.headers['x-ratelimit-reset-tokens'] === 'string') {
        const waitTime = parseInt(error.headers['x-ratelimit-reset-tokens']);
        const remainingTokens = error.headers['x-ratelimit-remaining-tokens'];
        const delayMs = 15000;
        logger.warn(
          `Waiting to retry in ${delayMs / 1000}s, token reset in: ${waitTime}s, remaining tokens: ${remainingTokens}`
        );
        await delay(delayMs);
        return await OpenAi.executeRequest(messageParamsWithHistory, stream, logLevel, functions, model);
      }
    }
    throw error;
  }

  private static async handleToolCalls(
    toolCalls: ChatCompletionMessageToolCall[],
    stream: boolean,
    currentFunctionCalls: number,
    history: MessageHistory,
    model?: string,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    logLevel: LogLevel = 'info',
    maxFunctionCalls: number = 50
  ): Promise<string | Readable> {
    if (currentFunctionCalls >= maxFunctionCalls) {
      throw new Error(`Max function calls (${maxFunctionCalls}) reached. Stopping execution.`);
    }

    // Create a message for the tool calls
    const toolCallMessage: ChatCompletionMessageParam = {
      role: 'assistant',
      content: null,
      tool_calls: toolCalls,
    };

    // Add the tool call message to the history
    history.push([toolCallMessage]);

    // Call the tools and get the responses
    const toolMessageParams = await this.callTools(logLevel, toolCalls, functions);

    // Add the tool responses to the history
    history.push(toolMessageParams);

    // Generate the next response
    return this.generateResponseHelper(
      [],
      stream,
      currentFunctionCalls + toolCalls.length,
      model,
      history,
      functions,
      messageModerators,
      logLevel,
      maxFunctionCalls
    );
  }

  private static async callTools(
    logLevel: LogLevel,
    toolCalls: ChatCompletionMessageToolCall[],
    functions?: Omit<Function, 'instructions'>[]
  ): Promise<ChatCompletionMessageParam[]> {
    const toolMessageParams: ChatCompletionMessageParam[] = (
      await Promise.all(
        toolCalls.map(
          async (toolCall) => await OpenAi.callFunction(logLevel, toolCall.function, toolCall.id, functions)
        )
      )
    ).reduce((acc, val) => acc.concat(val), []);

    return toolMessageParams;
  }

  private static async callFunction(
    logLevel: LogLevel,
    functionCall: ChatCompletionMessageToolCall.Function,
    toolCallId: string,
    functions?: Omit<Function, 'instructions'>[]
  ): Promise<ChatCompletionMessageParam[]> {
    const logger = new Logger('OpenAi.callFunction', logLevel);
    if (!functions) {
      const error = `Assistant attempted to call a function when no functions were provided`;
      logger.error(error);
      return [{ role: 'tool', tool_call_id: toolCallId, content: JSON.stringify({ error }) }];
    }

    functionCall.name = functionCall.name.split('.').pop() as string;
    const f = functions.find((f) => f.definition.name === functionCall.name);
    if (!f) {
      const error = `Assistant attempted to call nonexistent function: ${functionCall.name}`;
      logger.error(error);
      return [{ role: 'tool', tool_call_id: toolCallId, content: JSON.stringify({ error }) }];
    }

    try {
      const parsedArguments = JSON.parse(functionCall.arguments);
      logger.info(
        `Assistant calling function: (${toolCallId}) ${f.definition.name}(${JSON.stringify(parsedArguments, null, 2)})`,
        1000
      );
      const returnObject = await f.call(parsedArguments);

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
        logger.info(
          `Assistant called function: (${toolCallId}) ${f.definition.name} => ${JSON.stringify(messageParams, null, 2)}`,
          500
        );
      } else {
        // handle all other functions
        const serializedReturnObject = JSON.stringify(returnObject);
        returnObjectCompletionParams.push({
          role: 'tool',
          tool_call_id: toolCallId,
          content: serializedReturnObject,
        });
        logger.info(
          `Assistant called function: (${toolCallId}) ${f.definition.name} => ${JSON.stringify(returnObject, null, 2)}`,
          1000
        );
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
      const errorMessage = `Error occurred while executing function ${f.definition.name}: (${toolCallId}) ${error.message}`;
      logger.error(errorMessage);
      throw error;
    }
  }

  static async generateCode(
    messages: (string | ChatCompletionMessageParam)[],
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    includeSystemMessages: boolean = true,
    logLevel: LogLevel = 'info'
  ) {
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
    const resolvedHistory = history
      ? includeSystemMessages
        ? history.push(systemMessages)
        : history
      : includeSystemMessages
        ? new MessageHistory().push(systemMessages)
        : undefined;
    const code = await this.generateResponse(messages, model, resolvedHistory, functions, messageModerators, logLevel);
    return this.parseCodeFromMarkdown(code);
  }

  static async updateCode(
    code: string,
    description: string,
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    includeSystemMessages: boolean = true,
    logLevel: LogLevel = 'info'
  ) {
    return await this.generateCode(
      [this.updateCodeDescription(code, description)],
      model,
      history,
      functions,
      messageModerators,
      includeSystemMessages,
      logLevel
    );
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

  static async generateList(
    messages: (string | ChatCompletionMessageParam)[],
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    includeSystemMessages: boolean = true,
    logLevel: LogLevel = 'info'
  ): Promise<string[]> {
    const systemMessages: ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: 'Return only the list and exclude example usage, markdown and all explanations, comments and notes.',
      },
      { role: 'system', content: 'Separate each item in the list by a ;' },
    ];
    const resolvedHistory = history
      ? includeSystemMessages
        ? history.push(systemMessages)
        : history
      : includeSystemMessages
        ? new MessageHistory().push(systemMessages)
        : undefined;
    const list = await this.generateResponse(messages, model, resolvedHistory, functions, messageModerators, logLevel);
    return list.split(';').map((item) => item.trim());
  }
}
