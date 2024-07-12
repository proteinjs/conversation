import { OpenAI as OpenAIApi } from 'openai';
import { ChatCompletionMessageParam, ChatCompletion, ChatCompletionMessageToolCall } from 'openai/resources/chat';
import { LogLevel, Logger, isInstanceOf } from '@proteinjs/util';
import { MessageModerator } from './history/MessageModerator';
import { Function } from './Function';
import { MessageHistory } from './history/MessageHistory';
import { TiktokenModel } from 'tiktoken';
import { ChatCompletionMessageParamFactory } from './ChatCompletionMessageParamFactory';

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
    return await this.generateResponseHelper(
      messages,
      0,
      model,
      history,
      functions,
      messageModerators,
      logLevel,
      maxFunctionCalls
    );
  }

  static async generateResponseHelper(
    messages: (string | ChatCompletionMessageParam)[],
    currentFunctionCalls: number,
    model?: string,
    history?: MessageHistory,
    functions?: Omit<Function, 'instructions'>[],
    messageModerators?: MessageModerator[],
    logLevel: LogLevel = 'info',
    maxFunctionCalls: number = 50
  ): Promise<string> {
    const logger = new Logger('OpenAi.generateResponse', logLevel);
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
    const response = await OpenAi.executeRequest(messageParamsWithHistory, logLevel, functions, model);
    const responseMessage = response.choices[0].message;
    if (responseMessage.tool_calls) {
      if (currentFunctionCalls >= maxFunctionCalls) {
        throw new Error(`Max function calls (${maxFunctionCalls}) reached. Stopping execution.`);
      }

      messageParamsWithHistory.push([responseMessage]);
      const toolMessageParams = await this.callTools(logLevel, responseMessage.tool_calls, functions);
      messageParamsWithHistory.push([...toolMessageParams]);

      return await this.generateResponseHelper(
        [],
        currentFunctionCalls + responseMessage.tool_calls.length,
        model,
        messageParamsWithHistory,
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

    messageParamsWithHistory.push([responseMessage]);
    return responseText;
  }

  private static moderateHistory(history: MessageHistory, messageModerators: MessageModerator[]) {
    for (const messageModerator of messageModerators) {
      history.setMessages(messageModerator.observe(history.getMessages()));
    }

    return history;
  }

  private static async executeRequest(
    messageParamsWithHistory: MessageHistory,
    logLevel: LogLevel,
    functions?: Omit<Function, 'instructions'>[],
    model?: string
  ): Promise<ChatCompletion> {
    const logger = new Logger('OpenAi.executeRequest', logLevel);
    const openaiApi = new OpenAIApi();
    let response: ChatCompletion;
    try {
      const latestMessage = messageParamsWithHistory.getMessages()[messageParamsWithHistory.getMessages().length - 1];
      if (latestMessage.content) {
        logger.info(`Sending request: ${latestMessage.content}`);
      } else if (latestMessage.role == 'function') {
        logger.info(`Sending request: returning output of ${latestMessage.name} function`);
      } else {
        logger.info(`Sending request`);
      }
      logger.debug(`Sending messages: ${JSON.stringify(messageParamsWithHistory.getMessages(), null, 2)}`, true);
      response = await openaiApi.chat.completions.create({
        model: model ? model : DEFAULT_MODEL,
        temperature: 0,
        messages: messageParamsWithHistory.getMessages(),
        tools: functions?.map((f) => ({
          type: 'function',
          function: f.definition,
        })),
      });
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
    } catch (error: any) {
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
          return await OpenAi.executeRequest(messageParamsWithHistory, logLevel, functions, model);
        }
      }

      throw error;
    }

    return response;
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
      logger.info(`Assistant calling function: (${toolCallId}) ${f.definition.name}(${functionCall.arguments})`, 1000);
      const returnObject = await f.call(JSON.parse(functionCall.arguments));

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
          `Assistant called function: (${toolCallId}) ${f.definition.name} => ${JSON.stringify(messageParams)}`,
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
          `Assistant called function: (${toolCallId}) ${f.definition.name} => ${serializedReturnObject}`,
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
      const errorMessage = `Error occurred while executing function ${f.definition.name}: ${error.message}`;
      logger.error(errorMessage);
      return [{ role: 'tool', tool_call_id: toolCallId, content: JSON.stringify({ error: errorMessage }) }];
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
