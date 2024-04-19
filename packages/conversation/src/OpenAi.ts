import { OpenAI as OpenAIApi } from 'openai';
import { ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletion } from 'openai/resources/chat';
import { LogLevel, Logger } from '@proteinjs/util';
import { MessageModerator } from './history/MessageModerator';
import { Function } from './Function';
import { MessageHistory } from './history/MessageHistory';
import { TiktokenModel } from 'tiktoken';

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export const DEFAULT_MODEL: TiktokenModel = 'gpt-3.5-turbo';
export class OpenAi {
  static async generateResponse(messages: string[], model?: string, history?: MessageHistory, functions?: Function[], messageModerators?: MessageModerator[], logLevel: LogLevel = 'info'): Promise<string> {
    const logger = new Logger('OpenAi.generateResponse', logLevel);
    const messageParams: ChatCompletionMessageParam[] = messages.map(message => { return { role: 'user', content: message }});
    if (history)
      history.push(messageParams);
    let messageParamsWithHistory = history ? history : new MessageHistory().push(messageParams);
    if (messageModerators)
      messageParamsWithHistory = OpenAi.moderateHistory(messageParamsWithHistory, messageModerators);
    const response = await OpenAi.executeRequest(messageParamsWithHistory, logLevel, functions, model);
    const responseMessage = response.choices[0].message;
    if (responseMessage.function_call) {
      messageParamsWithHistory.push([responseMessage]);
      const functionReturnMessage = await this.callFunction(logLevel, responseMessage.function_call, functions);
      if (functionReturnMessage)
        messageParamsWithHistory.push([functionReturnMessage])
      return await this.generateResponse([], model, messageParamsWithHistory, functions, messageModerators, logLevel);
    }

    const responseText = responseMessage.content;
    if (!responseText) {
      logger.error(`Received response: ${JSON.stringify(response)}`);
      throw new Error(`Response was empty for messages: ${messages.join('\n')}`);
    }

    messageParamsWithHistory.push([responseMessage]);
    return responseText;
  }

  private static moderateHistory(history: MessageHistory, messageModerators: MessageModerator[]) {
    for (let messageModerator of messageModerators)
      history.setMessages(messageModerator.observe(history.getMessages()));

    return history;
  }

  private static async executeRequest(messageParamsWithHistory: MessageHistory, logLevel: LogLevel, functions?: Function[], model?: string): Promise<ChatCompletion> {
    const logger = new Logger('OpenAi.executeRequest', logLevel);
    const openaiApi = new OpenAIApi();
    let response: ChatCompletion;
    try {
      const latestMessage = messageParamsWithHistory.getMessages()[messageParamsWithHistory.getMessages().length - 1];
      if (latestMessage.content)
        logger.info(`Sending request: ${latestMessage.content}`);
      else if (latestMessage.role == 'function')
        logger.info(`Sending request: returning output of ${latestMessage.name} function`);
      else
        logger.info(`Sending request`);
      logger.debug(`Sending messages: ${JSON.stringify(messageParamsWithHistory.getMessages(), null, 2)}`, true);
      response = await openaiApi.chat.completions.create({
        model: model ? model : DEFAULT_MODEL,
        temperature: 0,
        messages: messageParamsWithHistory.getMessages(),
        functions: functions?.map(f => f.definition),
      });
      const responseMessage = response.choices[0].message;
      if (responseMessage.content)
        logger.info(`Received response: ${responseMessage.content}`);
      else if (responseMessage.function_call)
        logger.info(`Received response: call ${responseMessage.function_call.name} function`);
      else
        logger.info(`Received response`);
      if (response.usage)
        logger.info(JSON.stringify(response.usage));
      else
        logger.info(JSON.stringify(`Usage data missing`));
    } catch(error: any) {
      logger.info(`Received error response, error type: ${error.type}`);
      if (typeof error.status !== 'undefined' && error.status == 429) {
        if (error.type == 'tokens' && typeof error.headers['x-ratelimit-reset-tokens'] === 'string') {
          const waitTime = parseInt(error.headers['x-ratelimit-reset-tokens']);
          const remainingTokens = error.headers['x-ratelimit-remaining-tokens'];
          const delayMs = 15000;
          logger.warn(`Waiting to retry in ${delayMs/1000}s, token reset in: ${waitTime}s, remaining tokens: ${remainingTokens}`);
          await delay(delayMs);
          return await OpenAi.executeRequest(messageParamsWithHistory, logLevel, functions, model);
        }
      }

      throw error;
    }

    return response;
  }

  private static async callFunction(logLevel: LogLevel, functionCall: ChatCompletionMessage.FunctionCall, functions?: Function[]): Promise<ChatCompletionMessageParam|undefined> {
    const logger = new Logger('OpenAi.callFunction', logLevel);
    if (!functions) {
      const warning = `Assistant attempted to call a function when no functions were provided`;
      logger.warn(warning);
      const message: ChatCompletionMessageParam = { role: 'user', content: warning }
      return message;
    }

    functionCall.name = functionCall.name.split('.').pop() as string;
    const f = functions.find(f => f.definition.name === functionCall.name);
    if (!f) {
      const warning = `Assistant attempted to call nonexistent function: ${functionCall.name}`;
      logger.warn(warning);
      const message: ChatCompletionMessageParam = { role: 'user', content: warning }
      return message;
    }

    let returnObject = null;
    try {
      logger.info(`Assistant calling function: ${f.definition.name}(${functionCall.arguments})`);
      returnObject = JSON.stringify(await f.call(JSON.parse(functionCall.arguments)));
      logger.info(`Assistant called function: ${f.definition.name}(${functionCall.arguments}) => ${returnObject}`, 1000);
    } catch (error: any) {
      logger.error(error.message);
    }

    if (!returnObject)
      return;

    return {
      role: 'function', 
      name: f.definition.name, 
      content: returnObject,
    };
  }

  static async generateCode(messages: string[], model?: string, history?: MessageHistory, functions?: Function[], messageModerators?: MessageModerator[], includeSystemMessages: boolean = true, logLevel: LogLevel = 'info') {
    const systemMessages: ChatCompletionMessageParam[] = [
      { role: 'system', content: 'Return only the code and exclude example usage, markdown, explanations, comments and notes.' },
      { role: 'system', content: `Write code in typescript.` },
      { role: 'system', content: `Declare explicit types for all function parameters.` },
      { role: 'system', content: 'Export all functions and objects generated.' },
      { role: 'system', content: 'Do not omit function implementations.' },
    ];
    const resolvedHistory = history ? 
      includeSystemMessages ?
        history.push(systemMessages)
        :
        history
      :
      includeSystemMessages ?
        new MessageHistory().push(systemMessages)
        :
        undefined
    ;
    const code = await this.generateResponse(messages, model, resolvedHistory, functions, messageModerators, logLevel);
    return this.parseCodeFromMarkdown(code);
  }

  static async updateCode(code: string, description: string, model?: string, history?: MessageHistory, functions?: Function[], messageModerators?: MessageModerator[], includeSystemMessages: boolean = true, logLevel: LogLevel = 'info') {
    return await this.generateCode([this.updateCodeDescription(code, description)], model, history, functions, messageModerators, includeSystemMessages, logLevel);
  }

  static updateCodeDescription(code: string, description: string) {
    return `Update this code:\n\n${code}\n\n${description}`;
  }

  static parseCodeFromMarkdown(code: string) {
    if (!code.match(/```([\s\S]+?)```/g))
      return code;

    const filteredLines: string[] = [];
    let inCodeBlock = false;
    for (let line of code.split('\n')) {
      if (line.startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        if (!inCodeBlock)
          filteredLines.push('');
          
        continue;
      }

      if (inCodeBlock)
        filteredLines.push(line);
    }

    // remove the last '' that will become a \n
    // we only want spaces between code blocks
    filteredLines.pop();

    return filteredLines.join('\n');
  }

  static async generateList(messages: string[], model?: string, history?: MessageHistory, functions?: Function[], messageModerators?: MessageModerator[], includeSystemMessages: boolean = true, logLevel: LogLevel = 'info'): Promise<string[]> {
    const systemMessages: ChatCompletionMessageParam[] = [
      { role: 'system', content: 'Return only the list and exclude example usage, markdown and all explanations, comments and notes.' },
      { role: 'system', content: 'Separate each item in the list by a ;' },
    ];
    const resolvedHistory = history ? 
      includeSystemMessages ?
        history.push(systemMessages)
        :
        history
      :
      includeSystemMessages ?
        new MessageHistory().push(systemMessages)
        :
        undefined
    ;
    const list = await this.generateResponse(messages, model, resolvedHistory, functions, messageModerators, logLevel);
    return list.split(';').map(item => item.trim());
  }
}