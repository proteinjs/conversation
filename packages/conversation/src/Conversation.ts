import { ChatCompletionMessageParam } from 'openai/resources/chat';
import { DEFAULT_MODEL, OpenAi } from './OpenAi';
import { MessageHistory } from './history/MessageHistory';
import { Function } from './Function';
import { Logger, LogLevel } from '@proteinjs/util';
import { Fs } from '@proteinjs/util-node';
import { MessageModerator } from './history/MessageModerator';
import { ConversationModule } from './ConversationModule';
import { TiktokenModel, encoding_for_model } from 'tiktoken';
import { searchLibrariesFunctionName } from './fs/package/PackageFunctions';

export type ConversationParams = {
  name: string;
  modules?: ConversationModule[];
  logLevel?: LogLevel;
  limits?: {
    enforceLimits?: boolean;
    maxMessagesInHistory?: number;
    tokenLimit?: number;
  };
};

export class Conversation {
  private tokenLimit = 3000;
  private history;
  private systemMessages: ChatCompletionMessageParam[] = [];
  private functions: Function[] = [];
  private messageModerators: MessageModerator[] = [];
  private generatedCode = false;
  private generatedList = false;
  private logger: Logger;
  private params: ConversationParams;

  constructor(params: ConversationParams) {
    this.params = params;
    this.history = new MessageHistory({
      maxMessages: params.limits?.maxMessagesInHistory,
      enforceMessageLimit: params.limits?.enforceLimits,
    });
    this.logger = new Logger(params.name, params.logLevel);

    if (params.modules) {
      this.addModules(params.modules);
    }

    if (typeof params.limits?.enforceLimits === 'undefined' || params.limits.enforceLimits) {
      this.addFunctions('Conversation', [summarizeConversationHistoryFunction(this)]);
    }

    if (params.limits?.tokenLimit) {
      this.tokenLimit = params.limits.tokenLimit;
    }
  }

  private addModules(modules: ConversationModule[]) {
    for (const module of modules) {
      if (module.getSystemMessages().length < 1) {
        continue;
      }

      const moduleSystemMessages = module.getSystemMessages();
      const formattedSystemMessages = Array.isArray(moduleSystemMessages)
        ? moduleSystemMessages.join('. ')
        : moduleSystemMessages;

      this.addSystemMessagesToHistory([
        `The following are instructions from the ${module.getName()} module: ${formattedSystemMessages}`,
      ]);
      this.addFunctions(module.getName(), module.getFunctions());
      this.addMessageModerators(module.getMessageModerators());
    }
  }

  private addFunctions(moduleName: string, functions: Function[]) {
    this.functions.push(...functions);
    let functionInstructions = `The following are instructions from functions in the ${moduleName} module:`;
    let functionInstructionsAdded = false;
    for (const f of functions) {
      if (f.instructions) {
        if (!f.instructions || f.instructions.length < 1) {
          continue;
        }

        functionInstructionsAdded = true;
        const instructionsParagraph = f.instructions.join('. ');
        functionInstructions += ` ${f.definition.name}: ${instructionsParagraph}.`;
      }
    }

    if (!functionInstructionsAdded) {
      return;
    }

    this.addSystemMessagesToHistory([functionInstructions]);
  }

  private addMessageModerators(messageModerators: MessageModerator[]) {
    this.messageModerators.push(...messageModerators);
  }

  private async enforceTokenLimit(messages: (string | ChatCompletionMessageParam)[], model?: TiktokenModel) {
    if (this.params.limits?.enforceLimits === false) {
      return;
    }

    const resolvedModel = model ? model : DEFAULT_MODEL;
    const encoder = encoding_for_model(resolvedModel);
    const conversation =
      this.history.toString() +
      messages
        .map((message) => {
          if (typeof message === 'string') {
            return message;
          } else {
            // Extract content from ChatCompletionMessageParam
            const contentParts = Array.isArray(message.content) ? message.content : [message.content];
            return contentParts
              .map((part) => {
                if (typeof part === 'string') {
                  return part;
                } else if (part?.type === 'text') {
                  return part.text;
                } else {
                  return ''; // Handle non-text content types as empty string
                }
              })
              .join(' ');
          }
        })
        .join('. ');
    const encoded = encoder.encode(conversation);
    console.log(`current tokens: ${encoded.length}`);
    if (encoded.length < this.tokenLimit) {
      return;
    }

    const summarizeConversationRequest = `First, call the ${summarizeConversationHistoryFunctionName} function`;
    await OpenAi.generateResponse(
      [summarizeConversationRequest],
      model,
      this.history,
      this.functions,
      this.messageModerators,
      this.params.logLevel
    );
    const referenceSummaryRequest = `If there's a file mentioned in the conversation summary, find and read the file to better respond to my next request. If that doesn't find anything, call the ${searchLibrariesFunctionName} function on other keywords in the conversation summary to find a file to read`;
    await OpenAi.generateResponse(
      [referenceSummaryRequest],
      model,
      this.history,
      this.functions,
      this.messageModerators,
      this.params.logLevel
    );
  }

  summarizeConversationHistory(summary: string) {
    this.clearHistory();
    this.history.push([{ role: 'assistant', content: `Previous conversation summary: ${summary}` }]);
  }

  private clearHistory() {
    this.history = new MessageHistory();
    this.history.push(this.systemMessages);
  }

  addSystemMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'system', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addAssistantMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'assistant', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addUserMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'user', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addMessagesToHistory(messages: ChatCompletionMessageParam[], unshift = false) {
    const systemMessages = messages.filter((message) => message.role === 'system');
    if (unshift) {
      this.history.getMessages().unshift(...messages);
      this.history.prune();
      this.systemMessages.unshift(...systemMessages);
    } else {
      this.history.push(messages);
      this.systemMessages.push(...systemMessages);
    }
  }

  async generateResponse(messages: (string | ChatCompletionMessageParam)[], model?: TiktokenModel) {
    await this.enforceTokenLimit(messages, model);
    return await OpenAi.generateResponse(
      messages,
      model,
      this.history,
      this.functions,
      this.messageModerators,
      this.params.logLevel
    );
  }

  async generateStreamingResponse(
    messages: (string | ChatCompletionMessageParam)[],
    model?: TiktokenModel,
    abortSignal?: AbortSignal
  ) {
    await this.enforceTokenLimit(messages, model);
    return await OpenAi.generateStreamingResponse(
      messages,
      model,
      this.history,
      this.functions,
      this.messageModerators,
      abortSignal,
      this.params.logLevel
    );
  }

  async generateCode(description: string[], model?: TiktokenModel) {
    this.logger.info(`Generating code for description:\n${description.join('\n')}`);
    const code = await OpenAi.generateCode(
      description,
      model,
      this.history,
      this.functions,
      this.messageModerators,
      !this.generatedCode,
      this.params.logLevel
    );
    this.logger.info(`Generated code:\n${code.slice(0, 150)}${code.length > 150 ? '...' : ''}`);
    this.generatedCode = true;
    return code;
  }

  async updateCodeFromFile(
    codeToUpdateFilePath: string,
    dependencyCodeFilePaths: string[],
    description: string,
    model?: TiktokenModel
  ) {
    const codeToUpdate = await Fs.readFile(codeToUpdateFilePath);
    let dependencyDescription = `Assume the following exists:\n`;
    for (const dependencyCodeFilePath of dependencyCodeFilePaths) {
      const dependencCode = await Fs.readFile(dependencyCodeFilePath);
      dependencyDescription += dependencCode + '\n\n';
    }

    this.logger.info(`Updating code from file: ${codeToUpdateFilePath}`);
    return await this.updateCode(codeToUpdate, dependencyDescription + description, model);
  }

  async updateCode(code: string, description: string, model?: TiktokenModel) {
    this.logger.info(
      `Updating code:\n${code.slice(0, 150)}${code.length > 150 ? '...' : ''}\nFrom description: ${description}`
    );
    const updatedCode = await OpenAi.updateCode(
      code,
      description,
      model,
      this.history,
      this.functions,
      this.messageModerators,
      !this.generatedCode,
      this.params.logLevel
    );
    this.logger.info(`Updated code:\n${updatedCode.slice(0, 150)}${updatedCode.length > 150 ? '...' : ''}`);
    this.generatedCode = true;
    return updatedCode;
  }

  async generateList(description: string[], model?: TiktokenModel) {
    const list = await OpenAi.generateList(
      description,
      model,
      this.history,
      this.functions,
      this.messageModerators,
      !this.generatedList,
      this.params.logLevel
    );
    this.generatedList = true;
    return list;
  }
}

export const summarizeConversationHistoryFunctionName = 'summarizeConversationHistory';
export const summarizeConversationHistoryFunction = (conversation: Conversation) => {
  return {
    definition: {
      name: summarizeConversationHistoryFunctionName,
      description: 'Clear the conversation history and summarize what was in it',
      parameters: {
        type: 'object',
        properties: {
          summary: {
            type: 'string',
            description: 'A 1-3 sentence summary of the current chat history',
          },
        },
        required: ['summary'],
      },
    },
    call: async (params: { summary: string }) => conversation.summarizeConversationHistory(params.summary),
  };
};
