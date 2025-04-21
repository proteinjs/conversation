import { ChatCompletionMessageParam } from 'openai/resources/chat';
import { DEFAULT_MODEL, OpenAi } from './OpenAi';
import { MessageHistory } from './history/MessageHistory';
import { Function } from './Function';
import { Logger, LogLevel } from '@proteinjs/logger';
import { Fs } from '@proteinjs/util-node';
import { MessageModerator } from './history/MessageModerator';
import { ConversationModule } from './ConversationModule';
import { TiktokenModel, encoding_for_model } from 'tiktoken';
import { searchLibrariesFunctionName } from './fs/package/PackageFunctions';
import { UsageData } from './UsageData';

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
  private modulesProcessed = false;
  private processingModulesPromise: Promise<void> | null = null;

  constructor(params: ConversationParams) {
    this.params = params;
    this.history = new MessageHistory({
      maxMessages: params.limits?.maxMessagesInHistory,
      enforceMessageLimit: params.limits?.enforceLimits,
    });
    this.logger = new Logger({ name: params.name, logLevel: params.logLevel });

    if (typeof params.limits?.enforceLimits === 'undefined' || params.limits.enforceLimits) {
      this.addFunctions('Conversation', [summarizeConversationHistoryFunction(this)]);
    }

    if (params.limits?.tokenLimit) {
      this.tokenLimit = params.limits.tokenLimit;
    }
  }

  private async ensureModulesProcessed(): Promise<void> {
    // If modules are already processed, return immediately
    if (this.modulesProcessed) {
      return;
    }

    // If modules are currently being processed, wait for that to complete
    if (this.processingModulesPromise) {
      return this.processingModulesPromise;
    }

    // Start processing modules and keep a reference to the promise
    this.processingModulesPromise = this.processModules();

    try {
      await this.processingModulesPromise;
      this.modulesProcessed = true;
    } catch (error) {
      this.logger.error({ message: 'Error processing modules', obj: { error } });
      // Reset the promise so we can try again
      this.processingModulesPromise = null;
      throw error;
    }
  }

  private async processModules(): Promise<void> {
    if (!this.params.modules || this.params.modules.length === 0) {
      return;
    }

    for (const module of this.params.modules) {
      // Get system messages and handle potential Promise
      const moduleSystemMessagesResult = module.getSystemMessages();
      let moduleSystemMessages: string[] | string;

      // Check if the result is a Promise and await it if needed
      if (moduleSystemMessagesResult instanceof Promise) {
        moduleSystemMessages = await moduleSystemMessagesResult;
      } else {
        moduleSystemMessages = moduleSystemMessagesResult;
      }

      if (!moduleSystemMessages || (Array.isArray(moduleSystemMessages) && moduleSystemMessages.length < 1)) {
        continue;
      }

      const formattedSystemMessages = Array.isArray(moduleSystemMessages)
        ? moduleSystemMessages.join('. ')
        : moduleSystemMessages;

      this.addSystemMessagesToHistory([
        `The following are instructions from the ${module.getName()} module:\n${formattedSystemMessages}`,
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
    await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateResponse({ messages: [summarizeConversationRequest], model });
    const referenceSummaryRequest = `If there's a file mentioned in the conversation summary, find and read the file to better respond to my next request. If that doesn't find anything, call the ${searchLibrariesFunctionName} function on other keywords in the conversation summary to find a file to read`;
    await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateResponse({ messages: [referenceSummaryRequest], model });
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

  async generateResponse({
    messages,
    model,
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
  }) {
    await this.ensureModulesProcessed();
    await this.enforceTokenLimit(messages, model);
    return await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateResponse({ messages, model });
  }

  async generateStreamingResponse({
    messages,
    model,
    ...rest
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
    abortSignal?: AbortSignal;
    onUsageData?: (usageData: UsageData) => Promise<void>;
  }) {
    await this.ensureModulesProcessed();
    await this.enforceTokenLimit(messages, model);
    return await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateStreamingResponse({ messages, model, ...rest });
  }

  async generateCode({ description, model }: { description: string[]; model?: TiktokenModel }) {
    this.logger.debug({ message: `Generating code`, obj: { description } });
    await this.ensureModulesProcessed();
    const code = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateCode({
      messages: description,
      model,
      includeSystemMessages: !this.generatedCode,
    });
    this.logger.debug({ message: `Generated code`, obj: { code } });
    this.generatedCode = true;
    return code;
  }

  async updateCodeFromFile({
    codeToUpdateFilePath,
    dependencyCodeFilePaths,
    description,
    model,
  }: {
    codeToUpdateFilePath: string;
    dependencyCodeFilePaths: string[];
    description: string;
    model?: TiktokenModel;
  }) {
    await this.ensureModulesProcessed();
    const codeToUpdate = await Fs.readFile(codeToUpdateFilePath);
    let dependencyDescription = `Assume the following exists:\n`;
    for (const dependencyCodeFilePath of dependencyCodeFilePaths) {
      const dependencCode = await Fs.readFile(dependencyCodeFilePath);
      dependencyDescription += dependencCode + '\n\n';
    }

    this.logger.debug({ message: `Updating code from file`, obj: { codeToUpdateFilePath } });
    return await this.updateCode({ code: codeToUpdate, description: dependencyDescription + description, model });
  }

  async updateCode({ code, description, model }: { code: string; description: string; model?: TiktokenModel }) {
    this.logger.debug({ message: `Updating code`, obj: { description, code } });
    await this.ensureModulesProcessed();
    const updatedCode = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).updateCode({
      code,
      description,
      model,
      includeSystemMessages: !this.generatedCode,
    });
    this.logger.debug({ message: `Updated code`, obj: { updatedCode } });
    this.generatedCode = true;
    return updatedCode;
  }

  async generateList({ description, model }: { description: string[]; model?: TiktokenModel }) {
    await this.ensureModulesProcessed();
    const list = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateList({
      messages: description,
      model,
      includeSystemMessages: !this.generatedList,
    });
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
