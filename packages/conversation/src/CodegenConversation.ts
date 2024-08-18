import * as readline from 'readline-sync';
import { Conversation } from './Conversation';
import { KeywordToFilesIndexModuleFactory } from './fs/keyword_to_files_index/KeywordToFilesIndexModule';
import { ConversationTemplateModuleFactory } from './template/ConversationTemplateModule';
import { ConversationFsModuleFactory } from './fs/conversation_fs/ConversationFsModule';
import { PackageModuleFactory } from './fs/package/PackageModule';
import { ConversationModule, ConversationModuleFactory } from './ConversationModule';
import { Reset, textColorMap } from '@proteinjs/util';
import { GitModuleFactory } from './fs/git/GitModule';
import { TiktokenModel } from 'tiktoken';
import { searchLibrariesFunctionName, searchPackagesFunctionName } from './fs/package/PackageFunctions';
import { getRecentlyAccessedFilePathsFunctionName, readFilesFunctionName } from './fs/conversation_fs/FsFunctions';
import { searchFilesFunctionName } from './fs/keyword_to_files_index/KeywordToFilesIndexFunctions';

export class CodegenConversation {
  private static INITIAL_QUESTION = 'What would you like to create?';
  private static BOT_NAME = 'Alina';
  private static MODEL: TiktokenModel = 'gpt-4'; //'gpt-3.5-turbo-16k';
  private repoPath: string;

  constructor(repoPath: string) {
    this.repoPath = repoPath;
  }

  async start() {
    const conversation = await this.createConversation();
    conversation.addAssistantMessagesToHistory([CodegenConversation.INITIAL_QUESTION]);
    const initialUserInput = this.respondToUser(CodegenConversation.INITIAL_QUESTION);
    let responseObject = await conversation.generateResponse({
      messages: [initialUserInput],
      model: CodegenConversation.MODEL,
    });
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const userInput = this.respondToUser(responseObject.message);
      responseObject = await conversation.generateResponse({ messages: [userInput], model: CodegenConversation.MODEL });
    }
  }

  private async createConversation() {
    const conversation = new Conversation({
      name: this.constructor.name,
      modules: await this.getModules(),
      logLevel: 'info',
    });
    conversation.addSystemMessagesToHistory(this.getSystemMessages());
    return conversation;
  }

  private async getModules(): Promise<ConversationModule[]> {
    const moduleFactories: ConversationModuleFactory[] = [
      new ConversationFsModuleFactory(),
      new KeywordToFilesIndexModuleFactory(),
      new PackageModuleFactory(),
      new ConversationTemplateModuleFactory(),
      new GitModuleFactory(),
    ];
    const modules: ConversationModule[] = [];
    for (const moduleFactory of moduleFactories) {
      modules.push(await moduleFactory.createModule(this.repoPath));
    }

    return modules;
  }

  private getSystemMessages() {
    return [
      `We are going to have a conversation with the user to generate code`,
      `Await all function calls that return a promise`,
      `Try to repspond to the user with as few words as possible while still having a conversational tone`,
      `When generating code, export the objects you create inline; do not use 'export default' syntax`,
      // `After finding a file to work with, assume the user's following question pertains to that file and use ${readFilesFunctionName} to read the file if needed`,
      // `If a conversation summary exists, if you aren't already working with a file, use the most relevant keyword mentioned in the conversation summary to find a file to read (using the ${searchFilesFunctionName} function) and then respond to the user after reading the file`,
      // `Use the most relevant keyword mentioned in the conversation summary to find a file to read (using the ${searchFilesFunctionName} function) and then respond to the user after reading the file`,
      // `If the conversation summary indicates the user was working with a file get the file path (use the ${searchFilesFunctionName} function if needed) and read the file with the ${readFilesFunctionName} function. Use that file as context to respond to the user.`,
      //
      // `Use the conversation summary to identify a file as context for the user interaction`,
      // `Use the ${searchFilesFunctionName} function to find the file if needed; read the file if needed`,
      // `If the user is referring to a function, object, class, or type and you don't have the relevant file content, first inspect the conversation summary in the chat history (if it exists) to find a file name, and call the ${searchFilesFunctionName} function and read the file before responding to the user`,
      // `Before calling ${searchFilesFunctionName}, ${searchLibrariesFunctionName} or ${searchPackagesFunctionName}, use the conversation summary in the chat history to identify a file or keyword to search for instead; after reading that file, respond to the user's request`,
      //
      // `Use the ${getRecentlyAccessedFilePathsFunctionName} function find a file that might pertain to the user's request before searching files, libraries or packages; read that file then respond to the user`,
      // `When trying to locate code, use the ${getRecentlyAccessedFilePathsFunctionName} function to search recently accessed files first, then proceed to calling other functions: ${searchLibrariesFunctionName}, ${searchPackagesFunctionName}, ${searchFilesFunctionName}`,
      // `The conversation summary indicates files recently worked in as well`,
      // `If that doesn't yield results, proceed to calling the ${searchLibrariesFunctionName} function, then fall back to functions: ${searchPackagesFunctionName}, ${searchFilesFunctionName}`,
      //
      // `To find code, a file, or a library, call ${getRecentlyAccessedFilePathsFunctionName} and read the most recent file, after trying that call ${searchLibrariesFunctionName} then ${searchFilesFunctionName} to find a relevant file`,
      // `The file mentioned in the conversation summary should be read if we're not already working in a file`,
      // `If there is a conversation summary assistant message, use that to pick a file to read before responding to the user if not already working with a specific file`,
      // `Check for a previous conversation summary assistant message in the chat history; if there is one and it mentions a file the user was working with, call ${searchLibrariesFunctionName} to find the file path then call ${readFilesFunctionName} to read the file. Do this to build context before responding to the user`,
    ];
  }

  private respondToUser(message: string) {
    return readline.question(`${textColorMap.cyan}[${CodegenConversation.BOT_NAME}] ${message}${Reset}\n`);
  }
}
