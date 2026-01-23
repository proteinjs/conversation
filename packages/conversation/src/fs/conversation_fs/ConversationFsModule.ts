import { ConversationModule, ConversationModuleFactory } from '../../ConversationModule';
import { Function } from '../../Function';
import { searchFilesFunctionName } from '../keyword_to_files_index/KeywordToFilesIndexFunctions';
import { searchLibrariesFunctionName, searchPackagesFunctionName } from '../package/PackageFunctions';
import { ConversationFsModerator } from './ConversationFsModerator';
import {
  deleteFilesFunction,
  fsFunctions,
  getRecentlyAccessedFilePathsFunction,
  grepFunction,
  grepFunctionName,
  readFilesFunction,
  readFilesFunctionName,
  writeFilesFunction,
  writeFilesFunctionName,
} from './FsFunctions';

export class ConversationFsModule implements ConversationModule {
  private repoPath: string;
  private recentlyAccessedFilePaths: string[] = [];

  constructor(repoPath: string) {
    this.repoPath = repoPath;
  }

  getName(): string {
    return 'Conversation Fs';
  }

  getSystemMessages(): string[] {
    return [
      `Assume the current working directory is: ${this.repoPath} unless specified by the user`,
      `Pre-pend the current working directory as the base path to file paths when performing file operations, unless specified otherwise by the user`,
      `If the user asks to change the cwd, do not create a new folder, just fail if it's not a valid path`,
      `If the user wants to work in a package, use the ${searchPackagesFunctionName} function and take the directory from the package.json file path and make that the cwd`,
      `You have access to code in a local repo, you can read and write code to and from the file system with the ${readFilesFunctionName} function and the ${writeFilesFunctionName} function`,
      `Before writing to a file that already exists, read the file first and make your changes to its contents`,
      `When reading/writing a file in a specified package, join the package directory with the relative path to form the file path`,
      `When searching for source files, do not look in the dist or node_modules directories`,
      `If you don't know a file path, don't try to guess it, use the ${searchFilesFunctionName} function to find it`,
      `When searching for something (ie. a file to work with/in), unless more context is specified, use the ${grepFunctionName} function first, then fall back to functions: ${searchPackagesFunctionName}, ${searchFilesFunctionName}`,
      `After finding a file to work with, assume the user's following question pertains to that file and use ${readFilesFunctionName} to read the file if needed`,
    ];
  }

  getFunctions(): Function[] {
    return [
      readFilesFunction(this),
      writeFilesFunction(this),
      deleteFilesFunction(this),
      getRecentlyAccessedFilePathsFunction(this),
      grepFunction(this),
      ...fsFunctions,
    ];
  }

  getMessageModerators() {
    // return [new ConversationFsModerator()];
    return [];
  }

  pushRecentlyAccessedFilePath(filePaths: string[]) {
    this.recentlyAccessedFilePaths.push(...filePaths);
  }

  getRecentlyAccessedFilePaths() {
    return this.recentlyAccessedFilePaths;
  }

  getRepoPath(): string {
    return this.repoPath;
  }
}

export class ConversationFsModuleFactory implements ConversationModuleFactory {
  async createModule(repoPath: string): Promise<ConversationFsModule> {
    return new ConversationFsModule(repoPath);
  }
}
