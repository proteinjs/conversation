import { Logger } from '@proteinjs/util';
import { Fs } from '@proteinjs/util-node';
import { ConversationModule, ConversationModuleFactory } from '../../ConversationModule';
import { Function } from '../../Function';
import path from 'path';
import { searchFilesFunction, searchFilesFunctionName } from './KeywordToFilesIndexFunctions';

export type KeywordToFilesIndexModuleParams = {
  dir: string;
  keywordFilesIndex: { [keyword: string]: string[] /** file paths */ };
};

export class KeywordToFilesIndexModule implements ConversationModule {
  private logger = new Logger(this.constructor.name);
  params: KeywordToFilesIndexModuleParams;

  constructor(params: KeywordToFilesIndexModuleParams) {
    this.params = params;
  }

  getName(): string {
    return 'Keyword to files index';
  }

  searchFiles(params: { keyword: string }) {
    this.logger.info(`Searching for file, keyword: ${params.keyword}`);
    const filePaths = this.params.keywordFilesIndex[params.keyword];
    return filePaths || [];
  }

  getSystemMessages(): string[] {
    return [
      `If you're searching for something, use the ${searchFilesFunctionName} function to find a file matching the search string`,
    ];
  }

  getFunctions(): Function[] {
    return [searchFilesFunction(this)];
  }

  getMessageModerators() {
    return [];
  }
}

export class KeywordToFilesIndexModuleFactory implements ConversationModuleFactory {
  private logger = new Logger(this.constructor.name);

  async createModule(repoPath: string): Promise<KeywordToFilesIndexModule> {
    this.logger.debug(`Creating module for repo: ${repoPath}`);
    const repoParams: KeywordToFilesIndexModuleParams = { keywordFilesIndex: {}, dir: repoPath };
    repoParams.keywordFilesIndex = await this.createKeywordFilesIndex(repoPath, ['**/node-typescript-parser/**']);
    this.logger.debug(`Created module for repo: ${repoPath}`);
    return new KeywordToFilesIndexModule(repoParams);
  }

  /**
   * Create keyword-files index for the given base directory.
   *
   * @param baseDir - The directory to start the file search from.
   * @returns An index with keywords mapped to file paths.
   */
  async createKeywordFilesIndex(
    baseDir: string,
    globIgnorePatterns: string[] = []
  ): Promise<{ [keyword: string]: string[] }> {
    // Ensure the base directory has a trailing slash
    if (!baseDir.endsWith(path.sep)) {
      baseDir += path.sep;
    }

    // Get all file paths, recursively, excluding node_modules and dist directories
    const filePaths = await Fs.getFilePaths(baseDir, ['**/node_modules/**', '**/dist/**', ...globIgnorePatterns]);

    const keywordFilesIndex: { [keyword: string]: string[] } = {};

    // Process each file path
    for (const filePath of filePaths) {
      const fileName = path.parse(filePath).name; // Get file name without extension

      if (!keywordFilesIndex[fileName]) {
        keywordFilesIndex[fileName] = [];
      }

      this.logger.debug(`fileName: ${fileName}, filePath: ${filePath}`);
      keywordFilesIndex[fileName].push(filePath);
    }

    return keywordFilesIndex;
  }
}
