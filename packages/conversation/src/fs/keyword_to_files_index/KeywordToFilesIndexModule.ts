import { Logger } from '@proteinjs/logger';
import { Fs } from '@proteinjs/util-node';
import { ConversationModule, ConversationModuleFactory } from '../../ConversationModule';
import { Function } from '../../Function';
import path from 'path';
import { searchFilesFunction, searchFilesFunctionName } from './KeywordToFilesIndexFunctions';

export type KeywordToFilesIndexModuleParams = {
  dir: string;
  // Map from lowercase filename *stem* (no extension) → file paths
  keywordFilesIndex: { [keyword: string]: string[] };
};

export class KeywordToFilesIndexModule implements ConversationModule {
  private logger = new Logger({ name: this.constructor.name });
  params: KeywordToFilesIndexModuleParams;

  constructor(params: KeywordToFilesIndexModuleParams) {
    this.params = params;
  }

  getName(): string {
    return 'Keyword to files index';
  }

  /**
   * Case-insensitive file name search that ignores extension.
   */
  searchFiles(params: { keyword: string }) {
    this.logger.debug({ message: `Searching for file, keyword: ${params.keyword}` });
    const keywordLowerNoExtension = path.parse(params.keyword).name.toLowerCase();
    const filePaths = this.params.keywordFilesIndex[keywordLowerNoExtension];
    return filePaths || [];
  }

  getSystemMessages(): string[] {
    return [
      `If you're searching for something, use the ${searchFilesFunctionName} function to find a file (by name) matching the search string`,
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
  private logger = new Logger({ name: this.constructor.name });

  async createModule(repoPath: string): Promise<KeywordToFilesIndexModule> {
    this.logger.debug({ message: `Creating module for repo: ${repoPath}` });
    const repoParams: KeywordToFilesIndexModuleParams = { keywordFilesIndex: {}, dir: repoPath };
    repoParams.keywordFilesIndex = await this.createKeywordFilesIndex(repoPath, ['**/node-typescript-parser/**']);
    this.logger.debug({ message: `Created module for repo: ${repoPath}` });
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
      const fileNameLower = path.parse(filePath).name.toLowerCase(); // Get file name without extension

      if (!keywordFilesIndex[fileNameLower]) {
        keywordFilesIndex[fileNameLower] = [];
      }

      this.logger.debug({ message: `fileName: ${fileNameLower}, filePath: ${filePath}` });
      keywordFilesIndex[fileNameLower].push(filePath);
    }

    return keywordFilesIndex;
  }
}
