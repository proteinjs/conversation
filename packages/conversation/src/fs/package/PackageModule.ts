import { Fs } from '@proteinjs/util-node';
import { ConversationModule, ConversationModuleFactory } from '../../ConversationModule';
import { Function } from '../../Function';
import {
  packageFunctions,
  searchLibrariesFunction,
  searchLibrariesFunctionName,
  searchPackagesFunction,
  searchPackagesFunctionName,
} from './PackageFunctions';
import path from 'path';
import { searchFilesFunctionName } from '../keyword_to_files_index/KeywordToFilesIndexFunctions';
import { readFilesFunctionName } from '../conversation_fs/FsFunctions';

export type Library = {
  fileName: string;
  filePath: string;
  packageName: string;
};

export type LibraryImport = {
  importStatements: string[];
  typescriptDeclaration: string;
};

export class PackageModule implements ConversationModule {
  private repoPath: string;

  constructor(repoPath: string) {
    this.repoPath = repoPath;
  }

  getName(): string {
    return 'Package';
  }

  getSystemMessages(): string[] {
    return [
      `When generating code, prefer importing code from local packages`,
      `Use the ${searchPackagesFunctionName} function on every package you plan to install to derermine if the package is in the local repo; if it is, calculate the relative path from the cwd package to the package being installed, use that path as the version when installing the package`,
      // `When generating code, use the searchFiles function to find all file paths to index.ts files; these are the local apis we have access to`,
      // `When generating import statements, use the searchFiles function to find all file paths to package.json files; if importing from a local package, make sure you import via its package if it is not a local file to the package we're generating code in`,
      `When generating import statements from another package, do not use relative paths`,
    ];
  }

  getFunctions(): Function[] {
    return [...packageFunctions, searchPackagesFunction(this), searchLibrariesFunction(this)];
  }

  getMessageModerators() {
    return [];
  }

  /**
   * @param keyword either the package name or some substring
   * @return string[] of file paths
   */
  async searchPackages(keyword: string): Promise<string[]> {
    const matchingPackageJsonPaths: string[] = [];
    const packageJsonFilePaths = await Fs.getFilePathsMatchingGlob(this.repoPath, '**/package.json', [
      '**/node_modules/**',
      '**/dist/**',
    ]);
    const packageJsonFileMap = await Fs.readFiles(packageJsonFilePaths);
    for (const packageJsonFilePath of Object.keys(packageJsonFileMap)) {
      const packageJson = JSON.parse(packageJsonFileMap[packageJsonFilePath]);
      if (packageJson.name.toLowerCase().includes(keyword.toLocaleLowerCase())) {
        matchingPackageJsonPaths.push(packageJsonFilePath);
      }
    }

    return matchingPackageJsonPaths;
  }

  /**
   * Search packages in repo for file names that include keyword
   * @param keyword substring of file name to search for
   * @returns all libraries in the repo matching the keyword
   */
  async searchLibraries(keyword: string): Promise<Library[]> {
    const matchingLibraries: Library[] = [];
    const packageJsonFilePaths = await Fs.getFilePathsMatchingGlob(this.repoPath, '**/package.json', [
      '**/node_modules/**',
      '**/dist/**',
    ]);
    const packageJsonFileMap = await Fs.readFiles(packageJsonFilePaths);
    for (const packageJsonFilePath of Object.keys(packageJsonFileMap)) {
      const packageJson = JSON.parse(packageJsonFileMap[packageJsonFilePath]);
      const packageJsonFilePathParts = packageJsonFilePath.split(path.sep);
      packageJsonFilePathParts.pop();
      const packageDirectory = packageJsonFilePathParts.join(path.sep);
      const srcFilePaths = await Fs.getFilePathsMatchingGlob(path.join(packageDirectory, 'src'), '**/*.ts', [
        '**/node_modules/**',
        '**/dist/**',
      ]);
      for (const srcFilePath of srcFilePaths) {
        const fileNameWithExtension = path.basename(srcFilePath);
        if (fileNameWithExtension.includes(keyword)) {
          const fileName = path.basename(srcFilePath, path.extname(srcFilePath));
          matchingLibraries.push({
            fileName,
            filePath: srcFilePath,
            packageName: packageJson.name,
          });
        }
      }
    }

    return matchingLibraries;
  }
}

export class PackageModuleFactory implements ConversationModuleFactory {
  async createModule(repoPath: string): Promise<PackageModule> {
    return new PackageModule(repoPath);
  }
}
