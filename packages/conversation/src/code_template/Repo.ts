import { Logger } from '@proteinjs/logger';
import { FileDescriptor, Fs, PackageUtil } from '@proteinjs/util-node';
import fs from 'fs/promises';
import path from 'path';

export interface TsFile extends FileDescriptor {
  declaration: string;
}

export interface PackageInfo {
  packageJSON: any; // The content of the package.json
  dirPath: string;
  tsFiles: { [tsFilePath: string]: TsFile };
  fileDescriptors: FileDescriptor[];
}

export type SlimPackageInfo = Omit<PackageInfo, 'packageJSON' | 'tsFiles'>;

export type RepoParams = {
  packages: Record<string, PackageInfo>;
  slimPackages: Record<string, SlimPackageInfo>;
  tsFiles: { [tsFilePath: string]: TsFile };
  keywordFilesIndex: { [keyword: string]: string[] /** file paths */ };
};

export class Repo {
  private logger = new Logger({ name: this.constructor.name });
  params: RepoParams;

  constructor(params: RepoParams) {
    this.params = params;
  }

  packages() {
    return this.params.packages;
  }

  slimPackages() {
    return this.params.slimPackages;
  }

  tsFiles() {
    return this.params.tsFiles;
  }

  keywordFilesIndex() {
    return this.params.keywordFilesIndex;
  }

  searchFiles(params: { keyword: string }) {
    this.logger.info({ message: `Searching for file, keyword: ${params.keyword}` });
    const filePaths = this.keywordFilesIndex()[params.keyword];
    return filePaths || [];
  }

  getDeclarations(params: { tsFilePaths: string[] }) {
    const queriedDeclarations: { [tsFilePath: string]: string } = {};
    for (const tsFilePath of params.tsFilePaths) {
      queriedDeclarations[tsFilePath] = this.params.tsFiles[tsFilePath].declaration;
      this.logger.info({ message: `Accessed declaration: ${tsFilePath}` });
    }
    return queriedDeclarations;
  }
}

export class RepoFactory {
  private static LOGGER = new Logger({ name: 'RepoFactory' });

  public static async createRepo(dir: string): Promise<Repo> {
    this.LOGGER.info({ message: `Creating repo for dir: ${dir}` });
    const repoParams: RepoParams = { packages: {}, slimPackages: {}, tsFiles: {}, keywordFilesIndex: {} };

    async function traverse(dir: string) {
      const childrenNames = await fs.readdir(dir, { withFileTypes: true });
      const hasPackageJson = childrenNames.some((dirent) => dirent.name === 'package.json');
      if (hasPackageJson) {
        const packageContent = await fs.readFile(path.join(dir, 'package.json'), 'utf-8');
        const packageJSON = JSON.parse(packageContent);
        repoParams.packages[packageJSON.name] = {
          packageJSON: packageJSON,
          dirPath: dir,
          fileDescriptors: [],
          tsFiles: {},
        };
      }

      for (const dirent of childrenNames) {
        if (!dirent.isDirectory()) {
          continue;
        }

        // Exclude directories 'node_modules' and 'dist' right at the beginning
        if (dirent.name.includes('node_modules') || dirent.name.includes('dist')) {
          continue;
        }

        // Continue with the traversal if it's a directory
        const childPath = path.join(dir, dirent.name);
        await traverse(childPath);
      }
    }

    await traverse(dir);
    await RepoFactory.loadFiles(repoParams);
    Object.keys(repoParams.packages).forEach((packageName) => {
      const { packageJSON, tsFiles, ...slimPackage } = repoParams.packages[packageName];
      repoParams.slimPackages[packageName] = slimPackage;
    });
    this.LOGGER.info({ message: `Created repo for dir: ${dir}` });
    return new Repo(repoParams);
  }

  private static async loadFiles(repoParams: RepoParams) {
    for (const packageName of Object.keys(repoParams.packages)) {
      this.LOGGER.info({ message: `Loading files for package: ${packageName}` });
      const dirPath = repoParams.packages[packageName].dirPath;
      if (dirPath) {
        repoParams.packages[packageName].fileDescriptors.push(
          ...(await Fs.getFilesInDirectory(dirPath, ['node_modules', 'dist']))
        );
        for (const fileDescriptor of repoParams.packages[packageName].fileDescriptors) {
          const typescriptDeclaration = PackageUtil.generateTypescriptDeclarations({
            tsFilePaths: [fileDescriptor.path],
          })[fileDescriptor.path];
          const tsFile = Object.assign({ declaration: typescriptDeclaration }, fileDescriptor);
          repoParams.packages[packageName].tsFiles[fileDescriptor.path] = tsFile;
          repoParams.tsFiles[fileDescriptor.path] = tsFile;
          if (!repoParams.keywordFilesIndex[fileDescriptor.nameWithoutExtension]) {
            repoParams.keywordFilesIndex[fileDescriptor.nameWithoutExtension] = [];
          }

          repoParams.keywordFilesIndex[fileDescriptor.nameWithoutExtension].push(fileDescriptor.path);
        }
      }
    }
  }

  // private static async generateDirectoryMap(dir: string): Promise<TsFile> {
  //   const stat = await fs.stat(dir);
  //   let map: TsFile = {
  //     name: path.basename(dir),
  //     path: dir,
  //     type: stat.isDirectory() ? 'folder' : 'file'
  //   };

  //   if (stat.isDirectory()) {
  //     const childrenNames = await fs.readdir(dir);
  //     map.children = await Promise.all(
  //       childrenNames.map(child => RepoFactory.generateDirectoryMap(path.join(dir, child)))
  //     );
  //   } else if (dir.endsWith('.ts') || dir.endsWith('.tsx')) {
  //     const declarations = this.generateDeclarations({ tsFilePaths: [dir] });
  //     const declarationFilePath = path.join(path.dirname(dir), path.basename(dir, path.extname(dir))) + '.d.ts';
  //     // console.log(`dir: ${dir}\ndeclarationFilePath: ${declarationFilePath}`);
  //     if (declarations[declarationFilePath]) {
  //       map.declaration = declarations[declarationFilePath];
  //     }
  //   }

  //   return map;
  // }
}
