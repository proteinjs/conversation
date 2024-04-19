import { Logger } from '@proteinjs/util';
import { Fs, PackageUtil, Package } from '@proteinjs/util-node';
import { SourceFile } from './Code';

export type TemplateArgs = {
  srcPath: string,
  additionalPackages?: Package[],
  replacePackages?: boolean,
}

export abstract class CodeTemplate {
  protected logger = new Logger(this.constructor.name);
  protected templateArgs: TemplateArgs;

  constructor(templateArgs: TemplateArgs) {
    this.templateArgs = templateArgs;
  }

  abstract dependencyPackages(): Package[];
  abstract sourceFiles(): SourceFile[];

  async generate() {
    await PackageUtil.installPackages(this.resolvePackages());
    for (let sourceFile of this.sourceFiles()) {
      const filePath = Fs.baseContainedJoin(this.templateArgs.srcPath, sourceFile.relativePath);
      this.logger.info(`Generating source file: ${filePath}`);
      const code = await sourceFile.code.generate();
      await Fs.writeFiles([{ path: filePath, content: code }]);
      this.logger.info(`Generated source file: ${filePath}`);
    }
  }

  private resolvePackages() {
    const packages: Package[] = this.templateArgs.replacePackages ? [] : this.dependencyPackages();
    if (this.templateArgs.additionalPackages)
      packages.push(...this.templateArgs.additionalPackages);
    return packages;
  }
}