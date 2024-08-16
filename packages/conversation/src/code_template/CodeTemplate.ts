import { Logger } from '@proteinjs/logger';
import { Fs, PackageUtil, Package } from '@proteinjs/util-node';
import { SourceFile } from './Code';

export type TemplateArgs = {
  srcPath: string;
  additionalPackages?: Package[];
  replacePackages?: boolean;
};

export abstract class CodeTemplate {
  protected logger: Logger;
  protected templateArgs: TemplateArgs;

  constructor(templateArgs: TemplateArgs) {
    this.logger = new Logger({ name: this.constructor.name });
    this.templateArgs = templateArgs;
  }

  abstract dependencyPackages(): Package[];
  abstract sourceFiles(): SourceFile[];

  async generate() {
    await PackageUtil.installPackages(this.resolvePackages());
    for (const sourceFile of this.sourceFiles()) {
      const filePath = Fs.baseContainedJoin(this.templateArgs.srcPath, sourceFile.relativePath);
      this.logger.info({ message: `Generating source file: ${filePath}` });
      const code = await sourceFile.code.generate();
      await Fs.writeFiles([{ path: filePath, content: code }]);
      this.logger.info({ message: `Generated source file: ${filePath}` });
    }
  }

  private resolvePackages() {
    const packages: Package[] = this.templateArgs.replacePackages ? [] : this.dependencyPackages();
    if (this.templateArgs.additionalPackages) {
      packages.push(...this.templateArgs.additionalPackages);
    }
    return packages;
  }
}
