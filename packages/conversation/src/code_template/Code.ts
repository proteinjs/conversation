import { PackageUtil } from '@proteinjs/util-node';
import { Conversation } from '../Conversation';
import { Repo } from './Repo';

export type SourceFile = {
  relativePath: string;
  code: Code;
};

export type Import = {
  moduleNames: string[];
  importPathFromGeneratedFile: string;
  sourceFilePath: string;
};

export type CodeArgs = {
  conversation: Conversation;
  description: string[];
  imports?: Import[];
};

export class Code {
  private args: CodeArgs;

  constructor(args: CodeArgs) {
    this.args = args;
  }

  async generate(): Promise<string> {
    if (this.args.imports) {
      this.addImports(this.args.imports, this.args.conversation);
    }

    return await this.args.conversation.generateCode(this.args.description, 'gpt-4');
  }

  private addImports(imports: Import[], conversation: Conversation) {
    conversation.addSystemMessagesToHistory([
      this.declarationMessage(imports.map((d) => d.sourceFilePath)),
      this.importMessage(imports),
    ]);
  }

  private declarationMessage(tsFilePaths: string[]) {
    const declarationMap = PackageUtil.generateTypescriptDeclarations({
      tsFilePaths,
      includeDependencyDeclarations: true,
    });
    const declarations = Object.values(declarationMap).join('\n');
    return `Assume the following code exists in other files:\n${declarations}`;
  }

  private importMessage(imports: Omit<Import, 'filePath'>[]) {
    const importStatements = imports.map(
      (i) => `import { ${i.moduleNames.join(', ')} } from '${i.importPathFromGeneratedFile}'`
    );
    return `Add the following imports:\n${importStatements}`;
  }
}
