import { ConversationModule } from '../ConversationModule';
import { Function } from '../Function';
import { MessageModerator } from '../history/MessageModerator';

export class CodeTemplateModule implements ConversationModule {
  private static CODE_RESPONSE = 'Code with user input:';

  getName(): string {
    return 'Code Template';
  }

  getSystemMessages(): string[] {
    return [
      `If they want to create a function/class/object using an API we are familiar with, we will ask the user for the required information to fill in all mandatory parameters and ask them if they want to provide optional parameter values`,
      `Once we have gotten the values for all parameters, respond with '${CodeTemplateModule.CODE_RESPONSE}' followed by the code to instantiate/call the function/class/object with the user's responses for parameter values`,
      `If you're generating a call to a class that extends Template, require that the user provide Template's constructor parameters as well and combine them into the parameters passed into the base class you're instantiating`,
      `Make sure you ask for a srcPath and pass that in to the Template base class constructor arg`,
      `Surround generated code (not including imports) with a self-executing, anonymous async function like this (async function() =>{})()`,
    ];
  }
  getFunctions(): Function[] {
    return [];
  }
  getMessageModerators(): MessageModerator[] {
    return [];
  }
}

// if (response.includes(CodegenConversation.CODE_RESPONSE))
//         await this.generateCode(response, conversation);

// private async generateCode(message: string, conversation: Conversation) {
//   const code = OpenAi.parseCodeFromMarkdown(message);
//   const srcPathToken = 'TOKEN';
//   const responseSrcPath = await conversation.generateResponse([`Return the srcPath the user provided surrounded by the token ${srcPathToken}`], CodegenConversation.MODEL);
//   const srcPath = responseSrcPath.replace(/["'`]/g, '').match(/TOKEN(.*?)TOKEN/)?.[1];
//   if (!srcPath)
//     throw new Error(`Failed to parse responseSrcPath: ${responseSrcPath}`);
//   const codePath = path.join(process.cwd(), srcPath, 'template.ts');
//   await fs.ensureFile(codePath);
//   await fs.writeFile(codePath, code);
//   console.log(`Wrote file: ${codePath}`);
//   const command = 'node';
//   const args = [codePath];
//   const commandLog = `${command} ` + args.join(' ');
//   console.info(`Running command: ${commandLog}`);
//   await cmd(command, args, {OPENAI_API_KEY: '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'});
//   console.info(`Ran command: ${commandLog}`);
//   console.info(`Generated code from template: ${codePath}`);
// }
