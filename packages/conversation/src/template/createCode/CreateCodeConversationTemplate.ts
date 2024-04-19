import { fileOrDirectoryExistsFunction, readFilesFunctionName } from '../../fs/conversation_fs/FsFunctions';
import { generateTypescriptDeclarationsFunction, installPackagesFunction, searchLibrariesFunctionName, searchPackagesFunctionName } from '../../fs/package/PackageFunctions';
import { ConversationTemplate } from '../ConversationTemplate';

export const createCodeConversationTemplate: ConversationTemplate = {
  name: 'Create code',
  keywords: [
    'create code',
    'implement',
    'write code',
    'generate code',
    'write software',
    'build something',
  ],
  description: 'User and ai developing together',
  questions: [],
  instructions: async () => {
    return [
      `You are going to generate code for the user, follow these steps:`,
      `1. Confirm the package they want to work in, if the user didn't already provide one`,
      `1.a. Use the ${searchPackagesFunctionName} function to identify the package.json file path`,
      `1.b. Use the ${readFilesFunctionName} function to read the package.json file, reference this throughout the conversation in the fileSystem`,
      `1.c. Set the cwd to the directory of the package.json file`,
      `2. Ask for a file name to work in, if the user didn't alrady provide one`,
      `2.a. Whenever the user wants to create a new source file, default to creating it in the package src/ folder`,
      `2.b. Confirm the package-relative file path with the user. Only after the users confirms the path, create the file if it doesn't exist`,
      `2.b.1. Use the ${fileOrDirectoryExistsFunction.definition.name} function to confirm if a file exists`,
      `3. Once working in a file, ask the user what they'd like to create`,
      `3.a. If the user references a library, use the ${searchLibrariesFunctionName} function to identify local libraries that can be imported`,
      `3.a.1. Confirm the library file name and package name with the user, if they provide a different library or package name, repeat step 3.a.`,
      `3.a.2. Call the ${generateTypescriptDeclarationsFunction.definition.name} function to get the typescript declaration of the library file`,
      `3.a.3. Use the typescript declaration and the package name to add the import statements to the top of the file`,
      `3.a.4. Use the ${readFilesFunctionName} function on the package.json if it's not in the fileSystem`,
      `3.a.5. Check the pacakge.json dependencies, if the imported package is not already a dependency, use the ${installPackagesFunction.definition.name} function to install it`,
      `3.a.5.a. Use the ${searchPackagesFunctionName} on the import package to derermine if it's in the local repo; if it is, calculate the relative path from the cwd package to the package being installed, use that path as the version when installing the package`,
      `3.b. Generate the code the user asked to create, leveraging the imported library where appropriate`,
      `3.c. When writing the code to file, if updating an existing file, be sure to read the file first to not blow away existing code. Be sure to preserve comments as well.`,
      `4. Repeat 3. unless the user asks to switch packages or files`,
    ];
  }
}