import { KeywordToFilesIndexModule } from './KeywordToFilesIndexModule';

export const searchFilesFunctionName = 'searchFiles';
export const searchFilesFunction = (module: KeywordToFilesIndexModule) => {
  return {
    definition: {
      name: searchFilesFunctionName,
      description:
        'Get file paths to files whose name contains the keyword. Case-insensitive file name search that ignores extension.',
      parameters: {
        type: 'object',
        properties: {
          keyword: {
            type: 'string',
            description: 'Search files by name for this keyword',
          },
        },
        required: ['keyword'],
      },
    },
    call: async (params: { keyword: string }) => module.searchFiles(params),
    instructions: [
      `If the user is trying to interact with a file, but does not provide a path, you can find files whose name matches a keyword using the ${searchFilesFunctionName} function`,
      `Only call functions that take in filePaths with valid file paths, if you don't know the valid file path try and search for it by keyword with the ${searchFilesFunctionName} function`,
      `If the user references a file in a package without providing a path, use the ${searchFilesFunctionName} function on the keyword to find potentially relevant files, and choose the one that references the package name in its path`,
    ],
  };
};

// {
//   definition: {
//     name: 'getDeclarations',
//     description: 'Get the typescript declarations of files',
//     parameters: {
//       type: 'object',
//       properties: {
//         tsFilePaths: {
//           type: 'array',
//           description: 'Paths to the files',
//           items: {
//             type: 'string',
//           },
//         },
//         includeDependencyDeclarations: {
//           type: 'boolean',
//           description: 'if true, returns declarations for input tsFilePaths and all dependencies. defaults to false.'
//         },
//       },
//       required: ['tsFilePaths']
//     },
//   },
//   call: async (params: { tsFilePaths: string[] }) => this.repo.getDeclarations(params),
//   instructions: [
//     `Favor calling getDeclarations over readFiles if full file content is not needed`,
//   ],
// },
