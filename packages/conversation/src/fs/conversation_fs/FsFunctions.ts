import { File, Fs } from '@proteinjs/util-node';
import { Function } from '../../Function';
import { ConversationFsModule } from './ConversationFsModule';
import path from 'path';

const toRepoAbs = (mod: ConversationFsModule, p: string) => (path.isAbsolute(p) ? p : path.join(mod.getRepoPath(), p));

// If path doesn’t exist, try to resolve "<repo>/<basename>" to the actual file under repo
async function canonicalizePaths(mod: ConversationFsModule, paths: string[]): Promise<string[]> {
  const repo = mod.getRepoPath();
  const ignore = ['**/node_modules/**', '**/dist/**', '**/.git/**'];

  const out: string[] = [];
  for (const p of paths) {
    const abs = toRepoAbs(mod, p);
    if (await Fs.exists(abs)) {
      out.push(abs);
      continue;
    }
    const base = path.basename(p);
    if (!base) {
      out.push(abs);
      continue;
    }
    const parsed = path.parse(base); // { name, ext }
    const pattern = parsed.ext ? `**/${parsed.name}${parsed.ext}` : `**/${parsed.name}.*`;

    let matches: string[] = [];
    try {
      matches = await (Fs as any).getFilePathsMatchingGlob(repo, pattern, ignore);
    } catch {
      // fall through
    }

    if (matches.length === 1) {
      out.push(matches[0]);
    } else if (matches.length > 1) {
      // Prefer the shortest match (usually “src/...” beats deeper/duplicate locations)
      matches.sort((a, b) => a.length - b.length);
      out.push(matches[0]);
    } else {
      // No luck; keep the original absolute (will throw with a clear error)
      out.push(abs);
    }
  }
  return out;
}

export const readFilesFunctionName = 'readFiles';
export function readFilesFunction(fsModule: ConversationFsModule) {
  return {
    definition: {
      name: readFilesFunctionName,
      description: 'Get the content of files',
      parameters: {
        type: 'object',
        properties: {
          filePaths: {
            type: 'array',
            description: 'Paths to the files',
            items: { type: 'string' },
          },
        },
        required: ['filePaths'],
      },
    },
    call: async (params: { filePaths: string[] }) => {
      fsModule.pushRecentlyAccessedFilePath(params.filePaths);
      const absPaths = await canonicalizePaths(fsModule, params.filePaths);
      try {
        return await Fs.readFiles(absPaths);
      } catch (error: any) {
        return error.message;
      }
    },
    instructions: [`To read files from the local file system, use the ${readFilesFunctionName} function`],
  };
}

export const writeFilesFunctionName = 'writeFiles';
export function writeFilesFunction(fsModule: ConversationFsModule) {
  return {
    definition: {
      name: writeFilesFunctionName,
      description: 'Write files to the file system',
      parameters: {
        type: 'object',
        properties: {
          files: {
            type: 'array',
            items: {
              type: 'object',
              properties: { path: { type: 'string' }, content: { type: 'string' } },
              required: ['path', 'content'],
            },
          },
        },
        required: ['files'],
      },
    },
    call: async (params: { files: File[] }) => {
      fsModule.pushRecentlyAccessedFilePath(params.files.map((f) => f.path));
      const canon = await canonicalizePaths(
        fsModule,
        params.files.map((f) => f.path)
      );
      const absFiles = params.files.map((f, i) => ({ ...f, path: canon[i] }));
      try {
        return await Fs.writeFiles(absFiles);
      } catch (error: any) {
        return error.message;
      }
    },
    instructions: [`To write files to the local file system, use the ${writeFilesFunctionName} function`],
  };
}

export const getRecentlyAccessedFilePathsFunctionName = 'getRecentlyAccessedFilePaths';
export function getRecentlyAccessedFilePathsFunction(fsModule: ConversationFsModule) {
  return {
    definition: {
      name: getRecentlyAccessedFilePathsFunctionName,
      description: 'Get paths of files accessed during this conversation, in order from oldest to newest',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
    call: async () => fsModule.getRecentlyAccessedFilePaths(),
  };
}

const createFolderFunctionName = 'createFolder';
const createFolderFunction: Function = {
  definition: {
    name: createFolderFunctionName,
    description: 'Create a folder/directory',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path of the new directory',
        },
      },
      required: ['path'],
    },
  },
  call: async (params: { path: string }) => {
    try {
      await Fs.createFolder(params.path);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [`To create a folder on the local file system, use the ${createFolderFunctionName} function`],
};

export const fileOrDirectoryExistsFunctionName = 'fileOrDirectoryExists';
export const fileOrDirectoryExistsFunction: Function = {
  definition: {
    name: fileOrDirectoryExistsFunctionName,
    description: 'Check if a file or directory exists',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path of the file or directory',
        },
      },
      required: ['path'],
    },
  },
  call: async (params: { path: string }) => {
    try {
      return await Fs.exists(params.path);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [
    `To check if a file or folder exists on the local file system, use the ${fileOrDirectoryExistsFunctionName} function`,
  ],
};

export const getFilePathsMatchingGlobFunctionName = 'getFilePathsMatchingGlob';
const getFilePathsMatchingGlobFunction: Function = {
  definition: {
    name: getFilePathsMatchingGlobFunctionName,
    description: 'Get file paths matching a glob',
    parameters: {
      type: 'object',
      properties: {
        dirPrefix: {
          type: 'string',
          description: 'Directory to recursively search for files',
        },
        glob: {
          type: 'string',
          description: 'File matching pattern',
        },
        globIgnorePatterns: {
          type: 'array',
          description: 'Directories to ignore',
          items: {
            type: 'string',
          },
        },
      },
      required: ['dirPrefix', 'glob'],
    },
  },
  call: async (params: { dirPrefix: string; glob: string; globIgnorePatterns?: string[] }) => {
    try {
      await Fs.getFilePathsMatchingGlob(params.dirPrefix, params.glob, params.globIgnorePatterns);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [`To get file paths matching a glob, use the ${getFilePathsMatchingGlobFunctionName} function`],
};

export const renameFunctionName = 'renameFileOrDirectory';
const renameFunction: Function = {
  definition: {
    name: renameFunctionName,
    description: 'Rename a file or directory',
    parameters: {
      type: 'object',
      properties: {
        oldPath: {
          type: 'string',
          description: 'Original path of the file or directory',
        },
        newName: {
          type: 'string',
          description: 'New name for the file or directory',
        },
      },
      required: ['oldPath', 'newName'],
    },
  },
  call: async (params: { oldPath: string; newName: string }) => {
    try {
      await Fs.rename(params.oldPath, params.newName);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [`To rename a file or directory, use the ${renameFunctionName} function`],
};

export const copyFunctionName = 'copyFileOrDirectory';
const copyFunction: Function = {
  definition: {
    name: copyFunctionName,
    description: 'Copy a file or directory',
    parameters: {
      type: 'object',
      properties: {
        sourcePath: {
          type: 'string',
          description: 'Path of the source file or directory',
        },
        destinationPath: {
          type: 'string',
          description: 'Destination path for the copied file or directory',
        },
      },
      required: ['sourcePath', 'destinationPath'],
    },
  },
  call: async (params: { sourcePath: string; destinationPath: string }) => {
    try {
      await Fs.copy(params.sourcePath, params.destinationPath);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [`To copy a file or directory, use the ${copyFunctionName} function`],
};

export const moveFunctionName = 'moveFileOrDirectory';
const moveFunction: Function = {
  definition: {
    name: moveFunctionName,
    description: 'Move a file or directory',
    parameters: {
      type: 'object',
      properties: {
        sourcePath: {
          type: 'string',
          description: 'Path of the source file or directory',
        },
        destinationPath: {
          type: 'string',
          description: 'Destination path for the moved file or directory',
        },
      },
      required: ['sourcePath', 'destinationPath'],
    },
  },
  call: async (params: { sourcePath: string; destinationPath: string }) => {
    try {
      await Fs.move(params.sourcePath, params.destinationPath);
    } catch (error: any) {
      return error.message;
    }
  },
  instructions: [`To move a file or directory, use the ${moveFunctionName} function`],
};

export const grepFunctionName = 'grep';
export function grepFunction(fsModule: ConversationFsModule) {
  return {
    definition: {
      name: grepFunctionName,
      description:
        "Run system grep recursively (-F literal) within the repository and return raw stdout/stderr/code. Excludes node_modules, dist, and .git. Use 'maxResults' to cap output.",
      parameters: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description:
              'Literal text to search for (grep -F). For parentheses or special characters, pass them as-is; no regex needed.',
          },
          dir: {
            type: 'string',
            description:
              'Directory to search under. If relative, it is resolved against the repo root. Defaults to the repo root.',
          },
          maxResults: {
            type: 'number',
            description: 'Maximum number of matching lines to return (uses grep -m N).',
          },
        },
        required: ['pattern'],
      },
    },
    call: async (params: { pattern: string; dir?: string; maxResults?: number }) => {
      const repo = fsModule.getRepoPath();
      const cwd = params.dir ? toRepoAbs(fsModule, params.dir) : repo;
      try {
        return await Fs.grep({ pattern: params.pattern, dir: cwd, maxResults: params.maxResults });
      } catch (error: any) {
        return error.message;
      }
    },
    instructions: [
      `Use ${grepFunctionName} to search for literal text across the repo.`,
      `Prefer small 'maxResults' (e.g., 5-20) to avoid flooding the context.`,
      `Parse the returned stdout yourself (format: "<path>:<line>:<text>") to pick files to read.`,
    ],
  };
}

export const fsFunctions: Function[] = [
  createFolderFunction,
  fileOrDirectoryExistsFunction,
  getFilePathsMatchingGlobFunction,
  renameFunction,
  copyFunction,
  moveFunction,
];
