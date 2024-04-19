import { Package, PackageUtil } from '@proteinjs/util-node';
import { Function } from '../../Function';
import { PackageModule } from './PackageModule';

export const installPackagesFunctionName = 'installPackages';
export const installPackagesFunction: Function = {
  definition: {
    name: installPackagesFunctionName,
    description: 'Get the content of files',
    parameters: {
      type: 'object',
      properties: {
        packages: {
          type: 'array',
          description: 'Packages to install',
          items: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Name of package',
              },
              version: {
                type: 'string',
                description: 'Version of package',
              },
              exactVersion: {
                type: 'boolean',
                description: 'If true, freeze the version so subsequent installs do not bump minor versions',
              },
              development: {
                type: 'boolean',
                description: 'If true, install as a dev dependency',
              },
            },
            required: ['name'],
          },
        },
        cwdPath: {
          type: 'string',
          description: 'The directory to install packages in',
        }
      },
      required: ['packages', 'cwdPath']
    },
  },
  call: async (params: { packages: Package[], cwdPath: string }) => await PackageUtil.installPackages(params.packages, params.cwdPath),
  instructions: [
    `To install a package, use the ${installPackagesFunctionName} function`,
  ],
}

export const runPackageScriptFunctionName = 'runPackageScript';
const runPackageScriptFunction: Function = {
  definition: {
    name: runPackageScriptFunctionName,
    description: 'Run `npm run x`, where `x` is an existing script in a package.json',
    parameters: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Name of the script',
        },
        cwdPath: {
          type: 'string',
          description: 'If omitted, defaults to process.cwd',
        }
      },
      required: ['name']
    },
  },
  call: async (params: { name: string, cwdPath?: string }) => await PackageUtil.runPackageScript(params.name, params.cwdPath),
  instructions: [
    `To run a npm script (such as start, test, or watch), use the ${runPackageScriptFunctionName} function`,
  ],
}

export const searchPackagesFunctionName = 'searchPackages';
export function searchPackagesFunction(packageModule: PackageModule) {
  return {
    definition: {
      name: searchPackagesFunctionName,
      description: 'Get package.json file paths for package names that include the keyword',
      parameters: {
        type: 'object',
        properties: {
          keyword: {
            type: 'string',
            description: 'Keyword to match package names',
          }
        },
        required: ['keyword']
      },
    },
    call: async (params: { keyword: string }) => await packageModule.searchPackages(params.keyword),
    instructions: [
      `To search for packages in the local repo, use the ${searchPackagesFunctionName} function`,
    ],
  }
}

export const searchLibrariesFunctionName = 'searchLibraries';
export function searchLibrariesFunction(packageModule: PackageModule) {
  return {
    definition: {
      name: searchLibrariesFunctionName,
      description: 'Return libraries that match the keyword',
      parameters: {
        type: 'object',
        properties: {
          keyword: {
            type: 'string',
            description: 'Keyword to match file names',
          }
        },
        required: ['keyword']
      },
    },
    call: async (params: { keyword: string }) => await packageModule.searchLibraries(params.keyword),
    instructions: [
      `To search for libraries in the local repo, use the ${searchLibrariesFunctionName} function`,
    ],
  }
}

export const generateTypescriptDeclarationsFunctionName = 'generateTypescriptDesclarations';
export const generateTypescriptDeclarationsFunction = {
  definition: {
    name: generateTypescriptDeclarationsFunctionName,
    description: 'Return the typescript declarations for the files',
    parameters: {
      type: 'object',
      properties: {
        tsFilePaths: {
          type: 'array',
          description: 'File paths to generate declarations for',
          items: {
            type: 'string',
          }
        }
      },
      required: ['tsFilePaths']
    },
  },
  call: async (params: { tsFilePaths: string[] }) => PackageUtil.generateTypescriptDeclarations(params),
  instructions: [
    `To generate typescript declarations for a local file, use the ${generateTypescriptDeclarationsFunctionName} function`,
  ],
}

export const npmInstallFunctionName = 'npmInstall';
export const npmInstallFunction: Function = {
  definition: {
    name: npmInstallFunctionName,
    description: 'Run `npm install` in the specified directory',
    parameters: {
      type: 'object',
      properties: {
        cwdPath: {
          type: 'string',
          description: 'Directory to execute the command from',
        }
      },
      required: ['cwdPath']
    },
  },
  call: async (params: { cwdPath: string }) => await PackageUtil.npmInstall(params.cwdPath),
  instructions: [
    `To run npm install in a specific directory, use the ${npmInstallFunctionName} function`,
  ],
}

export const uninstallPackagesFunctionName = 'uninstallPackages';
export const uninstallPackagesFunction: Function = {
  definition: {
    name: uninstallPackagesFunctionName,
    description: 'Uninstall packages',
    parameters: {
      type: 'object',
      properties: {
        packageNames: {
          type: 'array',
          description: 'Packages to uninstall',
          items: {
            type: 'string',
            description: 'Name of package',
          },
        },
        cwdPath: {
          type: 'string',
          description: 'The directory to uninstall packages from',
        }
      },
      required: ['packageNames', 'cwdPath']
    },
  },
  call: async (params: { packageNames: string[], cwdPath: string }) => await PackageUtil.uninstallPackages(params.packageNames, params.cwdPath),
  instructions: [
    `To uninstall a package, use the ${uninstallPackagesFunctionName} function`,
  ],
}

export const packageFunctions: Function[] = [
  installPackagesFunction,
  runPackageScriptFunction,
  generateTypescriptDeclarationsFunction,
  npmInstallFunction,
  uninstallPackagesFunction,
]