import * as path from 'path'
import { PackageUtil, cmd, WorkspaceMetadata } from '@proteinjs/util-node'
import { Logger } from '@proteinjs/util'

export const testWorkspace = async (workspaceMetadata?: WorkspaceMetadata) => {
  const logger = new Logger('testWorkspace');
  const workspacePath = path.resolve(__dirname, '../../..'); // __dirname: build/dist
  const { packageMap, sortedPackageNames } = workspaceMetadata ? workspaceMetadata : await PackageUtil.getWorkspaceMetadata(workspacePath);
  const filteredPackageNames = sortedPackageNames.filter(packageName => !!packageMap[packageName].packageJson.scripts?.test && packageName != 'typescript-parser');

  logger.info(`> Testing ${filteredPackageNames.length} packages in workspace (${workspacePath})`);
  for (let packageName of filteredPackageNames) {
    const localPackage = packageMap[packageName];
    const packageDir = path.dirname(localPackage.filePath);
    if (!await PackageUtil.hasTests(packageDir))
      continue;

    await cmd('npm', ['run', 'test'], { cwd: packageDir });
  }
  logger.info(`> Finished testing ${filteredPackageNames.length} packages in workspace (${workspacePath})`);
}