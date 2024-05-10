import { RepoFactory } from '../../src/code_template/Repo';
// import { Table } from 'conversation/src/table';
import { Conversation } from '../../src/Conversation';
import { Fs } from '@proteinjs/util-node';

// test(`Return repo of the source files and the code they depend on`, async () => {
//   const tableTemplatePath = require.resolve('conversation/src/table.ts');
//   const repo = await new RepoFactory([tableTemplatePath]).create();
//   console.log(JSON.stringify(Object.keys(repo.declarations), null, 2));
// }, 60000);

// test(`It should be able to reference code`, async () => {

// }, 60000);

// test(`Interpret all parameters of a function`, async () => {

// }, 60000);

// test(`Gather parameter data from the user`, async () => {
//   const conversation = new Conversation();

// }, 60000);

// test(`Generate code from templates guided by user input`, async () => {

// }, 60000);

test(`Create repo`, async () => {
  // console.log(JSON.stringify((await RepoFactory.createRepo(`${process.cwd()}`)).params, null, 4));
  // console.log(JSON.stringify(await Fs.getFilesInDirectory(`${process.cwd()}`, ['node_modules', 'dist']), null, 4));
}, 60000);
