import { KeywordToFilesIndexModuleFactory } from '../src/fs/keyword_to_files_index/KeywordToFilesIndexModule';

test('Create keyword-files index', async() => {
  // Example usage
  const index = new KeywordToFilesIndexModuleFactory().createKeywordFilesIndex(`${process.cwd()}`)
  console.log(JSON.stringify(index, null, 2));
}, 60000)