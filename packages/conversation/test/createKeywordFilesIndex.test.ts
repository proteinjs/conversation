import { KeywordToFilesIndexSkillFactory } from '../src/fs/keyword_to_files_index/KeywordToFilesIndexSkill';

test('Create keyword-files index', async () => {
  // Example usage
  const index = new KeywordToFilesIndexSkillFactory().createKeywordFilesIndex(`${process.cwd()}`);
  console.log(JSON.stringify(index, null, 2));
}, 60000);
