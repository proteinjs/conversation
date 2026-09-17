import { OpenAi } from '../../src/OpenAi';
import { fixtureModelData } from '../conversation/fixtureModelData';

/**
 * Hits the real OpenAI API (requires OPENAI_API_KEY env var) — skipped without the key, like the
 * other live suites, so a run without credentials never fails on the client's constructor.
 */
const hasApiKey = !!process.env.OPENAI_API_KEY;
const testIfKey = hasApiKey ? test : test.skip;

testIfKey('generateList should return an array of numbers, counting to 10', async () => {
  const numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  const result = await new OpenAi({ modelData: fixtureModelData }).generateList({
    messages: [`Create a list of numbers spelled out, from 1 to 10`],
  });
  expect(result.map((s) => s.toLowerCase()).join(' ')).toBe(numbers.join(' '));
});
