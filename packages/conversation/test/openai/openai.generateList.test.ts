import { OpenAi } from '../../src/OpenAi';

test('generateList should return an array of numbers, counting to 10', async () => {
  const numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  const result = await new OpenAi().generateList({ messages: [`Create a list of numbers spelled out, from 1 to 10`] });
  expect(result.map((s) => s.toLowerCase()).join(' ')).toBe(numbers.join(' '));
});
