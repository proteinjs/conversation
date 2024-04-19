import { OpenAi } from "../../src/OpenAi";

const helloWorldCode = "console.log('hello world');";
const helloWorldWithTicksCode = "console.log('hello ```world');";
const logXCode = "const x = 'yo';\nconsole.log(x);";
const testOneBlock = "Sure! I'm a helpful chatbot.\n```typescript\n" + helloWorldCode + "\n```";
const testOneBlockWithExtraTicks = "Sure! I'm a helpful chatbot.\n```typescript\n" + helloWorldWithTicksCode + "\n```";
const testTwoBlocks = "Sure! I'm a helpful chatbot.\n```typescript\n" + helloWorldCode + "\n```\n\nI'm still really helpful.\n```typescript\n" + logXCode + "\n```\n\nMore unhelpful chat";

test('parseCodeFromMarkdown should parse 1 block of code', () => {
  expect(OpenAi.parseCodeFromMarkdown(testOneBlock)).toBe(helloWorldCode);
});

test('parseCodeFromMarkdown should parse 1 block of code that contains ticks', () => {
  expect(OpenAi.parseCodeFromMarkdown(testOneBlockWithExtraTicks)).toBe(helloWorldWithTicksCode);
});

test('parseCodeFromMarkdown should parse 1 block of code', () => {
  expect(OpenAi.parseCodeFromMarkdown(testTwoBlocks)).toBe(helloWorldCode + '\n\n' + logXCode);
});