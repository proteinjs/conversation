import { Conversation } from '../../src/Conversation';
import { z } from 'zod';

/**
 * Integration tests for Conversation.generateObject.
 *
 * These hit the real OpenAI API (requires OPENAI_API_KEY env var).
 * They verify structured output generation, schema strictification,
 * JSON repair, and usage data extraction.
 */

const hasApiKey = !!process.env.OPENAI_API_KEY;
const describeIfKey = hasApiKey ? describe : describe.skip;

const TEST_MODEL = 'gpt-4.1-nano';
const TIMEOUT = 60_000;

type CountryInfo = {
  name: string;
  population: number;
  continent: string;
};

type ColorList = {
  colors: string[];
};

type PersonProfile = {
  person: {
    firstName: string;
    lastName: string;
    age: number;
  };
  occupation: string;
};

describeIfKey('Conversation.generateObject', () => {
  test(
    'returns a typed object matching a Zod schema',
    async () => {
      const schema = z.object({
        name: z.string(),
        population: z.number(),
        continent: z.string(),
      });

      const conversation = new Conversation({ name: 'test-object-zod' });
      const result = await conversation.generateObject<CountryInfo>({
        messages: [
          'Give me info about France. Return the country name, population (approximate number), and continent.',
        ],
        model: TEST_MODEL,
        schema,
      });

      expect(result.object).toBeDefined();
      expect(typeof result.object.name).toBe('string');
      expect(typeof result.object.population).toBe('number');
      expect(typeof result.object.continent).toBe('string');
      expect(result.object.name.toLowerCase()).toContain('france');
      expect(result.object.population).toBeGreaterThan(1_000_000);

      // Usage should be populated
      expect(result.usage.totalTokenUsage.inputTokens).toBeGreaterThan(0);
      expect(result.usage.totalTokenUsage.outputTokens).toBeGreaterThan(0);
    },
    TIMEOUT
  );

  test(
    'returns a typed object matching a JSON Schema',
    async () => {
      const jsonSchema = {
        type: 'object',
        properties: {
          colors: {
            type: 'array',
            items: { type: 'string' },
          },
        },
        required: ['colors'],
      };

      const conversation = new Conversation({ name: 'test-object-json-schema' });
      const result = await conversation.generateObject<ColorList>({
        messages: ['List the 3 primary colors (red, blue, yellow).'],
        model: TEST_MODEL,
        schema: jsonSchema,
      });

      expect(result.object).toBeDefined();
      expect(Array.isArray(result.object.colors)).toBe(true);
      expect(result.object.colors.length).toBe(3);

      const lower = result.object.colors.map((c) => c.toLowerCase());
      expect(lower).toContain('red');
      expect(lower).toContain('blue');
      expect(lower).toContain('yellow');
    },
    TIMEOUT
  );

  test(
    'handles nested object schemas (strictification)',
    async () => {
      const schema = z.object({
        person: z.object({
          firstName: z.string(),
          lastName: z.string(),
          age: z.number(),
        }),
        occupation: z.string(),
      });

      const conversation = new Conversation({ name: 'test-object-nested' });
      const result = await conversation.generateObject<PersonProfile>({
        messages: [
          'Create a fictional person profile. Use firstName "Ada", lastName "Lovelace", age 36, occupation "Mathematician".',
        ],
        model: TEST_MODEL,
        schema,
      });

      expect(result.object.person).toBeDefined();
      expect(result.object.person.firstName).toBe('Ada');
      expect(result.object.person.lastName).toBe('Lovelace');
      expect(result.object.person.age).toBe(36);
      expect(result.object.occupation).toBe('Mathematician');
    },
    TIMEOUT
  );
});
