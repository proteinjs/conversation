import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { Conversation } from '../../src/Conversation';
import type { ConversationSkill } from '../../src/ConversationSkill';
import type { Function } from '../../src/Function';
import { fixtureModelData } from './fixtureModelData';

/**
 * A nullable type union reaches Google in the shape its API accepts — pinned on the wire.
 *
 * JSON Schema says "a list, or nothing" as `type: ['array', 'null']` with `items` beside it (a
 * skill's hand-written tool parameters say it that way; so does a structured answer's schema). The
 * Google API's `Schema` object has no type union: its `type` is ONE enum, nullability is a separate
 * `nullable` boolean, and `items` / `properties` / `required` are only legal next to the one type
 * they belong to. `@ai-sdk/google` converts every schema on the way out, and before 3.0.138 its
 * converter rendered the union as `anyOf: [{ type: 'array' }]` + `nullable: true` while `items`
 * stayed on the parent — which the API refuses as a whole (HTTP 400 INVALID_ARGUMENT:
 * `properties[steps].items: field predicate failed: $type == Type.ARRAY`). Once a skill carrying
 * such a tool was pinned in a chat, every later call on a Google model failed. Upstream fixed the
 * converter in 3.0.138 ("convert `type: [X, "null"]` to a plain nullable node so `items` and
 * `format` beside it stay valid for Gemini"); this library's floor on the provider admits nothing
 * older, and this suite reads the request body the API would receive — through the library's own
 * seams (the tool declarations `buildAiSdkTools` hands the SDK, the structured-output schema
 * `generateObject` hands it) — so the floor cannot drift back without a red. No network, no key.
 *
 * The schemas under test are the two shapes in the house that hit this: a tool parameter
 * `steps: { type: ['array', 'null'], items }` and a structured-output property
 * `reply: { type: ['object', 'null'], properties, required }`. They stay as authored — valid JSON
 * Schema — and the provider is what renders them for Google.
 */

const TIMEOUT = 30_000;

type OpenApiSchema = Record<string, unknown> & {
  type?: unknown;
  nullable?: boolean;
  anyOf?: unknown[];
  items?: unknown;
  properties?: Record<string, OpenApiSchema>;
  required?: string[];
};
type FunctionDeclaration = { name: string; description?: string; parameters?: OpenApiSchema };
type WireBody = {
  contents: unknown[];
  generationConfig?: Record<string, unknown> & { responseSchema?: OpenApiSchema; responseMimeType?: string };
  tools?: Array<{ functionDeclarations?: FunctionDeclaration[] }> | { functionDeclarations?: FunctionDeclaration[] };
};

const usageMetadata = { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 };

/** One streamed text answer, the way `streamGenerateContent?alt=sse` serves it. */
const sseText = (text: string): string =>
  `data: ${JSON.stringify({
    candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: 'STOP', index: 0 }],
    usageMetadata,
  })}\n\n`;

/** One non-streamed answer, the way `generateContent` serves it. */
const jsonText = (text: string): string =>
  JSON.stringify({
    candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: 'STOP', index: 0 }],
    usageMetadata,
  });

/** The provider over a scripted wire: every request body kept; the answer fits the door the request came through. */
class ScriptedGoogleWire {
  readonly requests: WireBody[] = [];
  readonly urls: string[] = [];

  constructor(private readonly answerText: string) {}

  readonly fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input instanceof Request ? input.url : input);
    this.urls.push(url);
    this.requests.push(JSON.parse(String(init?.body)) as WireBody);
    const streamed = url.includes('streamGenerateContent');
    return new Response(streamed ? sseText(this.answerText) : jsonText(this.answerText), {
      status: 200,
      headers: { 'content-type': streamed ? 'text/event-stream' : 'application/json' },
    });
  };

  model(modelId: string) {
    return createGoogleGenerativeAI({ apiKey: 'scripted', fetch: this.fetch as never })(modelId);
  }
}

/** The ticket tool's parameters as a skill writes them: a list of steps, or nothing; a string, or nothing. */
const createTicket: Function = {
  definition: {
    name: 'createTicket',
    description: 'File a ticket',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string' },
        steps: { type: ['array', 'null'], items: { type: 'string' } },
        expected: { type: ['string', 'null'] },
      },
      required: ['title'],
    },
  },
  call: async () => ({ id: 'T-1' }),
};

const skill: ConversationSkill = {
  getId: () => 'ticket-skill',
  getName: () => 'Support',
  getSystemMessages: () => [],
  getFunctions: () => [createTicket],
  getMessageModerators: () => [],
};

/** A structured answer's schema as a drafter writes it: a reply object, or nothing. */
const draftSchema = {
  type: 'object',
  properties: {
    reply: {
      type: ['object', 'null'],
      properties: { body: { type: 'string' } },
      required: ['body'],
    },
  },
  required: ['reply'],
};

const conversationOver = (skills: ConversationSkill[]) =>
  new Conversation({
    modelData: fixtureModelData,
    name: 'google-nullable-schemas',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills,
  });

/** The one tool declaration named, wherever the provider put the declarations in the body. */
const declarationNamed = (body: WireBody, name: string): FunctionDeclaration | undefined => {
  const sets = Array.isArray(body.tools) ? body.tools : body.tools ? [body.tools] : [];
  return sets.flatMap((set) => set.functionDeclarations ?? []).find((d) => d.name === name);
};

describe('Google nullable schemas — a type union leaves as one type plus nullable, its sub-schema beside it', () => {
  test(
    "a tool parameter `type: ['array','null']` with items goes out as { type: 'array', nullable: true, items }",
    async () => {
      const wire = new ScriptedGoogleWire('Noted.');
      const conversation = conversationOver([skill]);
      const result = await conversation.generateStream({
        messages: ['The login button does nothing.'],
        model: wire.model('gemini-3.5-flash'),
      });
      expect(await result.text).toBe('Noted.');

      expect(wire.requests).toHaveLength(1);
      const declaration = declarationNamed(wire.requests[0], 'createTicket');
      expect(declaration).toBeDefined();
      const properties = declaration?.parameters?.properties ?? {};
      // THE LAW: one type, `nullable`, and the sub-schema on the same node — no `anyOf` wrapper.
      expect(properties.steps).toEqual({ type: 'array', nullable: true, items: { type: 'string' } });
      expect(properties.expected).toEqual({ type: 'string', nullable: true });
      // The parent's own shape is untouched.
      expect(declaration?.parameters?.required).toEqual(['title']);
      expect(properties.title).toEqual({ type: 'string' });
    },
    TIMEOUT
  );

  test(
    "a structured-output property `type: ['object','null']` with properties goes out as { type: 'object', nullable: true, properties, required }",
    async () => {
      const wire = new ScriptedGoogleWire(JSON.stringify({ reply: { body: 'Thanks, filed.' } }));
      const conversation = conversationOver([]);
      const result = await conversation.generateObject<{ reply: { body: string } | null }>({
        messages: ['Draft the reply.'],
        model: wire.model('gemini-3.5-flash'),
        schema: draftSchema,
      });
      expect(result.object).toEqual({ reply: { body: 'Thanks, filed.' } });

      expect(wire.requests).toHaveLength(1);
      const generationConfig = wire.requests[0].generationConfig;
      expect(generationConfig?.responseMimeType).toBe('application/json');
      const reply = generationConfig?.responseSchema?.properties?.reply;
      // THE LAW, on the structured-output door: `properties` and `required` stay beside the one type.
      expect(reply).toEqual({
        type: 'object',
        nullable: true,
        properties: { body: { type: 'string' } },
        required: ['body'],
      });
      expect(generationConfig?.responseSchema?.required).toEqual(['reply']);
    },
    TIMEOUT
  );
});
