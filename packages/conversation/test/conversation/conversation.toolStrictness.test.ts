import { jsonSchema, tool, type ToolSet } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { OpenAiResponses } from '../../src/OpenAiResponses';
import { ToolStrictness } from '../../src/ToolStrictness';
import { fixtureModelData } from './fixtureModelData';

/**
 * A tool's OPTIONAL properties stay optional on every provider.
 *
 * OpenAI's Responses API treats a function tool that does not state `strict` as STRICT: it
 * rewrites the schema so every property is required, and the model then fills every optional
 * property of every tool with something — an empty string, `false`, the first enum value, a
 * made-up object. A tool that reads "this optional object is present" as a decision (a reminder's
 * `repeat`, an ask's `credential`) then acts on a value the model never chose. The library's tool
 * schemas are ordinary JSON Schema with optional properties, so the library states `strict: false`
 * for every function tool it hands OpenAI, on both of its OpenAI paths; a tool is strict only when
 * its own definition says so (the object tool-loop's `submit_result`, whose schema is rewritten
 * for strict mode on purpose).
 *
 * RED at the pre-fix library: the tools reach the OpenAI model with no `strict` at all (the
 * provider default applies), and `buildResponseTools` emits no `strict`.
 */

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

/** A tool with one required and two optional properties (one of them an optional OBJECT). */
const setReminder: Function = {
  definition: {
    name: 'setReminder',
    description: 'Set a reminder.',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string' },
        note: { type: 'string', description: 'Optional note.' },
        repeat: {
          type: 'object',
          description: 'Optional. Set ONLY for a repeating reminder.',
          properties: { every: { type: 'string', enum: ['day', 'week', 'month'] } },
          required: ['every'],
          additionalProperties: false,
        },
      },
      required: ['title'],
      additionalProperties: false,
    },
  },
  call: async () => 'ok',
};

const skill = (fns: Function[]): ConversationSkill => ({
  getId: () => 'tool-strictness-test-skill',
  getName: () => 'ToolStrictnessTestSkill',
  getSystemMessages: () => [],
  getFunctions: () => fns,
  getMessageModerators: () => [] as MessageModerator[],
});

type ProviderTool = { type: string; name: string; strict?: boolean; inputSchema?: { required?: string[] } };

/** The function tools a model of this id is handed by one `generateStream` call. */
const toolsHandedTo = async (modelId: string): Promise<ProviderTool[]> => {
  let handed: ProviderTool[] = [];
  const model = new MockLanguageModelV3({
    modelId,
    doStream: async (options: { tools?: ProviderTool[] }) => {
      handed = options.tools ?? [];
      return { stream: textStep('done') };
    },
  });
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'tool-strictness-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: [skill([setReminder])],
  });
  const result = await conversation.generateStream({ messages: ['set a reminder'], model: model as never });
  for await (const part of result.fullStream) {
    void part;
  }
  return handed.filter((tool) => tool.type === 'function');
};

type ResponseToolsInternals = {
  buildResponseTools(functions: Function[]): Array<{ name: string; strict?: boolean; parameters?: unknown }>;
};

/**
 * The polling adapter's tool builder. The OpenAI SDK client is constructed in the constructor and
 * needs an api-key env var to EXIST; no request is made here. The env var is restored at once so
 * live-gated suites in the same worker never see a bogus key.
 */
const responseToolsBuilder = (): ResponseToolsInternals => {
  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = 'test-key-never-used';
  try {
    return new OpenAiResponses({ modelData: fixtureModelData }) as unknown as ResponseToolsInternals;
  } finally {
    if (prevKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = prevKey;
    }
  }
};

describe('tool strictness — optional tool properties stay optional on every provider', () => {
  test('an OpenAI model is handed every function tool as strict: false, its schema untouched', async () => {
    const tools = await toolsHandedTo('gpt-5.6-sol');
    const reminder = tools.find((tool) => tool.name === 'setReminder');
    expect(reminder).toBeDefined();
    expect(reminder?.strict).toBe(false);
    // The optional properties are still optional in what the provider reads.
    expect(reminder?.inputSchema?.required).toEqual(['title']);
  });

  test("the other providers' tools carry no strictness statement (their wire is unchanged)", async () => {
    for (const modelId of ['claude-sonnet-5', 'gemini-3.1-pro-preview', 'grok-4.5']) {
      const tools = await toolsHandedTo(modelId);
      const reminder = tools.find((tool) => tool.name === 'setReminder');
      expect(reminder).toBeDefined();
      expect(reminder && 'strict' in reminder ? reminder.strict : undefined).toBeUndefined();
    }
  });

  test('the polling (background) OpenAI path states strict: false on every function tool too', () => {
    const tools = responseToolsBuilder().buildResponseTools([setReminder]);
    expect(tools).toHaveLength(1);
    expect(tools[0].strict).toBe(false);
    expect(tools[0].parameters).toEqual(setReminder.definition.parameters);
  });

  test('a tool whose own definition declares strict keeps it on both OpenAI paths', async () => {
    const strictTool: Function = {
      definition: { ...setReminder.definition, name: 'strictReminder', strict: true },
      call: async () => 'ok',
    };
    let handed: ProviderTool[] = [];
    const model = new MockLanguageModelV3({
      modelId: 'gpt-5.6-sol',
      doStream: async (options: { tools?: ProviderTool[] }) => {
        handed = options.tools ?? [];
        return { stream: textStep('done') };
      },
    });
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'tool-strictness-test',
      logLevel: 'error',
      limits: { enforceLimits: false },
      skills: [skill([strictTool])],
    });
    const result = await conversation.generateStream({ messages: ['set a reminder'], model: model as never });
    for await (const part of result.fullStream) {
      void part;
    }
    expect(handed.find((tool) => tool.name === 'strictReminder')?.strict).toBe(true);

    expect(responseToolsBuilder().buildResponseTools([strictTool])[0].strict).toBe(true);
  });

  test('the object tool loop on OpenAI: submit_result is declared strict (its schema is rewritten for it), the loop tools are not', async () => {
    let handed: ProviderTool[] = [];
    let calls = 0;
    const model = new MockLanguageModelV3({
      modelId: 'gpt-5.6-sol',
      doStream: async (options: { tools?: ProviderTool[] }) => {
        handed = options.tools ?? [];
        calls++;
        return {
          stream: convertArrayToReadableStream([
            { type: 'stream-start' as const, warnings: [] },
            {
              type: 'tool-call' as const,
              toolCallId: `tc-${calls}`,
              toolName: 'submit_result',
              input: '{"answer":"42","note":""}',
            },
            { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
          ]),
        };
      },
    });
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'tool-strictness-test',
      logLevel: 'error',
      limits: { enforceLimits: false },
    });
    const result = await conversation.generateObject<{ answer: string; note?: string }>({
      messages: ['Investigate, then answer.'],
      model: model as never,
      schema: {
        type: 'object',
        properties: { answer: { type: 'string' }, note: { type: 'string' } },
        required: ['answer'],
      },
      maxToolCalls: 5,
      tools: [setReminder],
    });
    expect(result.object.answer).toBe('42');

    const submit = handed.find((tool) => tool.name === 'submit_result');
    expect(submit?.strict).toBe(true);
    // Strict mode's own demand, made on purpose for this one tool: every property required.
    expect(submit?.inputSchema?.required).toEqual(['answer', 'note']);
    const reminder = handed.find((tool) => tool.name === 'setReminder');
    expect(reminder?.strict).toBe(false);
    expect(reminder?.inputSchema?.required).toEqual(['title']);
  });
});

/**
 * The two ways a function tool could still reach OpenAI without the statement.
 *
 * (1) A skill may hand in tools ALREADY BUILT as AI SDK tools through `getProviderDefinedTools`
 *     (a provider's native tool — or an ordinary function tool standing in for one on the other
 *     providers). Those never pass through the library's own tool builder.
 * (2) A bare model name no pattern recognizes is ROUTED to OpenAI by `resolveModel`. The provider
 *     the tools are stated for must be the provider that receives the call — the same routing
 *     decision — not a separate guess from the name (which answered "unknown" and said nothing).
 *
 * RED at the pre-fix library: the built tool reaches the OpenAI model with no `strict`; the
 * unrecognized model's request body carries tools with no `strict`.
 */
describe('tool strictness — every path to OpenAI', () => {
  /** A portable file reader with an OPTIONAL array — the shape a skill hands non-Anthropic providers. */
  const builtTools = (options?: { withNativeTool: boolean }): ToolSet =>
    ({
      ReadFile: tool({
        description: 'Read a file.',
        inputSchema: jsonSchema<{ path: string; viewRange?: number[] }>({
          type: 'object',
          properties: { path: { type: 'string' }, viewRange: { type: 'array', items: { type: 'number' } } },
          required: ['path'],
          additionalProperties: false,
        }),
        execute: async () => 'contents',
      }),
      StrictByChoice: tool({
        description: 'A tool whose author wrote its schema for strict mode.',
        inputSchema: jsonSchema<{ path: string }>({
          type: 'object',
          properties: { path: { type: 'string' } },
          required: ['path'],
          additionalProperties: false,
        }),
        strict: true,
        execute: async () => 'ok',
      }),
      // A provider's native tool: no function schema, nothing to be strict about.
      ...(options?.withNativeTool === false
        ? {}
        : { native_tool: { type: 'provider', id: 'acme.native_tool', args: { maxUses: 1 } } }),
    }) as unknown as ToolSet;

  const builtToolSkill = (tools: ToolSet): ConversationSkill => ({
    ...skill([]),
    getProviderDefinedTools: () => tools,
  });

  type WireTool = ProviderTool & { id?: string; args?: unknown };

  const handedWithBuiltTools = async (modelId: string, tools: ToolSet): Promise<WireTool[]> => {
    let handed: WireTool[] = [];
    const model = new MockLanguageModelV3({
      modelId,
      doStream: async (options: { tools?: WireTool[] }) => {
        handed = options.tools ?? [];
        return { stream: textStep('done') };
      },
    });
    const conversation = new Conversation({
      modelData: fixtureModelData,
      name: 'tool-strictness-test',
      logLevel: 'error',
      limits: { enforceLimits: false },
      skills: [builtToolSkill(tools)],
    });
    const result = await conversation.generateStream({ messages: ['read a file'], model: model as never });
    for await (const part of result.fullStream) {
      void part;
    }
    return handed;
  };

  test('a function tool a skill hands in already built is stated strict: false on OpenAI — its own strict: true is kept, a native provider tool is untouched, the skill’s objects are not mutated', async () => {
    const tools = builtTools();
    const handed = await handedWithBuiltTools('gpt-5.6-sol', tools);

    const readFile = handed.find((entry) => entry.name === 'ReadFile');
    expect(readFile?.type).toBe('function');
    expect(readFile?.strict).toBe(false);
    expect(readFile?.inputSchema?.required).toEqual(['path']);

    expect(handed.find((entry) => entry.name === 'StrictByChoice')?.strict).toBe(true);

    const native = handed.find((entry) => entry.name === 'native_tool');
    expect(native).toEqual({ type: 'provider', name: 'native_tool', id: 'acme.native_tool', args: { maxUses: 1 } });

    expect('strict' in (tools.ReadFile as object)).toBe(false);
    // The native tool is handed on as the very object the skill built — nothing is said on it.
    expect(ToolStrictness.statedOn('openai', tools).native_tool).toBe(tools.native_tool);
  });

  test("the other providers' built tools carry only what the skill said (their wire is unchanged)", async () => {
    for (const modelId of ['claude-sonnet-5', 'gemini-3.1-pro-preview', 'grok-4.5']) {
      const handed = await handedWithBuiltTools(modelId, builtTools());
      const readFile = handed.find((entry) => entry.name === 'ReadFile');
      expect(readFile).toBeDefined();
      expect(readFile && 'strict' in readFile ? readFile.strict : undefined).toBeUndefined();
      expect(handed.find((entry) => entry.name === 'StrictByChoice')?.strict).toBe(true);
    }
  });

  test('a model name no pattern recognizes is routed to OpenAI — and its tools are told what OpenAI is told', async () => {
    // The real routing: a STRING model id goes through `resolveModel`, which builds an OpenAI
    // Responses model for it. No request leaves the process — `fetch` is replaced for this test and
    // answers 400 once it has the request body; the key env var only has to exist.
    const prevKey = process.env.OPENAI_API_KEY;
    const prevFetch = globalThis.fetch;
    process.env.OPENAI_API_KEY = 'test-key-never-used';
    const bodies: Array<{ url: string; tools: Array<{ type: string; name: string; strict?: boolean }> }> = [];
    globalThis.fetch = (async (url: unknown, init?: { body?: unknown }) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { tools?: Array<{ type: string; name: string }> };
      bodies.push({ url: String(url), tools: body.tools ?? [] });
      return new Response(JSON.stringify({ error: { message: 'stop here', type: 'invalid_request_error' } }), {
        status: 400,
        headers: { 'content-type': 'application/json' },
      });
    }) as typeof fetch;
    try {
      const conversation = new Conversation({
        modelData: fixtureModelData,
        name: 'tool-strictness-test',
        logLevel: 'error',
        limits: { enforceLimits: false },
        skills: [{ ...skill([setReminder]), getProviderDefinedTools: () => builtTools({ withNativeTool: false }) }],
      });
      try {
        const result = await conversation.generateStream({
          messages: ['set a reminder'],
          model: 'acme-house-model-v2',
        });
        for await (const part of result.fullStream) {
          void part;
        }
      } catch {
        // The 400 is the point at which this test stops the call.
      }
    } finally {
      globalThis.fetch = prevFetch;
      if (prevKey === undefined) {
        delete process.env.OPENAI_API_KEY;
      } else {
        process.env.OPENAI_API_KEY = prevKey;
      }
    }

    expect(bodies.length).toBeGreaterThan(0);
    expect(bodies[0].url).toContain('api.openai.com');
    const functionTools = bodies[0].tools.filter((entry) => entry.type === 'function');
    expect(functionTools.map((entry) => entry.name).sort()).toEqual(['ReadFile', 'StrictByChoice', 'setReminder']);
    expect(functionTools.find((entry) => entry.name === 'setReminder')?.strict).toBe(false);
    expect(functionTools.find((entry) => entry.name === 'ReadFile')?.strict).toBe(false);
    expect(functionTools.find((entry) => entry.name === 'StrictByChoice')?.strict).toBe(true);
  });
});
