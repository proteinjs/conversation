import type { ChatCompletionMessageParam } from 'openai/resources/chat';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createXai } from '@ai-sdk/xai';
import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import { ChatCompletionMessageParamFactory } from '../../src/ChatCompletionMessageParamFactory';
import { Function } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { OpenAi } from '../../src/OpenAi';
import { OpenAiResponses } from '../../src/OpenAiResponses';
import { SkillDispatcherSkill } from '../../src/SkillDispatcherSkill';
import { UsageDataAccumulator } from '../../src/UsageData';
import { fixtureModelData } from './fixtureModelData';

/**
 * A picture returned by a tool reaches the model as a PICTURE when the tool is reached through
 * the skill dispatcher — on every provider, exactly as it does when the tool is called directly.
 *
 * A skill that is loaded on demand is reached through `useSkill` on its first turn. `useSkill`
 * used to JSON-stringify every non-string result, so a vision tool's result (a
 * `ChatCompletionMessageParamFactory`, or a bare content-part array) arrived at the executor as
 * TEXT: the model was handed the picture's base64 as a string — it never saw the picture, and the
 * bytes were counted (and billed) as input text. The executor's own conversion
 * (`SdkContentParts.extractContentPartsFromToolReturn`) never got the chance to run.
 *
 * Each case plays the same turn twice — the vision tool called DIRECTLY, and the same tool reached
 * through `useSkill` — and compares what the second request carries, first as the prompt the
 * library hands the provider adapter, then as the request body that adapter puts on the wire
 * (the shipped adapter, a captured `fetch`, no network).
 *
 * RED at the pre-fix library: through `useSkill` the tool result is `{ type: 'text' }` holding the
 * base64; the wire body carries it as a string; the text budget counts it.
 */

const TIMEOUT = 30_000;

/** A 1x1 PNG. */
const PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const PNG_DATA_URI = `data:image/png;base64,${PNG_BASE64}`;
const CAPTION = 'The screen as it looked.';

/** A vision tool's result: a factory SUBCLASS that holds the picture in its own fields, like a real skill's. */
class ScreenPictureFactory extends ChatCompletionMessageParamFactory {
  constructor(
    private readonly dataUri = PNG_DATA_URI,
    private readonly caption = CAPTION
  ) {
    super();
  }

  async create(): Promise<ChatCompletionMessageParam[]> {
    return [
      {
        role: 'user',
        content: [
          { type: 'text', text: this.caption },
          { type: 'image_url', image_url: { url: this.dataUri } },
        ],
      },
    ];
  }
}

const noParameters = { type: 'object' as const, properties: {}, additionalProperties: false };

const takeScreenshot = (result: () => unknown): Function => ({
  definition: { name: 'takeScreenshot', description: 'Take a screenshot.', parameters: noParameters },
  call: async () => result(),
});

const NOTE = { id: 7, title: 'a note', tags: ['x', 'y'] };
const readNote: Function = {
  definition: { name: 'readNote', description: 'Read a note.', parameters: noParameters },
  call: async () => NOTE,
};

const skill = (id: string, fns: Function[]): ConversationSkill => ({
  getId: () => id,
  getName: () => id,
  getSystemMessages: () => [],
  getFunctions: () => fns,
  getMessageModerators: () => [] as MessageModerator[],
});

const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};

type ToolCall = { id: string; name: string; input: unknown };

const toolCallStep = (calls: ToolCall[]) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    ...calls.map((call) => ({
      type: 'tool-call' as const,
      toolCallId: call.id,
      toolName: call.name,
      input: JSON.stringify(call.input),
    })),
    { type: 'finish' as const, finishReason: { unified: 'tool-calls' as const, raw: 'tool_use' }, usage },
  ]);

const textStep = (text: string) =>
  convertArrayToReadableStream([
    { type: 'stream-start' as const, warnings: [] },
    { type: 'text-start' as const, id: 't1' },
    { type: 'text-delta' as const, id: 't1', delta: text },
    { type: 'text-end' as const, id: 't1' },
    { type: 'finish' as const, finishReason: { unified: 'stop' as const, raw: 'stop' }, usage },
  ]);

type PromptPart = {
  type?: string;
  toolCallId?: string;
  toolName?: string;
  output?: { type?: string; value?: unknown };
  [key: string]: unknown;
};
type PromptMessage = { role: string; content: string | PromptPart[] };

/**
 * Play one turn: the model's first step makes `calls`, and the SECOND request's prompt — the one
 * that carries the tool results back to the model — is returned as the adapter receives it.
 */
const secondRequestPrompt = async (args: {
  family: string;
  modelId: string;
  skills: ConversationSkill[];
  calls: ToolCall[];
}): Promise<PromptMessage[]> => {
  const prompts: PromptMessage[][] = [];
  const model = new MockLanguageModelV3({
    provider: args.family,
    modelId: args.modelId,
    doStream: async (options: { prompt: unknown }) => {
      prompts.push(options.prompt as PromptMessage[]);
      return { stream: prompts.length === 1 ? toolCallStep(args.calls) : textStep('done') };
    },
  });
  const conversation = new Conversation({
    modelData: fixtureModelData,
    name: 'use-skill-structured-content-test',
    logLevel: 'error',
    limits: { enforceLimits: false },
    skills: args.skills,
  });
  const result = await conversation.generateStream({ messages: ['what is on my screen?'], model: model as never });
  for await (const part of result.fullStream) {
    void part;
  }
  expect(prompts).toHaveLength(2);
  return prompts[1];
};

const toolResult = (prompt: PromptMessage[], toolCallId: string): PromptPart => {
  for (const message of prompt) {
    if (message.role === 'tool' && Array.isArray(message.content)) {
      const part = message.content.find((p) => p.type === 'tool-result' && p.toolCallId === toolCallId);
      if (part) {
        return part;
      }
    }
  }
  throw new Error(`no tool result for ${toolCallId}`);
};

/** Everything the request carries AFTER the assistant's tool calls, with the tool's name blanked. */
const afterTheToolCalls = (prompt: PromptMessage[]): unknown => {
  const lastAssistant = prompt.map((m) => m.role).lastIndexOf('assistant');
  return JSON.parse(
    JSON.stringify(prompt.slice(lastAssistant + 1), (key, value) => (key === 'toolName' ? '<tool>' : value))
  );
};

type ConversationStatics = {
  pruneStaleToolImages(messages: PromptMessage[], keepLast: number, evictionBatch?: number): PromptMessage[];
  pruneToolResultsOverBudget(
    messages: PromptMessage[],
    budget: number,
    evictionFloorRatio?: number,
    countTokens?: (text: string) => number
  ): PromptMessage[];
};
const statics = Conversation as unknown as ConversationStatics;

/** The tool-result texts the library's text budget counts as input TEXT for this request. */
const textsCountedAsInput = (prompt: PromptMessage[]): string[] => {
  const counted: string[] = [];
  statics.pruneToolResultsOverBudget(prompt, 1_000_000, 0.75, (text) => {
    counted.push(text);
    return 1;
  });
  return counted;
};

/** How many tool results the library's image retention treats as PICTURES in this request. */
const resultsHeldAsPictures = (prompt: PromptMessage[]): number =>
  statics
    .pruneStaleToolImages(prompt, 0, 1)
    .flatMap((m) => (Array.isArray(m.content) ? m.content : []))
    .filter((p) => p.type === 'tool-result' && String(p.output?.value).includes('stale screenshot removed')).length;

/** The request body the SHIPPED provider adapter puts on the wire for `prompt` — no network. */
const wireBody = async (provider: ProviderCase, prompt: PromptMessage[]): Promise<Record<string, unknown>> => {
  let body: Record<string, unknown> | undefined;
  const capture = (async (_url: unknown, init?: { body?: unknown }) => {
    body = JSON.parse(String(init?.body));
    return new Response(JSON.stringify({ error: { message: 'captured', type: 'invalid_request_error' } }), {
      status: 400,
      headers: { 'content-type': 'application/json' },
    });
  }) as unknown as typeof fetch;
  const model = provider.adapter(capture) as { doGenerate(options: { prompt: unknown }): Promise<unknown> };
  await model.doGenerate({ prompt }).catch(() => undefined);
  if (!body) {
    throw new Error(`${provider.name}: the adapter sent no request`);
  }
  return body;
};

/** Every string anywhere in a JSON value. */
const stringsIn = (value: unknown): string[] => {
  if (typeof value === 'string') {
    return [value];
  }
  if (value && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).flatMap(stringsIn);
  }
  return [];
};

/** Every object anywhere in a JSON value. */
const objectsIn = (value: unknown): Array<Record<string, unknown>> => {
  if (!value || typeof value !== 'object') {
    return [];
  }
  const own = Array.isArray(value) ? [] : [value as Record<string, unknown>];
  return [...own, ...Object.values(value as Record<string, unknown>).flatMap(objectsIn)];
};

type ProviderCase = {
  name: string;
  family: string;
  modelId: string;
  /** xAI's adapter cannot carry a picture in a tool result: the library's existing redirect applies. */
  pictureRidesToolResult: boolean;
  adapter: (capture: typeof fetch) => unknown;
  /** The provider-native block that IS the picture on the wire. */
  isWirePicture: (block: Record<string, unknown>) => boolean;
};

const PROVIDERS: ProviderCase[] = [
  {
    name: 'anthropic',
    family: 'anthropic.messages',
    modelId: 'claude-sonnet-4-5',
    pictureRidesToolResult: true,
    adapter: (capture) => createAnthropic({ apiKey: 'test', fetch: capture })('claude-sonnet-4-5'),
    isWirePicture: (b) =>
      b.type === 'image' &&
      (b.source as Record<string, unknown>)?.type === 'base64' &&
      (b.source as Record<string, unknown>)?.media_type === 'image/png' &&
      (b.source as Record<string, unknown>)?.data === PNG_BASE64,
  },
  {
    name: 'openai (Responses)',
    family: 'openai.responses',
    modelId: 'gpt-5',
    pictureRidesToolResult: true,
    adapter: (capture) => createOpenAI({ apiKey: 'test', fetch: capture }).responses('gpt-5'),
    isWirePicture: (b) => b.type === 'input_image' && b.image_url === PNG_DATA_URI,
  },
  {
    name: 'google',
    family: 'google.generative-ai',
    modelId: 'gemini-2.5-pro',
    pictureRidesToolResult: true,
    adapter: (capture) => createGoogleGenerativeAI({ apiKey: 'test', fetch: capture })('gemini-2.5-pro'),
    isWirePicture: (b) =>
      (b.inlineData as Record<string, unknown>)?.mimeType === 'image/png' &&
      (b.inlineData as Record<string, unknown>)?.data === PNG_BASE64,
  },
  {
    name: 'xai (Responses)',
    family: 'xai.responses',
    modelId: 'grok-4',
    pictureRidesToolResult: false,
    adapter: (capture) => createXai({ apiKey: 'test', fetch: capture }).responses('grok-4'),
    isWirePicture: (b) => b.type === 'input_image' && b.image_url === PNG_DATA_URI,
  },
];

const VISION_RESULTS: Array<{ name: string; result: () => unknown }> = [
  { name: 'a ChatCompletionMessageParamFactory', result: () => new ScreenPictureFactory() },
  {
    name: 'a bare content-part array',
    result: () => [
      { type: 'text', text: CAPTION },
      { type: 'image_url', image_url: { url: PNG_DATA_URI } },
    ],
  },
];

describe('useSkill — a picture reaches the model as a picture, on every provider', () => {
  for (const provider of PROVIDERS) {
    for (const vision of VISION_RESULTS) {
      describe(`${provider.name} — ${vision.name}`, () => {
        const play = async () => {
          const direct = await secondRequestPrompt({
            ...provider,
            skills: [skill('vision', [takeScreenshot(vision.result)])],
            calls: [{ id: 'call-1', name: 'takeScreenshot', input: {} }],
          });
          const dispatched = await secondRequestPrompt({
            ...provider,
            skills: [new SkillDispatcherSkill([skill('vision', [takeScreenshot(vision.result)])])],
            calls: [{ id: 'call-1', name: 'useSkill', input: { skill: 'vision', tool: 'takeScreenshot', args: {} } }],
          });
          return { direct, dispatched };
        };

        it(
          'hands the adapter the same tool result as the direct call',
          async () => {
            const { direct, dispatched } = await play();
            const output = toolResult(dispatched, 'call-1').output;
            expect(output?.type).toBe('content');
            if (provider.pictureRidesToolResult) {
              expect(output?.value).toEqual([
                { type: 'text', text: CAPTION },
                { type: 'image-data', data: PNG_BASE64, mediaType: 'image/png' },
              ]);
            } else {
              // The library's existing redirect: the text stays in the tool result, the picture
              // rides a user message right after it.
              const values = output?.value as Array<{ type: string }>;
              expect(values.every((v) => v.type === 'text')).toBe(true);
              const last = dispatched[dispatched.length - 1];
              expect(last.role).toBe('user');
              expect(JSON.stringify(last.content)).toContain(PNG_BASE64);
            }
            expect(afterTheToolCalls(dispatched)).toEqual(afterTheToolCalls(direct));
          },
          TIMEOUT
        );

        it(
          'is accounted as a picture, never as input text',
          async () => {
            const { direct, dispatched } = await play();
            expect(textsCountedAsInput(dispatched).filter((text) => text.includes(PNG_BASE64))).toEqual([]);
            expect(textsCountedAsInput(dispatched)).toEqual(textsCountedAsInput(direct));
            expect(resultsHeldAsPictures(dispatched)).toBe(provider.pictureRidesToolResult ? 1 : 0);
            expect(resultsHeldAsPictures(dispatched)).toBe(resultsHeldAsPictures(direct));
          },
          TIMEOUT
        );

        it(
          'goes on the wire as the provider-native picture block, exactly once, never inside a string',
          async () => {
            const { direct, dispatched } = await play();
            const body = await wireBody(provider, dispatched);
            expect(objectsIn(body).filter(provider.isWirePicture)).toHaveLength(1);
            // The picture's bytes appear in no string other than the picture block's own field.
            const carriers = stringsIn(body).filter((s) => s.includes(PNG_BASE64));
            expect(carriers).toHaveLength(1);
            expect([PNG_BASE64, PNG_DATA_URI]).toContain(carriers[0]);
            // …and the same picture blocks the direct call puts on the wire.
            const directBody = await wireBody(provider, direct);
            expect(objectsIn(body).filter(provider.isWirePicture)).toEqual(
              objectsIn(directBody).filter(provider.isWirePicture)
            );
          },
          TIMEOUT
        );
      });
    }
  }
});

describe('useSkill — inside a multi-tool step', () => {
  for (const provider of PROVIDERS) {
    it(
      `${provider.name}: each result in the step keeps its own shape`,
      async () => {
        const dispatcher = new SkillDispatcherSkill([
          skill('vision', [takeScreenshot(() => new ScreenPictureFactory())]),
          skill('notes', [readNote]),
        ]);
        const pinnedNote: Function = { ...readNote, definition: { ...readNote.definition, name: 'readPinnedNote' } };
        const prompt = await secondRequestPrompt({
          ...provider,
          skills: [skill('pinned', [pinnedNote]), dispatcher],
          calls: [
            { id: 'call-direct', name: 'readPinnedNote', input: {} },
            { id: 'call-picture', name: 'useSkill', input: { skill: 'vision', tool: 'takeScreenshot', args: {} } },
            { id: 'call-object', name: 'useSkill', input: { skill: 'notes', tool: 'readNote', args: {} } },
          ],
        });

        // A plain object called directly: json, as always.
        expect(toolResult(prompt, 'call-direct').output).toEqual({ type: 'json', value: NOTE });
        // A plain object through useSkill: the same 2-space JSON text as before.
        expect(toolResult(prompt, 'call-object').output).toEqual({
          type: 'text',
          value: JSON.stringify(NOTE, null, 2),
        });
        // The picture through useSkill: structured content, beside the other two.
        const picture = toolResult(prompt, 'call-picture').output;
        expect(picture?.type).toBe('content');
        expect(textsCountedAsInput(prompt).filter((text) => text.includes(PNG_BASE64))).toEqual([]);

        const body = await wireBody(provider, prompt);
        expect(objectsIn(body).filter(provider.isWirePicture)).toHaveLength(1);
        expect(stringsIn(body).filter((s) => s.includes(PNG_BASE64))).toHaveLength(1);
      },
      TIMEOUT
    );
  }
});

/**
 * The library's two own OpenAI clients call tools themselves and each has its own, older handling
 * of a factory result (Chat Completions: the factory's messages follow the tool message; the
 * polled Responses client: the factory's TEXT only). Through `useSkill` each now gets the value
 * the direct call gives it — so each applies that same handling, not a new one.
 */
describe('useSkill — the library’s own OpenAI clients treat the result as the direct call’s', () => {
  const vision = () => skill('vision', [takeScreenshot(() => new ScreenPictureFactory())]);
  const dispatcherFunctions = () => new SkillDispatcherSkill([vision()]).getFunctions();
  const useSkillArguments = JSON.stringify({ skill: 'vision', tool: 'takeScreenshot', args: {} });

  it('Chat Completions client: the factory’s messages follow the tool message', async () => {
    type ChatInternals = {
      callFunction(
        call: { name: string; arguments: string },
        toolCallId: string,
        usage: UsageDataAccumulator,
        toolInvocations: unknown[]
      ): Promise<ChatCompletionMessageParam[]>;
    };
    const client = (functions: Function[]) =>
      new OpenAi({ modelData: fixtureModelData, functions, logLevel: 'error' }) as unknown as ChatInternals;
    const usageData = () => new UsageDataAccumulator({ model: 'gpt-4o', modelData: fixtureModelData });

    const direct = await client(vision().getFunctions()).callFunction(
      { name: 'takeScreenshot', arguments: '{}' },
      'call-1',
      usageData(),
      []
    );
    const dispatched = await client(dispatcherFunctions()).callFunction(
      { name: 'useSkill', arguments: useSkillArguments },
      'call-1',
      usageData(),
      []
    );

    expect(dispatched).toEqual(direct);
    expect(dispatched[0].role).toBe('tool');
    expect(JSON.stringify(dispatched[0])).not.toContain(PNG_BASE64);
    expect(dispatched[1]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: CAPTION },
        { type: 'image_url', image_url: { url: PNG_DATA_URI } },
      ],
    });
  });

  it('polled Responses client: the factory’s text, never the picture’s bytes as text', async () => {
    type ResponsesInternals = { formatToolReturn(returnObject: unknown): Promise<string> };
    // The OpenAI SDK client is built in the constructor and needs an api-key env var to EXIST; no
    // request is made. Restored at once so live-gated suites in the same worker never see it.
    const prevKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'test-key-never-used';
    let client: ResponsesInternals;
    try {
      client = new OpenAiResponses({ modelData: fixtureModelData }) as unknown as ResponsesInternals;
    } finally {
      if (prevKey === undefined) {
        delete process.env.OPENAI_API_KEY;
      } else {
        process.env.OPENAI_API_KEY = prevKey;
      }
    }
    const useSkill = dispatcherFunctions().find((f) => f.definition.name === 'useSkill')!;

    const direct = await client.formatToolReturn(await vision().getFunctions()[0].call({}));
    const dispatched = await client.formatToolReturn(await useSkill.call(JSON.parse(useSkillArguments)));

    expect(dispatched).toBe(direct);
    expect(dispatched).toContain(CAPTION);
    expect(dispatched).not.toContain(PNG_BASE64);
  });
});
