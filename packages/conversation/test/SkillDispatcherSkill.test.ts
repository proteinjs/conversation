import type { ChatCompletionContentPart, ChatCompletionMessageParam } from 'openai/resources/chat';
import { SkillDispatcherSkill } from '../src/SkillDispatcherSkill';
import { ConversationSkill } from '../src/ConversationSkill';
import { Function as ConvFunction, type ToolCallContext, type ToolPhase } from '../src/Function';
import { ChatCompletionMessageParamFactory } from '../src/ChatCompletionMessageParamFactory';
import { SdkContentParts } from '../src/sdkContentParts';
import { ToolBudget, type ToolBudgetConversion } from '../src/ToolBudget';

/**
 * Pure unit tests for SkillDispatcherSkill — no API calls, no model in the loop.
 *
 * Cover the four contracts the rest of the system relies on:
 *  - listAvailableSkills returns the catalog of unpinned skills
 *  - describeSkill renders instructions + tool catalog with JSON schemas
 *  - useSkill dispatches correctly + fires onSkillUsed for auto-pin
 *  - useSkill hands a structured-content result (a picture) through UNCHANGED, so the executor
 *    converts it exactly as it converts the same tool called directly
 *  - useSkill hands the dispatched tool the tool-call context (abort signal, phase reporter)
 *  - duplicate ids throw at construction time
 */

function makeSkill(opts: {
  id: string;
  name?: string;
  summary?: string;
  whenToUse?: string;
  instructions?: string | string[];
  functions?: ConvFunction[];
}): ConversationSkill {
  return {
    getId: () => opts.id,
    getName: () => opts.name ?? opts.id,
    getSummary: opts.summary ? () => opts.summary! : undefined,
    getWhenToUse: opts.whenToUse ? () => opts.whenToUse! : undefined,
    getSystemMessages: () => opts.instructions ?? '',
    getFunctions: () => opts.functions ?? [],
    getMessageModerators: () => [],
  };
}

function makeFn(name: string, description: string, callImpl: (args: any) => Promise<any>): ConvFunction {
  return {
    definition: {
      name,
      description,
      parameters: {
        type: 'object',
        properties: { value: { type: 'string', description: 'Some value.' } },
        required: ['value'],
        additionalProperties: false,
      },
    },
    call: callImpl,
  };
}

async function callTool(dispatcher: SkillDispatcherSkill, toolName: string, args: unknown): Promise<string> {
  return (await callToolRaw(dispatcher, toolName, args)) as string;
}

/** The tool's return value exactly as the executor receives it — no cast to string. */
async function callToolRaw(dispatcher: SkillDispatcherSkill, toolName: string, args: unknown): Promise<unknown> {
  const tool = dispatcher.getFunctions().find((f) => f.definition.name === toolName);
  if (!tool) {
    throw new Error(`Tool ${toolName} not exposed by dispatcher`);
  }
  return tool.call(args ?? {});
}

/** A 1x1 PNG. Small, but a real picture: what a vision tool hands back. */
const PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const PNG_DATA_URI = `data:image/png;base64,${PNG_BASE64}`;

const pictureParts = (): ChatCompletionContentPart[] => [
  { type: 'text', text: 'The screen as it looked.' },
  { type: 'image_url', image_url: { url: PNG_DATA_URI } },
];

/** A vision tool's result: a factory SUBCLASS carrying its own fields, like a real skill's. */
class ScreenPictureFactory extends ChatCompletionMessageParamFactory {
  constructor(
    private readonly dataUri: string,
    private readonly caption: string
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

describe('SkillDispatcherSkill', () => {
  describe('construction', () => {
    it('throws on duplicate skill ids', () => {
      const a = makeSkill({ id: 'same', name: 'A' });
      const b = makeSkill({ id: 'same', name: 'B' });
      expect(() => new SkillDispatcherSkill([a, b])).toThrow(/Duplicate skill id "same"/);
    });

    it('accepts an empty skill list and emits no functions or system message', () => {
      const dispatcher = new SkillDispatcherSkill([]);
      expect(dispatcher.size()).toBe(0);
      expect(dispatcher.getFunctions()).toEqual([]);
      expect(dispatcher.getSystemMessages()).toBe('');
    });

    it('exposes the three drill-down tools when skills are present', () => {
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'foo' })]);
      const names = dispatcher.getFunctions().map((f) => f.definition.name);
      expect(names).toEqual(['listAvailableSkills', 'describeSkill', 'useSkill']);
    });

    it('reports a stable getId and lists known skills (name + summary) in system messages', () => {
      const dispatcher = new SkillDispatcherSkill([
        makeSkill({ id: 'b', name: 'Beta', summary: 'does beta things' }),
        makeSkill({ id: 'a', name: 'Alpha', summary: 'does alpha things' }),
      ]);
      expect(dispatcher.getId()).toBe('skill-dispatcher');
      const msg = dispatcher.getSystemMessages();
      // Sorted by id; each rendered name-first with summary + id (the call key)
      // labeled at the end, so the model can match requests to skills up front.
      expect(msg).toContain('Alpha — does alpha things (id: `a`)');
      expect(msg).toContain('Beta — does beta things (id: `b`)');
      expect(msg.indexOf('(id: `a`)')).toBeLessThan(msg.indexOf('(id: `b`)'));
    });
  });

  describe('listAvailableSkills', () => {
    it('renders id, name and summary for each registered skill', async () => {
      const dispatcher = new SkillDispatcherSkill([
        makeSkill({ id: 'alpha', name: 'Alpha', summary: 'first skill' }),
        makeSkill({ id: 'beta', name: 'Beta' }),
      ]);
      const output = await callTool(dispatcher, 'listAvailableSkills', {});
      expect(output).toContain('**Alpha** — first skill (id: `alpha`)');
      expect(output).toContain('**Beta** (id: `beta`)');
      // No "—" without a summary
      expect(output).not.toContain('**Beta** —');
    });
  });

  describe('describeSkill', () => {
    it('returns an unknown-skill message for an id that is not registered', async () => {
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'known' })]);
      const output = await callTool(dispatcher, 'describeSkill', { skill: 'mystery' });
      expect(output).toContain('No skill with id "mystery"');
      expect(output).toContain('known');
    });

    it('renders instructions, when-to-use, and tools with JSON schemas', async () => {
      const fn = makeFn('doThing', 'Does a thing.', async () => 'ok');
      const dispatcher = new SkillDispatcherSkill([
        makeSkill({
          id: 'demo',
          name: 'Demo',
          summary: 'a demo skill',
          whenToUse: 'when you need to demo',
          instructions: 'Be careful with demos.',
          functions: [fn],
        }),
      ]);
      const output = await callTool(dispatcher, 'describeSkill', { skill: 'demo' });
      expect(output).toContain('# Demo');
      expect(output).toContain('**id:** `demo`');
      expect(output).toContain('**Summary:** a demo skill');
      expect(output).toContain('## When to use');
      expect(output).toContain('when you need to demo');
      expect(output).toContain('## Instructions');
      expect(output).toContain('Be careful with demos.');
      expect(output).toContain('### doThing');
      expect(output).toContain('Does a thing.');
      expect(output).toContain('```json');
      expect(output).toContain('"value"');
      expect(output).toContain('useSkill({ skill: "demo", tool: "<name>", args: { ... } })');
    });

    it('handles skills with no dispatcher-reachable tools', async () => {
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'empty', name: 'Empty' })]);
      const output = await callTool(dispatcher, 'describeSkill', { skill: 'empty' });
      expect(output).toContain('This skill has no dispatcher-reachable tools.');
    });

    it('returns a missing-argument message when skill is omitted', async () => {
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'a' })]);
      const output = await callTool(dispatcher, 'describeSkill', {});
      expect(output).toContain('Missing required argument: `skill`');
    });
  });

  describe('useSkill', () => {
    it('dispatches into the named tool and returns its result as a string', async () => {
      const fn = makeFn('echo', 'Echoes input.', async (args) => ({ got: args }));
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', name: 'Mod', functions: [fn] })]);
      const output = await callTool(dispatcher, 'useSkill', {
        skill: 'mod',
        tool: 'echo',
        args: { value: 'hi' },
      });
      expect(output).toContain('"got"');
      expect(output).toContain('"value": "hi"');
    });

    it('returns a string result directly without JSON.stringify wrapping', async () => {
      const fn = makeFn('plain', 'Plain string result.', async () => 'just a string');
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })]);
      const output = await callTool(dispatcher, 'useSkill', {
        skill: 'mod',
        tool: 'plain',
        args: {},
      });
      expect(output).toBe('just a string');
    });

    it('fires onSkillUsed exactly once per dispatch', async () => {
      const fn = makeFn('a', 'a', async () => 'ok');
      const onSkillUsed = jest.fn();
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })], { onSkillUsed });
      await callTool(dispatcher, 'useSkill', { skill: 'mod', tool: 'a', args: {} });
      expect(onSkillUsed).toHaveBeenCalledTimes(1);
      expect(onSkillUsed).toHaveBeenCalledWith('mod');
    });

    it('swallows onSkillUsed errors so the dispatch result still returns', async () => {
      const fn = makeFn('a', 'a', async () => 'dispatched');
      const onSkillUsed = jest.fn(() => {
        throw new Error('auto-pin failed');
      });
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })], { onSkillUsed });
      const output = await callTool(dispatcher, 'useSkill', { skill: 'mod', tool: 'a', args: {} });
      expect(output).toBe('dispatched');
    });

    it('returns an unknown-skill message for a missing skill id', async () => {
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'a' })]);
      const output = await callTool(dispatcher, 'useSkill', {
        skill: 'nope',
        tool: 'x',
        args: {},
      });
      expect(output).toContain('No skill with id "nope"');
    });

    it('returns an unknown-tool message listing available tool names', async () => {
      const fn = makeFn('exists', 'e', async () => 'ok');
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })]);
      const output = await callTool(dispatcher, 'useSkill', {
        skill: 'mod',
        tool: 'missing',
        args: {},
      });
      expect(output).toContain('has no dispatcher-reachable tool named "missing"');
      expect(output).toContain('Available tools: exists');
    });

    it('wraps tool errors into a readable result and still fires onSkillUsed=false', async () => {
      const fn = makeFn('broken', 'b', async () => {
        throw new Error('boom');
      });
      const onSkillUsed = jest.fn();
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })], { onSkillUsed });
      const output = await callTool(dispatcher, 'useSkill', { skill: 'mod', tool: 'broken', args: {} });
      expect(output).toContain('Error invoking mod.broken: boom');
      // Tool errored — we don't pin a skill that failed.
      expect(onSkillUsed).not.toHaveBeenCalled();
    });
  });

  describe('useSkill — structured content passes through unchanged', () => {
    const visionDispatcher = (result: () => unknown, options?: { onSkillUsed?: (id: string) => void }) =>
      new SkillDispatcherSkill(
        [
          makeSkill({
            id: 'vision',
            name: 'Vision',
            functions: [makeFn('look', 'Look at the screen.', async () => result())],
          }),
        ],
        options
      );
    const look = (dispatcher: SkillDispatcherSkill) =>
      callToolRaw(dispatcher, 'useSkill', { skill: 'vision', tool: 'look', args: {} });

    it('hands a ChatCompletionMessageParamFactory subclass through as the SAME object, not its JSON', async () => {
      const factory = new ScreenPictureFactory(PNG_DATA_URI, 'The screen as it looked.');
      const output = await look(visionDispatcher(() => factory));
      expect(output).toBe(factory);
      // What the executor then makes of it: the picture, as the direct call would have produced.
      expect(await SdkContentParts.extractContentPartsFromToolReturn(output)).toEqual(pictureParts());
    });

    it('hands a bare content-part array through as the SAME array, not its JSON', async () => {
      const parts = pictureParts();
      const output = await look(visionDispatcher(() => parts));
      expect(output).toBe(parts);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(output)).toEqual(pictureParts());
    });

    it('hands a structurally-typed factory (a create() that is not an instance) through unchanged', async () => {
      const foreign = { create: async () => pictureParts() };
      const output = await look(visionDispatcher(() => foreign));
      expect(output).toBe(foreign);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(output)).toEqual(pictureParts());
    });

    it('hands through an array whose FIRST part is the picture', async () => {
      const parts: ChatCompletionContentPart[] = [{ type: 'image_url', image_url: { url: PNG_DATA_URI } }];
      const output = await look(visionDispatcher(() => parts));
      expect(output).toBe(parts);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(output)).toEqual(parts);
    });

    it('hands through a structurally-typed factory whose create() is synchronous and yields messages', async () => {
      const foreign = { create: () => [{ role: 'user', content: 'The screen as it looked.' }] };
      const output = await look(visionDispatcher(() => foreign));
      expect(output).toBe(foreign);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(output)).toEqual([
        { type: 'text', text: 'The screen as it looked.' },
      ]);
    });

    it('never calls create() itself — the executor owns the one conversion', async () => {
      const factory = new ScreenPictureFactory(PNG_DATA_URI, 'once');
      const create = jest.spyOn(factory, 'create');
      await look(visionDispatcher(() => factory));
      expect(create).not.toHaveBeenCalled();
    });

    it('passes a picture through a dispatcher nested inside a dispatcher', async () => {
      const factory = new ScreenPictureFactory(PNG_DATA_URI, 'nested');
      const outer = new SkillDispatcherSkill([visionDispatcher(() => factory)]);
      const output = await callToolRaw(outer, 'useSkill', {
        skill: 'skill-dispatcher',
        tool: 'useSkill',
        args: { skill: 'vision', tool: 'look', args: {} },
      });
      expect(output).toBe(factory);
    });

    it('still pins the skill when the result is a picture', async () => {
      const onSkillUsed = jest.fn();
      await look(visionDispatcher(() => pictureParts(), { onSkillUsed }));
      expect(onSkillUsed).toHaveBeenCalledTimes(1);
      expect(onSkillUsed).toHaveBeenCalledWith('vision');
    });

    it("gives the dispatched tool's outcome hook the tool's own result, as a direct call does", async () => {
      const factory = new ScreenPictureFactory(PNG_DATA_URI, 'outcome');
      const seen: unknown[] = [];
      const fn: ConvFunction = {
        ...makeFn('look', 'l', async () => factory),
        getTimelineOutcome: (_args: unknown, result: unknown) => {
          seen.push(result);
          return undefined;
        },
      };
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'vision', functions: [fn] })]);
      const useSkill = dispatcher.getFunctions().find((f) => f.definition.name === 'useSkill')!;
      const result = await useSkill.call({ skill: 'vision', tool: 'look', args: {} });
      await useSkill.getTimelineOutcome!({ skill: 'vision', tool: 'look', args: {} }, result);
      expect(seen).toEqual([factory]);
    });
  });

  describe('useSkill — everything else is returned exactly as before', () => {
    const dispatcherReturning = (result: () => unknown) =>
      new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [makeFn('t', 't', async () => result())] })]);
    const run = (result: () => unknown) =>
      callToolRaw(dispatcherReturning(result), 'useSkill', { skill: 'mod', tool: 't', args: {} });

    it('a plain object is stringified byte-for-byte as before (2-space JSON)', async () => {
      const value = { id: 7, nested: { list: [1, 'two', null], flag: false }, text: 'a "quoted" line\n' };
      expect(await run(() => value)).toBe(JSON.stringify(value, null, 2));
      expect(await run(() => ({ got: { value: 'hi' } }))).toBe('{\n  "got": {\n    "value": "hi"\n  }\n}');
    });

    it('an object that merely LOOKS picture-ish (a type field, no array) is still stringified', async () => {
      const value = { type: 'image_url', image_url: { url: PNG_DATA_URI } };
      expect(await run(() => value)).toBe(JSON.stringify(value, null, 2));
    });

    // Only a `create` that can be CALLED makes a factory. A record with a field that happens to be
    // named `create` is data.
    it('a record whose `create` field is a string is still stringified', async () => {
      const value = { create: 'a new record', id: 1 };
      expect(await run(() => value)).toBe('{\n  "create": "a new record",\n  "id": 1\n}');
      expect(SdkContentParts.isStructuredToolReturn(value)).toBe(false);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(value)).toBeUndefined();
    });

    it('a record whose `create` field is null is still stringified', async () => {
      const value = { create: null };
      expect(await run(() => value)).toBe('{\n  "create": null\n}');
      expect(SdkContentParts.isStructuredToolReturn(value)).toBe(false);
      expect(await SdkContentParts.extractContentPartsFromToolReturn(value)).toBeUndefined();
    });

    const ORDINARY_DATA: Array<[string, unknown]> = [
      [
        'a list of plain records',
        [
          { id: 1, name: 'first' },
          { id: 2, name: 'second' },
        ],
      ],
      ['one record whose `type` is "text" (an object, not a list)', { type: 'text', id: 'r1', title: 'A note' }],
      ['an object that holds parts under a key', { parts: [{ type: 'text', text: 'inside' }] }],
      ['a list of strings', ['first', 'second']],
      ['a list holding null', [null]],
      ['a Date', new Date(0)],
    ];

    for (const [name, value] of ORDINARY_DATA) {
      it(`${name} is ordinary data: 2-space JSON text, and the detector refuses it`, async () => {
        const output = await run(() => value);
        expect(typeof output).toBe('string');
        expect(output).toBe(JSON.stringify(value, null, 2));
        expect(SdkContentParts.isStructuredToolReturn(value)).toBe(false);
        expect(await SdkContentParts.extractContentPartsFromToolReturn(value)).toBeUndefined();
      });
    }

    it('arrays that are not content parts are still stringified', async () => {
      expect(await run(() => [])).toBe('[]');
      expect(await run(() => [1, 2])).toBe(JSON.stringify([1, 2], null, 2));
      expect(await run(() => [{ id: 1 }, { type: 'text', text: 'second' }])).toBe(
        JSON.stringify([{ id: 1 }, { type: 'text', text: 'second' }], null, 2)
      );
    });

    it('a string passes as itself — including one that holds JSON', async () => {
      expect(await run(() => 'just a string')).toBe('just a string');
      expect(await run(() => '{"a":1}')).toBe('{"a":1}');
      expect(await run(() => '')).toBe('');
    });

    it('undefined stays undefined and null stays the text "null"', async () => {
      expect(await run(() => undefined)).toBeUndefined();
      expect(await run(() => null)).toBe('null');
    });

    it('numbers and booleans are stringified as before', async () => {
      expect(await run(() => 42)).toBe('42');
      expect(await run(() => false)).toBe('false');
    });

    it('a value JSON cannot hold falls back to String(), as before', async () => {
      const circular: Record<string, unknown> = {};
      circular.self = circular;
      expect(await run(() => circular)).toBe('[object Object]');
    });

    it("a thrown error is still reported as text, and a picture tool's throw does not pin", async () => {
      const onSkillUsed = jest.fn();
      const dispatcher = new SkillDispatcherSkill(
        [
          makeSkill({
            id: 'vision',
            functions: [
              makeFn('look', 'l', async () => {
                throw new Error('no screen to look at');
              }),
            ],
          }),
        ],
        { onSkillUsed }
      );
      expect(await callToolRaw(dispatcher, 'useSkill', { skill: 'vision', tool: 'look', args: {} })).toBe(
        'Error invoking vision.look: no screen to look at'
      );
      expect(onSkillUsed).not.toHaveBeenCalled();
    });
  });

  // A list is content parts only when EVERY element is a part the extractor can carry to the model:
  // a text part with a string `text`, an image_url part with a non-empty `image_url.url`. Reading
  // only the first element's `type` took a list of records for parts, dropped every record, and
  // handed the model an empty result.
  describe('useSkill — a list is content parts only when every element is a valid part', () => {
    const dispatcherReturning = (result: () => unknown) =>
      new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [makeFn('t', 't', async () => result())] })]);
    const run = (result: () => unknown) =>
      callToolRaw(dispatcherReturning(result), 'useSkill', { skill: 'mod', tool: 't', args: {} });

    const NOT_PARTS: Array<[string, unknown[]]> = [
      ['records typed "text" (no `text` field)', [{ type: 'text', id: 'r1', title: 'A note' }]],
      ['records typed "file"', [{ type: 'file', id: 'f1', name: 'report.pdf', size: 12 }]],
      ['records typed "image"', [{ type: 'image', id: 'i1', name: 'diagram.png' }]],
      ['records typed "image_url" (no `image_url.url`)', [{ type: 'image_url', id: 'u1', name: 'banner.png' }]],
      ['records typed "input_audio"', [{ type: 'input_audio', id: 'a1', name: 'memo.wav' }]],
      ['a text part whose `text` is not a string', [{ type: 'text', text: 5 }]],
      ['an image_url part with no url', [{ type: 'image_url', image_url: {} }]],
      ['an image_url part with an empty url', [{ type: 'image_url', image_url: { url: '' } }]],
      // Part kinds the extractor cannot carry yet: data, rather than an empty result.
      ['an `image` part', [{ type: 'image', image: PNG_DATA_URI }]],
      ['a `file` part', [{ type: 'file', file: { file_data: PNG_DATA_URI, filename: 'diagram.png' } }]],
      ['an `input_audio` part', [{ type: 'input_audio', input_audio: { data: 'AAAA', format: 'wav' } }]],
      // The strict reading of a mixed list: one element that is not a valid part makes it data.
      ['valid parts followed by a record', [...pictureParts(), { type: 'text', id: 'r1', title: 'A note' }]],
      [
        'valid parts followed by a part kind the extractor cannot carry',
        [...pictureParts(), { type: 'file', file: {} }],
      ],
      ['valid parts followed by null', [...pictureParts(), null]],
      // A hole in a sparse list is an element too: it maps to no part, so the list is data.
      ['a sparse list that is all holes', new Array(2)],
      ['a valid part beside a hole', Object.assign(new Array(2), [pictureParts()[0]])],
    ];

    for (const [name, value] of NOT_PARTS) {
      it(`${name}: ordinary data — 2-space JSON text, never an empty result`, async () => {
        expect(await run(() => value)).toBe(JSON.stringify(value, null, 2));
        expect(SdkContentParts.isStructuredToolReturn(value)).toBe(false);
        expect(await SdkContentParts.extractContentPartsFromToolReturn(value)).toBeUndefined();
      });
    }

    it('every list the detector accepts maps to as many parts as it has elements', async () => {
      const accepted: unknown[][] = [
        pictureParts(),
        [{ type: 'text', text: 'only words' }],
        [{ type: 'text', text: '' }],
        [{ type: 'image_url', image_url: { url: 'https://example.com/diagram.png' } }],
      ];
      for (const parts of accepted) {
        expect(await run(() => parts)).toBe(parts);
        const extracted = await SdkContentParts.extractContentPartsFromToolReturn(parts);
        expect(extracted).toBe(parts);
        expect(SdkContentParts.toToolResultContentParts(extracted!)).toHaveLength(parts.length);
      }
    });

    it('a record that has the FULL shape of a part is a part — shape is all there is to go on', async () => {
      const parts = [{ type: 'text', text: 'The body of the note.', id: 'r1' }];
      expect(await run(() => parts)).toBe(parts);
      expect(SdkContentParts.toToolResultContentParts(parts as ChatCompletionContentPart[])).toEqual([
        { type: 'text', text: 'The body of the note.' },
      ]);
    });
  });

  // The executor hands a tool its call context beside its arguments: the call's own abort signal
  // (Stop) and the phase reporter. A tool reached through `useSkill` is still that tool — it gets
  // the same context, so it can stop and report phases on its first turn too.
  describe('useSkill — the tool-call context reaches the dispatched tool', () => {
    const useSkillTool = (dispatcher: SkillDispatcherSkill) =>
      dispatcher.getFunctions().find((f) => f.definition.name === 'useSkill')!;
    const contextFor = (controller: AbortController, phases: ToolPhase[] = []): ToolCallContext => ({
      signal: controller.signal,
      onPhase: (phase) => phases.push(phase),
    });
    /** A skill whose one tool records every context it is called with. */
    const recordingSkill = (received: Array<ToolCallContext | undefined>) => {
      const fn: ConvFunction = {
        ...makeFn('work', 'w', async () => 'ok'),
        call: async (_args: unknown, ctx?: ToolCallContext) => {
          received.push(ctx);
          return 'ok';
        },
      };
      return makeSkill({ id: 'mod', functions: [fn] });
    };

    it('hands the dispatched tool the SAME context object the executor handed useSkill', async () => {
      const received: Array<ToolCallContext | undefined> = [];
      const ctx = contextFor(new AbortController());
      await useSkillTool(new SkillDispatcherSkill([recordingSkill(received)])).call(
        { skill: 'mod', tool: 'work', args: {} },
        ctx
      );
      expect(received).toHaveLength(1);
      expect(received[0]).toBe(ctx);
    });

    it('a call made without a context still reaches the tool without one', async () => {
      const received: Array<ToolCallContext | undefined> = [];
      await useSkillTool(new SkillDispatcherSkill([recordingSkill(received)])).call({
        skill: 'mod',
        tool: 'work',
        args: {},
      });
      expect(received).toEqual([undefined]);
    });

    it('an aborted signal is visible to the dispatched tool, and its phases reach the reporter', async () => {
      const controller = new AbortController();
      controller.abort();
      const phases: ToolPhase[] = [];
      const fn: ConvFunction = {
        ...makeFn('work', 'w', async () => 'ok'),
        call: async (_args: unknown, ctx?: ToolCallContext) => {
          ctx?.onPhase({ on: 'Checking whether to go on' });
          return ctx?.signal.aborted ? 'stopped before any work' : 'did the work';
        },
      };
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })]);
      const output = await useSkillTool(dispatcher).call(
        { skill: 'mod', tool: 'work', args: {} },
        contextFor(controller, phases)
      );
      expect(output).toBe('stopped before any work');
      expect(phases).toEqual([{ on: 'Checking whether to go on' }]);
    });

    it('forwards the context through a dispatcher nested inside a dispatcher', async () => {
      const received: Array<ToolCallContext | undefined> = [];
      const ctx = contextFor(new AbortController());
      const outer = new SkillDispatcherSkill([new SkillDispatcherSkill([recordingSkill(received)])]);
      await useSkillTool(outer).call(
        { skill: 'skill-dispatcher', tool: 'useSkill', args: { skill: 'mod', tool: 'work', args: {} } },
        ctx
      );
      expect(received[0]).toBe(ctx);
    });

    it('under a budgeted executor, a long dispatched tool names its phase and Stop ends it', async () => {
      const fn: ConvFunction = {
        ...makeFn('work', 'w', async () => 'ok'),
        call: (_args: unknown, ctx?: ToolCallContext) =>
          new Promise<string>((resolve) => {
            ctx?.onPhase({ on: 'Setting up the workspace' });
            ctx?.signal.addEventListener('abort', () => resolve('stopped by the signal'));
          }),
      };
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'mod', functions: [fn] })]);
      const conversions: ToolBudgetConversion[] = [];
      const outcome = await new ToolBudget({
        host: {
          softBudgetMs: 20,
          convert: async (call) => {
            conversions.push(call);
            return { jobId: 'job-1' };
          },
        },
        fn: useSkillTool(dispatcher),
        toolCallId: 'call-1',
        input: { skill: 'mod', tool: 'work', args: {} },
      }).run();

      // Past its budget the call became a background job, named by the DISPATCHED tool's phase…
      expect(outcome.kind).toBe('converted');
      expect(conversions).toHaveLength(1);
      expect(conversions[0].phase).toEqual({ on: 'Setting up the workspace' });
      // …and the job's Stop reaches the dispatched tool, which ends.
      conversions[0].abort();
      expect(await conversions[0].promise).toBe('stopped by the signal');
    });
  });

  describe('useSkill timeline', () => {
    const useSkillTool = (dispatcher: SkillDispatcherSkill) =>
      dispatcher.getFunctions().find((f) => f.definition.name === 'useSkill')!;

    it("carries the dispatched tool's own subject line when the tool provides one", async () => {
      const fn: ConvFunction = {
        ...makeFn('createEntry', 'c', async () => 'ok'),
        getTimelineDetail: (args: { title?: string }) => args.title,
      };
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'journal', name: 'Journal', functions: [fn] })]);
      const detail = await useSkillTool(dispatcher).getTimelineDetail!({
        skill: 'journal',
        tool: 'createEntry',
        args: { title: 'Monday morning' },
      });
      expect(detail).toBe('Monday morning');
    });

    it('falls back to the display name and tool — never the skill id — when the tool has no subject', async () => {
      const fn = makeFn('plain', 'p', async () => 'ok');
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'j-9f1c', name: 'Journal', functions: [fn] })]);
      const detail = await useSkillTool(dispatcher).getTimelineDetail!({ skill: 'j-9f1c', tool: 'plain', args: {} });
      expect(detail).toBe('Journal → plain');
      // An empty subject from the tool also falls back.
      const blank: ConvFunction = { ...makeFn('blank', 'b', async () => 'ok'), getTimelineDetail: () => '' };
      const dispatcher2 = new SkillDispatcherSkill([makeSkill({ id: 'j', name: 'Journal', functions: [blank] })]);
      expect(await useSkillTool(dispatcher2).getTimelineDetail!({ skill: 'j', tool: 'blank', args: {} })).toBe(
        'Journal → blank'
      );
    });

    it("settles with the dispatched tool's outcome (ok/detail), keeping its own node name", async () => {
      const fn: ConvFunction = {
        ...makeFn('save', 's', async () => 'Refused: nothing saved'),
        getTimelineOutcome: (_args: unknown, result: unknown) =>
          typeof result === 'string' && result.startsWith('Refused')
            ? { ok: false, name: 'save:refused', detail: 'not saved' }
            : undefined,
      };
      const dispatcher = new SkillDispatcherSkill([makeSkill({ id: 'j', name: 'Journal', functions: [fn] })]);
      const outcome = await useSkillTool(dispatcher).getTimelineOutcome!(
        { skill: 'j', tool: 'save', args: {} },
        'Refused: nothing saved'
      );
      expect(outcome).toEqual({ ok: false, detail: 'not saved' });
      expect(
        await useSkillTool(dispatcher).getTimelineOutcome!({ skill: 'j', tool: 'save', args: {} }, 'saved')
      ).toBeUndefined();
      const plain = new SkillDispatcherSkill([makeSkill({ id: 'j', functions: [makeFn('x', 'x', async () => 'ok')] })]);
      expect(await useSkillTool(plain).getTimelineOutcome!({ skill: 'j', tool: 'x', args: {} }, 'ok')).toBeUndefined();
    });
  });
});
