import { SkillDispatcherSkill } from '../src/SkillDispatcherSkill';
import { ConversationSkill } from '../src/ConversationSkill';
import { Function as ConvFunction } from '../src/Function';

/**
 * Pure unit tests for SkillDispatcherSkill — no API calls, no model in the loop.
 *
 * Cover the four contracts the rest of the system relies on:
 *  - listAvailableSkills returns the catalog of unpinned skills
 *  - describeSkill renders instructions + tool catalog with JSON schemas
 *  - useSkill dispatches correctly + fires onSkillUsed for auto-pin
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
  const tool = dispatcher.getFunctions().find((f) => f.definition.name === toolName);
  if (!tool) {
    throw new Error(`Tool ${toolName} not exposed by dispatcher`);
  }
  return (await tool.call(args ?? {})) as string;
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
});
