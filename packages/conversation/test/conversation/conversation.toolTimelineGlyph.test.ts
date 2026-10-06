import { Conversation } from '../../src/Conversation';
import { ConversationSkill } from '../../src/ConversationSkill';
import type { Function, ToolTimelineDetail } from '../../src/Function';
import { MessageModerator } from '../../src/history/MessageModerator';
import { fixtureModelData } from './fixtureModelData';

/**
 * A tool's timeline detail carries its entity's TYPE identity as a FACE (`ToolTimelineGlyph`): the
 * type's icon WHOLE — its name, its style and every drawing term the producing domain declares on
 * it — the type's hue, and the type's id when the producer has one. This package never reads the
 * glyph: the harness hands the detail through exactly as the tool composed it, so a drawing term
 * this package does not know (`letterform` below) reaches the rendering layer intact. The flat
 * form (the icon's NAME on `icon`) stays admitted for its dated window.
 *
 * The faces below are typed through `ToolTimelineDetail` itself, so at a type that admits only the
 * flat form this suite fails to compile (the face refused) rather than failing to import.
 */
type ConversationInternals = {
  ensureSkillsProcessed(): Promise<void>;
  resolveToolTimelineDetail(toolName: string, input: unknown): Promise<string | ToolTimelineDetail | undefined>;
};

const FACE: NonNullable<ToolTimelineDetail['glyph']> = {
  id: 'type-note',
  icon: { name: 'text', style: 'regular', letterform: true },
  color: '#6BACEC',
};

const skill = (fns: Function[]): ConversationSkill => ({
  getId: () => 'tool-timeline-glyph-test-skill',
  getName: () => 'ToolTimelineGlyphTestSkill',
  getSystemMessages: () => [],
  getFunctions: () => fns,
  getMessageModerators: () => [] as MessageModerator[],
});

const tool = (detail: ToolTimelineDetail): Function => ({
  definition: {
    name: 'editNote',
    description: 'Edit a note',
    parameters: { type: 'object', properties: { noteId: { type: 'string' } } },
  },
  call: async () => 'ok',
  getTimelineDetail: () => detail,
});

async function resolved(detail: ToolTimelineDetail): Promise<string | ToolTimelineDetail | undefined> {
  const harness = new Conversation({
    modelData: fixtureModelData,
    name: 'test-toolTimelineGlyph',
    logLevel: 'error',
    skills: [skill([tool(detail)])],
  }) as unknown as ConversationInternals;
  await harness.ensureSkillsProcessed();
  return harness.resolveToolTimelineDetail('editNote', { noteId: '1' });
}

describe('ToolTimelineDetail.glyph — the type face', () => {
  it('a tool hands the face whole and the harness passes it through: the icon with its drawing terms, the hue, the id', async () => {
    const detail = await resolved({ text: 'Trail notes', href: 'app://note?id=1', glyph: FACE });
    expect(detail).toEqual({
      text: 'Trail notes',
      href: 'app://note?id=1',
      glyph: { id: 'type-note', icon: { name: 'text', style: 'regular', letterform: true }, color: '#6BACEC' },
    });
  });

  it('a face without an id is a face — the id is the producer’s to have, never the renderer’s to need', async () => {
    const detail = await resolved({ text: 'Summit trip', glyph: { icon: { name: 'plane' }, color: '#DFA53E' } });
    expect(detail).toEqual({ text: 'Summit trip', glyph: { icon: { name: 'plane' }, color: '#DFA53E' } });
  });

  it('the flat form is still admitted for its dated window and passes through as composed', async () => {
    const detail = await resolved({ text: 'Old row', glyph: { icon: 'text', style: 'regular', color: '#6BACEC' } });
    expect(detail).toEqual({ text: 'Old row', glyph: { icon: 'text', style: 'regular', color: '#6BACEC' } });
  });
});
