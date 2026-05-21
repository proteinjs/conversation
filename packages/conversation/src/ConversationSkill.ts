import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';

export interface ConversationSkill {
  getName(): string;
  /** Return array of strings that will be formatted with periods in between or return a preformatted string */
  getSystemMessages(): string[] | string | Promise<string[] | string>;
  getFunctions(): Function[];
  getMessageModerators(): MessageModerator[];
}

export interface ConversationSkillFactory {
  createSkill(repoPath: string): Promise<ConversationSkill>;
}
