import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';

export interface ConversationModule {
  getName(): string;
  /** Return array of strings that will be formatted with periods in between or return a preformatted string */
  getSystemMessages(): string[] | string;
  getFunctions(): Function[];
  getMessageModerators(): MessageModerator[];
}

export interface ConversationModuleFactory {
  createModule(repoPath: string): Promise<ConversationModule>;
}
