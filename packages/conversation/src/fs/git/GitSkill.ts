import { gitFunctions } from '@proteinjs/util-node';
import { ConversationSkill, ConversationSkillFactory } from '../../ConversationSkill';
import { Function } from '../../Function';

export class GitSkill implements ConversationSkill {
  private repoPath: string;

  constructor(repoPath: string) {
    this.repoPath = repoPath;
  }

  getId(): string {
    return 'git';
  }

  getName(): string {
    return 'Git';
  }

  getSystemMessages(): string[] {
    return [
      `After we make code changes (write to files), ask the user if they'd like to commit and sync those changes`,
      `If they want to commit, offer them 3 generated messages that summarize the changes to choose from, as well as the option to write a custom message`,
      `The generated messages should be 1-3 sentences in length, and each generated option should be a different length of summary`,
    ];
  }

  getFunctions(): Function[] {
    return [...gitFunctions];
  }

  getMessageModerators() {
    return [];
  }
}

export class GitSkillFactory implements ConversationSkillFactory {
  async createSkill(repoPath: string): Promise<GitSkill> {
    return new GitSkill(repoPath);
  }
}
