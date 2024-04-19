import { Logger } from '@proteinjs/util';
import { ConversationTemplate } from './ConversationTemplate';
import { createPackageConversationTemplate } from './createPackage/CreatePackageConversationTemplate';
import { ConversationModule, ConversationModuleFactory } from '../ConversationModule';
import { getConversationTemplateFunction, getConversationTemplateFunctionName, searchConversationTemplatesFunction, searchConversationTemplatesFunctionName } from './ConversationTemplateFunctions';
import { createCodeConversationTemplate } from './createCode/CreateCodeConversationTemplate';
import { createAppTemplate } from './createApp/CreateAppTemplate';

const conversationTemplates: ConversationTemplate[] = [
  createPackageConversationTemplate,
  createCodeConversationTemplate,
  createAppTemplate,
];

export type ConversationTemplateModuleParams = {
  conversationTemplates: { [conversationTemplateName: string]: ConversationTemplate},
  conversationTemplateKeywordIndex: { [keyword: string]: string[] /** conversationTemplateNames */ }
}

export class ConversationTemplateModule implements ConversationModule {
  private logger = new Logger(this.constructor.name);
  params: ConversationTemplateModuleParams;

  constructor(params: ConversationTemplateModuleParams) {
    this.params = params;
  }

  getName(): string {
    return 'Conversation Template';
  }

  searchConversationTemplates(keyword: string) {
    this.logger.info(`Searching for conversation template, keyword: ${keyword}`);
    const conversationNames = this.params.conversationTemplateKeywordIndex[keyword];
    return conversationNames || [];
  }

  async getConversationTemplate(conversationTemplateName: string): Promise<Omit<ConversationTemplate, 'instructions'> & { instructions: string[] }> {
    const conversationTemplate = this.params.conversationTemplates[conversationTemplateName];
    if (!conversationTemplate)
      return {} as any;

    const instructions = await conversationTemplate.instructions();
    return Object.assign(conversationTemplate, { instructions });
  }

  getSystemMessages() {
    return [
      `Whenever the user wants to create or talk about something, fisrt use the ${searchConversationTemplatesFunctionName} function to identify if there are relevant conversation templates to use`,
      `Use the ${getConversationTemplateFunctionName} function to get the conversation template`,
      `Once you've identified a conversation template that's relevant to the conversation, ask the user if they'd like to use that conversation template`,
      `If they want to engage in the templated conversation, ask only the conversation template questions (template.questions), then use the user's answers to carry out the conversation template instructions (template.instructions)`,
    ];
  }

  getFunctions() {
    return [
      searchConversationTemplatesFunction(this),
      getConversationTemplateFunction(this),
    ];
  }

  getMessageModerators() {
    return [];
  }
}

export class ConversationTemplateModuleFactory implements ConversationModuleFactory {
  async createModule(repoPath: string) {
    const params: ConversationTemplateModuleParams = { conversationTemplates: {}, conversationTemplateKeywordIndex: {} };
    for (let conversationTemplate of conversationTemplates) {
      params.conversationTemplates[conversationTemplate.name] = conversationTemplate;
      for (let keyword of conversationTemplate.keywords) {
        if (!params.conversationTemplateKeywordIndex[keyword])
          params.conversationTemplateKeywordIndex[keyword] = [];

        params.conversationTemplateKeywordIndex[keyword].push(conversationTemplate.name);
      }
    }

    return new ConversationTemplateModule(params);
  }
}