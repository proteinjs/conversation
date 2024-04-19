import { ConversationTemplateModule } from './ConversationTemplateModule';

export const searchConversationTemplatesFunctionName = 'searchConversationTemplates';
export const searchConversationTemplatesFunction = (repo: ConversationTemplateModule) => {
  return {
    definition: {
      name: searchConversationTemplatesFunctionName,
      description: 'Get the conversation template names for templates matching the keyword',
      parameters: {
        type: 'object',
        properties: {
          keyword: {
            type: 'string',
            description: 'Search for conversation template names that match this keyword',
          },
        },
        required: ['keyword']
      },
    },
    call: async (params: { keyword: string }) => repo.searchConversationTemplates(params.keyword),
  }
}

export const getConversationTemplateFunctionName = 'getConversationTemplate';
export const getConversationTemplateFunction = (repo: ConversationTemplateModule) => {
  return {
    definition: {
      name: getConversationTemplateFunctionName,
      description: 'Get the conversation template matching the name',
      parameters: {
        type: 'object',
        properties: {
          conversationTemplateName: {
            type: 'string',
            description: 'Get the conversation template that has this name',
          },
        },
        required: ['conversationTemplateName']
      },
    },
    call: async (params: { conversationTemplateName: string }) => await repo.getConversationTemplate(params.conversationTemplateName),
  }
}