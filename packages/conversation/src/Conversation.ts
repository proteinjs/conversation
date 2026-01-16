import { ChatCompletionMessageParam } from 'openai/resources/chat';
import { DEFAULT_MODEL, OpenAi, ToolInvocationProgressEvent } from './OpenAi';
import { OpenAI as OpenAIApi } from 'openai';
import { MessageHistory } from './history/MessageHistory';
import { Function } from './Function';
import { Logger, LogLevel } from '@proteinjs/logger';
import { Fs } from '@proteinjs/util-node';
import { MessageModerator } from './history/MessageModerator';
import { ConversationModule } from './ConversationModule';
import { TiktokenModel, encoding_for_model } from 'tiktoken';
import { searchLibrariesFunctionName } from './fs/package/PackageFunctions';
import { UsageData } from './UsageData';
import type { ModelMessage, LanguageModel } from 'ai';
import { generateObject as aiGenerateObject, jsonSchema } from 'ai';

export type ConversationParams = {
  name: string;
  modules?: ConversationModule[];
  logLevel?: LogLevel;
  limits?: {
    enforceLimits?: boolean;
    maxMessagesInHistory?: number;
    tokenLimit?: number;
  };
};

/** Object-only generation (no tool calls in this run). */
export type GenerateObjectParams<S> = {
  /** Same input contract as generateResponse */
  messages: (string | ChatCompletionMessageParam)[];

  /** A ready AI SDK model, e.g., openai('gpt-5') / openai('gpt-4o') */
  model: LanguageModel;

  abortSignal?: AbortSignal;

  /** Zod schema or JSON Schema */
  schema: S;

  /** Sampling & limits */
  temperature?: number;
  topP?: number;
  maxTokens?: number;

  /** Usage callback */
  onUsageData?: (usageData: UsageData) => Promise<void>;

  /** Append final JSON to history as assistant text; default true */
  recordInHistory?: boolean;

  /** Per-call override for reasoning effort (reasoning models only). */
  reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;
};

export type GenerateObjectOutcome<T> = {
  object: T; // validated final object
  usageData: UsageData;
};

export class Conversation {
  private tokenLimit = 50000;
  private history;
  private systemMessages: ChatCompletionMessageParam[] = [];
  private functions: Function[] = [];
  private messageModerators: MessageModerator[] = [];
  private generatedCode = false;
  private generatedList = false;
  private logger: Logger;
  private params: ConversationParams;
  private modulesProcessed = false;
  private processingModulesPromise: Promise<void> | null = null;

  constructor(params: ConversationParams) {
    this.params = params;
    this.history = new MessageHistory({
      maxMessages: params.limits?.maxMessagesInHistory,
      enforceMessageLimit: params.limits?.enforceLimits,
    });
    this.logger = new Logger({ name: params.name, logLevel: params.logLevel });

    if (params?.limits?.enforceLimits) {
      this.addFunctions('Conversation', [summarizeConversationHistoryFunction(this)]);
    }

    if (params.limits?.tokenLimit) {
      this.tokenLimit = params.limits.tokenLimit;
    }
  }

  private async ensureModulesProcessed(): Promise<void> {
    // If modules are already processed, return immediately
    if (this.modulesProcessed) {
      return;
    }

    // If modules are currently being processed, wait for that to complete
    if (this.processingModulesPromise) {
      return this.processingModulesPromise;
    }

    // Start processing modules and keep a reference to the promise
    this.processingModulesPromise = this.processModules();

    try {
      await this.processingModulesPromise;
      this.modulesProcessed = true;
    } catch (error) {
      this.logger.error({ message: 'Error processing modules', obj: { error } });
      // Reset the promise so we can try again
      this.processingModulesPromise = null;
      throw error;
    }
  }

  private async processModules(): Promise<void> {
    if (!this.params.modules || this.params.modules.length === 0) {
      return;
    }

    for (const module of this.params.modules) {
      // Get system messages and handle potential Promise
      const moduleSystemMessagesResult = module.getSystemMessages();
      let moduleSystemMessages: string[] | string;

      // Check if the result is a Promise and await it if needed
      if (moduleSystemMessagesResult instanceof Promise) {
        moduleSystemMessages = await moduleSystemMessagesResult;
      } else {
        moduleSystemMessages = moduleSystemMessagesResult;
      }

      if (!moduleSystemMessages || (Array.isArray(moduleSystemMessages) && moduleSystemMessages.length < 1)) {
        continue;
      }

      const formattedSystemMessages = Array.isArray(moduleSystemMessages)
        ? moduleSystemMessages.join('. ')
        : moduleSystemMessages;

      this.addSystemMessagesToHistory([
        `The following are instructions from the ${module.getName()} module:\n${formattedSystemMessages}`,
      ]);
      this.addFunctions(module.getName(), module.getFunctions());
      this.addMessageModerators(module.getMessageModerators());
    }
  }

  private addFunctions(moduleName: string, functions: Function[]) {
    this.functions.push(...functions);
    let functionInstructions = `The following are instructions from functions in the ${moduleName} module:`;
    let functionInstructionsAdded = false;
    for (const f of functions) {
      if (f.instructions) {
        if (!f.instructions || f.instructions.length < 1) {
          continue;
        }

        functionInstructionsAdded = true;
        const instructionsParagraph = f.instructions.join('. ');
        functionInstructions += ` ${f.definition.name}: ${instructionsParagraph}.`;
      }
    }

    if (!functionInstructionsAdded) {
      return;
    }

    this.addSystemMessagesToHistory([functionInstructions]);
  }

  private addMessageModerators(messageModerators: MessageModerator[]) {
    this.messageModerators.push(...messageModerators);
  }

  private async enforceTokenLimit(messages: (string | ChatCompletionMessageParam)[], model?: TiktokenModel) {
    if (!this.params.limits?.enforceLimits) {
      return;
    }

    const resolvedModel = model ? model : DEFAULT_MODEL;
    const encoder = encoding_for_model(resolvedModel);
    const conversation =
      this.history.toString() +
      messages
        .map((message) => {
          if (typeof message === 'string') {
            return message;
          } else {
            // Extract content from ChatCompletionMessageParam
            const contentParts = Array.isArray(message.content) ? message.content : [message.content];
            return contentParts
              .map((part) => {
                if (typeof part === 'string') {
                  return part;
                } else if (part?.type === 'text') {
                  return part.text;
                } else {
                  return ''; // Handle non-text content types as empty string
                }
              })
              .join(' ');
          }
        })
        .join('. ');
    const encoded = encoder.encode(conversation);
    console.log(`current tokens: ${encoded.length}`);
    if (encoded.length < this.tokenLimit) {
      return;
    }

    const summarizeConversationRequest = `First, call the ${summarizeConversationHistoryFunctionName} function`;
    await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateResponse({ messages: [summarizeConversationRequest], model });
    const referenceSummaryRequest = `If there's a file mentioned in the conversation summary, find and read the file to better respond to my next request. If that doesn't find anything, call the ${searchLibrariesFunctionName} function on other keywords in the conversation summary to find a file to read`;
    await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateResponse({ messages: [referenceSummaryRequest], model });
  }

  summarizeConversationHistory(summary: string) {
    this.clearHistory();
    this.history.push([{ role: 'assistant', content: `Previous conversation summary: ${summary}` }]);
  }

  private clearHistory() {
    this.history = new MessageHistory();
    this.history.push(this.systemMessages);
  }

  addSystemMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'system', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addAssistantMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'assistant', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addUserMessagesToHistory(messages: string[], unshift = false) {
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((message) => {
      return { role: 'user', content: message };
    });
    this.addMessagesToHistory(chatCompletions, unshift);
  }

  addMessagesToHistory(messages: ChatCompletionMessageParam[], unshift = false) {
    const systemMessages = messages.filter((message) => message.role === 'system');
    if (unshift) {
      this.history.getMessages().unshift(...messages);
      this.history.prune();
      this.systemMessages.unshift(...systemMessages);
    } else {
      this.history.push(messages);
      this.systemMessages.push(...systemMessages);
    }
  }

  async generateResponse({
    messages,
    model,
    maxToolCalls,
    ...rest
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
    abortSignal?: AbortSignal;
    onUsageData?: (usageData: UsageData) => Promise<void>;
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
    reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;
    maxToolCalls?: number;
  }) {
    await this.ensureModulesProcessed();
    await this.enforceTokenLimit(messages, model);

    this.logger.debug({ message: `=============== Conversation.generateResponse (start) ===============` });
    this.logger.debug({ message: `Message history`, obj: { history: this.history.getMessages(), messages } });
    this.logger.debug({ message: `=============== Conversation.generateResponse (end) ===============` });

    return await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
      ...(typeof maxToolCalls !== 'undefined' ? { maxFunctionCalls: maxToolCalls } : {}),
    }).generateResponse({ messages, model, ...rest });
  }

  async generateStreamingResponse({
    messages,
    model,
    maxToolCalls,
    ...rest
  }: {
    messages: (string | ChatCompletionMessageParam)[];
    model?: TiktokenModel;
    abortSignal?: AbortSignal;
    onUsageData?: (usageData: UsageData) => Promise<void>;
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
    reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;
    maxToolCalls?: number;
  }) {
    await this.ensureModulesProcessed();
    await this.enforceTokenLimit(messages, model);
    return await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
      ...(typeof maxToolCalls !== 'undefined' ? { maxFunctionCalls: maxToolCalls } : {}),
    }).generateStreamingResponse({ messages, model, ...rest });
  }

  /**
   * Generate a validated JSON object (no tools in this run).
   * Uses AI SDK `generateObject` which leverages provider-native structured outputs when available.
   */
  async generateObject<T>({
    messages,
    model,
    abortSignal,
    schema,
    temperature,
    topP,
    maxTokens,
    onUsageData,
    recordInHistory = true,
    reasoningEffort,
  }: GenerateObjectParams<unknown>): Promise<GenerateObjectOutcome<T>> {
    await this.ensureModulesProcessed();

    const combined: ModelMessage[] = [
      ...this.toModelMessages(this.history.getMessages()),
      ...this.toModelMessages(messages),
    ];

    // Schema normalization (Zod OR JSON Schema supported)
    const isZod =
      schema &&
      (typeof (schema as any).safeParse === 'function' ||
        (!!(schema as any)._def && typeof (schema as any)._def.typeName === 'string'));
    const normalizedSchema = isZod ? (schema as any) : jsonSchema(this.strictifyJsonSchema(schema as any));

    this.logger.debug({ message: `=============== Conversation.generateObject (start) ===============` });
    this.logger.debug({ message: `Message history`, obj: { messages: combined } });
    this.logger.debug({ message: `=============== Conversation.generateObject (end) ===============` });

    const result = await aiGenerateObject({
      model,
      abortSignal,
      messages: combined,
      schema: normalizedSchema,
      providerOptions: {
        openai: {
          strictJsonSchema: true,
          reasoningEffort,
        },
      },
      maxOutputTokens: maxTokens,
      temperature,
      topP,
      experimental_repairText: async ({ text }: any) => {
        const cleaned = String(text ?? '')
          .trim()
          .replace(/^```(?:json)?/i, '')
          .replace(/```$/, '');
        try {
          JSON.parse(cleaned);
          return cleaned;
        } catch {
          return null;
        }
      },
    } as any);

    // Record user messages to history (parity with other methods)
    const chatCompletions: ChatCompletionMessageParam[] = messages.map((m) =>
      typeof m === 'string' ? ({ role: 'user', content: m } as ChatCompletionMessageParam) : m
    );
    this.addMessagesToHistory(chatCompletions);

    // Optionally persist the final JSON in history
    if (recordInHistory) {
      try {
        const toRecord = typeof result?.object === 'object' ? JSON.stringify(result.object) : '';
        if (toRecord) {
          this.addAssistantMessagesToHistory([toRecord]);
        }
      } catch {
        /* ignore */
      }
    }

    const usageData = this.processUsageData({
      result,
      model,
    });

    if (onUsageData) {
      await onUsageData(usageData);
    }

    return {
      object: (result?.object ?? ({} as any)) as T,
      usageData,
    };
  }

  /** Convert (string | ChatCompletionMessageParam)[] -> AI SDK ModelMessage[] */
  private toModelMessages(input: (string | ChatCompletionMessageParam)[]): ModelMessage[] {
    return input.map((m) => {
      if (typeof m === 'string') {
        return { role: 'user', content: m };
      }
      const text = Array.isArray(m.content)
        ? m.content.map((p: any) => (typeof p === 'string' ? p : p?.text ?? '')).join('\n')
        : (m.content as string | undefined) ?? '';
      const role = m.role === 'system' || m.role === 'user' || m.role === 'assistant' ? m.role : 'user';
      return { role, content: text };
    });
  }

  /**
   * Strictifies a plain JSON Schema for OpenAI Structured Outputs (strict mode):
   *  - Ensures every object has `additionalProperties: false`
   *  - Ensures every object has a `required` array that includes **all** keys in `properties`
   *  - Adds missing `type: "object"` / `type: "array"` where implied by keywords
   */
  private strictifyJsonSchema(schema: any): any {
    const root = JSON.parse(JSON.stringify(schema));

    const visit = (node: any) => {
      if (!node || typeof node !== 'object') {
        return;
      }

      // If keywords imply a type but it's missing, add it (helps downstream validators)
      if (!node.type) {
        if (node.properties || node.additionalProperties || node.patternProperties) {
          node.type = 'object';
        } else if (node.items || node.prefixItems) {
          node.type = 'array';
        }
      }

      const types = Array.isArray(node.type) ? node.type : node.type ? [node.type] : [];

      // Objects: enforce strict requirements
      if (types.includes('object')) {
        // 1) additionalProperties: false
        if (node.additionalProperties !== false) {
          node.additionalProperties = false;
        }

        // 2) required must exist and include every key in properties
        if (node.properties && typeof node.properties === 'object') {
          const propKeys = Object.keys(node.properties);
          const currentReq: string[] = Array.isArray(node.required) ? node.required.slice() : [];
          const union = Array.from(new Set([...currentReq, ...propKeys]));
          node.required = union;

          // Recurse into each property schema
          for (const k of propKeys) {
            visit(node.properties[k]);
          }
        }

        // Recurse into patternProperties
        if (node.patternProperties && typeof node.patternProperties === 'object') {
          for (const k of Object.keys(node.patternProperties)) {
            visit(node.patternProperties[k]);
          }
        }

        // Recurse into $defs / definitions
        for (const defsKey of ['$defs', 'definitions']) {
          if (node[defsKey] && typeof node[defsKey] === 'object') {
            for (const key of Object.keys(node[defsKey])) {
              visit(node[defsKey][key]);
            }
          }
        }
      }

      // Arrays: recurse into items/prefixItems
      if (types.includes('array')) {
        if (node.items) {
          if (Array.isArray(node.items)) {
            node.items.forEach(visit);
          } else {
            visit(node.items);
          }
        }
        if (Array.isArray(node.prefixItems)) {
          node.prefixItems.forEach(visit);
        }
      }

      // Combinators
      for (const k of ['oneOf', 'anyOf', 'allOf']) {
        if (Array.isArray(node[k])) {
          node[k].forEach(visit);
        }
      }

      // Negation
      if (node.not) {
        visit(node.not);
      }
    };

    visit(root);
    return root;
  }

  // ---- Usage + provider metadata normalization ----

  private processUsageData(args: {
    result: any;
    model?: LanguageModel;
    toolCounts?: Map<string, number>;
    toolLedgerLen?: number;
  }): UsageData {
    const { result, model, toolCounts, toolLedgerLen } = args;

    // Try several shapes used by AI SDK / providers
    const u: any = result?.usage ?? result?.response?.usage ?? result?.response?.metadata?.usage;

    // Provider-specific extras (OpenAI Responses variants)
    const { cachedInputTokens, reasoningTokens } = this.extractOpenAiUsageDetails?.(result) ?? {};

    const input = Number.isFinite(u?.inputTokens) ? Number(u.inputTokens) : 0;
    const reasoning = Number.isFinite(reasoningTokens) ? Number(reasoningTokens) : 0;
    const output = Number.isFinite(u?.outputTokens) ? Number(u.outputTokens) : 0;
    const total = Number.isFinite(u?.totalTokens) ? Number(u.totalTokens) : input + output;
    const cached = Number.isFinite(cachedInputTokens) ? Number(cachedInputTokens) : 0;

    // Resolve model id for pricing/telemetry
    const modelId: any =
      (model as any)?.modelId ??
      result?.response?.providerMetadata?.openai?.model ??
      result?.providerMetadata?.openai?.model ??
      result?.response?.model ??
      undefined;

    const tokenUsage = {
      promptTokens: input,
      reasoningTokens: reasoning,
      cachedPromptTokens: cached,
      completionTokens: output,
      totalTokens: total,
    };

    const callsPerTool = toolCounts ? Object.fromEntries(toolCounts) : {};
    const totalToolCalls =
      typeof toolLedgerLen === 'number' ? toolLedgerLen : Object.values(callsPerTool).reduce((a, b) => a + (b || 0), 0);

    return {
      model: modelId,
      initialRequestTokenUsage: { ...tokenUsage },
      totalTokenUsage: { ...tokenUsage },
      totalRequestsToAssistant: 1,
      totalToolCalls,
      callsPerTool,
    };
  }

  // Pull OpenAI-specific cached/extra usage from provider metadata or raw usage.
  // Safe across providers; returns undefined if not available.
  private extractOpenAiUsageDetails(result: any): {
    cachedInputTokens?: number;
    reasoningTokens?: number;
  } {
    try {
      const md = result?.providerMetadata?.openai ?? result?.response?.providerMetadata?.openai;
      const usage = md?.usage ?? result?.response?.usage ?? result?.usage;

      // OpenAI Responses API has used different shapes over time; try both:
      const cachedInputTokens =
        usage?.input_tokens_details?.cached_tokens ??
        usage?.prompt_tokens_details?.cached_tokens ??
        usage?.cached_input_tokens;

      // Reasoning tokens (when available on reasoning models)
      const reasoningTokens =
        usage?.output_tokens_details?.reasoning_tokens ??
        usage?.completion_tokens_details?.reasoning_tokens ??
        usage?.reasoning_tokens;

      return {
        cachedInputTokens: typeof cachedInputTokens === 'number' ? cachedInputTokens : undefined,
        reasoningTokens: typeof reasoningTokens === 'number' ? reasoningTokens : undefined,
      };
    } catch {
      return {};
    }
  }

  async generateCode({ description, model }: { description: string[]; model?: TiktokenModel }) {
    this.logger.debug({ message: `Generating code`, obj: { description } });
    await this.ensureModulesProcessed();
    const code = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateCode({
      messages: description,
      model,
      includeSystemMessages: !this.generatedCode,
    });
    this.logger.debug({ message: `Generated code`, obj: { code } });
    this.generatedCode = true;
    return code;
  }

  async updateCodeFromFile({
    codeToUpdateFilePath,
    dependencyCodeFilePaths,
    description,
    model,
  }: {
    codeToUpdateFilePath: string;
    dependencyCodeFilePaths: string[];
    description: string;
    model?: TiktokenModel;
  }) {
    await this.ensureModulesProcessed();
    const codeToUpdate = await Fs.readFile(codeToUpdateFilePath);
    let dependencyDescription = `Assume the following exists:\n`;
    for (const dependencyCodeFilePath of dependencyCodeFilePaths) {
      const dependencCode = await Fs.readFile(dependencyCodeFilePath);
      dependencyDescription += dependencCode + '\n\n';
    }

    this.logger.debug({ message: `Updating code from file`, obj: { codeToUpdateFilePath } });
    return await this.updateCode({ code: codeToUpdate, description: dependencyDescription + description, model });
  }

  async updateCode({ code, description, model }: { code: string; description: string; model?: TiktokenModel }) {
    this.logger.debug({ message: `Updating code`, obj: { description, code } });
    await this.ensureModulesProcessed();
    const updatedCode = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).updateCode({
      code,
      description,
      model,
      includeSystemMessages: !this.generatedCode,
    });
    this.logger.debug({ message: `Updated code`, obj: { updatedCode } });
    this.generatedCode = true;
    return updatedCode;
  }

  async generateList({ description, model }: { description: string[]; model?: TiktokenModel }) {
    await this.ensureModulesProcessed();
    const list = await new OpenAi({
      history: this.history,
      functions: this.functions,
      messageModerators: this.messageModerators,
      logLevel: this.params.logLevel,
    }).generateList({
      messages: description,
      model,
      includeSystemMessages: !this.generatedList,
    });
    this.generatedList = true;
    return list;
  }
}

export const summarizeConversationHistoryFunctionName = 'summarizeConversationHistory';
export const summarizeConversationHistoryFunction = (conversation: Conversation) => {
  return {
    definition: {
      name: summarizeConversationHistoryFunctionName,
      description: 'Clear the conversation history and summarize what was in it',
      parameters: {
        type: 'object',
        properties: {
          summary: {
            type: 'string',
            description: 'A 1-3 sentence summary of the current chat history',
          },
        },
        required: ['summary'],
      },
    },
    call: async (params: { summary: string }) => conversation.summarizeConversationHistory(params.summary),
  };
};
