import type { LanguageModel, ToolSet, LanguageModelUsage, ReasoningOutput } from 'ai';
import type { LanguageModelV3Source } from '@ai-sdk/provider';
import { streamText, generateObject as aiGenerateObject, jsonSchema, stepCountIs } from 'ai';
import type { RepairTextFunction } from 'ai';
import { Logger, LogLevel } from '@proteinjs/logger';
import { ConversationModule } from './ConversationModule';
import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';
import { MessageHistory } from './history/MessageHistory';
import { UsageData, UsageDataAccumulator, TokenUsage } from './UsageData';
import { resolveModel, inferProvider } from './resolveModel';
import type { ToolInvocationProgressEvent, ToolInvocationResult } from './OpenAi';
import type { OpenAiResponses, OpenAiServiceTier } from './OpenAiResponses';
import type { ChatCompletionMessageParam } from 'openai/resources/chat';
import { TiktokenModel } from 'tiktoken';

// Re-export for convenience
export type { ToolInvocationProgressEvent, ToolInvocationResult } from './OpenAi';

// ────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────

export type ConversationParams = {
  name: string;
  modules?: ConversationModule[];
  logLevel?: LogLevel;
  defaultModel?: LanguageModel | string;
  limits?: {
    enforceLimits?: boolean;
    maxMessagesInHistory?: number;
    tokenLimit?: number;
  };
};

/** Message format accepted by Conversation methods. */
export type ConversationMessage =
  | string
  | {
      role: 'system' | 'user' | 'assistant' | 'developer' | 'tool' | 'function' | (string & {});
      content: string | null | unknown;
    };

/** @deprecated Use `GenerateObjectResult` instead. */
export type GenerateObjectOutcome<T> = GenerateObjectResult<T>;

export type ReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh';

export type GenerateStreamParams = {
  messages: ConversationMessage[];
  model?: LanguageModel | string;
  reasoningEffort?: ReasoningEffort;
  webSearch?: boolean;
  tools?: Function[];
  onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
  onUsageData?: (usageData: UsageData) => Promise<void>;
  abortSignal?: AbortSignal;
  maxToolCalls?: number;

  // OpenAI-specific
  backgroundMode?: boolean;
  serviceTier?: OpenAiServiceTier;
  maxBackgroundWaitMs?: number;
};

/** The result of generateStream. All properties are available immediately for streaming consumption. */
export type StreamResult = {
  /** Async iterable of text content chunks. */
  textStream: AsyncIterable<string>;
  /** Async iterable of reasoning/thinking chunks (empty if model doesn't support reasoning). */
  reasoningStream: AsyncIterable<string>;
  /** Resolves to the full text when generation completes. */
  text: Promise<string>;
  /** Resolves to the full reasoning text when generation completes. */
  reasoning: Promise<string>;
  /** Resolves to source citations (web search results, etc.). */
  sources: Promise<StreamSource[]>;
  /** Resolves to usage data when generation completes. */
  usage: Promise<UsageData>;
  /** Resolves to tool invocation results. */
  toolInvocations: Promise<ToolInvocationResult[]>;
};

export type StreamSource = {
  url?: string;
  title?: string;
};

export type GenerateObjectParams<T> = {
  messages: ConversationMessage[];
  model?: LanguageModel | string;
  schema: any; // Zod schema or JSON Schema
  reasoningEffort?: ReasoningEffort;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  abortSignal?: AbortSignal;
  onUsageData?: (usageData: UsageData) => Promise<void>;
  recordInHistory?: boolean;

  // OpenAI-specific
  backgroundMode?: boolean;
  serviceTier?: OpenAiServiceTier;
  maxBackgroundWaitMs?: number;
};

export type GenerateObjectResult<T> = {
  object: T;
  usage: UsageData;
  reasoning?: string;
  toolInvocations: ToolInvocationResult[];
};

export type GenerateResponseResult = {
  text: string;
  reasoning?: string;
  sources: StreamSource[];
  usage: UsageData;
  toolInvocations: ToolInvocationResult[];
};

// ────────────────────────────────────────────────────────────────
// Default constants
// ────────────────────────────────────────────────────────────────

const DEFAULT_MODEL = 'gpt-4o' as TiktokenModel;
const DEFAULT_TOKEN_LIMIT = 50_000;

// ────────────────────────────────────────────────────────────────
// Conversation class
// ────────────────────────────────────────────────────────────────

export class Conversation {
  private tokenLimit: number;
  private history: MessageHistory;
  private systemMessages: ConversationMessage[] = [];
  private functions: Function[] = [];
  private messageModerators: MessageModerator[] = [];
  private logger: Logger;
  private params: ConversationParams;
  private modulesProcessed = false;
  private processingModulesPromise: Promise<void> | null = null;

  constructor(params: ConversationParams) {
    this.params = params;
    this.tokenLimit = params.limits?.tokenLimit ?? DEFAULT_TOKEN_LIMIT;
    this.history = new MessageHistory({
      maxMessages: params.limits?.maxMessagesInHistory,
      enforceMessageLimit: params.limits?.enforceLimits,
    });
    this.logger = new Logger({ name: params.name, logLevel: params.logLevel });
  }

  // ────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────

  /**
   * Stream a text response from the model.
   *
   * Returns a `StreamResult` with async iterables for text and reasoning chunks,
   * plus promises that resolve when generation completes.
   *
   * For OpenAI models with high reasoning effort or pro models, this may
   * fall back to background/polling mode via `OpenAiResponses` and return
   * the full result as a single-chunk stream.
   */
  async generateStream(params: GenerateStreamParams): Promise<StreamResult> {
    await this.ensureModulesProcessed();

    const model = this.resolveModelInstance(params.model);
    const modelString = this.getModelString(params.model);
    const provider = inferProvider(params.model ?? this.params.defaultModel ?? DEFAULT_MODEL);

    // Check if we should use background/polling mode (OpenAI-specific)
    if (provider === 'openai' && this.shouldUseBackgroundMode(modelString, params)) {
      return this.generateStreamViaPolling(params, modelString);
    }

    // Build messages for the AI SDK
    const messages = this.buildAiSdkMessages(params.messages);

    // Build tools from module functions + any extra tools
    const allFunctions = [...this.functions, ...(params.tools ?? [])];
    const tools = this.buildAiSdkTools(allFunctions);

    // Build provider options
    const providerOptions = this.buildProviderOptions(provider, params);

    const result = streamText({
      model,
      messages,
      tools: Object.keys(tools).length > 0 ? tools : undefined,
      stopWhen: stepCountIs(params.maxToolCalls ?? 50),
      abortSignal: params.abortSignal,
      providerOptions,
      ...(params.webSearch && provider === 'openai' ? { toolChoice: 'auto' as const } : {}),
    });

    // Build the StreamResult
    const usagePromise = this.buildUsagePromise(result, modelString, params);
    const toolInvocationsPromise = this.buildToolInvocationsPromise(result);

    return {
      textStream: result.textStream,
      reasoningStream: this.extractReasoningStream(result),
      text: Promise.resolve(result.text),
      reasoning: Promise.resolve(result.reasoning).then((parts: ReasoningOutput[]) =>
        parts
          ? parts
              .filter((part) => part.type === 'reasoning')
              .map((part) => part.text)
              .join('')
          : ''
      ),
      sources: Promise.resolve(result.sources).then((s: LanguageModelV3Source[]) =>
        (s ?? []).map((source) => ({
          url: source.sourceType === 'url' ? source.url : undefined,
          title: source.sourceType === 'url' ? source.title : undefined,
        }))
      ),
      usage: usagePromise,
      toolInvocations: toolInvocationsPromise,
    };
  }

  /**
   * Generate a strongly-typed structured object from the model.
   *
   * This is promise-based (not streaming-first) to guarantee the
   * type contract. Reasoning is available on the result after completion.
   *
   * For OpenAI models with high reasoning or pro models, this uses
   * `OpenAiResponses` with background/polling mode.
   */
  async generateObject<T>(params: GenerateObjectParams<T>): Promise<GenerateObjectResult<T>> {
    await this.ensureModulesProcessed();

    const model = this.resolveModelInstance(params.model);
    const modelString = this.getModelString(params.model);
    const provider = inferProvider(params.model ?? this.params.defaultModel ?? DEFAULT_MODEL);

    // Check if we should use background/polling mode (OpenAI-specific)
    if (provider === 'openai' && this.shouldUseBackgroundMode(modelString, params)) {
      return this.generateObjectViaPolling(params, modelString);
    }

    const messages = this.buildAiSdkMessages(params.messages);

    // Schema normalization
    const isZod = this.isZodSchema(params.schema);
    const normalizedSchema = isZod ? params.schema : jsonSchema(this.strictifyJsonSchema(params.schema));

    const result = await aiGenerateObject({
      model,
      messages,
      schema: normalizedSchema,
      abortSignal: params.abortSignal,
      maxOutputTokens: params.maxTokens,
      temperature: params.temperature,
      topP: params.topP,
      providerOptions: this.buildProviderOptions(provider, params),
      experimental_repairText: (async ({ text }) => {
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
      }) as RepairTextFunction,
    });

    // Record in history
    if (params.recordInHistory !== false) {
      try {
        const toRecord = typeof result?.object === 'object' ? JSON.stringify(result.object) : '';
        if (toRecord) {
          this.addAssistantMessagesToHistory([toRecord]);
        }
      } catch {
        /* ignore */
      }
    }

    const usage = this.processAiSdkUsage(result, modelString);

    if (params.onUsageData) {
      await params.onUsageData(usage);
    }

    // Extract reasoning if available
    const reasoning = this.extractReasoningFromResult(result);

    return {
      object: (result?.object ?? {}) as T,
      usage,
      reasoning: reasoning || undefined,
      toolInvocations: [],
    };
  }

  /**
   * Non-streaming convenience: generates a text response and waits for completion.
   */
  async generateResponse(params: GenerateStreamParams): Promise<GenerateResponseResult> {
    const stream = await this.generateStream(params);
    const [text, reasoning, sources, usage, toolInvocations] = await Promise.all([
      stream.text,
      stream.reasoning,
      stream.sources,
      stream.usage,
      stream.toolInvocations,
    ]);
    return { text, reasoning: reasoning || undefined, sources, usage, toolInvocations };
  }

  // ────────────────────────────────────────────────────────────
  // History management (public, for callers like ThoughtConversation)
  // ────────────────────────────────────────────────────────────

  addSystemMessagesToHistory(messages: string[], unshift = false) {
    const formatted: ConversationMessage[] = messages.map((m) => ({ role: 'system' as const, content: m }));
    this.addMessagesToHistory(formatted, unshift);
  }

  addAssistantMessagesToHistory(messages: string[], unshift = false) {
    const formatted: ConversationMessage[] = messages.map((m) => ({ role: 'assistant' as const, content: m }));
    this.addMessagesToHistory(formatted, unshift);
  }

  addUserMessagesToHistory(messages: string[], unshift = false) {
    const formatted: ConversationMessage[] = messages.map((m) => ({ role: 'user' as const, content: m }));
    this.addMessagesToHistory(formatted, unshift);
  }

  addMessagesToHistory(messages: ConversationMessage[], unshift = false) {
    // Convert to the format MessageHistory expects (ChatCompletionMessageParam-like)
    const historyMessages = messages.map((m) => {
      if (typeof m === 'string') {
        return { role: 'user' as const, content: m };
      }
      return m;
    });

    const systemMsgs = historyMessages.filter((m) => m.role === 'system');

    if (unshift) {
      this.history.getMessages().unshift(...(historyMessages as any[]));
      this.history.prune();
      this.systemMessages.unshift(...systemMsgs);
    } else {
      this.history.push(historyMessages as any[]);
      this.systemMessages.push(...systemMsgs);
    }
  }

  // ────────────────────────────────────────────────────────────
  // Module system
  // ────────────────────────────────────────────────────────────

  private async ensureModulesProcessed(): Promise<void> {
    if (this.modulesProcessed) {
      return;
    }
    if (this.processingModulesPromise) {
      return this.processingModulesPromise;
    }

    this.processingModulesPromise = this.processModules();
    try {
      await this.processingModulesPromise;
      this.modulesProcessed = true;
    } catch (error) {
      this.logger.error({ message: 'Error processing modules', obj: { error } });
      this.processingModulesPromise = null;
      throw error;
    }
  }

  private async processModules(): Promise<void> {
    if (!this.params.modules || this.params.modules.length === 0) {
      return;
    }

    for (const module of this.params.modules) {
      const moduleName = module.getName();

      // System messages
      const rawSystem = await Promise.resolve(module.getSystemMessages());
      const sysArr = Array.isArray(rawSystem) ? rawSystem : rawSystem ? [rawSystem] : [];
      const trimmed = sysArr.map((s) => String(s ?? '').trim()).filter(Boolean);

      if (trimmed.length > 0) {
        const formatted = trimmed.join('. ');
        this.addSystemMessagesToHistory([
          `The following are instructions from the ${moduleName} module:\n${formatted}`,
        ]);
      }

      // Functions
      const moduleFunctions = module.getFunctions();
      this.functions.push(...moduleFunctions);

      // Function instructions
      let functionInstructions = `The following are instructions from functions in the ${moduleName} module:`;
      let hasInstructions = false;
      for (const f of moduleFunctions) {
        if (f.instructions && f.instructions.length > 0) {
          hasInstructions = true;
          const paragraph = f.instructions.join('. ');
          functionInstructions += ` ${f.definition.name}: ${paragraph}.`;
        }
      }
      if (hasInstructions) {
        this.addSystemMessagesToHistory([functionInstructions]);
      }

      // Message moderators
      this.messageModerators.push(...module.getMessageModerators());
    }
  }

  // ────────────────────────────────────────────────────────────
  // AI SDK message building
  // ────────────────────────────────────────────────────────────

  private buildAiSdkMessages(
    callMessages: ConversationMessage[]
  ): Array<{ role: 'system' | 'user' | 'assistant'; content: string }> {
    const result: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [];

    // Add history messages
    for (const msg of this.history.getMessages()) {
      const m = msg as any;
      const rawRole = String(m.role ?? 'user');
      // Map non-standard roles to the closest AI SDK role
      const role = (rawRole === 'system' ? 'system' : rawRole === 'assistant' ? 'assistant' : 'user') as
        | 'system'
        | 'user'
        | 'assistant';
      const content = typeof m.content === 'string' ? m.content : this.extractTextFromContent(m.content);
      if (content.trim()) {
        result.push({ role, content });
      }
    }

    // Add call messages
    for (const msg of callMessages) {
      if (typeof msg === 'string') {
        result.push({ role: 'user', content: msg });
      } else {
        const rawRole = String(msg.role ?? 'user');
        const role = (rawRole === 'system' ? 'system' : rawRole === 'assistant' ? 'assistant' : 'user') as
          | 'system'
          | 'user'
          | 'assistant';
        result.push({ role, content: typeof msg.content === 'string' ? msg.content : '' });
      }
    }

    return result;
  }

  private extractTextFromContent(content: any): string {
    if (typeof content === 'string') {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .map((p: any) => {
          if (typeof p === 'string') {
            return p;
          }
          if (p?.type === 'text') {
            return p.text;
          }
          return '';
        })
        .join('\n');
    }
    return '';
  }

  // ────────────────────────────────────────────────────────────
  // AI SDK tool building
  // ────────────────────────────────────────────────────────────

  private buildAiSdkTools(functions: Function[]): ToolSet {
    const tools: ToolSet = {};

    for (const f of functions) {
      const def = f.definition;
      if (!def?.name) {
        continue;
      }

      tools[def.name] = {
        description: def.description,
        inputSchema: jsonSchema(this.normalizeToolParameters(def.parameters)),
        execute: async (args: any) => {
          const result = await f.call(args);
          if (typeof result === 'undefined') {
            return { result: 'Function executed successfully' };
          }
          return result;
        },
      } as any;
    }

    return tools;
  }

  /**
   * Normalize tool parameter schemas to ensure they are valid JSON Schema
   * with `type: "object"`. Handles missing, null, or invalid schemas
   * (e.g. `type: "None"` which some functions produce).
   */
  private normalizeToolParameters(parameters: any): Record<string, any> {
    const emptySchema = { type: 'object', properties: {} };

    if (!parameters || typeof parameters !== 'object') {
      return emptySchema;
    }

    // If type is missing, not a string, or not a valid JSON Schema type, default to object
    const validTypes = ['object', 'array', 'string', 'number', 'integer', 'boolean', 'null'];
    if (
      !parameters.type ||
      typeof parameters.type !== 'string' ||
      !validTypes.includes(parameters.type.toLowerCase())
    ) {
      return { ...emptySchema, ...parameters, type: 'object' };
    }

    return parameters;
  }

  // ────────────────────────────────────────────────────────────
  // Provider options
  // ────────────────────────────────────────────────────────────

  private buildProviderOptions(
    provider: string,
    params: { reasoningEffort?: ReasoningEffort; webSearch?: boolean; serviceTier?: OpenAiServiceTier }
  ): Record<string, any> {
    const options: Record<string, any> = {};

    if (provider === 'openai') {
      const openaiOpts: Record<string, any> = {};
      if (params.reasoningEffort) {
        openaiOpts.reasoningEffort = params.reasoningEffort;
      }
      if (params.serviceTier) {
        openaiOpts.serviceTier = params.serviceTier;
      }
      options.openai = openaiOpts;
    }

    if (provider === 'anthropic') {
      const anthropicOpts: Record<string, any> = {};
      if (params.reasoningEffort) {
        // Map reasoning effort to Anthropic's thinking budget
        const budgetMap: Record<string, number> = {
          low: 2048,
          medium: 8192,
          high: 16384,
          xhigh: 32768,
        };
        anthropicOpts.thinking = {
          type: 'enabled',
          budgetTokens: budgetMap[params.reasoningEffort] ?? 8192,
        };
      }
      options.anthropic = anthropicOpts;
    }

    return options;
  }

  // ────────────────────────────────────────────────────────────
  // Background/polling escape hatch (OpenAI-specific)
  // ────────────────────────────────────────────────────────────

  private shouldUseBackgroundMode(
    modelString: string,
    params: { backgroundMode?: boolean; reasoningEffort?: ReasoningEffort }
  ): boolean {
    if (typeof params.backgroundMode === 'boolean') {
      return params.backgroundMode;
    }
    if (this.isProModel(modelString)) {
      return true;
    }
    if (this.isHighReasoningEffort(params.reasoningEffort)) {
      return true;
    }
    return false;
  }

  private isProModel(model: string): boolean {
    return /(^|[-_.])pro($|[-_.])/.test(String(model ?? '').toLowerCase());
  }

  private isHighReasoningEffort(effort?: ReasoningEffort): boolean {
    return effort === 'high' || effort === 'xhigh';
  }

  /**
   * Fall back to OpenAiResponses for background/polling mode.
   * Returns a StreamResult where the text arrives as a single chunk after polling completes.
   */
  private async generateStreamViaPolling(params: GenerateStreamParams, modelString: string): Promise<StreamResult> {
    const responses = this.createOpenAiResponses(params);

    // Convert messages to the format OpenAiResponses expects
    const messages = this.convertToOpenAiMessages(params.messages);

    const result = await responses.generateText({
      messages,
      model: modelString as TiktokenModel,
      abortSignal: params.abortSignal,
      onToolInvocation: params.onToolInvocation,
      onUsageData: params.onUsageData,
      reasoningEffort: params.reasoningEffort,
      maxToolCalls: params.maxToolCalls,
      backgroundMode: params.backgroundMode,
      maxBackgroundWaitMs: params.maxBackgroundWaitMs,
      serviceTier: params.serviceTier,
    });

    // Wrap the polling result as a StreamResult
    const text = result.message;
    const usage = result.usagedata;
    const toolInvocations = result.toolInvocations;

    return {
      textStream: (async function* () {
        yield text;
      })(),
      reasoningStream: (async function* () {
        // Reasoning not available via polling mode
      })(),
      text: Promise.resolve(text),
      reasoning: Promise.resolve(''),
      sources: Promise.resolve([]),
      usage: Promise.resolve(usage),
      toolInvocations: Promise.resolve(toolInvocations),
    };
  }

  /**
   * Fall back to OpenAiResponses for generateObject with background/polling.
   */
  private async generateObjectViaPolling<T>(
    params: GenerateObjectParams<T>,
    modelString: string
  ): Promise<GenerateObjectResult<T>> {
    const responses = this.createOpenAiResponses(params);

    const messages = this.convertToOpenAiMessages(params.messages);

    const result = await responses.generateObject<T>({
      messages,
      model: modelString as TiktokenModel,
      schema: params.schema,
      abortSignal: params.abortSignal,
      onUsageData: params.onUsageData,
      reasoningEffort: params.reasoningEffort,
      temperature: params.temperature,
      topP: params.topP,
      maxTokens: params.maxTokens,
      backgroundMode: params.backgroundMode,
      maxBackgroundWaitMs: params.maxBackgroundWaitMs,
      serviceTier: params.serviceTier,
    });

    return {
      object: result.object,
      usage: result.usageData,
      reasoning: undefined,
      toolInvocations: [],
    };
  }

  private createOpenAiResponses(params: {
    serviceTier?: OpenAiServiceTier;
    maxBackgroundWaitMs?: number;
  }): OpenAiResponses {
    // Lazy require to avoid circular dependency and keep OpenAiResponses optional
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { OpenAiResponses: OAIResponses } = require('./OpenAiResponses');
    return new OAIResponses({
      modules: this.params.modules,
      logLevel: this.params.logLevel,
      defaultModel: this.getModelString(this.params.defaultModel) as TiktokenModel,
    });
  }

  private convertToOpenAiMessages(messages: ConversationMessage[]): Array<string | ChatCompletionMessageParam> {
    const result: Array<string | ChatCompletionMessageParam> = [];

    // Include history
    for (const msg of this.history.getMessages()) {
      const m = msg as any;
      result.push({
        role: m.role as 'system' | 'user' | 'assistant',
        content: typeof m.content === 'string' ? m.content : this.extractTextFromContent(m.content),
      });
    }

    // Include call messages
    for (const msg of messages) {
      if (typeof msg === 'string') {
        result.push(msg);
      } else {
        result.push({
          role: msg.role as 'system' | 'user' | 'assistant',
          content: msg.content as string,
        });
      }
    }

    return result;
  }

  // ────────────────────────────────────────────────────────────
  // Model resolution
  // ────────────────────────────────────────────────────────────

  private resolveModelInstance(model?: LanguageModel | string): LanguageModel {
    const m = model ?? this.params.defaultModel ?? DEFAULT_MODEL;
    return resolveModel(m);
  }

  private getModelString(model?: LanguageModel | string): string {
    if (!model) {
      const def = this.params.defaultModel;
      if (!def) {
        return DEFAULT_MODEL;
      }
      if (typeof def === 'string') {
        return def;
      }
      return (def as any).modelId ?? DEFAULT_MODEL;
    }
    if (typeof model === 'string') {
      return model;
    }
    return (model as any).modelId ?? 'unknown';
  }

  // ────────────────────────────────────────────────────────────
  // Usage processing
  // ────────────────────────────────────────────────────────────

  /**
   * Build a usage promise from a streaming result.
   * Uses `totalUsage` (accumulated across all steps in a tool-call loop)
   * and populates tool call stats from the steps.
   */
  private async buildUsagePromise(
    result: {
      totalUsage: PromiseLike<LanguageModelUsage>;
      steps: PromiseLike<Array<{ toolCalls?: Array<{ toolName?: string }> }>>;
    },
    modelString: string,
    params: GenerateStreamParams
  ): Promise<UsageData> {
    const [sdkUsage, steps] = await Promise.all([result.totalUsage, result.steps]);
    const usage = this.mapSdkUsage(sdkUsage, modelString, steps);

    if (params.onUsageData) {
      await params.onUsageData(usage);
    }

    return usage;
  }

  private async buildToolInvocationsPromise(result: {
    steps: PromiseLike<
      Array<{
        toolCalls?: Array<{ toolCallId?: string; toolName?: string; args?: unknown }>;
        toolResults?: Array<{ toolCallId?: string; result?: unknown }>;
      }>
    >;
  }): Promise<ToolInvocationResult[]> {
    const steps = await result.steps;
    const invocations: ToolInvocationResult[] = [];

    for (const step of steps ?? []) {
      for (const toolCall of step.toolCalls ?? []) {
        invocations.push({
          id: toolCall.toolCallId ?? '',
          name: toolCall.toolName ?? '',
          startedAt: new Date(),
          finishedAt: new Date(),
          input: toolCall.args,
          ok: true,
          data: (step.toolResults ?? []).find((r) => r.toolCallId === toolCall.toolCallId)?.result,
        });
      }
    }

    return invocations;
  }

  /**
   * Map AI SDK's `LanguageModelUsage` to our `UsageData`.
   *
   * The AI SDK v6 provides cached/reasoning token breakdowns directly in
   * `LanguageModelUsage.inputTokenDetails` and `outputTokenDetails`, so we
   * use those first and only fall back to provider metadata for older providers.
   */
  private mapSdkUsage(
    sdkUsage: LanguageModelUsage,
    modelString: string,
    steps?: Array<{ toolCalls?: Array<{ toolName?: string }> }>
  ): UsageData {
    const inputTokens = sdkUsage?.inputTokens ?? 0;
    const outputTokens = sdkUsage?.outputTokens ?? 0;
    const totalTokens = sdkUsage?.totalTokens ?? inputTokens + outputTokens;

    // AI SDK v6 provides structured token details
    const cachedInputTokens = sdkUsage?.inputTokenDetails?.cacheReadTokens ?? 0;
    const reasoningTokens = sdkUsage?.outputTokenDetails?.reasoningTokens ?? 0;

    const tokenUsage: TokenUsage = {
      inputTokens,
      cachedInputTokens,
      reasoningTokens,
      outputTokens,
      totalTokens,
    };

    // Count steps as individual requests to the assistant
    const stepCount = steps?.length ?? 1;
    const acc = new UsageDataAccumulator({ model: modelString as TiktokenModel });
    acc.addTokenUsage(tokenUsage);

    // Populate tool call stats from steps
    const callsPerTool: Record<string, number> = {};
    let totalToolCalls = 0;
    for (const step of steps ?? []) {
      for (const toolCall of step.toolCalls ?? []) {
        const name = toolCall.toolName ?? 'unknown';
        callsPerTool[name] = (callsPerTool[name] ?? 0) + 1;
        totalToolCalls++;
      }
    }

    return {
      ...acc.usageData,
      totalRequestsToAssistant: stepCount,
      callsPerTool,
      totalToolCalls,
    };
  }

  /**
   * Process usage from a generateObject result (single-step, no tool calls).
   */
  private processAiSdkUsage(result: { usage: LanguageModelUsage }, modelString: string): UsageData {
    return this.mapSdkUsage(result.usage, modelString);
  }

  // ────────────────────────────────────────────────────────────
  // Reasoning stream extraction
  // ────────────────────────────────────────────────────────────

  private extractReasoningStream(result: any): AsyncIterable<string> {
    // The AI SDK exposes reasoning via result.reasoning (promise) and
    // result.fullStream which includes reasoning-delta events.
    // We create an async iterable that yields reasoning text from fullStream.
    return {
      [Symbol.asyncIterator]() {
        const fullStream = result.fullStream;
        const reader = fullStream[Symbol.asyncIterator]();

        return {
          async next(): Promise<IteratorResult<string>> {
            // eslint-disable-next-line no-constant-condition
            while (true) {
              const { value, done } = await reader.next();
              if (done) {
                return { done: true, value: undefined };
              }

              if (value.type === 'reasoning' && value.textDelta) {
                return { done: false, value: value.textDelta };
              }
            }
          },
        };
      },
    };
  }

  private extractReasoningFromResult(result: any): string {
    try {
      // Try to get reasoning from provider metadata or response
      const reasoning = result?.reasoning;
      if (typeof reasoning === 'string') {
        return reasoning;
      }
      if (Array.isArray(reasoning)) {
        return reasoning
          .filter((r: any) => r.type === 'reasoning')
          .map((r: any) => r.text)
          .join('');
      }
    } catch {
      // ignore
    }
    return '';
  }

  // ────────────────────────────────────────────────────────────
  // Schema utilities
  // ────────────────────────────────────────────────────────────

  private isZodSchema(schema: unknown): boolean {
    if (!schema || (typeof schema !== 'object' && typeof schema !== 'function')) {
      return false;
    }
    return (
      typeof (schema as any).safeParse === 'function' ||
      (!!(schema as any)._def && typeof (schema as any)._def.typeName === 'string')
    );
  }

  /**
   * Strictifies a JSON Schema for OpenAI Structured Outputs (strict mode).
   */
  private strictifyJsonSchema(schema: any): any {
    const root = JSON.parse(JSON.stringify(schema ?? {}));

    const visit = (node: any) => {
      if (!node || typeof node !== 'object') {
        return;
      }

      if (!node.type) {
        if (node.properties || node.additionalProperties || node.patternProperties) {
          node.type = 'object';
        } else if (node.items || node.prefixItems) {
          node.type = 'array';
        }
      }

      const types = Array.isArray(node.type) ? node.type : node.type ? [node.type] : [];

      if (types.includes('object')) {
        if (node.additionalProperties !== false) {
          node.additionalProperties = false;
        }
        if (node.properties && typeof node.properties === 'object') {
          const propKeys = Object.keys(node.properties);
          const currentReq: string[] = Array.isArray(node.required) ? node.required.slice() : [];
          node.required = Array.from(new Set([...currentReq, ...propKeys]));
          for (const k of propKeys) {
            visit(node.properties[k]);
          }
        }
        if (node.patternProperties && typeof node.patternProperties === 'object') {
          for (const k of Object.keys(node.patternProperties)) {
            visit(node.patternProperties[k]);
          }
        }
        for (const defsKey of ['$defs', 'definitions']) {
          if (node[defsKey] && typeof node[defsKey] === 'object') {
            for (const key of Object.keys(node[defsKey])) {
              visit(node[defsKey][key]);
            }
          }
        }
      }

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

      for (const k of ['oneOf', 'anyOf', 'allOf']) {
        if (Array.isArray(node[k])) {
          node[k].forEach(visit);
        }
      }
      if (node.not) {
        visit(node.not);
      }
    };

    visit(root);
    return root;
  }
}
