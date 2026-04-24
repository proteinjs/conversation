import type { LanguageModel, ToolSet, LanguageModelUsage, ReasoningOutput, ModelMessage } from 'ai';
import type { ImagePart, TextPart, FilePart } from '@ai-sdk/provider-utils';
import type { LanguageModelV3Source } from '@ai-sdk/provider';
import { streamText, generateObject as aiGenerateObject, jsonSchema, stepCountIs } from 'ai';
import { SdkContentParts } from './sdkContentParts';
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
      content?: string | null | unknown;
    };

/** @deprecated Use `GenerateObjectResult` instead. */
export type GenerateObjectOutcome<T> = GenerateObjectResult<T>;

export type ReasoningEffort = 'auto' | 'none' | 'low' | 'medium' | 'high' | 'max' | 'xhigh';

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

/** A single part emitted by the interleaved full stream. */
export type StreamPart =
  | { type: 'text-delta'; textDelta: string }
  | { type: 'reasoning-start' }
  | { type: 'reasoning-delta'; textDelta: string }
  | { type: 'reasoning-end' }
  | { type: 'source'; source: StreamSource }
  | { type: 'tool-call'; toolName: string };

/** The result of generateStream. All properties are available immediately for streaming consumption. */
export type StreamResult = {
  /** Async iterable of text content chunks. */
  textStream: AsyncIterable<string>;
  /** Async iterable of reasoning/thinking chunks (empty if model doesn't support reasoning). */
  reasoningStream: AsyncIterable<string>;
  /**
   * Interleaved stream of all parts (text, reasoning, sources) for real-time
   * consumption. Prefer this over consuming `textStream` and `reasoningStream`
   * separately, since those may share the same underlying data source and
   * cannot be consumed concurrently.
   */
  fullStream: AsyncIterable<StreamPart>;
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

    this.logger.info({
      message: `generateStream`,
      obj: { model: modelString, provider, reasoningEffort: params.reasoningEffort, webSearch: params.webSearch },
    });

    // Check if we should use background/polling mode (OpenAI-specific)
    if (provider === 'openai' && this.shouldUseBackgroundMode(modelString, params)) {
      return this.generateStreamViaPolling(params, modelString);
    }

    // Build messages for the AI SDK
    let messages = this.buildAiSdkMessages(params.messages);

    // Many providers (Anthropic, Google) require all system messages at the
    // beginning of the conversation. Reorder so system messages come first,
    // preserving relative order within each group.
    if (provider === 'google' || provider === 'anthropic') {
      const system = messages.filter((m) => m.role === 'system');
      const nonSystem = messages.filter((m) => m.role !== 'system');
      messages = [...system, ...nonSystem];
    }

    // Build tools from module functions + any extra tools. For providers
    // whose tool-result adapter strips image content (xAI — see
    // `buildAiSdkTools`' imageRedirect comment), pass a shared map that
    // lets `execute` stash images by toolCallId and `prepareStep` splice
    // them into a follow-up user message. Other providers skip this
    // machinery entirely.
    const needsUserMessageImageInjection = provider === 'xai';
    const pendingImageInjections = needsUserMessageImageInjection
      ? new Map<string, Array<TextPart | ImagePart | FilePart>>()
      : undefined;
    const allFunctions = [...this.functions, ...(params.tools ?? [])];
    const tools = this.buildAiSdkTools(allFunctions, { pendingImageInjections });

    // Build provider options
    const providerOptions = this.buildProviderOptions(provider, params, modelString);

    // Include web search tools. For providers with true tool-use search
    // (Anthropic, OpenAI), always include so the model can autonomously
    // decide when to search.  For grounding-based providers (Google), only
    // include when the user explicitly requests search via the toggle.
    const webSearchTools = this.getWebSearchTools(provider, modelString, params.webSearch);

    const allTools = { ...tools, ...webSearchTools };

    const result = streamText({
      model,
      messages,
      tools: Object.keys(allTools).length > 0 ? allTools : undefined,
      stopWhen: stepCountIs(params.maxToolCalls ?? 50),
      abortSignal: params.abortSignal,
      providerOptions,
      prepareStep: pendingImageInjections
        ? ({ messages: stepMessages }) => this.injectPendingImageUserMessages(stepMessages, pendingImageInjections)
        : undefined,
    });

    // Build the StreamResult
    const usagePromise = this.buildUsagePromise(result, modelString, params);
    const toolInvocationsPromise = this.buildToolInvocationsPromise(result);

    // IMPORTANT: We must NOT eagerly evaluate result.text, result.reasoning,
    // or result.sources here. In AI SDK v6, these getters trigger internal
    // stream consumers that compete with result.fullStream for the same
    // underlying data. Eager evaluation causes text/reasoning from later
    // tool-call steps to be consumed by the promise path instead of
    // fullStream, resulting in missing content on the streaming path.
    //
    // Instead, we use lazy helpers that only start consuming when the
    // promise is actually awaited (i.e. in generateResponse). The catch
    // handlers still prevent unhandled rejections on abort.

    // Lazy promise factories — only trigger AI SDK stream consumption on access
    const lazySafeText = () =>
      Promise.resolve(result.text).catch((err) => {
        this.logger.error({ message: 'Error resolving text from stream', obj: { error: err?.message ?? err } });
        return '';
      });
    const lazySafeReasoning = () =>
      Promise.resolve(result.reasoning)
        .then((parts: ReasoningOutput[]) =>
          parts
            ? parts
                .filter((part) => part.type === 'reasoning')
                .map((part) => part.text)
                .join('')
            : ''
        )
        .catch(() => '');
    const lazySafeSources = () =>
      Promise.resolve(result.sources)
        .then((s: LanguageModelV3Source[]) =>
          (s ?? []).map((source) => ({
            url: source.sourceType === 'url' ? source.url : undefined,
            title: source.sourceType === 'url' ? source.title : undefined,
          }))
        )
        .catch(() => [] as StreamSource[]);

    const safeUsage = usagePromise.catch(
      () =>
        ({
          model: modelString,
          initialRequestTokenUsage: {
            inputTokens: 0,
            cachedInputTokens: 0,
            reasoningTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
          },
          initialRequestCostUsd: { inputUsd: 0, cachedInputUsd: 0, reasoningUsd: 0, outputUsd: 0, totalUsd: 0 },
          totalTokenUsage: {
            inputTokens: 0,
            cachedInputTokens: 0,
            reasoningTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
          },
          totalCostUsd: { inputUsd: 0, cachedInputUsd: 0, reasoningUsd: 0, outputUsd: 0, totalUsd: 0 },
          totalRequestsToAssistant: 0,
          callsPerTool: {},
          totalToolCalls: 0,
        }) as UsageData
    );
    const safeToolInvocations = toolInvocationsPromise.catch(() => [] as ToolInvocationResult[]);

    // Cache for lazy promises — ensures each getter returns the same promise
    let _textPromise: Promise<string> | undefined;
    let _reasoningPromise: Promise<string> | undefined;
    let _sourcesPromise: Promise<StreamSource[]> | undefined;

    // We still need to catch finishReason etc. to prevent unhandled rejections,
    // but these don't compete with fullStream.
    Promise.resolve(result.finishReason).catch(() => {});
    Promise.resolve((result as any).rawFinishReason).catch(() => {});
    Promise.resolve(result.response).catch(() => {});

    return {
      textStream: result.textStream,
      reasoningStream: (async function* () {
        // Reasoning is available via the promise after generation completes.
        // For real-time streaming, use fullStream instead.
      })(),
      fullStream: this.mapFullStream(result.fullStream),
      // Lazy getters: only start consuming the AI SDK stream when accessed.
      // This prevents dual-consumption when the caller uses fullStream instead.
      get text() {
        return (_textPromise ??= lazySafeText());
      },
      get reasoning() {
        return (_reasoningPromise ??= lazySafeReasoning());
      },
      get sources() {
        return (_sourcesPromise ??= lazySafeSources());
      },
      usage: safeUsage,
      toolInvocations: safeToolInvocations,
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

    let messages = this.buildAiSdkMessages(params.messages);

    // Google requires all system messages at the beginning
    if (provider === 'google') {
      const system = messages.filter((m) => m.role === 'system');
      const nonSystem = messages.filter((m) => m.role !== 'system');
      messages = [...system, ...nonSystem];
    }

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
      providerOptions: this.buildProviderOptions(provider, params, modelString),
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

  private buildAiSdkMessages(callMessages: ConversationMessage[]): ModelMessage[] {
    const result: ModelMessage[] = [];

    // Add history messages
    for (const msg of this.history.getMessages()) {
      const built = this.toModelMessage(msg as unknown as Record<string, unknown>);
      if (built) {
        result.push(built);
      }
    }

    // Add call messages
    for (const msg of callMessages) {
      if (typeof msg === 'string') {
        if (msg.trim()) {
          result.push({ role: 'user', content: msg });
        }
        continue;
      }
      const built = this.toModelMessage(msg as unknown as Record<string, unknown>);
      if (built) {
        result.push(built);
      }
    }

    return result;
  }

  /**
   * Map a loose ConversationMessage-shape to a Vercel AI SDK ModelMessage,
   * preserving structured content (text + image + file parts) for user/assistant
   * roles. System messages are flattened to string since the SDK only accepts
   * string content there.
   *
   * Returns `undefined` for empty messages — callers skip those to avoid sending
   * empty content to providers that reject it.
   */
  private toModelMessage(msg: Record<string, unknown>): ModelMessage | undefined {
    const rawRole = String(msg.role ?? 'user');
    const role = (rawRole === 'system' ? 'system' : rawRole === 'assistant' ? 'assistant' : 'user') as
      | 'system'
      | 'user'
      | 'assistant';
    const rawContent = msg.content;

    // System messages: the SDK only accepts string content here.
    if (role === 'system') {
      const text = this.flattenContentToText(rawContent);
      return text.trim() ? ({ role, content: text } as ModelMessage) : undefined;
    }

    // Structured content: keep image / file parts alongside text.
    if (Array.isArray(rawContent)) {
      const parts = SdkContentParts.toUserContentParts(rawContent);
      if (parts.length > 0) {
        return { role, content: parts } as ModelMessage;
      }
      // Array produced no mappable parts — fall through to string handling.
    }

    const text = this.flattenContentToText(rawContent);
    return text.trim() ? ({ role, content: text } as ModelMessage) : undefined;
  }

  private flattenContentToText(content: unknown): string {
    if (typeof content === 'string') {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .map((p: unknown) => {
          if (typeof p === 'string') {
            return p;
          }
          if (p && typeof p === 'object' && (p as { type?: unknown }).type === 'text') {
            return String((p as { text?: unknown }).text ?? '');
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

  /**
   * Build the AI SDK tool set from our `Function` array, wiring in the
   * multimodal tool-result plumbing and (optionally) an xAI-style image
   * redirect: when a tool returns image parts and the provider's adapter can't
   * carry images inside a `role:'tool'` message, the images are stashed in
   * `pendingImageInjections` keyed by toolCallId and a text-only tool result
   * is sent back. A `prepareStep` callback on `streamText` then splices those
   * images into a synthetic `role:'user'` message right after the tool
   * message, so the model sees them via the well-tested user-content path.
   */
  private buildAiSdkTools(
    functions: Function[],
    options?: {
      pendingImageInjections?: Map<string, Array<TextPart | ImagePart | FilePart>>;
    }
  ): ToolSet {
    const tools: ToolSet = {};
    const pendingImageInjections = options?.pendingImageInjections;
    const imageRedirectEnabled = !!pendingImageInjections;

    // Sentinel for tool returns that produced multimodal content parts.
    // The execute() function must return a "bare" result (the SDK treats
    // non-string returns as JSON payloads), and the structured
    // `ToolResultOutput` shape is only respected when it comes out of the
    // `toModelOutput` hook (see ai/dist/index.js `createToolModelOutput`:
    // when `toModelOutput` is absent, the SDK wraps output as
    // `{type: 'json', value: toJSONValue(output)}` — i.e. our careful
    // `{type:'content', value: [...image-data...]}` would get re-wrapped
    // inside a json payload and the model would see only metadata).
    // Solution: resolve content parts eagerly inside execute(), stash them
    // on a sentinel, and let toModelOutput project them into the SDK's
    // ToolResultOutput shape for the provider adapter.
    const MULTIMODAL_SENTINEL = Symbol.for('conversation.tool.multimodal');
    const debugToolResults = !!process.env.CONVERSATION_DEBUG_TOOL_RESULTS;
    const logger = this.logger;

    for (const f of functions) {
      const def = f.definition;
      if (!def?.name) {
        continue;
      }
      tools[def.name] = {
        description: def.description,
        inputSchema: jsonSchema(this.normalizeToolParameters(def.parameters)),
        execute: async (args: any, executionOptions: { toolCallId: string }) => {
          const result = await f.call(args);
          if (typeof result === 'undefined') {
            return { result: 'Function executed successfully' };
          }
          // If the tool returned OpenAI-shape content parts (directly or via a
          // `ChatCompletionMessageParamFactory` like `getFiles`), stash them
          // on a sentinel object that `toModelOutput` will unwrap. Tools
          // returning primitives / plain JSON objects are unaffected.
          const contentParts = await SdkContentParts.extractContentPartsFromToolReturn(result);
          if (!contentParts || contentParts.length === 0) {
            return result;
          }
          const toolResultParts = SdkContentParts.toToolResultContentParts(contentParts);

          // xAI-style redirect: when the provider adapter can't transport
          // image content inside a tool result (xAI collapses `type: 'content'`
          // to JSON.stringify in its chat-completions adapter — see
          // node_modules/@ai-sdk/xai/dist/index.js ~line 134), peel the image
          // parts off into `pendingImageInjections`. A `prepareStep` hook on
          // `streamText` splices them into a synthetic user message right
          // after this tool's result, so the model sees the image through the
          // user-content path (which xAI handles correctly). Non-image parts
          // stay in the tool result so the tool-call/result pairing is intact.
          if (imageRedirectEnabled) {
            const textOnlyParts = toolResultParts.filter((p) => p.type === 'text');
            const hasImages = textOnlyParts.length !== toolResultParts.length;
            if (hasImages) {
              pendingImageInjections!.set(
                executionOptions.toolCallId,
                SdkContentParts.toUserContentParts(contentParts)
              );
              const placeholderParts = [
                ...textOnlyParts,
                {
                  type: 'text' as const,
                  text: 'The file content has been attached as a user message immediately following this tool result. Read the image(s) there to answer.',
                },
              ];
              return { [MULTIMODAL_SENTINEL]: placeholderParts };
            }
          }

          return { [MULTIMODAL_SENTINEL]: toolResultParts };
        },
        toModelOutput: ({ output }: { output: unknown }) => {
          // Sentinel path: execute() resolved content parts; project them
          // into a structured multimodal ToolResultOutput.
          if (output && typeof output === 'object' && MULTIMODAL_SENTINEL in (output as object)) {
            const sdkParts = (output as Record<symbol, unknown>)[MULTIMODAL_SENTINEL] as ReturnType<
              typeof SdkContentParts.toToolResultContentParts
            >;
            if (debugToolResults) {
              logger.info({
                message: `tool-result multimodal payload`,
                obj: {
                  toolName: def.name,
                  parts: sdkParts.map((p) => {
                    if (p.type === 'image-data') {
                      return { type: p.type, mediaType: p.mediaType, bytes: p.data.length };
                    }
                    if (p.type === 'image-url') {
                      return { type: p.type, url: p.url };
                    }
                    return { type: p.type, textLength: p.text?.length ?? 0 };
                  }),
                },
              });
            }
            return { type: 'content', value: sdkParts };
          }
          // Default path: preserve the SDK's historical behavior — string →
          // text, everything else → json (stringified).
          return typeof output === 'string'
            ? { type: 'text', value: output }
            : { type: 'json', value: (output ?? null) as unknown as any };
        },
      } as any;
    }

    return tools;
  }

  /**
   * `prepareStep` callback used only for providers whose tool-result adapter
   * strips image content (xAI today). Walks the outgoing messages, finds any
   * tool-result parts whose `toolCallId` has stashed image parts pending in
   * `pendingImageInjections`, and inserts a `role: 'user'` message carrying
   * those image parts directly after the tool message. The map entry is
   * cleared once the injection is emitted so the same images aren't
   * re-injected on subsequent steps.
   */
  private injectPendingImageUserMessages(
    messages: ModelMessage[],
    pendingImageInjections: Map<string, Array<TextPart | ImagePart | FilePart>>
  ): { messages: ModelMessage[] } {
    if (pendingImageInjections.size === 0) {
      return { messages };
    }

    const out: ModelMessage[] = [];
    for (const msg of messages) {
      out.push(msg);
      if (msg.role !== 'tool' || !Array.isArray(msg.content)) {
        continue;
      }
      const injected: Array<TextPart | ImagePart | FilePart> = [];
      for (const part of msg.content) {
        const toolCallId = (part as { toolCallId?: string }).toolCallId;
        if (
          (part as { type?: string }).type === 'tool-result' &&
          toolCallId &&
          pendingImageInjections.has(toolCallId)
        ) {
          injected.push(...pendingImageInjections.get(toolCallId)!);
          pendingImageInjections.delete(toolCallId);
        }
      }
      if (injected.length > 0) {
        out.push({ role: 'user', content: injected } as ModelMessage);
      }
    }
    return { messages: out };
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
    params: { reasoningEffort?: ReasoningEffort; webSearch?: boolean; serviceTier?: OpenAiServiceTier },
    modelString?: string
  ): Record<string, any> {
    const options: Record<string, any> = {};
    const effort = params.reasoningEffort;

    if (provider === 'openai') {
      const openaiOpts: Record<string, any> = {};
      if (effort && effort !== 'auto') {
        // OpenAI accepts: none | low | medium | high | xhigh
        // 'max' → 'xhigh' (OpenAI's highest)
        openaiOpts.reasoningEffort = effort === 'max' ? 'xhigh' : effort;
      }
      // 'auto': omit reasoningEffort — let OpenAI use its default reasoning behavior
      if (params.serviceTier) {
        openaiOpts.serviceTier = params.serviceTier;
      }
      options.openai = openaiOpts;
    }

    if (provider === 'anthropic') {
      const anthropicOpts: Record<string, any> = {};
      const isHaiku = modelString ? /haiku/i.test(modelString) : false;
      if (effort === 'auto') {
        if (isHaiku) {
          // Haiku 4.5 supports extended thinking (budget-based) but NOT adaptive.
          // Auto → enable with a moderate budget and let the model decide how much to use.
          anthropicOpts.thinking = { type: 'enabled', budgetTokens: 10000 };
        } else {
          // Opus 4.7 + Sonnet 4.6 support adaptive thinking — model decides effort.
          anthropicOpts.thinking = { type: 'adaptive' };
        }
      } else if (effort && effort !== 'none') {
        if (isHaiku) {
          // Haiku 4.5 supports extended thinking (budget-based) but NOT adaptive.
          // Map effort levels to budget_tokens: low → 5k, medium → 10k, high → 50k
          const budgetMap: Record<string, number> = { low: 5000, medium: 10000, high: 50000 };
          anthropicOpts.thinking = { type: 'enabled', budgetTokens: budgetMap[effort] ?? 10000 };
        } else {
          // Opus 4.7 + Sonnet 4.6 (and 4.5) support adaptive thinking with effort.
          // Anthropic accepts effort: low | medium | high | xhigh | max
          // ('xhigh' was added in Opus 4.7 — sits between high and max.)
          anthropicOpts.thinking = { type: 'adaptive' };
          anthropicOpts.effort = effort;
        }
      }
      options.anthropic = anthropicOpts;
    }

    if (provider === 'google') {
      const googleOpts: Record<string, any> = {};
      if (effort === 'auto') {
        // Auto: enable thinking without specifying level — model decides
        googleOpts.thinkingConfig = {};
      } else if (effort && effort !== 'none') {
        // Google accepts thinkingLevel: minimal | low | medium | high
        // Our 'max'/'xhigh' have no Google equivalent → map to 'high'
        const levelMap: Record<string, string> = {
          low: 'low',
          medium: 'medium',
          high: 'high',
          xhigh: 'high',
          max: 'high',
        };
        googleOpts.thinkingConfig = {
          thinkingLevel: levelMap[effort] ?? 'medium',
        };
      }
      options.google = googleOpts;
    }

    if (provider === 'xai') {
      const xaiOpts: Record<string, any> = {};
      // Only models with reasoning support accept the reasoningEffort parameter.
      // Models like grok-4 (no "-fast" suffix) reject it with a 400 error.
      const xaiSupportsReasoning = modelString ? /fast/i.test(modelString) : false;
      if (effort && effort !== 'none' && effort !== 'auto' && xaiSupportsReasoning) {
        // xAI accepts: low | high
        // Map everything to the closest valid value
        const xaiEffort = effort === 'low' ? 'low' : 'high';
        xaiOpts.reasoningEffort = xaiEffort;
      }
      // 'auto': omit reasoningEffort — let xAI use its default reasoning behavior
      options.xai = xaiOpts;
    }

    return options;
  }

  /**
   * Returns provider-specific web search tools.
   *
   * Each provider SDK exposes a web search tool factory that creates a
   * provider-executed tool (the model calls it server-side; we just pass
   * the tool definition into `streamText`).
   */
  private getWebSearchTools(provider: string, modelString: string, _webSearchRequested?: boolean): ToolSet {
    try {
      // Models that don't support programmatic tool calling can't use web search tools.
      // Haiku 4.5 and nano-class models are excluded.
      if (/nano/i.test(modelString) || /haiku/i.test(modelString)) {
        return {};
      }

      switch (provider) {
        // Tool-use search: always included so the model can decide when to search
        case 'openai': {
          const { openai } = require('@ai-sdk/openai');
          return { web_search: openai.tools.webSearch() };
        }
        case 'anthropic': {
          const { anthropic } = require('@ai-sdk/anthropic');
          return { web_search: anthropic.tools.webSearch_20260209() };
        }
        // Google: grounding-based search is currently broken in @ai-sdk/google@3.0.43.
        // Re-enable when the SDK is updated. When working, it should be gated on
        // webSearchRequested since it grounds *every* response when present.
        // case 'google': { ... }
        default:
          return {};
      }
    } catch (error) {
      this.logger.error({ message: `Web search tool not available for provider: ${provider}`, error });
      return {};
    }
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
    return effort === 'high' || effort === 'xhigh' || effort === 'max';
  }

  /**
   * Map our ReasoningEffort to OpenAI's accepted values.
   * OpenAI accepts: none | low | medium | high | xhigh
   * 'max' → 'xhigh' (OpenAI's highest).
   */
  private mapReasoningEffortForOpenAi(
    effort?: ReasoningEffort
  ): 'none' | 'low' | 'medium' | 'high' | 'xhigh' | undefined {
    if (!effort || effort === 'auto') {
      return undefined;
    }
    if (effort === 'max') {
      return 'xhigh';
    }
    return effort as 'none' | 'low' | 'medium' | 'high' | 'xhigh';
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
      reasoningEffort: this.mapReasoningEffortForOpenAi(params.reasoningEffort),
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
      fullStream: (async function* () {
        yield { type: 'text-delta' as const, textDelta: text };
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
      reasoningEffort: this.mapReasoningEffortForOpenAi(params.reasoningEffort),
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
        content: typeof m.content === 'string' ? m.content : this.flattenContentToText(m.content),
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
  // Full stream mapping
  // ────────────────────────────────────────────────────────────

  /**
   * Maps the AI SDK's `fullStream` (which emits all event types) into our
   * `StreamPart` union. This is the primary way to consume streaming output
   * in real-time, since it yields text, reasoning, and source events in the
   * order the model produces them.
   */
  private mapFullStream(aiSdkFullStream: AsyncIterable<any>): AsyncIterable<StreamPart> {
    const logger = this.logger;
    return {
      async *[Symbol.asyncIterator]() {
        const partCounts: Record<string, number> = {};
        try {
          for await (const part of aiSdkFullStream) {
            const partType = part.type ?? 'unknown';
            partCounts[partType] = (partCounts[partType] ?? 0) + 1;

            if (part.type === 'text-delta') {
              // AI SDK v6 emits text-delta with `delta` or `text` property (not `textDelta`)
              const textContent = part.textDelta ?? part.delta ?? part.text;
              if (textContent) {
                yield { type: 'text-delta' as const, textDelta: textContent };
              }
            } else if (part.type === 'reasoning-start') {
              yield { type: 'reasoning-start' as const };
            } else if (part.type === 'reasoning-delta') {
              // AI SDK v6 emits reasoning-delta with `delta` or `text` property
              const reasoningText = part.delta ?? part.text ?? part.textDelta;
              if (reasoningText) {
                yield { type: 'reasoning-delta' as const, textDelta: reasoningText };
              }
            } else if (part.type === 'reasoning-end') {
              yield { type: 'reasoning-end' as const };
            } else if (part.type === 'tool-call') {
              yield { type: 'tool-call' as const, toolName: part.toolName ?? 'unknown' };
            } else if (part.type === 'source') {
              yield {
                type: 'source' as const,
                source: {
                  url: part.sourceType === 'url' ? part.url : undefined,
                  title: part.sourceType === 'url' ? part.title : undefined,
                },
              };
            }
          }
        } finally {
          logger.info({ message: 'mapFullStream completed', obj: { partCounts } });
        }
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
