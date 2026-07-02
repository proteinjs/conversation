import type { LanguageModel, ToolSet, LanguageModelUsage, ReasoningOutput, ModelMessage } from 'ai';
import type { ImagePart, TextPart, FilePart } from '@ai-sdk/provider-utils';
import type { LanguageModelV3Source } from '@ai-sdk/provider';
import { streamText, generateObject as aiGenerateObject, jsonSchema, stepCountIs, hasToolCall } from 'ai';
import { SdkContentParts } from './sdkContentParts';
import type { RepairTextFunction } from 'ai';
import { Logger, LogLevel } from '@proteinjs/logger';
import { ConversationSkill } from './ConversationSkill';
import { Function } from './Function';
import { MessageModerator } from './history/MessageModerator';
import { MessageHistory } from './history/MessageHistory';
import { UsageData, UsageDataAccumulator, TokenUsage } from './UsageData';
import { resolveModel, inferProvider } from './resolveModel';
import { LlmTransportRetry } from './LlmTransportRetry';
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
  skills?: ConversationSkill[];
  logLevel?: LogLevel;
  defaultModel?: LanguageModel | string;
  limits?: {
    enforceLimits?: boolean;
    maxMessagesInHistory?: number;
    tokenLimit?: number;
  };
  /**
   * Keep only the most recent N image-bearing tool results in what is sent to
   * the model; older ones are replaced with a text placeholder. A stateless
   * per-step projection of the outbound request (never mutates history) for
   * image-heavy tool loops — e.g. computer use, where every action returns a
   * screenshot and stale frames carry little signal (the app is live; the model
   * can always look again). Evicts in batches (hysteresis) so the message
   * prefix stays prompt-cache-stable between evictions. User-message images are
   * never touched. Off when unset.
   */
  toolImageRetention?: number;
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
  /**
   * Fired after EACH step (tool-call round) with the cumulative usage so far,
   * for live in-flight token/cost display. The running sum reconciles exactly to
   * the final `onUsageData` (which maps `result.totalUsage`). Only meaningful for
   * multi-step requests — a single-step request fires this once, at the end.
   */
  onPartialUsageData?: (usageData: UsageData) => Promise<void>;
  abortSignal?: AbortSignal;
  maxToolCalls?: number;
  /**
   * Tool names that END the loop when called: the step the tool is called in
   * is the last step, by construction. For turn-ending tools (e.g. a flow's
   * `askQuestion`) — instructions alone don't stop a model from continuing to
   * work after asking, which strands the recorded ask out of sync with the
   * work that follows it.
   */
  stopOnToolCalls?: string[];

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
  | {
      type: 'tool-call';
      toolName: string;
      /**
       * A short, human-meaningful subject for the call when one can be derived
       * from the tool input (e.g. a web-search query, a created space/thought
       * title) — used to personalize the call's node in the thinking timeline.
       */
      detail?: string;
      /**
       * True when this is a provider-defined tool (e.g. Anthropic's native
       * `text_editor` / `bash`) rather than a custom function tool. Custom tools
       * surface through the `onToolInvocation` callback, so stream consumers that
       * also listen there skip them here to avoid double-counting; provider-defined
       * tools have no callback and are surfaced ONLY through this stream part.
       */
      providerDefined: boolean;
    };

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
  private skillsProcessed = false;
  private processingSkillsPromise: Promise<void> | null = null;
  // Invisible bounded retries for transient transport failures on EVERY LLM request this conversation
  // makes (each tool-loop step included) — see resolveModelInstance. The SDKs' own retries are disabled
  // (maxRetries: 0 at the streamText/generateObject call sites) so exactly one layer owns retrying.
  private transportRetry = new LlmTransportRetry();

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
    await this.ensureSkillsProcessed();

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

    // Build tools from skill functions + any extra tools. For providers
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
    const tools = this.buildAiSdkTools(allFunctions, {
      pendingImageInjections,
      onToolInvocation: params.onToolInvocation,
    });

    // Build provider options
    const providerOptions = this.buildProviderOptions(provider, params, modelString);

    // Include web search tools. For providers with true tool-use search
    // (Anthropic, OpenAI), always include so the model can autonomously
    // decide when to search.  For grounding-based providers (Google), only
    // include when the user explicitly requests search via the toggle.
    const webSearchTools = this.getWebSearchTools(provider, modelString, params.webSearch);

    // Provider-defined tools contributed by skills (e.g. Anthropic's native
    // text_editor / bash). Injected directly here — like webSearchTools —
    // rather than through buildAiSdkTools, since they are AI SDK provider
    // tools, not our `Function` shape.
    const skillProviderTools = this.getSkillProviderDefinedTools(provider);

    const allTools = { ...tools, ...webSearchTools, ...skillProviderTools };

    // When the user toggles search on, force the search tool on the first
    // step so the toggle has a consistent "guarantee a search this turn"
    // meaning across providers. After step 1 the model returns to default
    // (auto) tool choice for subsequent steps.
    const webSearchToolChoice = this.getWebSearchToolChoice(provider, webSearchTools, params.webSearch);

    // Cumulative per-step usage for live in-flight reporting. Each step (tool-call
    // round) is a separate billed call, so summing step usage reconciles exactly
    // to the final `result.totalUsage`.
    let cumIn = 0;
    let cumOut = 0;
    let cumTotal = 0;
    let cumCacheRead = 0;
    let cumCacheWrite = 0;
    let cumReason = 0;
    let cumSteps = 0;

    const result = streamText({
      model,
      messages,
      tools: Object.keys(allTools).length > 0 ? allTools : undefined,
      toolChoice: webSearchToolChoice,
      stopWhen: [
        stepCountIs(params.maxToolCalls ?? 50),
        ...(params.stopOnToolCalls ?? []).map((name) => hasToolCall(name)),
      ],
      // Retries are owned by LlmTransportRetry (the wrapped model) — disable the SDK's own layer so
      // budgets don't stack multiplicatively.
      maxRetries: 0,
      abortSignal: params.abortSignal,
      providerOptions,
      prepareStep: ({ messages: stepMessages }) => {
        let next = stepMessages;
        if (pendingImageInjections) {
          next = this.injectPendingImageUserMessages(next, pendingImageInjections).messages;
        }
        if (this.params.toolImageRetention != null) {
          next = Conversation.pruneStaleToolImages(next, this.params.toolImageRetention);
        }
        if (provider === 'anthropic') {
          // Runs for every step including the first, so this is the single seam
          // where outgoing Anthropic requests get cache breakpoints (after pruning —
          // marks must land on the final per-step messages).
          next = Conversation.applyAnthropicPromptCaching(next);
        }
        // Coerce non-object tool-call inputs LAST, so it sees the final per-step
        // messages (after image inject/prune + cache marking).
        return { messages: this.sanitizeToolCallInputs(next) };
      },
      onStepFinish: params.onPartialUsageData
        ? async (step) => {
            const su = step.usage;
            cumIn += su?.inputTokens ?? 0;
            cumOut += su?.outputTokens ?? 0;
            cumTotal += su?.totalTokens ?? (su?.inputTokens ?? 0) + (su?.outputTokens ?? 0);
            cumCacheRead += su?.inputTokenDetails?.cacheReadTokens ?? 0;
            cumCacheWrite += su?.inputTokenDetails?.cacheWriteTokens ?? 0;
            cumReason += su?.outputTokenDetails?.reasoningTokens ?? 0;
            cumSteps += 1;
            const partial = this.mapSdkUsage(
              {
                inputTokens: cumIn,
                outputTokens: cumOut,
                totalTokens: cumTotal,
                inputTokenDetails: { cacheReadTokens: cumCacheRead, cacheWriteTokens: cumCacheWrite },
                outputTokenDetails: { reasoningTokens: cumReason },
              } as LanguageModelUsage,
              modelString,
              Array.from({ length: cumSteps }, () => ({}))
            );
            await params.onPartialUsageData!(partial);
          }
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
            cacheWriteTokens: 0,
            reasoningTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
          },
          initialRequestCostUsd: { inputUsd: 0, cachedInputUsd: 0, reasoningUsd: 0, outputUsd: 0, totalUsd: 0 },
          totalTokenUsage: {
            inputTokens: 0,
            cachedInputTokens: 0,
            cacheWriteTokens: 0,
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
    await this.ensureSkillsProcessed();

    const model = this.resolveModelInstance(params.model);
    const modelString = this.getModelString(params.model);
    const provider = inferProvider(params.model ?? this.params.defaultModel ?? DEFAULT_MODEL);

    // Check if we should use background/polling mode (OpenAI-specific)
    if (provider === 'openai' && this.shouldUseBackgroundMode(modelString, params)) {
      return this.generateObjectViaPolling(params, modelString);
    }

    let messages = this.buildAiSdkMessages(params.messages);

    // Google and Anthropic require all system messages at the beginning
    // (Anthropic's converter rejects non-leading system messages outright —
    // and callers legitimately mix them in, e.g. a flow's system preludes
    // appended after recorded conversation history). Mirrors generateStream.
    if (provider === 'google' || provider === 'anthropic') {
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
      // Retries are owned by LlmTransportRetry (the wrapped model) — disable the SDK's own layer.
      maxRetries: 0,
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
  // Skill system
  // ────────────────────────────────────────────────────────────

  private async ensureSkillsProcessed(): Promise<void> {
    if (this.skillsProcessed) {
      return;
    }
    if (this.processingSkillsPromise) {
      return this.processingSkillsPromise;
    }

    this.processingSkillsPromise = this.processSkills();
    try {
      await this.processingSkillsPromise;
      this.skillsProcessed = true;
    } catch (error) {
      this.logger.error({ message: 'Error processing skills', obj: { error } });
      this.processingSkillsPromise = null;
      throw error;
    }
  }

  private async processSkills(): Promise<void> {
    if (!this.params.skills || this.params.skills.length === 0) {
      return;
    }

    for (const skill of this.params.skills) {
      const skillName = skill.getName();

      // System messages
      const rawSystem = await Promise.resolve(skill.getSystemMessages());
      const sysArr = Array.isArray(rawSystem) ? rawSystem : rawSystem ? [rawSystem] : [];
      const trimmed = sysArr.map((s) => String(s ?? '').trim()).filter(Boolean);

      if (trimmed.length > 0) {
        const formatted = trimmed.join('. ');
        this.addSystemMessagesToHistory([`The following are instructions from the ${skillName} skill:\n${formatted}`]);
      }

      // Functions
      const skillFunctions = skill.getFunctions();
      this.functions.push(...skillFunctions);

      // Function instructions
      let functionInstructions = `The following are instructions from functions in the ${skillName} skill:`;
      let hasInstructions = false;
      for (const f of skillFunctions) {
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
      this.messageModerators.push(...skill.getMessageModerators());
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
      onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
    }
  ): ToolSet {
    const tools: ToolSet = {};
    const pendingImageInjections = options?.pendingImageInjections;
    const onToolInvocation = options?.onToolInvocation;
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
          const toolStartedAt = new Date();
          const timelineDetail = await this.resolveToolTimelineDetail(def.name, args);
          onToolInvocation?.({
            type: 'started',
            id: executionOptions.toolCallId,
            name: def.name,
            startedAt: toolStartedAt,
            input: args,
            detail: timelineDetail,
          });
          let result: unknown;
          try {
            result = await f.call(args);
          } catch (toolError) {
            onToolInvocation?.({
              type: 'finished',
              result: {
                id: executionOptions.toolCallId,
                name: def.name,
                startedAt: toolStartedAt,
                finishedAt: new Date(),
                input: args,
                ok: false,
                error: {
                  message: toolError instanceof Error ? toolError.message : String(toolError),
                  stack: toolError instanceof Error ? toolError.stack : undefined,
                },
              },
            });
            throw toolError;
          }
          onToolInvocation?.({
            type: 'finished',
            result: {
              id: executionOptions.toolCallId,
              name: def.name,
              startedAt: toolStartedAt,
              finishedAt: new Date(),
              input: args,
              ok: true,
              data: result,
            },
          });
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
   * Guarantee every assistant tool-call carries an OBJECT `input` before it goes on the wire.
   *
   * When the model emits empty/malformed arguments for a tool that has required fields, the AI SDK
   * marks the call invalid and leaves its `input` as the raw string (e.g. `""`) — then copies that
   * verbatim into the assistant message it re-sends on the next (continuation) step. The Anthropic
   * adapter forwards it as-is, and the API rejects the whole request:
   * `messages.N.content.M.tool_use.input: Input should be an object` (HTTP 400) — killing the turn.
   *
   * We can't force the model to always emit valid args, so we enforce the on-wire invariant here
   * (the layer we own): coerce any non-object tool-call input to `{}`. The call was already flagged
   * invalid and carries a tool-error result, so the model still sees the failure and recovers — the
   * request just no longer crashes. Runs on every step; continuations are where a prior step's
   * invalid tool-call gets re-serialized.
   */
  private sanitizeToolCallInputs(messages: ModelMessage[]): ModelMessage[] {
    let mutated = false;
    const out = messages.map((msg) => {
      if (msg.role !== 'assistant' || !Array.isArray(msg.content)) {
        return msg;
      }
      let contentMutated = false;
      const content = msg.content.map((part) => {
        if ((part as { type?: string }).type !== 'tool-call') {
          return part;
        }
        const input = (part as { input?: unknown }).input;
        const isObject = typeof input === 'object' && input !== null && !Array.isArray(input);
        if (isObject) {
          return part;
        }
        contentMutated = true;
        return { ...(part as object), input: {} };
      });
      if (!contentMutated) {
        return msg;
      }
      mutated = true;
      return { ...msg, content } as ModelMessage;
    });
    return mutated ? out : messages;
  }

  /**
   * Map a `computer` tool action to a timeline display category + human detail
   * (e.g. `click` + `(200,125)`, `type` + `"hello…"`). Returns undefined when the
   * input doesn't look like a computer action — the caller falls back to the
   * plain tool name.
   */
  private static describeComputerAction(
    input: Record<string, unknown>
  ): { suffix: string; detail?: string } | undefined {
    const action = typeof input.action === 'string' ? input.action : undefined;
    if (!action) {
      return undefined;
    }
    const at = Array.isArray(input.coordinate) ? `(${input.coordinate[0]},${input.coordinate[1]})` : undefined;
    const text = typeof input.text === 'string' ? input.text : undefined;
    const truncate = (value: string, max = 40) => (value.length > max ? `${value.slice(0, max)}…` : value);
    switch (action) {
      case 'screenshot':
        return { suffix: 'screenshot' };
      case 'left_click':
      case 'right_click':
      case 'middle_click':
      case 'double_click':
      case 'triple_click':
        return { suffix: 'click', detail: at };
      case 'type':
        return { suffix: 'type', detail: text ? `"${truncate(text)}"` : undefined };
      case 'key':
      case 'hold_key':
        return { suffix: 'key', detail: text };
      case 'scroll': {
        const direction = typeof input.scroll_direction === 'string' ? input.scroll_direction : '';
        const amount = typeof input.scroll_amount === 'number' ? ` ×${input.scroll_amount}` : '';
        return { suffix: 'scroll', detail: `${direction}${amount}`.trim() || undefined };
      }
      case 'left_click_drag': {
        const from = Array.isArray(input.start_coordinate)
          ? `(${input.start_coordinate[0]},${input.start_coordinate[1]})`
          : undefined;
        return { suffix: 'drag', detail: from && at ? `${from} → ${at}` : at };
      }
      case 'mouse_move':
      case 'cursor_position':
        return { suffix: 'move', detail: at };
      case 'wait':
        return { suffix: 'wait', detail: typeof input.duration === 'number' ? `${input.duration}s` : undefined };
      default:
        return { suffix: action.replace(/_/g, '-'), detail: at };
    }
  }

  /**
   * Keep only the most recent `keepLast` image-bearing tool results in the
   * outgoing messages; older ones have their image output replaced with a text
   * placeholder. A stateless per-step projection (`prepareStep`) — persisted and
   * in-memory history are never mutated. Only tool-result outputs are touched;
   * images in user messages are preserved. See `ConversationParams.toolImageRetention`.
   */
  private static pruneStaleToolImages(
    messages: ModelMessage[],
    keepLast: number,
    // Hysteresis: only evict once the excess reaches a batch, so the message
    // prefix stays byte-stable between evictions (prompt-cache friendly) instead
    // of shifting by one image every step.
    evictionBatch = 4
  ): ModelMessage[] {
    type ToolOutput = { type?: string; value?: unknown };
    type ToolResultLike = { type?: string; output?: ToolOutput };
    const hasImageOutput = (part: ToolResultLike): boolean => {
      if (part.type !== 'tool-result' || !part.output || part.output.type !== 'content') {
        return false;
      }
      const value = part.output.value;
      return (
        Array.isArray(value) &&
        value.some((p: { type?: string }) => p?.type === 'media' || p?.type === 'file-data' || p?.type === 'image-data')
      );
    };

    // Pass 1: count image-bearing tool results so we know which fall outside the keep window.
    let imageCount = 0;
    for (const msg of messages) {
      if (msg.role !== 'tool' || !Array.isArray(msg.content)) {
        continue;
      }
      for (const part of msg.content) {
        if (hasImageOutput(part as ToolResultLike)) {
          imageCount++;
        }
      }
    }
    const keep = Math.max(0, keepLast);
    if (imageCount < keep + Math.max(1, evictionBatch)) {
      return messages;
    }
    const pruneCount = imageCount - keep;

    // Pass 2: replace the oldest `pruneCount` image outputs with a text placeholder.
    let pruned = 0;
    return messages.map((msg) => {
      if (msg.role !== 'tool' || !Array.isArray(msg.content) || pruned >= pruneCount) {
        return msg;
      }
      let changed = false;
      const content = msg.content.map((part) => {
        if (pruned < pruneCount && hasImageOutput(part as ToolResultLike)) {
          pruned++;
          changed = true;
          return {
            ...(part as object),
            output: {
              type: 'text',
              value: '[stale screenshot removed — superseded by more recent screenshots]',
            },
          };
        }
        return part;
      });
      return changed ? ({ ...msg, content } as ModelMessage) : msg;
    });
  }

  /**
   * Anthropic prompt caching: mark cache breakpoints on the outgoing messages
   * so each request reuses the previous one's prefix instead of re-reading the
   * whole conversation at full input price (agentic loops resend the entire
   * prefix every step — uncached, that dominates turn cost).
   *
   * Breakpoints (≤3 of Anthropic's max 4):
   *  - the last system message — caches tools + system, stable across turns;
   *  - the last TWO non-system messages — a rolling pair: the next request's
   *    penultimate breakpoint is this request's last one, so the longest
   *    cached prefix is re-read every step even as the transcript grows.
   *
   * Marks from earlier steps persist on message objects, so unmarked messages
   * are STRIPPED of stale breakpoints — past 4 total Anthropic rejects the
   * request. Stateless per-step projection like `pruneStaleToolImages`:
   * persisted and in-memory history are never mutated.
   */
  private static applyAnthropicPromptCaching(messages: ModelMessage[]): ModelMessage[] {
    type WithProviderOptions = { providerOptions?: Record<string, Record<string, unknown>> };
    const cacheControl = { type: 'ephemeral' as const };

    const markIndexes = new Set<number>();
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'system') {
        markIndexes.add(i);
        break;
      }
    }
    let rollingMarks = 0;
    for (let i = messages.length - 1; i >= 0 && rollingMarks < 2; i--) {
      if (messages[i].role === 'system') {
        continue;
      }
      markIndexes.add(i);
      rollingMarks++;
    }

    return messages.map((msg, i) => {
      const prev = (msg as WithProviderOptions).providerOptions;
      if (markIndexes.has(i)) {
        return {
          ...msg,
          providerOptions: { ...prev, anthropic: { ...(prev?.anthropic ?? {}), cacheControl } },
        } as ModelMessage;
      }
      if (!prev?.anthropic || !('cacheControl' in prev.anthropic)) {
        return msg;
      }
      const { cacheControl: _stale, ...restAnthropic } = prev.anthropic;
      const nextOptions: Record<string, Record<string, unknown>> = { ...prev };
      if (Object.keys(restAnthropic).length > 0) {
        nextOptions.anthropic = restAnthropic;
      } else {
        delete nextOptions.anthropic;
      }
      return {
        ...msg,
        providerOptions: Object.keys(nextOptions).length > 0 ? nextOptions : undefined,
      } as ModelMessage;
    });
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
      // Always request reasoning summary text. The Responses API (used by
      // resolveModel for OpenAI) only emits `reasoning-delta` stream chunks
      // when `reasoningSummary` is set; default is no summary. Honored on
      // reasoning models; harmlessly ignored on non-reasoning models.
      openaiOpts.reasoningSummary = 'auto';
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
          // Haiku supports extended thinking (budget-based) but NOT adaptive.
          // Auto → enable with a moderate budget and let the model decide how much to use.
          anthropicOpts.thinking = { type: 'enabled', budgetTokens: 10000 };
        } else {
          // Opus + Sonnet support adaptive thinking — the model decides effort.
          // display: 'summarized' is required to stream reasoning text (the API
          // defaults to 'omitted'); passing it makes the behavior explicit.
          anthropicOpts.thinking = { type: 'adaptive', display: 'summarized' };
        }
      } else if (effort && effort !== 'none') {
        if (isHaiku) {
          // Haiku supports extended thinking (budget-based) but NOT adaptive.
          // Map effort levels to budget_tokens: low → 5k, medium → 10k, high → 50k
          const budgetMap: Record<string, number> = { low: 5000, medium: 10000, high: 50000 };
          anthropicOpts.thinking = { type: 'enabled', budgetTokens: budgetMap[effort] ?? 10000 };
        } else {
          // Opus + Sonnet support adaptive thinking with explicit effort.
          // Anthropic accepts effort: low | medium | high | xhigh | max
          // ('xhigh' sits between high and max.)
          anthropicOpts.thinking = { type: 'adaptive', display: 'summarized' };
          anthropicOpts.effort = effort;
        }
      }
      options.anthropic = anthropicOpts;
    }

    if (provider === 'google') {
      const googleOpts: Record<string, any> = {};
      // includeThoughts is required for Gemini to stream `thought_summary`
      // events back; without it the model computes reasoning internally but
      // doesn't surface any text. Equivalent to Anthropic's
      // `display: 'summarized'` and OpenAI's `reasoningSummary: 'auto'`.
      // Always-on, except when effort is explicitly 'none'.
      if (effort === 'auto') {
        // Auto: enable thinking with summaries; let Gemini choose the level.
        googleOpts.thinkingConfig = { includeThoughts: true };
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
          includeThoughts: true,
          thinkingLevel: levelMap[effort] ?? 'medium',
        };
      }
      options.google = googleOpts;
    }

    if (provider === 'xai') {
      const xaiOpts: Record<string, any> = {};
      // Only models with reasoning support accept the reasoningEffort parameter.
      // Models like grok-4 (no "-fast" suffix) reject it with a 400 error;
      // the model decides effort internally.
      const xaiSupportsReasoning = modelString ? /fast/i.test(modelString) : false;
      if (effort && effort !== 'none' && effort !== 'auto' && xaiSupportsReasoning) {
        // xAI accepts: low | high (Responses also accepts 'medium')
        // Map everything to the closest valid value.
        const xaiEffort = effort === 'low' ? 'low' : 'high';
        xaiOpts.reasoningEffort = xaiEffort;
      }
      // Live Search is enabled via the `webSearch` tool factory on the
      // Responses endpoint (handled in getWebSearchTools). The old
      // Chat Completions `searchParameters` API was deprecated by xAI
      // — it now returns 410 with "switch to the Agent Tools API".
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
  private getWebSearchTools(provider: string, modelString: string, webSearchRequested?: boolean): ToolSet {
    try {
      // Models that don't support programmatic tool calling can't use web search tools.
      // Haiku and nano-class models are excluded.
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
          // Deliberately the 2025 web search, not the agentic `webSearch_20260209`.
          // `@ai-sdk/anthropic`'s replay converter (`convertToAnthropicMessagesPrompt`)
          // only knows `webSearch_20250305OutputSchema` — it was never taught to
          // round-trip 2026 server-tool results. With `webSearch_20260209` the model
          // also gets a server-side `code_execution` tool; on step 2+ of a multi-step
          // turn the converter drops those result blocks, orphaning the `server_tool_use`
          // and 400-ing the turn. The 2025 tool keeps send/parse/replay on the one
          // version the SDK fully supports. Revisit once the SDK round-trips 20260209.
          return { web_search: anthropic.tools.webSearch_20250305() };
        }
        case 'google': {
          // Google's search is *grounding-based*, not a model-called tool:
          // attaching `google_search` forces grounding on every response in
          // the turn. So we only attach it when the user explicitly toggled
          // search on. For Gemini 3.0+ this composes cleanly with custom
          // function tools (`@ai-sdk/google` builds a combined toolConfig
          // with `functionCallingConfig: VALIDATED`); earlier Geminis would
          // drop function tools, but we only ship Gemini 3.x.
          if (!webSearchRequested) {
            return {};
          }
          const { google } = require('@ai-sdk/google');
          // The `{}` is required — `googleSearch`'s factory destructures
          // its arg, so passing `undefined` throws. Empty object = default
          // grounding behavior with no extra filters (no time range, etc.).
          return { google_search: google.tools.googleSearch({}) };
        }
        case 'xai': {
          // All xAI models now route through Responses (see resolveModel),
          // so the `webSearch` tool factory works uniformly. Always attach
          // it so the model can search when the prompt warrants — same
          // pattern as OpenAI/Anthropic. The webSearch toggle is a no-op
          // for xAI just like it is for those providers.
          const { xai } = require('@ai-sdk/xai');
          return { web_search: xai.tools.webSearch() };
        }
        default:
          return {};
      }
    } catch (error) {
      this.logger.error({ message: `Web search tool not available for provider: ${provider}`, error });
      return {};
    }
  }

  /**
   * Collects provider-defined tools contributed by skills via the optional
   * `ConversationSkill.getProviderDefinedTools` hook.
   *
   * Unlike skill `Function`s — which are converted by `buildAiSdkTools` — these
   * are already AI SDK provider tools (e.g. Anthropic's native `text_editor` /
   * `bash`) and are merged straight into the tool set passed to `streamText`.
   * A skill returns only the tools the active `provider` natively supports.
   */
  private getSkillProviderDefinedTools(provider: string): ToolSet {
    const skills = this.params.skills;
    if (!skills || skills.length === 0) {
      return {};
    }

    const result: ToolSet = {};
    for (const skill of skills) {
      if (typeof skill.getProviderDefinedTools !== 'function') {
        continue;
      }
      try {
        Object.assign(result, skill.getProviderDefinedTools(provider));
      } catch (error) {
        this.logger.error({
          message: `Error collecting provider-defined tools from skill: ${skill.getName()}`,
          error,
        });
      }
    }
    return result;
  }

  /**
   * Returns the `toolChoice` value to use when the user has toggled web
   * search on. The toggle's contract: "guarantee a search this turn." We
   * deliver that by forcing the search tool as the first step's tool call,
   * after which the model returns to default (auto) tool selection.
   *
   * Provider notes:
   * - OpenAI / Anthropic / xAI: search is a model-called tool. Set
   *   `toolChoice: { type: 'tool', toolName: 'web_search' }` to force.
   * - Google: search is grounding-based — attaching `googleSearch` already
   *   forces it on every response (no model choice involved). So
   *   toolChoice is irrelevant; we omit it.
   * - Toggle off, or tool unavailable (e.g. Haiku/nano excluded models):
   *   return `undefined` so the SDK falls back to its default (auto).
   */
  private getWebSearchToolChoice(
    provider: string,
    webSearchTools: ToolSet,
    webSearchRequested?: boolean
  ): { type: 'tool'; toolName: string } | undefined {
    if (!webSearchRequested) {
      return undefined;
    }
    if (provider === 'google') {
      // googleSearch is auto-invoked by the API once attached; toolChoice is
      // a no-op for it.
      return undefined;
    }
    const toolName = Object.keys(webSearchTools)[0];
    if (!toolName) {
      // Model class doesn't have a search tool wired (e.g. nano/haiku).
      return undefined;
    }
    return { type: 'tool', toolName };
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
      skills: this.params.skills,
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
    // The single transport choke point: streamText, generateObject, and every per-step tool-loop
    // request run through the wrapped model, so transient provider failures retry invisibly here.
    return this.transportRetry.wrap(resolveModel(m) as never);
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
   * The AI SDK v6 normalizes the cached/reasoning/cache-write token breakdowns
   * across providers into `LanguageModelUsage.inputTokenDetails` and
   * `outputTokenDetails`, so we read them directly from there.
   */
  private mapSdkUsage(
    sdkUsage: LanguageModelUsage,
    modelString: string,
    steps?: Array<{ toolCalls?: Array<{ toolName?: string }> }>
  ): UsageData {
    const inputTokens = sdkUsage?.inputTokens ?? 0;
    const outputTokens = sdkUsage?.outputTokens ?? 0;
    const totalTokens = sdkUsage?.totalTokens ?? inputTokens + outputTokens;

    // AI SDK v6 provides structured token details. cacheReadTokens (cheap) and
    // cacheWriteTokens (premium, e.g. Anthropic cache_creation) are both carved
    // out of inputTokens and priced differently in calculateUsageCostUsd.
    const cachedInputTokens = sdkUsage?.inputTokenDetails?.cacheReadTokens ?? 0;
    const cacheWriteTokens = sdkUsage?.inputTokenDetails?.cacheWriteTokens ?? 0;
    const reasoningTokens = sdkUsage?.outputTokenDetails?.reasoningTokens ?? 0;

    const tokenUsage: TokenUsage = {
      inputTokens,
      cachedInputTokens,
      cacheWriteTokens,
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
  /**
   * Resolve the curated one-line timeline detail for a tool call — prefer the tool's own
   * `getTimelineDetail` (it can name an entity), fall back to a generic detail from the input.
   * Used by `buildAiSdkTools` so flow tool-progress events carry the same detail `mapFullStream`
   * surfaces on the non-flow streaming path. Best-effort: never throws.
   */
  private async resolveToolTimelineDetail(toolName: string, input: unknown): Promise<string | undefined> {
    try {
      const fn = this.functions.find((f) => f.definition.name === toolName);
      if (fn?.getTimelineDetail) {
        const detail = await fn.getTimelineDetail(input);
        if (detail) {
          return detail;
        }
      }
    } catch {
      // detail is best-effort — never let it break a tool call
    }
    return deriveToolCallDetail(input);
  }

  // ────────────────────────────────────────────────────────────
  // Full-stream mapping
  // ────────────────────────────────────────────────────────────

  private mapFullStream(aiSdkFullStream: AsyncIterable<any>): AsyncIterable<StreamPart> {
    const logger = this.logger;
    const functions = this.functions;
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
              const toolName = part.toolName ?? 'unknown';
              // A custom function tool resolves to one of our `functions`; a
              // provider-defined tool (Anthropic `text_editor` / `bash`) does not.
              // This is the robust custom-vs-provider signal (independent of the
              // exact provider tool name, which may be e.g. `str_replace_based_edit_tool`).
              const fn = functions.find((f) => f.definition.name === toolName);
              // Prefer the tool's own detail resolver (it can name an entity by
              // id); fall back to a generic detail derived from the input.
              let detail: string | undefined;
              try {
                if (fn?.getTimelineDetail) {
                  detail = (await fn.getTimelineDetail(part.input)) || undefined;
                }
              } catch {
                // detail is best-effort — never let it break the stream
              }
              // The provider text-editor multiplexes operations behind one tool (`command` verb +
              // `path`); suffix non-view operations so the display layer can label views (a file OR
              // a directory) apart from edits. `bash` carries `command` alone, so it's untouched.
              const input = (part.input ?? {}) as Record<string, unknown>;
              const editorOp =
                !fn && typeof input.command === 'string' && typeof (input.path ?? input.file_path) === 'string'
                  ? input.command
                  : undefined;
              // The provider computer tool likewise multiplexes every action behind one
              // name; suffix by action category (+ a human detail like coordinates or
              // typed text) so a long browser session reads as distinct steps in the
              // timeline instead of an opaque run of `computer` calls.
              const computerAction =
                !fn && toolName === 'computer' ? Conversation.describeComputerAction(input) : undefined;
              yield {
                type: 'tool-call' as const,
                toolName: computerAction
                  ? `${toolName}:${computerAction.suffix}`
                  : editorOp && editorOp !== 'view'
                    ? `${toolName}:edit`
                    : toolName,
                detail: detail ?? computerAction?.detail ?? deriveToolCallDetail(part.input),
                providerDefined: !fn,
              };
            } else if (part.type === 'source') {
              yield {
                type: 'source' as const,
                source: {
                  url: part.sourceType === 'url' ? part.url : undefined,
                  title: part.sourceType === 'url' ? part.title : undefined,
                },
              };
            } else if (part.type === 'error') {
              // The AI SDK never throws from streamText — failures arrive as `error` parts. Before this,
              // a transport failure surviving the retry layer SILENTLY truncated the stream: consumers saw
              // an empty message + zero usage and treated it as a (bogus) successful result. Surfacing it
              // lets the visible layers own it — FlowRunner's task retry, then the blocker-ask.
              const cause = (part as { error?: unknown }).error;
              throw cause instanceof Error
                ? cause
                : new Error(String((cause as { message?: string })?.message ?? cause ?? 'LLM stream error'));
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

/**
 * Best-effort extraction of a short, human-meaningful subject from a tool
 * call's input — used to personalize the call's node in the thinking timeline
 * (e.g. "Searched the web · 'best SaaS billing practices'").
 *
 * Kept generic (matches common input field names rather than specific tool
 * names) so the framework stays app-agnostic. Tools whose subject is only an
 * id, not a name, resolve their detail elsewhere; this just returns undefined.
 */
function deriveToolCallDetail(input: unknown): string | undefined {
  if (!input || typeof input !== 'object') {
    return undefined;
  }
  const obj = input as Record<string, unknown>;
  // Collapse to a single trimmed line (commands/patterns can be multi-line); the display layer
  // truncates. The audience is developers, so the raw argument (pattern / command / path) is the
  // reliable, useful signal — we only surface fields we recognize, never raw JSON.
  const oneLine = (v: unknown): string | undefined => {
    if (typeof v !== 'string') {
      return undefined;
    }
    const s = v.replace(/\s+/g, ' ').trim();
    return s || undefined;
  };

  // Common dev-tool arguments, most-salient first.
  const command = oneLine(obj.command); // bash shell line / text-editor operation verb
  const filePath = oneLine(obj.file_path) ?? oneLine(obj.filePath) ?? oneLine(obj.path); // read / write / edit
  // `text_editor` carries BOTH: `command` is the operation verb (view/str_replace/insert/create) and
  // `path` is the file — the file is the meaningful subject, so it wins. `bash` carries `command`
  // alone (the actual shell line, no path), so it falls through to the command.
  if (command && filePath) {
    return filePath;
  }
  if (command) {
    return command;
  }
  const pattern = oneLine(obj.pattern) ?? oneLine(obj.glob); // grep / glob
  if (pattern) {
    const path = oneLine(obj.path);
    return path ? `${pattern} — ${path}` : pattern;
  }
  const query = oneLine(obj.query); // search-style query
  if (query) {
    return query;
  }
  if (filePath) {
    return filePath;
  }
  const url = oneLine(obj.url); // fetch
  if (url) {
    return url;
  }
  const title = oneLine(obj.title); // created entity
  if (title) {
    return title;
  }
  // A markdown document — use its first heading as the title.
  if (typeof obj.markdown === 'string') {
    const heading = /^#{1,6}\s+(.+)$/m.exec(obj.markdown);
    if (heading?.[1]?.trim()) {
      return heading[1].trim();
    }
  }
  return undefined;
}
