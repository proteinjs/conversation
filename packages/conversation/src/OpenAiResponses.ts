import { OpenAI as OpenAIApi } from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat';
import { Logger, LogLevel } from '@proteinjs/logger';
import type { ConversationModule } from './ConversationModule';
import type { Function } from './Function';
import { UsageData, UsageDataAccumulator } from './UsageData';
import { ChatCompletionMessageParamFactory } from './ChatCompletionMessageParamFactory';
import type { GenerateResponseReturn, ToolInvocationProgressEvent, ToolInvocationResult } from './OpenAi';
import { DEFAULT_MODEL } from './OpenAi';

export const DEFAULT_RESPONSES_MODEL = 'gpt-5.2';
export const DEFAULT_MAX_TOOL_CALLS = 50;

export type OpenAiResponsesParams = {
  modules?: ConversationModule[];
  /** If provided, only these functions will be exposed to the model. */
  allowedFunctionNames?: string[];
  logLevel?: LogLevel;

  /** Default model when none is provided per call. */
  defaultModel?: string;

  /** Default cap for tool calls (per call). */
  maxToolCalls?: number;
};

export type GenerateTextParams = {
  messages: (string | ChatCompletionMessageParam)[];
  model?: string;

  abortSignal?: AbortSignal;

  /** Sampling & limits */
  temperature?: number;
  topP?: number;
  maxTokens?: number;

  /** Optional realtime hook for tool-call lifecycle (started/finished). */
  onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;

  /** Usage callback */
  onUsageData?: (usageData: UsageData) => Promise<void>;

  /** Per-call override for reasoning effort (reasoning models only). */
  reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;

  /** Hard cap for custom function tool calls executed by this wrapper. */
  maxToolCalls?: number;

  /** If true, run using Responses API background mode (polling). */
  backgroundMode?: boolean;
};

export type ResponsesGenerateObjectParams<S> = {
  messages: (string | ChatCompletionMessageParam)[];
  model?: string;

  abortSignal?: AbortSignal;

  /** Zod schema or JSON Schema */
  schema: S;

  /** Sampling & limits */
  temperature?: number;
  topP?: number;
  maxTokens?: number;

  /** Optional realtime hook for tool-call lifecycle (started/finished). */
  onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;

  /** Usage callback */
  onUsageData?: (usageData: UsageData) => Promise<void>;

  /** Per-call override for reasoning effort (reasoning models only). */
  reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;

  /** Hard cap for custom function tool calls executed by this wrapper. */
  maxToolCalls?: number;

  /** If true, run using Responses API background mode (polling). */
  backgroundMode?: boolean;
};

/**
 * OpenAI Responses API wrapper (tool-loop + usage tracking + ConversationModules).
 * - Uses Responses API directly
 * - Supports custom function tools (tool calling loop)
 * - Supports structured outputs (JSON schema / Zod)
 * - Tracks usage + tool calls using existing types
 * - Supports background mode (polling)
 * - Supports ConversationModules (system messages + tool registration)
 */
export class OpenAiResponses {
  private readonly client: OpenAIApi;
  private readonly logger: Logger;

  private readonly modules: ConversationModule[];
  private readonly allowedFunctionNames?: string[];
  private readonly defaultModel: string;
  private readonly defaultMaxToolCalls: number;

  private modulesProcessed = false;
  private processingModulesPromise: Promise<void> | null = null;

  private systemMessages: string[] = [];
  private functions: Function[] = [];

  constructor(opts: OpenAiResponsesParams = {}) {
    this.client = new OpenAIApi();
    this.logger = new Logger({ name: 'OpenAiResponses', logLevel: opts.logLevel });

    this.modules = opts.modules ?? [];
    this.allowedFunctionNames = opts.allowedFunctionNames;

    this.defaultModel = (opts.defaultModel ?? DEFAULT_RESPONSES_MODEL).trim();
    this.defaultMaxToolCalls = typeof opts.maxToolCalls === 'number' ? opts.maxToolCalls : DEFAULT_MAX_TOOL_CALLS;
  }

  /** Plain text generation (supports tool calling). */
  async generateText(args: GenerateTextParams): Promise<GenerateResponseReturn> {
    await this.ensureModulesProcessed();

    const model = this.resolveModel(args.model);
    const backgroundMode = this.resolveBackgroundMode({
      requested: args.backgroundMode,
      model,
      reasoningEffort: args.reasoningEffort,
    });

    const maxToolCalls = typeof args.maxToolCalls === 'number' ? args.maxToolCalls : this.defaultMaxToolCalls;

    const result = await this.run({
      model,
      messages: args.messages,
      temperature: args.temperature,
      topP: args.topP,
      maxTokens: args.maxTokens,
      abortSignal: args.abortSignal,
      onToolInvocation: args.onToolInvocation,
      reasoningEffort: args.reasoningEffort,
      maxToolCalls,
      backgroundMode,
      textFormat: undefined,
    });

    if (args.onUsageData) {
      await args.onUsageData(result.usagedata);
    }

    return result;
  }

  /** Back-compat alias for callers that use `generateResponse`. */
  async generateResponse(args: GenerateTextParams): Promise<GenerateResponseReturn> {
    return this.generateText(args);
  }

  /** Structured object generation (supports tool calling). */
  async generateObject<T>(args: ResponsesGenerateObjectParams<unknown>): Promise<{ object: T; usageData: UsageData }> {
    await this.ensureModulesProcessed();

    const model = this.resolveModel(args.model);
    const backgroundMode = this.resolveBackgroundMode({
      requested: args.backgroundMode,
      model,
      reasoningEffort: args.reasoningEffort,
    });

    const maxToolCalls = typeof args.maxToolCalls === 'number' ? args.maxToolCalls : this.defaultMaxToolCalls;
    const textFormat = this.buildTextFormat(args.schema);

    const result = await this.run({
      model,
      messages: args.messages,
      temperature: args.temperature,
      topP: args.topP,
      maxTokens: args.maxTokens,
      abortSignal: args.abortSignal,
      onToolInvocation: args.onToolInvocation,
      reasoningEffort: args.reasoningEffort,
      maxToolCalls,
      backgroundMode,
      textFormat,
    });

    const object = this.parseAndValidateStructuredOutput<T>(result.message, args.schema, {
      model,
      maxOutputTokens: args.maxTokens,
    });

    const outcome = {
      object,
      usageData: result.usagedata,
    };

    if (args.onUsageData) {
      await args.onUsageData(outcome.usageData);
    }

    return outcome;
  }

  // -----------------------------------------
  // Core runner (tool loop)
  // -----------------------------------------

  private async run(args: {
    model: string;
    messages: (string | ChatCompletionMessageParam)[];

    temperature?: number;
    topP?: number;
    maxTokens?: number;

    abortSignal?: AbortSignal;
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;

    reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;

    maxToolCalls: number;
    backgroundMode: boolean;

    textFormat?: unknown;
  }): Promise<GenerateResponseReturn> {
    // UsageDataAccumulator is typed around TiktokenModel; keep accumulator model stable,
    // and (optionally) report the actual model via upstream telemetry if you later choose to.
    const usage = new UsageDataAccumulator({ model: DEFAULT_MODEL });
    const toolInvocations: ToolInvocationResult[] = [];

    const tools = this.buildResponseTools(this.functions);

    const { instructions, input } = this.buildInstructionsAndInput(args.messages);

    let toolCallsExecuted = 0;
    let previousResponseId: string | undefined;
    let nextInput: unknown = input;

    for (;;) {
      const response = await this.createResponseAndMaybeWait({
        model: args.model,
        // Always pass instructions; they are not carried over with previous_response_id.
        instructions,
        input: nextInput,
        previousResponseId,
        tools,
        temperature: args.temperature,
        topP: args.topP,
        maxTokens: args.maxTokens,
        reasoningEffort: args.reasoningEffort,
        textFormat: args.textFormat,
        backgroundMode: args.backgroundMode,
        abortSignal: args.abortSignal,
      });

      this.addUsageFromResponse(response, usage);

      // For structured outputs we should not attempt to parse incomplete/failed/cancelled responses.
      // For plain-text generation, we allow "incomplete" to pass through (partial output),
      // but still fail on other non-completed statuses.
      this.throwIfResponseUnusable(response as any, {
        allowIncomplete: !args.textFormat,
        model: args.model,
        maxOutputTokens: args.maxTokens,
      });

      const functionCalls = this.extractFunctionCalls(response);
      if (functionCalls.length < 1) {
        const message = this.extractAssistantText(response);
        if (!message) {
          throw new Error(`Response was empty`);
        }
        return { message, usagedata: usage.usageData, toolInvocations };
      }

      if (toolCallsExecuted + functionCalls.length > args.maxToolCalls) {
        throw new Error(`Max tool calls (${args.maxToolCalls}) reached. Stopping execution.`);
      }

      if (!response.id) {
        throw new Error(`Responses API did not return an id for a tool-calling response.`);
      }

      const toolOutputs = await this.executeFunctionCalls({
        calls: functionCalls,
        functions: this.functions,
        usage,
        toolInvocations,
        onToolInvocation: args.onToolInvocation,
      });

      toolCallsExecuted += functionCalls.length;

      previousResponseId = response.id;
      nextInput = toolOutputs;

      this.logger.debug({
        message: `Tool loop continuing`,
        obj: { toolCallsExecuted, lastToolCallCount: functionCalls.length, responseId: previousResponseId },
      });
    }
  }

  private throwIfResponseUnusable(
    response: any,
    opts: { allowIncomplete: boolean; model?: string; maxOutputTokens?: number }
  ): void {
    const statusRaw = typeof response?.status === 'string' ? String(response.status) : '';
    const status = statusRaw.toLowerCase();

    if (!status || status === 'completed') {
      return;
    }

    if (status === 'incomplete' && opts.allowIncomplete) {
      return;
    }

    const id = typeof response?.id === 'string' ? response.id : '';
    const reason = response?.incomplete_details?.reason;
    const apiErr = response?.error;

    const directOutputText = typeof response?.output_text === 'string' ? response.output_text : '';
    const assistantText = this.extractAssistantText(response as any);

    const outTextLen = directOutputText ? directOutputText.length : 0;
    const assistantLen = assistantText ? assistantText.length : 0;

    const usage = response?.usage;
    const inputTokens = typeof usage?.input_tokens === 'number' ? usage.input_tokens : undefined;
    const outputTokens = typeof usage?.output_tokens === 'number' ? usage.output_tokens : undefined;
    const totalTokens =
      typeof usage?.total_tokens === 'number'
        ? usage.total_tokens
        : typeof inputTokens === 'number' && typeof outputTokens === 'number'
          ? inputTokens + outputTokens
          : undefined;

    let msg = `Responses API returned status="${status}"`;
    if (id) {
      msg += ` (id=${id})`;
    }
    msg += `.`;

    const details: Record<string, unknown> = {
      response_id: id || undefined,
      status,
      model: typeof opts.model === 'string' && opts.model.trim() ? opts.model : undefined,
      max_output_tokens: typeof opts.maxOutputTokens === 'number' ? opts.maxOutputTokens : undefined,

      incomplete_reason: typeof reason === 'string' && reason.trim() ? reason : undefined,
      api_error: apiErr ?? undefined,

      usage_input_tokens: inputTokens,
      usage_output_tokens: outputTokens,
      usage_total_tokens: totalTokens,

      output_text_len: outTextLen || undefined,
      output_text_tail: outTextLen > 0 ? truncateTail(directOutputText, 400) : undefined,

      assistant_text_len: assistantLen || undefined,
      assistant_text_tail: assistantLen > 0 ? truncateTail(assistantText, 400) : undefined,
    };

    const extra: string[] = [];
    if (details.model) {
      extra.push(`model=${details.model}`);
    }
    if (typeof details.max_output_tokens === 'number') {
      extra.push(`max_output_tokens=${details.max_output_tokens}`);
    }
    if (details.incomplete_reason) {
      extra.push(`reason=${details.incomplete_reason}`);
    }
    if (typeof details.output_text_len === 'number') {
      extra.push(`output_text_len=${details.output_text_len}`);
    }
    if (typeof details.assistant_text_len === 'number') {
      extra.push(`assistant_text_len=${details.assistant_text_len}`);
    }

    if (extra.length > 0) {
      msg += ` ${extra.join(' ')}.`;
    }

    throw new OpenAiResponsesError({
      code: 'RESPONSE_STATUS',
      message: msg,
      details,
    });
  }

  private toOpenAiApiError(
    error: unknown,
    meta: {
      operation: 'responses.create' | 'responses.retrieve';
      model?: string;
      reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;
      backgroundMode?: boolean;
      responseId?: string;
      previousResponseId?: string;
      pollAttempt?: number;
    }
  ): OpenAiResponsesError {
    const status = extractHttpStatus(error);
    const requestId = extractRequestId(error);
    const retryable = isRetryableHttpStatus(status);

    const errMsg = error instanceof Error ? error.message : String(error ?? '');
    const errName = error instanceof Error ? error.name : undefined;

    let msg = `OpenAI ${meta.operation} failed.`;
    const extra: string[] = [];

    if (typeof status === 'number') {
      extra.push(`status=${status}`);
    }
    if (requestId) {
      extra.push(`requestId=${requestId}`);
    }
    if (meta.responseId) {
      extra.push(`responseId=${meta.responseId}`);
    }
    if (meta.backgroundMode) {
      extra.push(`background=true`);
    }
    if (typeof meta.pollAttempt === 'number') {
      extra.push(`pollAttempt=${meta.pollAttempt}`);
    }
    if (typeof meta.model === 'string' && meta.model.trim()) {
      extra.push(`model=${meta.model.trim()}`);
    }
    if (meta.reasoningEffort) {
      extra.push(`reasoningEffort=${meta.reasoningEffort}`);
    }

    if (extra.length > 0) {
      msg += ` ${extra.join(' ')}.`;
    }
    if (errMsg) {
      msg += ` error=${JSON.stringify(errMsg)}.`;
    }

    const details: Record<string, unknown> = {
      operation: meta.operation,
      status: typeof status === 'number' ? status : undefined,
      request_id: requestId,
      response_id: meta.responseId,
      previous_response_id: meta.previousResponseId,
      background: meta.backgroundMode ? true : undefined,
      poll_attempt: meta.pollAttempt,
      model: typeof meta.model === 'string' && meta.model.trim() ? meta.model.trim() : undefined,
      reasoning_effort: meta.reasoningEffort,
      error_name: errName,
    };

    return new OpenAiResponsesError({
      code: 'OPENAI_API',
      message: msg,
      details,
      cause: error,
      retryable,
    });
  }

  private async createResponseAndMaybeWait(args: {
    model: string;
    instructions?: string;
    input: unknown;
    previousResponseId?: string;

    tools: Array<{ type: 'function'; name: string; description?: string; parameters?: unknown; strict?: boolean }>;
    temperature?: number;
    topP?: number;
    maxTokens?: number;
    reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;

    textFormat?: unknown;

    backgroundMode: boolean;
    abortSignal?: AbortSignal;
  }): Promise<{
    id?: string;
    status?: string;
    output_text?: string;
    output?: unknown[];
    usage?: unknown;
  }> {
    const body: Record<string, unknown> = {
      model: args.model,
      input: args.input,
    };

    if (args.instructions) {
      body.instructions = args.instructions;
    }

    if (args.previousResponseId) {
      body.previous_response_id = args.previousResponseId;
    }

    if (args.tools.length > 0) {
      body.tools = args.tools;
    }

    if (typeof args.temperature === 'number') {
      body.temperature = args.temperature;
    }
    if (typeof args.topP === 'number') {
      body.top_p = args.topP;
    }
    if (typeof args.maxTokens === 'number') {
      body.max_output_tokens = args.maxTokens;
    }
    if (args.reasoningEffort) {
      body.reasoning = { effort: args.reasoningEffort };
    }
    if (args.textFormat) {
      body.text = { format: args.textFormat };
    }

    if (args.backgroundMode) {
      body.background = true;
      body.store = true;
    }

    let created: any;
    try {
      created = await this.client.responses.create(
        body as never,
        args.abortSignal ? { signal: args.abortSignal } : undefined
      );
    } catch (error: unknown) {
      throw this.toOpenAiApiError(error, {
        operation: 'responses.create',
        model: args.model,
        reasoningEffort: args.reasoningEffort,
        backgroundMode: args.backgroundMode,
        previousResponseId: args.previousResponseId,
      });
    }

    if (!args.backgroundMode) {
      return created as unknown as {
        id?: string;
        status?: string;
        output_text?: string;
        output?: unknown[];
        usage?: unknown;
      };
    }

    if (!created?.id) {
      return created as unknown as {
        id?: string;
        status?: string;
        output_text?: string;
        output?: unknown[];
        usage?: unknown;
      };
    }

    return await this.waitForCompletion(created.id, args.abortSignal, {
      model: args.model,
      reasoningEffort: args.reasoningEffort,
    });
  }

  private async waitForCompletion(
    responseId: string,
    abortSignal?: AbortSignal,
    ctx?: { model?: string; reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort }
  ): Promise<{
    id?: string;
    status?: string;
    output_text?: string;
    output?: unknown[];
    usage?: unknown;
  }> {
    let delayMs = 500;
    let pollAttempt = 0;

    for (;;) {
      if (abortSignal?.aborted) {
        throw new Error(`Request aborted`);
      }

      pollAttempt += 1;

      let resp: any;
      try {
        resp = await this.client.responses.retrieve(
          responseId,
          undefined,
          abortSignal ? { signal: abortSignal } : undefined
        );
      } catch (error: unknown) {
        throw this.toOpenAiApiError(error, {
          operation: 'responses.retrieve',
          model: ctx?.model,
          reasoningEffort: ctx?.reasoningEffort,
          backgroundMode: true,
          responseId,
          pollAttempt,
        });
      }

      const status = typeof (resp as any)?.status === 'string' ? String((resp as any).status).toLowerCase() : '';
      if (status === 'completed' || status === 'failed' || status === 'cancelled' || status === 'incomplete') {
        return resp as unknown as {
          id?: string;
          status?: string;
          output_text?: string;
          output?: unknown[];
          usage?: unknown;
        };
      }

      this.logger.debug({ message: `Polling response`, obj: { responseId, status, delayMs } });

      await sleep(delayMs);
      delayMs = Math.min(5000, Math.floor(delayMs * 1.5));
    }
  }

  // -----------------------------------------
  // Tool calls
  // -----------------------------------------

  private buildResponseTools(
    functions: Function[]
  ): Array<{ type: 'function'; name: string; description?: string; parameters?: unknown; strict?: boolean }> {
    const tools: Array<{
      type: 'function';
      name: string;
      description?: string;
      parameters?: unknown;
      strict?: boolean;
    }> = [];

    if (!functions || functions.length < 1) {
      return tools;
    }

    for (const f of functions) {
      const def = f.definition;
      if (!def?.name) {
        continue;
      }

      tools.push({
        type: 'function',
        name: def.name,
        description: def.description,
        parameters: def.parameters,
        // strict: true,
      });
    }

    return tools;
  }

  private extractFunctionCalls(response: { output?: unknown[] }): Array<{
    type: 'function_call';
    call_id: string;
    name: string;
    arguments: string;
  }> {
    const out = Array.isArray(response.output) ? response.output : [];
    const calls: Array<{ type: 'function_call'; call_id: string; name: string; arguments: string }> = [];

    for (const item of out) {
      if (!item || typeof item !== 'object') {
        continue;
      }
      const rec = item as Record<string, unknown>;
      if (rec.type !== 'function_call') {
        continue;
      }

      const call_id = typeof rec.call_id === 'string' ? rec.call_id : '';
      const name = typeof rec.name === 'string' ? rec.name : '';
      const args = typeof rec.arguments === 'string' ? rec.arguments : '';

      if (!call_id || !name) {
        continue;
      }

      calls.push({ type: 'function_call', call_id, name, arguments: args });
    }

    return calls;
  }

  private async executeFunctionCalls(args: {
    calls: Array<{ type: 'function_call'; call_id: string; name: string; arguments: string }>;
    functions: Function[];
    usage: UsageDataAccumulator;
    toolInvocations: ToolInvocationResult[];
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
  }): Promise<Array<{ type: 'function_call_output'; call_id: string; output: string }>> {
    const outputs: Array<{ type: 'function_call_output'; call_id: string; output: string }> = [];

    for (const call of args.calls) {
      outputs.push(
        await this.executeFunctionCall({
          call,
          functions: args.functions,
          usage: args.usage,
          toolInvocations: args.toolInvocations,
          onToolInvocation: args.onToolInvocation,
        })
      );
    }

    return outputs;
  }

  private async executeFunctionCall(args: {
    call: { call_id: string; name: string; arguments: string };
    functions: Function[];
    usage: UsageDataAccumulator;
    toolInvocations: ToolInvocationResult[];
    onToolInvocation?: (evt: ToolInvocationProgressEvent) => void;
  }): Promise<{ type: 'function_call_output'; call_id: string; output: string }> {
    const callId = args.call.call_id;
    const rawName = args.call.name;
    const shortName = rawName.split('.').pop() ?? rawName;

    const functionToCall =
      args.functions.find((fx) => fx.definition.name === rawName) ??
      args.functions.find((fx) => (fx.definition.name.split('.').pop() ?? fx.definition.name) === shortName);

    const startedAt = new Date();

    let parsedArgs: unknown;
    try {
      parsedArgs = JSON.parse(args.call.arguments ?? '{}');
    } catch {
      parsedArgs = args.call.arguments;
    }

    args.onToolInvocation?.({
      type: 'started',
      id: callId,
      name: functionToCall?.definition?.name ?? shortName,
      startedAt,
      input: parsedArgs,
    });

    if (!functionToCall) {
      const finishedAt = new Date();
      const rec: ToolInvocationResult = {
        id: callId,
        name: shortName,
        startedAt,
        finishedAt,
        input: parsedArgs,
        ok: false,
        error: { message: `Assistant attempted to call nonexistent function` },
      };
      args.toolInvocations.push(rec);
      args.onToolInvocation?.({ type: 'finished', result: rec });

      return {
        type: 'function_call_output',
        call_id: callId,
        output: JSON.stringify({ error: rec.error?.message, functionName: shortName }),
      };
    }

    try {
      let argsObj: unknown;
      try {
        argsObj = JSON.parse(args.call.arguments ?? '{}');
      } catch {
        argsObj = {};
      }

      args.usage.recordToolCall(functionToCall.definition.name);

      const returnObject = await functionToCall.call(argsObj);
      const finishedAt = new Date();

      const rec: ToolInvocationResult = {
        id: callId,
        name: functionToCall.definition.name,
        startedAt,
        finishedAt,
        input: argsObj,
        ok: true,
        data: returnObject,
      };
      args.toolInvocations.push(rec);
      args.onToolInvocation?.({ type: 'finished', result: rec });

      const output = await this.formatToolReturn(returnObject);

      return {
        type: 'function_call_output',
        call_id: callId,
        output,
      };
    } catch (error: unknown) {
      const finishedAt = new Date();

      const errMessage = error instanceof Error ? error.message : String(error);
      const errStack = error instanceof Error ? error.stack : undefined;

      const rec: ToolInvocationResult = {
        id: callId,
        name: functionToCall.definition.name,
        startedAt,
        finishedAt,
        input: parsedArgs,
        ok: false,
        error: { message: errMessage, stack: errStack },
      };
      args.toolInvocations.push(rec);
      args.onToolInvocation?.({ type: 'finished', result: rec });

      throw error;
    }
  }

  private async formatToolReturn(returnObject: unknown): Promise<string> {
    if (typeof returnObject === 'undefined') {
      return JSON.stringify({ result: 'Function with no return value executed successfully' });
    }

    if (returnObject instanceof ChatCompletionMessageParamFactory) {
      const messageParams = await returnObject.create();
      const normalized = (messageParams ?? [])
        .map((m) => ({
          role: m.role,
          content: this.extractTextContent(m.content),
        }))
        .filter((m) => typeof m.content === 'string' && m.content.trim().length > 0);

      return JSON.stringify({ messages: normalized });
    }

    return JSON.stringify(returnObject);
  }

  // -----------------------------------------
  // Usage + text extraction
  // -----------------------------------------

  private addUsageFromResponse(response: { usage?: unknown }, usage: UsageDataAccumulator): void {
    const u = response.usage;
    if (!u || typeof u !== 'object') {
      return;
    }

    const rec = u as Record<string, unknown>;
    const input = typeof rec.input_tokens === 'number' ? rec.input_tokens : 0;
    const output = typeof rec.output_tokens === 'number' ? rec.output_tokens : 0;
    const total = typeof rec.total_tokens === 'number' ? rec.total_tokens : input + output;

    let cached = 0;
    let reasoning = 0;

    const inputDetails = rec.input_tokens_details;
    if (inputDetails && typeof inputDetails === 'object') {
      const id = inputDetails as Record<string, unknown>;
      cached = typeof id.cached_tokens === 'number' ? id.cached_tokens : 0;
    }

    const outputDetails = rec.output_tokens_details;
    if (outputDetails && typeof outputDetails === 'object') {
      const od = outputDetails as Record<string, unknown>;
      reasoning = typeof od.reasoning_tokens === 'number' ? od.reasoning_tokens : 0;
    }

    usage.addTokenUsage({
      promptTokens: input,
      cachedPromptTokens: cached,
      completionTokens: output,
      reasoningTokens: reasoning,
      totalTokens: total,
    });
  }

  private extractAssistantText(response: { output_text?: string; output?: unknown[] }): string {
    const out = Array.isArray(response.output) ? response.output : [];

    let lastJoined = '';

    for (const item of out) {
      if (!item || typeof item !== 'object') {
        continue;
      }
      const rec = item as Record<string, unknown>;
      if (rec.type !== 'message') {
        continue;
      }
      if (rec.role !== 'assistant') {
        continue;
      }

      const contentRaw = rec.content;
      if (!Array.isArray(contentRaw)) {
        continue;
      }

      const pieces: string[] = [];
      for (const c of contentRaw) {
        if (!c || typeof c !== 'object') {
          continue;
        }
        const part = c as Record<string, unknown>;
        if (part.type !== 'output_text') {
          continue;
        }
        const t = part.text;
        if (typeof t === 'string' && t.trim()) {
          pieces.push(t);
        }
      }

      const joined = pieces.join('\n').trim();
      if (joined) {
        lastJoined = joined;
      }
    }

    if (lastJoined) {
      return lastJoined;
    }

    const direct = typeof response.output_text === 'string' ? response.output_text.trim() : '';
    if (direct) {
      return direct;
    }

    return '';
  }

  // -----------------------------------------
  // Structured outputs (JSON schema / Zod)
  // -----------------------------------------

  private buildTextFormat(schema: unknown): unknown {
    if (this.isZodSchema(schema)) {
      // Prefer the official helper when schema is Zod.
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require('openai/helpers/zod');
      return mod.zodTextFormat(schema, 'output');
    }

    return {
      type: 'json_schema',
      name: 'output',
      strict: true,
      schema: this.strictifyJsonSchema(schema),
    };
  }

  private parseAndValidateStructuredOutput<T>(
    text: string,
    schema: unknown,
    ctx?: { model?: string; maxOutputTokens?: number }
  ): T {
    const parsed = this.parseJson(text, ctx);

    if (this.isZodSchema(schema)) {
      const res = schema.safeParse(parsed);
      if (!res?.success) {
        throw new Error(`Structured output failed schema validation`);
      }
      return res.data as T;
    }

    return parsed as T;
  }

  private isZodSchema(schema: unknown): schema is { safeParse: (input: unknown) => { success: boolean; data?: any } } {
    if (!schema || (typeof schema !== 'object' && typeof schema !== 'function')) {
      return false;
    }
    return typeof (schema as any).safeParse === 'function';
  }

  private parseJson(text: string, ctx?: { model?: string; maxOutputTokens?: number }): any {
    const cleaned = String(text ?? '')
      .trim()
      .replace(/^```(?:json)?/i, '')
      .replace(/```$/i, '')
      .trim();

    try {
      return JSON.parse(cleaned);
    } catch (err1: unknown) {
      const firstErrMsg = err1 instanceof Error ? err1.message : String(err1);

      const s = cleaned;
      const firstObj = s.indexOf('{');
      const firstArr = s.indexOf('[');
      const start = firstObj === -1 ? firstArr : firstArr === -1 ? firstObj : Math.min(firstObj, firstArr);

      const lastObj = s.lastIndexOf('}');
      const lastArr = s.lastIndexOf(']');
      const end = Math.max(lastObj, lastArr);

      if (start >= 0 && end > start) {
        const candidate = s.slice(start, end + 1);
        try {
          return JSON.parse(candidate);
        } catch (err2: unknown) {
          const secondErrMsg = err2 instanceof Error ? err2.message : String(err2);

          const pos2rel = extractJsonParsePosition(secondErrMsg);
          const pos2 = typeof pos2rel === 'number' ? start + pos2rel : undefined;

          const pos1 = extractJsonParsePosition(firstErrMsg);
          const pos = typeof pos2 === 'number' ? pos2 : pos1;

          const lc = extractJsonParseLineCol(secondErrMsg) ?? extractJsonParseLineCol(firstErrMsg);

          const details: Record<string, unknown> = {
            model: typeof ctx?.model === 'string' && ctx.model.trim() ? ctx.model : undefined,
            max_output_tokens: typeof ctx?.maxOutputTokens === 'number' ? ctx.maxOutputTokens : undefined,

            cleaned_len: s.length,
            cleaned_head: truncateHead(s, 250),
            cleaned_tail: truncateTail(s, 500),

            json_start: start,
            json_end: end,
            json_candidate_len: candidate.length,

            first_error: firstErrMsg,
            second_error: secondErrMsg,

            error_pos: typeof pos === 'number' ? pos : undefined,
            error_line: lc?.line,
            error_column: lc?.column,
            error_context: typeof pos === 'number' ? snippetAround(s, pos, 160) : undefined,
          };

          const msg =
            `Failed to parse model output as JSON. ` +
            `cleaned_len=${s.length} json_start=${start} json_end=${end}. ` +
            `first_error=${JSON.stringify(firstErrMsg)} second_error=${JSON.stringify(secondErrMsg)}.`;

          throw new OpenAiResponsesError({
            code: 'JSON_PARSE',
            message: msg,
            details,
            cause: err2,
          });
        }
      }

      const pos = extractJsonParsePosition(firstErrMsg);
      const lc = extractJsonParseLineCol(firstErrMsg);

      const details: Record<string, unknown> = {
        model: typeof ctx?.model === 'string' && ctx.model.trim() ? ctx.model : undefined,
        max_output_tokens: typeof ctx?.maxOutputTokens === 'number' ? ctx.maxOutputTokens : undefined,

        cleaned_len: s.length,
        cleaned_head: truncateHead(s, 250),
        cleaned_tail: truncateTail(s, 500),

        json_start: start >= 0 ? start : undefined,
        json_end: end >= 0 ? end : undefined,

        first_error: firstErrMsg,

        error_pos: typeof pos === 'number' ? pos : undefined,
        error_line: lc?.line,
        error_column: lc?.column,
        error_context: typeof pos === 'number' ? snippetAround(s, pos, 160) : undefined,
      };

      const msg =
        `Failed to parse model output as JSON. ` +
        `cleaned_len=${s.length}. ` +
        `error=${JSON.stringify(firstErrMsg)}.`;

      throw new OpenAiResponsesError({
        code: 'JSON_PARSE',
        message: msg,
        details,
        cause: err1,
      });
    }
  }

  /**
   * Strictifies a plain JSON Schema for OpenAI Structured Outputs (strict mode):
   *  - Ensures every object has `additionalProperties: false`
   *  - Ensures every object has a `required` array that includes **all** keys in `properties`
   *  - Adds missing `type: "object"` / `type: "array"` where implied by keywords
   */
  private strictifyJsonSchema(schema: unknown): any {
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

  // -----------------------------------------
  // Messages + modules
  // -----------------------------------------

  private buildInstructionsAndInput(messages: (string | ChatCompletionMessageParam)[]): {
    instructions?: string;
    input: Array<{ role: 'user' | 'assistant'; content: string }>;
  } {
    const instructionsParts: string[] = [];
    instructionsParts.push(...this.systemMessages);

    const input: Array<{ role: 'user' | 'assistant'; content: string }> = [];

    for (const m of messages) {
      const msg: ChatCompletionMessageParam =
        typeof m === 'string' ? ({ role: 'user', content: m } as ChatCompletionMessageParam) : m;

      if (msg.role === 'system') {
        const c = this.extractTextContent(msg.content).trim();
        if (c) {
          instructionsParts.push(c);
        }
        continue;
      }

      if (msg.role === 'tool') {
        continue;
      }

      const role: 'user' | 'assistant' = msg.role === 'assistant' ? 'assistant' : 'user';
      const content = this.extractTextContent(msg.content).trim();
      if (!content) {
        continue;
      }

      input.push({ role, content });
    }

    const instructions =
      instructionsParts.map((s) => String(s ?? '').trim()).filter(Boolean).length > 0
        ? instructionsParts
            .map((s) => String(s ?? '').trim())
            .filter(Boolean)
            .join('\n\n')
        : undefined;

    return { instructions, input };
  }

  private extractTextContent(content: ChatCompletionMessageParam['content']): string {
    if (typeof content === 'string') {
      return content;
    }
    if (!content) {
      return '';
    }
    if (Array.isArray(content)) {
      return content
        .map((p: any) => {
          if (typeof p === 'string') {
            return p;
          }
          if (p?.type === 'text' && typeof p?.text === 'string') {
            return p.text;
          }
          return '';
        })
        .join('\n');
    }
    return '';
  }

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
    } catch (error: unknown) {
      this.processingModulesPromise = null;
      throw error;
    }
  }

  private async processModules(): Promise<void> {
    if (!this.modules || this.modules.length < 1) {
      return;
    }

    for (const module of this.modules) {
      const moduleName = module.getName();

      const rawSystem = await Promise.resolve(module.getSystemMessages());
      const sysArr = Array.isArray(rawSystem) ? rawSystem : rawSystem ? [rawSystem] : [];
      const trimmed = sysArr.map((s) => String(s ?? '').trim()).filter(Boolean);

      if (trimmed.length > 0) {
        const formatted = trimmed.join('. ');
        this.systemMessages.push(`The following are instructions from the ${moduleName} module:\n${formatted}`);
      }

      const moduleFunctions = module.getFunctions();
      const filtered = this.filterFunctions(moduleFunctions);
      this.functions.push(...filtered);

      const fnInstructions = this.buildFunctionInstructionsMessage(moduleName, filtered);
      if (fnInstructions) {
        this.systemMessages.push(fnInstructions);
      }
    }
  }

  private filterFunctions(functions: Function[]): Function[] {
    if (!this.allowedFunctionNames || this.allowedFunctionNames.length < 1) {
      return functions;
    }

    const allow = new Set(this.allowedFunctionNames.map((n) => String(n).trim()).filter(Boolean));
    return functions.filter((f) => {
      const name = String(f.definition?.name ?? '').trim();
      if (!name) {
        return false;
      }
      const short = name.split('.').pop() ?? name;
      return allow.has(name) || allow.has(short);
    });
  }

  private buildFunctionInstructionsMessage(moduleName: string, functions: Function[]): string | null {
    let msg = `The following are instructions from functions in the ${moduleName} module:`;
    let added = false;

    for (const f of functions) {
      const name = String(f.definition?.name ?? '').trim();
      const instructions = f.instructions;
      if (!name || !instructions || instructions.length < 1) {
        continue;
      }

      const paragraph = instructions
        .map((s) => String(s ?? '').trim())
        .filter(Boolean)
        .join('. ');
      if (!paragraph) {
        continue;
      }

      added = true;
      msg += ` ${name}: ${paragraph}.`;
    }

    return added ? msg : null;
  }

  // -----------------------------------------
  // Model/background defaults
  // -----------------------------------------

  private resolveModel(model?: string): string {
    const m = (model ?? this.defaultModel).trim();
    return m.length > 0 ? m : DEFAULT_RESPONSES_MODEL;
  }

  private resolveBackgroundMode(args: {
    requested?: boolean;
    model: string;
    reasoningEffort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort;
  }): boolean {
    if (typeof args.requested === 'boolean') {
      return args.requested;
    }
    if (this.isProModel(args.model)) {
      return true;
    }
    if (this.isHighReasoningEffort(args.reasoningEffort)) {
      return true;
    }
    return false;
  }

  private isProModel(model: string): boolean {
    const m = String(model ?? '').toLowerCase();
    return /(^|[-_.])pro($|[-_.])/.test(m);
  }

  private isHighReasoningEffort(effort?: OpenAIApi.Chat.Completions.ChatCompletionReasoningEffort): boolean {
    const v = String(effort ?? '').toLowerCase();
    return v === 'high' || v === 'xhigh';
  }
}

export type OpenAiResponsesErrorCode = 'OPENAI_API' | 'RESPONSE_STATUS' | 'JSON_PARSE';

export class OpenAiResponsesError extends Error {
  public readonly code: OpenAiResponsesErrorCode;
  public readonly details: Record<string, unknown>;
  public readonly cause?: unknown;
  public readonly retryable: boolean;

  constructor(args: {
    code: OpenAiResponsesErrorCode;
    message: string;
    details?: Record<string, unknown>;
    cause?: unknown;
    retryable?: boolean;
  }) {
    super(args.message);
    this.name = 'OpenAiResponsesError';
    this.code = args.code;
    this.details = args.details ?? {};
    this.cause = args.cause;
    this.retryable = typeof args.retryable === 'boolean' ? args.retryable : true;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

function truncateHead(text: string, max: number): string {
  const s = String(text ?? '');
  if (max <= 0) {
    return '';
  }
  if (s.length <= max) {
    return s;
  }
  return s.slice(0, max) + '...';
}

function truncateTail(text: string, max: number): string {
  const s = String(text ?? '');
  if (max <= 0) {
    return '';
  }
  if (s.length <= max) {
    return s;
  }
  return '...' + s.slice(s.length - max);
}

function extractJsonParsePosition(errMsg: string): number | undefined {
  const m = String(errMsg ?? '').match(/at position\s+(\d+)/i);
  if (!m) {
    return undefined;
  }
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : undefined;
}

function extractJsonParseLineCol(errMsg: string): { line?: number; column?: number } | undefined {
  const m = String(errMsg ?? '').match(/line\s+(\d+)\s+column\s+(\d+)/i);
  if (!m) {
    return undefined;
  }
  const line = Number(m[1]);
  const column = Number(m[2]);
  return {
    line: Number.isFinite(line) ? line : undefined,
    column: Number.isFinite(column) ? column : undefined,
  };
}

function snippetAround(text: string, pos: number, radius: number): string {
  const s = String(text ?? '');
  const p = Math.max(0, Math.min(s.length, Number.isFinite(pos) ? pos : 0));
  const r = Math.max(0, radius);

  const start = Math.max(0, p - r);
  const end = Math.min(s.length, p + r);

  const before = s.slice(start, p);
  const after = s.slice(p, end);

  const left = start > 0 ? '...' : '';
  const right = end < s.length ? '...' : '';

  return `${left}${before}<<HERE>>${after}${right}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractHttpStatus(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') {
    return undefined;
  }
  const rec = error as Record<string, unknown>;
  const status = rec.status;
  if (typeof status === 'number' && Number.isFinite(status)) {
    return status;
  }
  const statusCode = rec.statusCode;
  if (typeof statusCode === 'number' && Number.isFinite(statusCode)) {
    return statusCode;
  }
  return undefined;
}

function extractRequestId(error: unknown): string | undefined {
  if (!error || typeof error !== 'object') {
    return undefined;
  }
  const rec = error as Record<string, unknown>;

  const direct = rec.request_id ?? rec.requestId;
  if (typeof direct === 'string' && direct.trim()) {
    return direct.trim();
  }

  const headers = rec.headers as any;
  if (!headers) {
    return undefined;
  }

  if (typeof headers.get === 'function') {
    const v = headers.get('x-request-id');
    return typeof v === 'string' && v.trim() ? v.trim() : undefined;
  }

  if (typeof headers === 'object' && !Array.isArray(headers)) {
    for (const k of Object.keys(headers)) {
      if (String(k).toLowerCase() !== 'x-request-id') {
        continue;
      }
      const v = (headers as any)[k];
      return typeof v === 'string' && v.trim() ? v.trim() : undefined;
    }
  }

  return undefined;
}

function isRetryableHttpStatus(status: number | undefined): boolean {
  if (typeof status !== 'number') {
    return true;
  }
  if (status === 408 || status === 409 || status === 429) {
    return true;
  }
  if (status >= 500) {
    return true;
  }
  return false;
}
