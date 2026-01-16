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

    const object = this.parseAndValidateStructuredOutput<T>(result.message, args.schema);

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
        instructions: previousResponseId ? undefined : instructions,
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

    const created = await this.client.responses.create(
      body as never,
      args.abortSignal ? { signal: args.abortSignal } : undefined
    );

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

    return await this.waitForCompletion(created.id, args.abortSignal);
  }

  private async waitForCompletion(
    responseId: string,
    abortSignal?: AbortSignal
  ): Promise<{
    id?: string;
    status?: string;
    output_text?: string;
    output?: unknown[];
    usage?: unknown;
  }> {
    let delayMs = 500;

    for (;;) {
      if (abortSignal?.aborted) {
        throw new Error(`Request aborted`);
      }

      const resp = await this.client.responses.retrieve(
        responseId,
        undefined,
        abortSignal ? { signal: abortSignal } : undefined
      );

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
    const direct = typeof response.output_text === 'string' ? response.output_text.trim() : '';
    if (direct) {
      return direct;
    }

    const out = Array.isArray(response.output) ? response.output : [];
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
        return joined;
      }
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

  private parseAndValidateStructuredOutput<T>(text: string, schema: unknown): T {
    const parsed = this.parseJson(text);

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

  private parseJson(text: string): any {
    const cleaned = String(text ?? '')
      .trim()
      .replace(/^```(?:json)?/i, '')
      .replace(/```$/i, '')
      .trim();

    try {
      return JSON.parse(cleaned);
    } catch {
      const s = cleaned;
      const firstObj = s.indexOf('{');
      const firstArr = s.indexOf('[');
      const start = firstObj === -1 ? firstArr : firstArr === -1 ? firstObj : Math.min(firstObj, firstArr);

      const lastObj = s.lastIndexOf('}');
      const lastArr = s.lastIndexOf(']');
      const end = Math.max(lastObj, lastArr);

      if (start >= 0 && end > start) {
        return JSON.parse(s.slice(start, end + 1));
      }

      throw new Error(`Failed to parse model output as JSON`);
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
