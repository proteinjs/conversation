import { AISDKError } from 'ai';
import { APIError as OpenAiSdkError } from 'openai';
import { ErrorLine, LogLineErrors } from '@proteinjs/logger';
import { ProviderLogPayloads } from './ProviderLogPayloads';
import { ProviderBillingError, providerErrorCodes } from './ProviderBillingError';
import { TransientProviderError } from './TransientProviderError';

/** What the marker knows about the call that failed; each part is optional. */
export type ProviderFailureCall = {
  /** The failing model's id. */
  modelId?: string;
  /** The provider as the client library names it (`anthropic.messages`); its family is what prints. */
  provider?: string;
  /**
   * What `error` was worded FROM, when the library built it itself out of a provider's payload
   * (the typed errors wrap the transport's error; an error part's raw payload becomes an Error).
   * Its presence declares `error` provider-worded whatever its class.
   */
  wordedFrom?: unknown;
};

/**
 * The ONE owner of how a model provider's failure reads on a LOG LINE: the error's name, the
 * HTTP status, the provider, the model, the vendor's own error code and a sentence of this
 * library's — never the request body, the response body or the response headers.
 *
 * The client libraries keep all three on the errors they raise (`APICallError.requestBodyValues`
 * is the conversation being sent; `responseBody` and `responseHeaders` beside it; an answer that
 * did not parse keeps the model's `text`), as ordinary enumerable fields — so a log line that is
 * handed such an error WHOLE carries them. They stay on what the library THROWS, exactly as they
 * were: callers classify on them (retry hints, billing shapes, decline routing). What changes is
 * what is PRINTED. Every provider-worded error that leaves the library is marked with the logger
 * (`LogLineErrors.mark`, which never touches the error); from then on any line about it — this
 * library's or a caller's, as the line's `error` or anywhere inside its `obj` — carries the
 * stand-in described above.
 *
 * WHAT IS MARKED: an error of the AI SDK's family (`APICallError` and its siblings), an error of
 * the OpenAI SDK's, the library's own typed errors that wrap one (TransientProviderError,
 * ProviderBillingError), a raw provider payload (what a stream's error part carries), and an
 * error the library built from such a payload (`wordedFrom`). An error that is nobody's payload
 * (a dropped connection's `TypeError`, an abort) keeps its own words.
 *
 * WHERE: the transport choke point (LlmTransportRetry — every error it judges, passes through or
 * wraps) and the places `Conversation` turns a failed call or a stream's error part into a throw.
 *
 * The provider's text rides a line only behind the dev-only payload switch (ProviderLogPayloads):
 * with both gates open a marked error prints as it is. Asked at each line, like the switch.
 */
export class ProviderFailureLine {
  /** A vendor code as it may ride a line: an identifier, never prose. */
  private static readonly CODE_SHAPE = /^[A-Za-z0-9_.-]{1,64}$/;
  /** Wrapper words that classify nothing (`{ type: 'error', error: { type: 'overloaded_error' } }`). */
  private static readonly EMPTY_CODES = new Set(['error']);
  private static readonly MAX_CODES = 4;
  private static readonly NO_STATUS = 'the call failed without an HTTP status';
  private static readonly SENTENCES_BY_STATUS: { [statusCode: number]: string } = {
    400: 'the provider rejected the request as invalid',
    401: 'the provider did not accept the credentials',
    402: 'the provider reported a billing or payment problem',
    403: 'the provider refused permission for the call',
    404: 'the provider does not know what the request names (a model or a route)',
    408: 'the provider timed the request out',
    409: 'the provider reported a conflict',
    413: 'the request is larger than the provider accepts',
    422: 'the provider could not process the request as sent',
    429: 'the provider is limiting the rate of requests',
    500: 'the provider failed internally',
    502: 'the provider could not be reached through its gateway',
    503: 'the provider is unavailable',
    504: 'the provider did not answer in time',
    529: 'the provider is overloaded',
  };
  private static readonly SENTENCES_BY_NAME: { [name: string]: string } = {
    AI_NoObjectGeneratedError: 'the answer did not parse into the requested shape',
    AI_JSONParseError: 'the answer was not the JSON the client expected',
    AI_TypeValidationError: 'the answer did not match the shape the client expected',
    AI_InvalidResponseDataError: 'the provider answered with data the client could not read',
    AI_EmptyResponseBodyError: 'the provider answered with an empty body',
    AI_NoContentGeneratedError: 'the provider answered with no content',
    AI_InvalidPromptError: 'the client refused the prompt before sending it',
    AI_MessageConversionError: 'the client could not convert a message before sending it',
    AI_InvalidArgumentError: 'the client refused an argument of the call before sending it',
    AI_LoadAPIKeyError: 'a provider credential is missing',
    AI_LoadSettingError: 'a provider setting is missing',
    AI_NoSuchModelError: 'the client does not know the model',
    AI_UnsupportedFunctionalityError: 'the model does not support what the call asked for',
    AI_NoSuchToolError: 'the model called a tool the call did not offer',
    AI_InvalidToolInputError: 'the model called a tool with input that does not match its schema',
    AI_ToolCallRepairError: 'a malformed tool call could not be repaired',
  };

  /**
   * Marks `error` — when it is provider-worded (see the class comment) — as never printing its
   * own text, and answers the same error. The FIRST door an error leaves through knows it best:
   * an error already marked keeps its line. What a typed error wraps is marked with it. Never
   * throws; never touches the error.
   */
  static mark<T>(error: T, call: ProviderFailureCall = {}): T {
    if (call.wordedFrom === undefined && !ProviderFailureLine.isProviderWorded(error)) {
      return error;
    }
    // Only what the line needs is kept beside the error: a mark lives as long as the error does.
    const known: ProviderFailureCall = { modelId: call.modelId, provider: call.provider, wordedFrom: call.wordedFrom };
    if (!LogLineErrors.isMarked(error)) {
      LogLineErrors.mark(error, () =>
        ProviderLogPayloads.enabled() ? undefined : ProviderFailureLine.lineOf(error, known)
      );
    }
    const wrapped = known.wordedFrom ?? ProviderFailureLine.wrappedBy(error);
    if (wrapped !== undefined && wrapped !== error) {
      ProviderFailureLine.mark(wrapped, {
        modelId: known.modelId ?? ProviderFailureLine.stringOf(error, 'modelId'),
        provider: known.provider,
      });
    }
    return error;
  }

  private static isProviderWorded(error: unknown): boolean {
    if (typeof error !== 'object' || error === null) {
      return false;
    }
    return (
      !(error instanceof Error) ||
      AISDKError.isInstance(error) ||
      error instanceof OpenAiSdkError ||
      TransientProviderError.isInstance(error) ||
      ProviderBillingError.isInstance(error)
    );
  }

  /** The transport's error under one of the library's typed errors. */
  private static wrappedBy(error: unknown): unknown {
    return TransientProviderError.isInstance(error) || ProviderBillingError.isInstance(error) ? error.cause : undefined;
  }

  private static lineOf(error: unknown, call: ProviderFailureCall): ErrorLine {
    const source = call.wordedFrom ?? ProviderFailureLine.wrappedBy(error) ?? error;
    const statusCode = ProviderFailureLine.statusOf(error) ?? ProviderFailureLine.statusOf(source);
    const codes = ProviderFailureLine.codesOf(source);
    const modelId = call.modelId ?? ProviderFailureLine.stringOf(error, 'modelId');
    const provider = ProviderFailureLine.familyOf(call.provider);
    const name = ProviderFailureLine.stringOf(error, 'name') ?? ProviderFailureLine.stringOf(source, 'name');
    const named = [statusCode !== undefined ? `HTTP ${statusCode}` : name, codes[0]].filter(Boolean).join(', ');
    return {
      code: codes[0] ?? statusCode,
      sentence: `${ProviderFailureLine.whatFailed(error)}${named ? ` (${named})` : ''}${
        modelId ? ` on ${modelId}` : ''
      }: ${ProviderFailureLine.sentenceOf(statusCode, ProviderFailureLine.stringOf(source, 'name') ?? name)}`,
      facts: {
        ...(statusCode !== undefined ? { statusCode } : {}),
        ...(provider ? { provider } : {}),
        ...(modelId ? { modelId } : {}),
        ...(codes.length > 0 ? { providerErrorCodes: codes } : {}),
        ...ProviderFailureLine.retryableOf(source),
      },
    };
  }

  private static whatFailed(error: unknown): string {
    if (TransientProviderError.isInstance(error)) {
      return 'The model provider stayed unavailable past the retry budget';
    }
    if (ProviderBillingError.isInstance(error)) {
      return 'The model provider reported a billing or credit failure';
    }
    return 'The model call failed';
  }

  private static sentenceOf(statusCode: number | undefined, name: string | undefined): string {
    if (statusCode === undefined) {
      return (name && ProviderFailureLine.SENTENCES_BY_NAME[name]) || ProviderFailureLine.NO_STATUS;
    }
    const known = ProviderFailureLine.SENTENCES_BY_STATUS[statusCode];
    if (known) {
      return known;
    }
    return statusCode >= 500 ? 'the provider failed internally' : 'the provider rejected the request';
  }

  /** `statusCode` (the AI SDK's, the typed errors') or `status` (the OpenAI SDK's). */
  private static statusOf(error: unknown): number | undefined {
    const record = ProviderFailureLine.recordOf(error);
    return [record?.statusCode, record?.status].find((each): each is number => typeof each === 'number');
  }

  /** The vendor's own codes, most specific first: identifiers only, a few at most. */
  private static codesOf(source: unknown): string[] {
    try {
      const codes = providerErrorCodes(source).filter(
        (code) => ProviderFailureLine.CODE_SHAPE.test(code) && !ProviderFailureLine.EMPTY_CODES.has(code.toLowerCase())
      );
      return codes.filter((code, index) => codes.indexOf(code) === index).slice(0, ProviderFailureLine.MAX_CODES);
    } catch {
      return [];
    }
  }

  private static retryableOf(source: unknown): { isRetryable?: boolean } {
    const isRetryable = ProviderFailureLine.recordOf(source)?.isRetryable;
    return typeof isRetryable === 'boolean' ? { isRetryable } : {};
  }

  /** `anthropic.messages` → `anthropic`. */
  private static familyOf(provider: string | undefined): string | undefined {
    const family = (provider ?? '').split('.')[0].trim().toLowerCase();
    return ProviderFailureLine.CODE_SHAPE.test(family) ? family : undefined;
  }

  private static stringOf(error: unknown, member: 'name' | 'modelId'): string | undefined {
    const value = ProviderFailureLine.recordOf(error)?.[member];
    return typeof value === 'string' && value ? value : undefined;
  }

  private static recordOf(error: unknown): { [member: string]: unknown } | undefined {
    return typeof error === 'object' && error !== null ? (error as { [member: string]: unknown }) : undefined;
  }
}
