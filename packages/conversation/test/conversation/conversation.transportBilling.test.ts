import { APICallError } from 'ai';
import { MockLanguageModelV3, convertArrayToReadableStream } from 'ai/test';
import { Conversation } from '../../src/Conversation';
import { LlmTransportRetry } from '../../src/LlmTransportRetry';
import { TransientProviderError } from '../../src/TransientProviderError';
import { ProviderBillingError, classifyProviderBillingError } from '../../src/ProviderBillingError';

/**
 * D1 (plans/FLOW_RESILIENCE.md §9.2): the BILLING class at the transport choke point. Every
 * shape below was verified against the LIVE vendor docs on 2026-08-31 (see the detection table
 * in ProviderBillingError.ts). The load-bearing property: billing failures surface IMMEDIATELY
 * as `ProviderBillingError` — never retried (the 429-riding shapes were previously mis-binned
 * as rate limits and burned the whole retry budget against a dead wallet), and never conflated
 * with `TransientProviderError` (outer layers route the two differently).
 *
 * No network, no API keys — APICallError fixtures carry the providers' real documented bodies.
 */

const TIMEOUT = 30_000;

const isRetryableSdk = (e: unknown) => APICallError.isInstance(e) && e.isRetryable === true;

/** Run `fn` expecting a ProviderBillingError of the given type; returns it for further assertions. */
const expectBillingRejection = async (fn: () => Promise<unknown>, providerErrorType: string) => {
  let surfaced: unknown;
  try {
    await fn();
  } catch (error) {
    surfaced = error;
  }
  expect(ProviderBillingError.isInstance(surfaced)).toBe(true);
  expect((surfaced as ProviderBillingError).providerErrorType).toBe(providerErrorType);
  return surfaced as ProviderBillingError;
};

const apiError = (args: { message: string; statusCode: number; responseBody: string }) =>
  new APICallError({
    message: args.message,
    url: 'https://provider.test',
    requestBodyValues: {},
    statusCode: args.statusCode,
    responseHeaders: {},
    responseBody: args.responseBody,
  });

/** OpenAI credit exhaustion — 429 with `insufficient_quota`/`credit_balance_exhausted` (live docs 2026-08-31). */
const openAiCreditExhausted = () =>
  apiError({
    message: 'Your organization has no prepaid credits remaining.',
    statusCode: 429,
    responseBody: JSON.stringify({
      error: {
        message: 'Your organization has no prepaid credits remaining.',
        type: 'insufficient_quota',
        param: null,
        code: 'credit_balance_exhausted',
      },
    }),
  });

/** OpenAI legacy shape — 429 whose only marker is error.type `insufficient_quota`. */
const openAiInsufficientQuota = () =>
  apiError({
    message: 'You exceeded your current quota, please check your plan and billing details.',
    statusCode: 429,
    responseBody: JSON.stringify({
      error: {
        message: 'You exceeded your current quota, please check your plan and billing details.',
        type: 'insufficient_quota',
        param: null,
        code: null,
      },
    }),
  });

/** Anthropic billing/payment problem — HTTP 402 `billing_error` (live docs 2026-08-31; was 403 in older lore). */
const anthropicBillingError = () =>
  apiError({
    message: 'There is an issue with your billing or payment information.',
    statusCode: 402,
    responseBody: JSON.stringify({
      type: 'error',
      error: { type: 'billing_error', message: 'There is an issue with your billing or payment information.' },
    }),
  });

/** Anthropic credit exhaustion — 400 `invalid_request_error`, message-sniffed (fragile-by-necessity). */
const anthropicCreditBalance = () =>
  apiError({
    message: 'Your credit balance is too low to access the Anthropic API.',
    statusCode: 400,
    responseBody: JSON.stringify({
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.',
      },
    }),
  });

/**
 * Anthropic tier spend cap — 429 whose error.type is `rate_limit_error` (the mis-bin trap); the
 * documented discriminator is `error.details.error_code: enforced_spend_limit_reached`.
 */
const anthropicSpendCap = () =>
  apiError({
    message: 'You have reached your API usage limits: your organization has crossed its monthly API usage threshold.',
    statusCode: 429,
    responseBody: JSON.stringify({
      type: 'error',
      error: {
        type: 'rate_limit_error',
        message:
          'You have reached your API usage limits: your organization has crossed its monthly API usage threshold. You will regain access on 2026-09-01 at 00:00 UTC.',
        details: { error_code: 'enforced_spend_limit_reached' },
      },
    }),
  });

/** Google daily/free-tier quota exhaustion — 429 RESOURCE_EXHAUSTED WITH QuotaFailure PerDay details. */
const googleDailyQuota = () =>
  apiError({
    message: 'Resource has been exhausted (e.g. check quota).',
    statusCode: 429,
    responseBody: JSON.stringify({
      error: {
        code: 429,
        message: 'Resource has been exhausted (e.g. check quota).',
        status: 'RESOURCE_EXHAUSTED',
        details: [
          {
            '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
            violations: [
              {
                quotaMetric: 'generativelanguage.googleapis.com/generate_content_free_tier_requests',
                quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
              },
            ],
          },
        ],
      },
    }),
  });

/** Google ORDINARY rate limit — the SAME ambiguous 429 RESOURCE_EXHAUSTED, no quota details. */
const googleRateLimit = () =>
  apiError({
    message: 'Resource has been exhausted (e.g. check quota).',
    statusCode: 429,
    responseBody: JSON.stringify({
      error: { code: 429, message: 'Resource has been exhausted (e.g. check quota).', status: 'RESOURCE_EXHAUSTED' },
    }),
  });

/** A plain rate limit (Anthropic shape, no billing discriminator) — must KEEP retrying. */
const plainRateLimit = () =>
  apiError({
    message: 'Rate limited',
    statusCode: 429,
    responseBody: JSON.stringify({
      type: 'error',
      error: { type: 'rate_limit_error', message: 'Number of requests has exceeded your rate limit.' },
    }),
  });

describe('classifyProviderBillingError — the detection table', () => {
  test('every verified billing row classifies, with its providerErrorType', () => {
    expect(classifyProviderBillingError(openAiCreditExhausted())).toBe('credit_balance_exhausted');
    expect(classifyProviderBillingError(openAiInsufficientQuota())).toBe('insufficient_quota');
    expect(classifyProviderBillingError(anthropicBillingError())).toBe('billing_error');
    expect(classifyProviderBillingError(anthropicCreditBalance())).toBe('credit_balance_too_low');
    expect(classifyProviderBillingError(anthropicSpendCap())).toBe('enforced_spend_limit_reached');
    expect(classifyProviderBillingError(googleDailyQuota())).toBe('quota_exhausted_daily');
    // A bare 402 with NO recognizable body — Payment Required is categorically billing.
    expect(
      classifyProviderBillingError(apiError({ message: 'Payment Required', statusCode: 402, responseBody: '' }))
    ).toBe('billing_error');
  });

  test('non-billing shapes stay unclassified — a bare rate limit is NEVER billing', () => {
    expect(classifyProviderBillingError(plainRateLimit())).toBeUndefined();
    expect(classifyProviderBillingError(googleRateLimit())).toBeUndefined();
    // Merely billing-ish prose without a named row stays terminal-semantic (conservative default).
    expect(
      classifyProviderBillingError(
        apiError({ message: 'please check your billing details with support', statusCode: 400, responseBody: '' })
      )
    ).toBeUndefined();
    expect(classifyProviderBillingError(new Error('billing is weird'))).toBeUndefined();
    expect(classifyProviderBillingError(undefined)).toBeUndefined();
  });

  test('mid-stream raw payloads (no APICallError, no HTTP status) classify by their code strings', () => {
    // OpenAI nests under `error`; Anthropic sends flat — both walk.
    expect(classifyProviderBillingError({ error: { type: 'insufficient_quota', code: 'insufficient_quota' } })).toBe(
      'insufficient_quota'
    );
    expect(classifyProviderBillingError({ type: 'billing_error', message: 'billing problem' })).toBe('billing_error');
    expect(classifyProviderBillingError({ type: 'rate_limit_error', message: 'slow down' })).toBeUndefined();
  });
});

describe('LlmTransportRetry — billing failures are never retried and surface typed', () => {
  test(
    'a 429-riding billing error surfaces as ProviderBillingError on the FIRST attempt (the mis-bin fix)',
    async () => {
      // Pre-fix this exact shape was classified retryable by status (429) and RETRIED — a second
      // call would have "succeeded", proving the burn; the budget path then wrapped it
      // TransientProviderError ("they're rate-limiting requests") — the standing lie.
      let calls = 0;
      const retry = new LlmTransportRetry({ budgetMs: 5_000 });
      let surfaced: unknown;
      try {
        await retry.run(
          () => {
            calls++;
            if (calls === 1) {
              throw openAiCreditExhausted();
            }
            return 'should never be reached by a retry';
          },
          { isRetryable: isRetryableSdk, modelId: 'gpt-5' }
        );
      } catch (error) {
        surfaced = error;
      }
      expect(calls).toBe(1); // never retried — no burn against a dead wallet
      expect(ProviderBillingError.isInstance(surfaced)).toBe(true);
      expect(TransientProviderError.isInstance(surfaced)).toBe(false);
      const billing = surfaced as ProviderBillingError;
      expect(billing.providerErrorType).toBe('credit_balance_exhausted');
      expect(billing.statusCode).toBe(429);
      expect(billing.modelId).toBe('gpt-5');
      expect(billing.cause).toBeInstanceOf(APICallError);
    },
    TIMEOUT
  );

  test(
    "Anthropic's spend-cap 429 (rate_limit_error + enforced_spend_limit_reached) surfaces billing, not a retry loop",
    async () => {
      let calls = 0;
      const retry = new LlmTransportRetry({ budgetMs: 5_000 });
      await expectBillingRejection(
        () =>
          retry.run(
            () => {
              calls++;
              throw anthropicSpendCap();
            },
            { isRetryable: isRetryableSdk, modelId: 'claude-opus-5' }
          ),
        'enforced_spend_limit_reached'
      );
      expect(calls).toBe(1);
    },
    TIMEOUT
  );

  test(
    'the Anthropic 400 credit-balance message surfaces ProviderBillingError (was a bare semantic error)',
    async () => {
      const retry = new LlmTransportRetry({ budgetMs: 5_000 });
      let surfaced: unknown;
      try {
        await retry.run(
          () => {
            throw anthropicCreditBalance();
          },
          { isRetryable: isRetryableSdk }
        );
      } catch (error) {
        surfaced = error;
      }
      expect(ProviderBillingError.isInstance(surfaced)).toBe(true);
      expect((surfaced as ProviderBillingError).providerErrorType).toBe('credit_balance_too_low');
    },
    TIMEOUT
  );

  test(
    'a 402 billing_error surfaces billing; a PLAIN 429 rate limit still retries and recovers (no regression)',
    async () => {
      const retry = new LlmTransportRetry({ budgetMs: 10_000 });
      await expectBillingRejection(
        () =>
          retry.run(
            () => {
              throw anthropicBillingError();
            },
            { isRetryable: isRetryableSdk }
          ),
        'billing_error'
      );

      let calls = 0;
      const value = await retry.run(
        () => {
          calls++;
          if (calls === 1) {
            throw plainRateLimit();
          }
          return 'recovered';
        },
        { isRetryable: isRetryableSdk }
      );
      expect(value).toBe('recovered');
      expect(calls).toBe(2);
    },
    TIMEOUT
  );

  test(
    "Google's ambiguous 429: bare RESOURCE_EXHAUSTED retries (rate limit); PerDay QuotaFailure surfaces billing",
    async () => {
      const retry = new LlmTransportRetry({ budgetMs: 10_000 });
      let calls = 0;
      const value = await retry.run(
        () => {
          calls++;
          if (calls === 1) {
            throw googleRateLimit();
          }
          return 'recovered';
        },
        { isRetryable: isRetryableSdk }
      );
      expect(value).toBe('recovered');

      await expectBillingRejection(
        () =>
          retry.run(
            () => {
              throw googleDailyQuota();
            },
            { isRetryable: isRetryableSdk }
          ),
        'quota_exhausted_daily'
      );
    },
    TIMEOUT
  );
});

describe('Conversation wiring — billing surfaces through the real middleware, generate and stream', () => {
  const newConversation = () =>
    new Conversation({ name: 'transport-billing-test', logLevel: 'error', limits: { enforceLimits: false } });

  const usage = {
    inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 1, text: 1, reasoning: 0 },
  };

  test(
    'generateObject: a billing failure surfaces typed on attempt 1 through Conversation → wrap → model',
    async () => {
      let calls = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          calls++;
          throw openAiInsufficientQuota();
        },
      });
      let surfaced: unknown;
      try {
        await newConversation().generateObject<{ answer: string }>({
          messages: ['give me the answer'],
          model: model as never,
          schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
        });
      } catch (error) {
        surfaced = error;
      }
      expect(calls).toBe(1);
      expect(ProviderBillingError.isInstance(surfaced)).toBe(true);
    },
    TIMEOUT
  );

  test(
    'stream: a mid-stream billing error part surfaces wrapped on the part, with no replay of the stream',
    async () => {
      let streams = 0;
      const model = new MockLanguageModelV3({
        doStream: async () => {
          streams++;
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start' as const, warnings: [] },
              {
                type: 'error' as const,
                error: {
                  type: 'error',
                  error: { type: 'insufficient_quota', code: 'insufficient_quota', message: 'quota exhausted' },
                },
              },
            ]),
          };
        },
      });
      // The raw wrapped-model seam (the transport contract itself — the existing mid-stream
      // transient test's idiom): the error PART carries the typed billing error.
      const wrapped = new LlmTransportRetry({ budgetMs: 5_000 }).wrap(model as never);
      const { stream } = await wrapped.doStream({ prompt: [] } as never);
      const reader = stream.getReader();
      const parts: Array<{ type: string; error?: unknown }> = [];
      for (;;) {
        const result = await reader.read();
        if (result.done) {
          break;
        }
        parts.push(result.value as { type: string; error?: unknown });
      }
      expect(streams).toBe(1); // never re-initiated — a dead wallet is not replayable weather
      const errorPart = parts.find((p) => p.type === 'error');
      expect(errorPart).toBeTruthy();
      expect(ProviderBillingError.isInstance(errorPart!.error)).toBe(true);
      expect((errorPart!.error as ProviderBillingError).providerErrorType).toBe('insufficient_quota');
    },
    TIMEOUT
  );
});
