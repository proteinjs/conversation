/**
 * Cross-package instance marker (the TransientProviderError/APICallError pattern): `Symbol.for`
 * resolves to the same symbol in every copy of this module, so `isInstance` holds even when a
 * consumer's dependency tree carries a different build of `@proteinjs/conversation`.
 */
const MARKER = Symbol.for('@proteinjs/conversation.ProviderBillingError');

/**
 * A PROVIDER-side BILLING/CREDIT failure: the account behind the call is out of credits, past a
 * spend limit, or has a payment problem. Typed at the ONE transport choke point
 * (`LlmTransportRetry` — detection runs BEFORE the retryable check, because two of the shapes
 * ride HTTP 429 and would otherwise be mis-binned as rate limits and retried forever against a
 * dead wallet). Never retried by the transport — nothing about a billing state heals inside a
 * 90-second retry budget — and never conflated with `TransientProviderError`: outer layers route
 * the two differently (outage → timed park ladder; billing → immediate owner-routed ask, no
 * timer — plans/FLOW_RESILIENCE.md §9.2 D1/D2).
 *
 * The detection table lives in {@link classifyProviderBillingError}. Conservative default: an
 * unrecognized billing-ish shape is NOT classified billing — it stays on the semantic/terminal
 * path (visible, never silently parked). Detection never demands exact schemas (no
 * strictification): every walk is optional-field sniffing over whatever the provider sent.
 */
export class ProviderBillingError extends Error {
  /** HTTP status of the underlying failure when known (400, 402, 429, …). */
  readonly statusCode?: number;
  /** The matched detection-table row (e.g. 'insufficient_quota', 'billing_error') — for records/routing. */
  readonly providerErrorType: string;
  /** The original error exactly as the transport surfaced it. */
  readonly cause: unknown;
  /** The failing model's id when the transport knew it. */
  readonly modelId?: string;

  constructor(args: {
    message: string;
    cause: unknown;
    providerErrorType: string;
    statusCode?: number;
    modelId?: string;
  }) {
    super(args.message);
    this.name = 'ProviderBillingError';
    this.cause = args.cause;
    this.providerErrorType = args.providerErrorType;
    this.statusCode = args.statusCode;
    this.modelId = args.modelId;
    Object.defineProperty(this, MARKER, { value: true, enumerable: false });
  }

  static isInstance(error: unknown): error is ProviderBillingError {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[MARKER] === true;
  }

  /** Wrap the transport's original error (idempotent), preserving its message and HTTP status. */
  static wrap(error: unknown, providerErrorType: string, modelId?: string): ProviderBillingError {
    if (ProviderBillingError.isInstance(error)) {
      return error;
    }
    const message =
      error instanceof Error ? error.message : typeof error === 'string' ? error : JSON.stringify(error ?? null);
    const statusCode = (error as { statusCode?: unknown } | null | undefined)?.statusCode;
    return new ProviderBillingError({
      message: `Model provider billing/credit failure (${providerErrorType}): ${message}`,
      cause: error,
      providerErrorType,
      ...(typeof statusCode === 'number' ? { statusCode } : {}),
      ...(modelId ? { modelId } : {}),
    });
  }
}

/**
 * Error type/code strings a provider uses ONLY for billing/credit/spend-limit states — a hit on
 * any collected code/type string is a billing verdict regardless of HTTP status. Verified against
 * the LIVE vendor docs on 2026-08-31 (the AssemblyAI rule generalized — never memorized shapes;
 * re-verify on vendor bumps):
 *
 * Anthropic (platform.claude.com/docs/en/api/errors + /api/rate-limits):
 * - `billing_error` — HTTP 402: "an issue with your billing or payment information".
 * - `enforced_spend_limit_reached` — the tier monthly spend cap. Rides HTTP 429 with error.type
 *   `rate_limit_error` (NO retry-after header) and names itself only in
 *   `error.details.error_code` — the documented way to tell it from a real rate limit. This is
 *   Anthropic's own 429 mis-bin case, the mirror of OpenAI's insufficient_quota.
 *
 * OpenAI (developers.openai.com/api/docs/guides/error-codes):
 * - `insufficient_quota` — the broader billing error.type; still documented, rides 429.
 * - `credit_balance_exhausted` — 429 error.code: "no prepaid credits remaining".
 * - `organization_spend_limit_exceeded` / `project_spend_limit_exceeded` — 429 error.code:
 *   enforced spend limits.
 * - `organization_usage_limit_exceeded` — 429 error.code: the OpenAI-assigned usage limit.
 * - `billing_hard_limit_reached` — legacy billing code, still in the wild; kept deliberately.
 */
const BILLING_ERROR_CODES: ReadonlySet<string> = new Set([
  'billing_error',
  'enforced_spend_limit_reached',
  'insufficient_quota',
  'credit_balance_exhausted',
  'organization_spend_limit_exceeded',
  'project_spend_limit_exceeded',
  'organization_usage_limit_exceeded',
  'billing_hard_limit_reached',
]);

/**
 * Anthropic message sniffs — named fragile-by-necessity (Anthropic does not type these two as
 * distinct error codes; both ride 400 `invalid_request_error`, verified live 2026-08-31):
 * - the classic credit exhaustion: message contains "credit balance is too low";
 * - a SELF-SET spend limit: message begins "You have reached your specified API usage limits"
 *   (or "…specified workspace API usage limits") — documented verbatim prefixes.
 * Each row maps the sniff to the providerErrorType it reports.
 */
const MESSAGE_SNIFFS: ReadonlyArray<{ test: (text: string) => boolean; type: string }> = [
  { test: (t) => t.includes('credit balance is too low'), type: 'credit_balance_too_low' },
  {
    test: (t) => t.includes('you have reached your specified api usage limits'),
    type: 'specified_spend_limit_reached',
  },
  {
    test: (t) => t.includes('you have reached your specified workspace api usage limits'),
    type: 'specified_spend_limit_reached',
  },
];

/**
 * THE billing detection table (plans/FLOW_RESILIENCE.md §9.2 D1) — returns the matched
 * providerErrorType, or undefined for everything else. One classifier for both transport paths
 * (initiation `APICallError`s and raw mid-stream error payloads).
 *
 * Rows, in match order:
 * 1. HTTP 402 (Payment Required) — categorically billing for any provider (Anthropic types it
 *    `billing_error`).
 * 2. Any collected code/type string in {@link BILLING_ERROR_CODES} — the typed rows for
 *    Anthropic + OpenAI above. Strings are collected by an optional-field walk over the error
 *    object, its parsed `data`/`responseBody`, and nested `error`/`details` shapes — never a
 *    schema demand.
 * 3. The Anthropic message sniffs ({@link MESSAGE_SNIFFS}).
 * 4. Google (ai.google.dev — verified 2026-08-31: quota exhaustion and ordinary rate limits
 *    share ONE ambiguous shape, 429 `RESOURCE_EXHAUSTED`): billing ONLY when QuotaFailure
 *    details name a per-day or free-tier quota (a state the transport budget cannot outwait);
 *    a bare RESOURCE_EXHAUSTED stays transient — the rate-limit ladder is the status quo and
 *    mis-flagging a real rate limit as billing would fire a false ops alert.
 *
 * Everything else — including shapes that merely SMELL billing-ish — returns undefined and stays
 * on the semantic/terminal path: visible, never silently parked (the conservative default).
 */
export function classifyProviderBillingError(error: unknown): string | undefined {
  if (typeof error !== 'object' || error === null) {
    return undefined;
  }
  const statusCode = (error as { statusCode?: unknown }).statusCode;
  if (statusCode === 402) {
    return 'billing_error';
  }
  const payloads = collectPayloads(error);
  const codes = payloads.flatMap((p) => collectCodeStrings(p, 0));
  const billingCode = codes.find((code) => BILLING_ERROR_CODES.has(code.toLowerCase()));
  if (billingCode) {
    return billingCode.toLowerCase();
  }
  const text = payloads
    .flatMap((p) => collectMessageStrings(p, 0))
    .join(' \n ')
    .toLowerCase();
  const sniff = MESSAGE_SNIFFS.find((row) => row.test(text));
  if (sniff) {
    return sniff.type;
  }
  // Google row: RESOURCE_EXHAUSTED + a per-day/free-tier quota id in the QuotaFailure details.
  if (codes.some((code) => code.toUpperCase() === 'RESOURCE_EXHAUSTED')) {
    const quotaIds = payloads.flatMap((p) => collectQuotaStrings(p, 0)).join(' ');
    if (/perday|free[_-]?tier|freetier/i.test(quotaIds)) {
      return 'quota_exhausted_daily';
    }
  }
  return undefined;
}

// ─── optional-field walks (module-private; bounded depth, never throw) ───────────────────────

/** The error object plus every parsed body attached to it (`data`, JSON `responseBody`), and message-embedded JSON. */
function collectPayloads(error: object): object[] {
  const payloads: object[] = [error];
  const data = (error as { data?: unknown }).data;
  if (typeof data === 'object' && data !== null) {
    payloads.push(data);
  }
  const responseBody = (error as { responseBody?: unknown }).responseBody;
  if (typeof responseBody === 'string' && responseBody.trim().startsWith('{')) {
    try {
      const parsed = JSON.parse(responseBody);
      if (typeof parsed === 'object' && parsed !== null) {
        payloads.push(parsed);
      }
    } catch {
      // not JSON — the message walk still sees the raw string via the error's own message
    }
  }
  return payloads;
}

/** `type`/`code`/`error_code`/`status`/`reason` strings, flat and nested under `error`/`details`. */
function collectCodeStrings(value: unknown, depth: number): string[] {
  if (typeof value !== 'object' || value === null || depth > 4) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => collectCodeStrings(item, depth + 1));
  }
  const record = value as Record<string, unknown>;
  // Specificity order: `error_code`/`code` name the exact condition, `type` is the broad class
  // (OpenAI's credit body carries type `insufficient_quota` AND code `credit_balance_exhausted`
  // — the reported row should be the specific one), `status`/`reason` are the gRPC-ish fields.
  const own = [record.error_code, record.errorCode, record.code, record.type, record.status, record.reason].filter(
    (v): v is string => typeof v === 'string'
  );
  return [...own, ...collectCodeStrings(record.error, depth + 1), ...collectCodeStrings(record.details, depth + 1)];
}

/** `message` strings plus a raw string `responseBody`, flat and nested under `error`. */
function collectMessageStrings(value: unknown, depth: number): string[] {
  if (typeof value !== 'object' || value === null || depth > 4) {
    return [];
  }
  const record = value as Record<string, unknown>;
  const own = [record.message, record.responseBody].filter((v): v is string => typeof v === 'string');
  return [...own, ...collectMessageStrings(record.error, depth + 1)];
}

/** Quota identifiers from Google `QuotaFailure` details: `quotaId`/`quotaMetric` (+ violation walks). */
function collectQuotaStrings(value: unknown, depth: number): string[] {
  if (typeof value !== 'object' || value === null || depth > 5) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => collectQuotaStrings(item, depth + 1));
  }
  const record = value as Record<string, unknown>;
  const own = [record.quotaId, record.quotaMetric].filter((v): v is string => typeof v === 'string');
  return [
    ...own,
    ...collectQuotaStrings(record.error, depth + 1),
    ...collectQuotaStrings(record.details, depth + 1),
    ...collectQuotaStrings(record.violations, depth + 1),
  ];
}
