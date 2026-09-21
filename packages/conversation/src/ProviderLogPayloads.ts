/**
 * The ONE owner of the dev-only switch that lets a provider's error print AS IT IS on a log
 * line — with the request body, the response body and the response headers the client library
 * keeps on it.
 *
 * By default no line carries them (ProviderFailureLine): a request body is the conversation
 * itself, and a log line outlives and out-travels the conversation it came from. On a developer's
 * machine the payload is what locates a bug. So the switch: when BOTH `DEVELOPMENT` is set (the
 * dev-server switch, never set in a production image) AND `CONVERSATION_LOG_PROVIDER_PAYLOADS=1`,
 * a provider error prints whole. With either unset: never. Both are read at each line, so a
 * process cannot hold a stale answer.
 *
 * The switch changes what is PRINTED and nothing else: what the library throws is the same with
 * the switch on or off.
 */
export class ProviderLogPayloads {
  /** The dev-server switch: the first gate. */
  static readonly DEVELOPMENT_VAR = 'DEVELOPMENT';
  /** The payload switch: the second gate, on only as exactly `1`. */
  static readonly SWITCH_VAR = 'CONVERSATION_LOG_PROVIDER_PAYLOADS';

  /** Whether a provider error may print whole right now — both gates, read now. */
  static enabled(): boolean {
    const env = typeof process !== 'undefined' ? process.env : undefined;
    return !!env?.[ProviderLogPayloads.DEVELOPMENT_VAR] && env?.[ProviderLogPayloads.SWITCH_VAR] === '1';
  }
}
