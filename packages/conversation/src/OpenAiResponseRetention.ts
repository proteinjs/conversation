import { OpenAiModelRules } from './OpenAiModelRules';

/**
 * THE ONE RULE for what OpenAI keeps of a Responses request the library makes: nothing. Every
 * request — the streaming path (`Conversation`, through `@ai-sdk/openai`'s responses model) and
 * the buffered adapter (`OpenAiResponses`) — spreads {@link stateless} into its request, so it
 * carries `store: false` and OpenAI retains no response past the request (the create reference's
 * `store`: "Whether to store the generated model response for later retrieval via API. Defaults
 * to true when omitted. If set to true, response data will be stored for at least 30 days").
 *
 * A stateless request has no server-side response to chain to (`previous_response_id` refers
 * to a stored one), so a multi-step tool loop carries its own transcript: the reasoning items a
 * step returns come back with `encrypted_content` — "populated by default for reasoning items
 * returned by `POST /v1/responses`" in stateless mode — and are replayed verbatim in the next
 * request's input beside the function calls and their outputs (the reasoning guide, "Preserve
 * reasoning without stored responses"). On a reasoning model the encrypted reasoning is also
 * asked for by name (`include: ["reasoning.encrypted_content"]`, "accepted for compatibility"),
 * so the contract is stated on the request rather than inherited from a server default; a
 * non-reasoning model has no reasoning items to keep and asks for nothing.
 *
 * Reference: https://developers.openai.com/api/reference/resources/responses/methods/create
 * (`store`, `include`, the reasoning item's `encrypted_content`) and
 * https://developers.openai.com/api/docs/guides/reasoning ("Preserve reasoning without stored
 * responses").
 */
export class OpenAiResponseRetention {
  /**
   * The request fields that make a request to `modelId` stateless. The names are the same on
   * both paths (the SDK body and the `@ai-sdk/openai` provider options): `store`, `include`.
   */
  static stateless(modelId: string): { store: false; include?: ['reasoning.encrypted_content'] } {
    return OpenAiModelRules.reasons(modelId)
      ? { store: false, include: ['reasoning.encrypted_content'] }
      : { store: false };
  }
}
