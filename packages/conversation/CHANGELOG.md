# Change Log

All notable changes to this project will be documented in this file.
See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [6.2.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@6.2.0...@proteinjs/conversation@6.2.1) (2026-09-04)


### Bug Fixes

* **conversation:** mid-turn notes wait while the assistant turn is open on a server tool — the drain is not consumed at a step boundary whose last assistant message carries a provider-executed call with no result (R7 finding 9: the follow-up that killed the research turn) ([197d410](https://github.com/proteinjs/conversation/commit/197d41064e6a01b6c6d9a2954750dba5d0948de7))





# [6.2.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@6.1.0...@proteinjs/conversation@6.2.0) (2026-09-04)


### Features

* **conversation): the tool-call budget — a call still running past N converts IN PLACE into a background job at the executor (plans/FREE_AGENT.md §M.3 part 1). ToolBudget owns the one clock (CONVERSATION_TOOL_SOFT_BUDGET_MS, default 5 s; HARD default 30 min passed to the host): GenerateStreamParams.toolBudget is the host hook — the §2.8 running-job lookup before a call runs (a repeat reads "already running as job {id}", never executes twice; dedupe: false opts out, dedupeKey(args) names the identity, the arguments' stable hash is the default), then f.call(args, { signal, onPhase }) raced against N (background: true → t = 0); past N convert() receives the running promise, its abort, the phase relay, the dedupe key, the HARD ceiling and the job's visible title, and the model reads the §2.4 hand-off sentence as the call's result so the loop reaches its next step boundary — where the next input drains — instead of waiting behind the tool. The promise runs on. ToolInvocationResult.converted and the tool-settled part's converted { jobId, title } mark the hand-off for the timeline. The job's NAME is a task in plain English (founder ruling 2026-09-04 22:00Z:** every budgeted tool's schema is offered a `task` label the executor splits off before the tool runs; the title is the model's label, else the tool's last phase words, else "Working on {subject}" / "Working in the background" — never the tool's name (pinned). An unbudgeted host is byte-identical to before. Function gains call(args, ctx?), background, hardBudgetMs, dedupe, dedupeKey. Tests: toolBudget.test.ts (14) — red at `await f.call(args)` (6 of 12; the boundary at 6 s), bites: the race removed, the lookup skipped, background ignored. ([6c6dfdd](https://github.com/proteinjs/conversation/commit/6c6dfdddba54545bf92a576162e146e6d2531762))





# [6.1.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@6.0.2...@proteinjs/conversation@6.1.0) (2026-09-03)


### Features

* **conversation:** per-step usage on UsageData — `steps` lists every loop step's provider-reported usage (input, cache read/write, reasoning, output, tool calls) beside the summed totals, mapped from the SDK's StepResult.usage in mapSdkUsage and concatenated by aggregateUsageData; the partial-usage placeholders (no per-step usage) produce no list, never fabricated zero rows. The seam a downstream ledger needs to split the FIRST request (cross-turn prompt-cache read) from the later re-reads of the same prefix — the summed totals cannot. ([b4854a7](https://github.com/proteinjs/conversation/commit/b4854a7da322891d2fda0466520a614db0bf81da))





## [6.0.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@6.0.1...@proteinjs/conversation@6.0.2) (2026-09-02)


### Bug Fixes

* **conversation:** stream liveness guard no longer counts local tool execution as model silence ([021e46c](https://github.com/proteinjs/conversation/commit/021e46c2cbfb709779d30b2772af7ec6fcdaf08a))





## [6.0.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@6.0.0...@proteinjs/conversation@6.0.1) (2026-09-02)


### Bug Fixes

* abort-tagged gave-up — the stop verdict rides the transport retry activity (prod 2026-09-02, the unstoppable-spinner defect's transport leg). An abort that lands mid-retry (pre-verdict, or during the backoff sleep — the announced attempt never runs) now reports gave-up with aborted: true, so wait-rendering consumers settle the provider wait with STOP words instead of the bogus provider-outage verdict ('didn't recover' for an end the user chose); a budget-exhausted gave-up stays untagged (the counter-pin keeps the discriminator honest). Retry semantics unchanged — the abort exits, budgets, and typed exhaustion are exactly as before; this is the observability contract growing one field. New suite conversation.transportRetryAbort.test.ts pins the one-act-stop contract at this layer across all four legs: stream initiation (thrown 529), mid-stream error part, the tool-loop continuation call, and run()/doGenerate — no attempt after the abort, prompt stream settle, abort-tagged gave-up. RED stated: the suite fails at the pre-fix contract (the aborted field does not exist — compile-red at every tag pin). Bites verified red then restored: abort wiring dropped from the verdict path (options.abortSignal nulled) reddened all four abort pins (retries continued past the stop; streams timed out); the aborted tag dropped from the post-sleep emit reddened the four tag pins. Transport estate green: transportRetry + transportRetryActivity + transportBilling + the new suite, 35 tests; full test/conversation folder 158 passed / 35 key-gated skips; tsc clean; eslint + prettier clean on touched files. ([6412ce3](https://github.com/proteinjs/conversation/commit/6412ce31f1bf138caccc1d0321638f5823778ccb))





# [6.0.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.23.0...@proteinjs/conversation@6.0.0) (2026-09-02)


* feat!: pricing tables leave the transport — ConversationParams requires a ModelDataResolver (model-catalog-as-data step 1) ([8d82fa9](https://github.com/proteinjs/conversation/commit/8d82fa9d2964a0a50470e9ef1fda83759a6082fd))


### BREAKING CHANGES

* ConversationParams (and OpenAiResponsesParams,
OpenAiParams, CodegenConversation's constructor,
UsageDataAccumulatorParams) now require modelData: ModelDataResolver,
and the pricing tables + isModelPriced are no longer exported — pass the
platform's resolver (n3xa: modelDataResolver from @n3xah/chat-common) or
your own.





# [5.23.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.22.0...@proteinjs/conversation@5.23.0) (2026-09-01)


### Features

* price claude-sonnet-5 — $2/$10, the launch intro pricing made permanent ([221cf6c](https://github.com/proteinjs/conversation/commit/221cf6cb782642b9e614c9c278c0bcbf27ba0d04))





# [5.22.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.21.0...@proteinjs/conversation@5.22.0) (2026-09-01)


### Features

* claude-fable-5-1 support — pricing + the forced-tool-choice gate ([fff361d](https://github.com/proteinjs/conversation/commit/fff361d3a8e61625a2d8147df6f4512fa88ab6fe))





# [5.21.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.20.0...@proteinjs/conversation@5.21.0) (2026-09-01)


### Features

* ProviderBillingError — the billing/credit class typed at the transport choke point (FLOW_RESILIENCE wave D, D1). Detection runs BEFORE the retryable check, because the worst shapes ride HTTP 429 and were mis-binned as rate limits and retried against a dead wallet: OpenAI's insufficient_quota family (error.code credit_balance_exhausted / organization_spend_limit_exceeded / project_spend_limit_exceeded / organization_usage_limit_exceeded, legacy billing_hard_limit_reached kept) and Anthropic's tier spend cap (rate_limit_error whose only discriminator is error.details.error_code enforced_spend_limit_reached, no retry-after). Every detection row was verified against the LIVE vendor docs at build (2026-08-31) and corrected the memorized lore: Anthropic billing_error is HTTP 402 (was remembered as 403; any 402 is categorically billing), the credit-balance and self-set spend-limit messages ride 400 invalid_request_error and are message-sniffed with the documented prefixes (named fragile-by-necessity), and Google's 429 RESOURCE_EXHAUSTED stays TRANSIENT unless QuotaFailure details name a PerDay/FreeTier quota — a bare mis-flag would fire a false billing alert on an ordinary rate limit. Never retried (nothing about a billing state heals inside a backoff), wrapped as ProviderBillingError (Symbol.for cross-copy marker, carries providerErrorType/statusCode/modelId/cause) on both transport paths — thrown calls and pre-output stream error parts; unknown billing-ish shapes stay semantic/terminal, never silently parked; no schema strictification anywhere (optional-field walks only). Outer layers route the type to the credit park (flow wave D D2). 10 outcome tests red-first (8 red at pre-fix code: the 429 shapes observably RETRIED; the Anthropic 400 surfaced as a bare semantic error); bites verified red then restored: the 402 row (which exposed a missing bare-402 case — added), the message sniffs, the Google quota guard inverted (a bare rate limit mis-billed), and the stream-part wrap; pre-existing transport suites 20/20 green, conversation folder 148 green (35 key-gated skips). ([2a2d54d](https://github.com/proteinjs/conversation/commit/2a2d54d532b0b59ab1c1ff0d94b4518c1abbedd6))





# [5.20.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.19.0...@proteinjs/conversation@5.20.0) (2026-08-27)


### Features

* transport-retry activity observer — LlmTransportRetry reports retrying/recovered/gave-up per wrapped call and generateStream exposes onTransportRetry, so visible surfaces (the chat turn's provider wait node) can render the transient-overload retries the transport layer absorbs; observational only — retry semantics, budgets, and typed exhaustion are unchanged ([3ceff7d](https://github.com/proteinjs/conversation/commit/3ceff7dbf5b4a027b72bdca528c526523e9584f2))





# [5.19.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.18.2...@proteinjs/conversation@5.19.0) (2026-08-27)


### Features

* surface stripped OpenAI citation runs as house source entries (url_citation -> sources, deduped by url) ([a80a325](https://github.com/proteinjs/conversation/commit/a80a325bed5563a3b3e579adc2b66899b26d09a1))





## [5.18.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.18.1...@proteinjs/conversation@5.18.2) (2026-08-27)


### Bug Fixes

* strip OpenAI Responses in-band citation markers from every assistant-text egress ([45fb134](https://github.com/proteinjs/conversation/commit/45fb134af48afe7214f5c917f6e9929e06b630ff))





## [5.18.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.18.0...@proteinjs/conversation@5.18.1) (2026-08-26)


### Bug Fixes

* generateObject tool loop — prompt caching + streamed dispatch ([6f92caa](https://github.com/proteinjs/conversation/commit/6f92caae3df386ba22e24184e00a3ee96ef42412))





# [5.18.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.17.2...@proteinjs/conversation@5.18.0) (2026-08-18)


### Features

* maxTotalTokens — a loop-seam token ceiling for unattended callers ([26fe9ed](https://github.com/proteinjs/conversation/commit/26fe9ed4f54570cf1fd000ee68ec6f786d7cc1f9))





## [5.17.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.17.1...@proteinjs/conversation@5.17.2) (2026-08-15)


### Bug Fixes

* scrub hardcoded API key from commented example — env reference instead ([dc103e9](https://github.com/proteinjs/conversation/commit/dc103e9a82430a375703947e37ab234ce217a2f3))





## [5.17.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.17.0...@proteinjs/conversation@5.17.1) (2026-08-13)

**Note:** Version bump only for package @proteinjs/conversation





# [5.17.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.16.0...@proteinjs/conversation@5.17.0) (2026-08-11)


### Features

* tool timeline details accept a serializable glyph ([19f50b4](https://github.com/proteinjs/conversation/commit/19f50b4f817987f7935fd87999b0683deb5765ec))
* tool-call stream parts carry the full argument object ([0d7e495](https://github.com/proteinjs/conversation/commit/0d7e495c19fee7109888fc3bf63ada1c83836b41))





# [5.15.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.14.1...@proteinjs/conversation@5.15.0) (2026-07-31)


### Features

* provider identity on TransientProviderError + shared outage-copy helpers ([07eca09](https://github.com/proteinjs/conversation/commit/07eca095ddd575f39ac934e809ef864585d034a6))





# [5.14.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.13.1...@proteinjs/conversation@5.14.0) (2026-07-30)


### Features

* tag budget-exhausted transient transport errors as TransientProviderError ([750c403](https://github.com/proteinjs/conversation/commit/750c40365fad5321d78228f1ee8587b9e77ed4c9))





## [5.13.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.13.0...@proteinjs/conversation@5.13.1) (2026-07-29)


### Bug Fixes

* turn-scoped restart eligibility + per-step finish vocabulary ([36956c9](https://github.com/proteinjs/conversation/commit/36956c950dffb8b711b6f661f43cb0a56b39fa5c))





# [5.13.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.12.0...@proteinjs/conversation@5.13.0) (2026-07-28)


### Features

* thinking-phase restart folds mid-turn notes into ONE re-planned answer ([598dca2](https://github.com/proteinjs/conversation/commit/598dca2ac0a8e2f5ee84d1e0656e1d916281707a))





# [5.12.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.11.1...@proteinjs/conversation@5.12.0) (2026-07-28)


### Features

* outcome-aware timeline relabeling (Function.getTimelineOutcome) ([2455e1d](https://github.com/proteinjs/conversation/commit/2455e1d4eed663a911d2c2e30aacebac14584fa0))





## [5.11.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.11.0...@proteinjs/conversation@5.11.1) (2026-07-25)


### Bug Fixes

* gate exit-note absorption behind absorbExitNotes (chat-lane opt-in) ([f057071](https://github.com/proteinjs/conversation/commit/f05707195b575ae0e7006a34bc3d14c1e62f2abd))





# [5.11.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.10.0...@proteinjs/conversation@5.11.0) (2026-07-25)


### Features

* steering hold + linkable tool timeline details ([4408460](https://github.com/proteinjs/conversation/commit/4408460c71ca6f3fb9b62acdba52af814275f1ce))





# [5.10.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.9.0...@proteinjs/conversation@5.10.0) (2026-07-24)


### Features

* claude-opus-5 pricing row ([c77a88a](https://github.com/proteinjs/conversation/commit/c77a88a82a787530b38fa5154cddb2be7c745eb7))





# [5.9.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.8.2...@proteinjs/conversation@5.9.0) (2026-07-24)


### Features

* hard input-cap guard for capped models (haiku 200K) at the dispatch seams ([8f6a2ec](https://github.com/proteinjs/conversation/commit/8f6a2ec76a75915bada59a06a9056d5d642de183))





## [5.8.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.8.1...@proteinjs/conversation@5.8.2) (2026-07-23)


### Bug Fixes

* OpenAI-only strict-mode schema rewriting; bump @ai-sdk/anthropic to 3.0.100 ([f8191b4](https://github.com/proteinjs/conversation/commit/f8191b474de1c305351f4aa66f97b2c4afd1bf90))





# [5.8.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.7.0...@proteinjs/conversation@5.8.0) (2026-07-21)


### Bug Fixes

* liveness guards on model calls — silent connection death can no longer hang forever ([2000906](https://github.com/proteinjs/conversation/commit/200090683cb5ccc2e35ff0901f2438282de83c76))


### Features

* generateObject supports an investigate-then-answer tool loop ([787be5b](https://github.com/proteinjs/conversation/commit/787be5bb7d3163ab25747b92fb09ca54f228da95))





# [5.7.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.6.0...@proteinjs/conversation@5.7.0) (2026-07-11)


### Features

* injected-context splice hook, empty-messages normalization, GPT-5.6/5.5-Pro pricing ([f01096d](https://github.com/proteinjs/conversation/commit/f01096d35fe5f5eb5dcecd1c2733a47241736697))





# [5.6.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.5.0...@proteinjs/conversation@5.6.0) (2026-07-10)


### Features

* pricing for claude-fable-5 and grok-4.5 ([44f7fca](https://github.com/proteinjs/conversation/commit/44f7fca2b8afa851b194190de5fa4ee2fba08ebf))





# [5.5.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.4.0...@proteinjs/conversation@5.5.0) (2026-07-10)


### Features

* toolResultTokenBudget guard + pre-output mid-stream transport retry + live-API test hardening ([7ab9fe0](https://github.com/proteinjs/conversation/commit/7ab9fe0a2e31a6d8303cb45b10d3667e93b55c97))





# [5.4.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.3.1...@proteinjs/conversation@5.4.0) (2026-07-09)


### Features

* **conversation:** PROMPT_DUMP_DIR debug affordance for prompt-cache diagnosis ([90f2f2b](https://github.com/proteinjs/conversation/commit/90f2f2beb51c0d3b3a49ad8ad65676f7ed0c5a7f))





## [5.3.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.3.0...@proteinjs/conversation@5.3.1) (2026-07-03)


### Bug Fixes

* generateStream tool invocations report real per-call outcomes ([8dd2785](https://github.com/proteinjs/conversation/commit/8dd27858b248c99d56b93fe04cbaf2df6948b46a))





# [5.3.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.2.1...@proteinjs/conversation@5.3.0) (2026-07-02)


### Features

* LlmTransportRetry — invisible bounded retries for transient LLM transport failures ([d148dd6](https://github.com/proteinjs/conversation/commit/d148dd6ff1d457eaa83c588d2c94072ce44676a6))





## [5.2.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.2.0...@proteinjs/conversation@5.2.1) (2026-06-20)


### Bug Fixes

* coerce non-object tool-call input to {} before the wire ([34b5d69](https://github.com/proteinjs/conversation/commit/34b5d69fbfd1a47d8dffffd5ff2ae64f02c7cf7c))





# [5.2.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.1.0...@proteinjs/conversation@5.2.0) (2026-06-19)


### Features

* prompt caching, tool-image pruning, and live usage for agentic loops ([e014a6c](https://github.com/proteinjs/conversation/commit/e014a6c16d99b838570400da0484e42c54cde560))
* track cache-write tokens and fix usage cost accounting ([a4bdeaf](https://github.com/proteinjs/conversation/commit/a4bdeaf110d9fcdeeb918b2f1f7cd8edadd04565))





# [5.1.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@5.0.0...@proteinjs/conversation@5.1.0) (2026-06-11)


### Features

* split text-editor view vs edit tool names in the stream ([5e306cf](https://github.com/proteinjs/conversation/commit/5e306cf509fac4198dd886382861f160a548617d))





# [5.0.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.7.0...@proteinjs/conversation@5.0.0) (2026-06-10)


* feat!: require getId() on ConversationSkill ([ff6c1e2](https://github.com/proteinjs/conversation/commit/ff6c1e289805c816d12c4c8bdb74cc3cf69936e7))
* refactor!: rename concrete *Module skills to *Skill ([c311cd4](https://github.com/proteinjs/conversation/commit/c311cd406e2013bb0c3ce6cbe4db93068c944972))


### Features

* add provider-defined tool injection to Conversation ([45d2d58](https://github.com/proteinjs/conversation/commit/45d2d5880873135d3cd9a5413229b8767727307a))
* add SkillDispatcherSkill for cross-provider lazy skill loading ([95689f6](https://github.com/proteinjs/conversation/commit/95689f6620d948fe764b02c6033e117f0ceb9b84))
* **conversation:** emit tool-invocation events from the tool loop ([32ae9b0](https://github.com/proteinjs/conversation/commit/32ae9b07c0caa7bc819ebf6c4a95839538a431a0))
* surface provider-defined tool calls + capture reasoning ([682a240](https://github.com/proteinjs/conversation/commit/682a240597d5037fb5b81915dbfbea29ee1e4b46))
* surface skill name+summary in dispatcher catalog so the model engages skills ([62535b5](https://github.com/proteinjs/conversation/commit/62535b5bed099c7e04f5467607df4f44215c3d2d))


### BREAKING CHANGES

* ConversationSkill.getId(): string is now required.
External implementers must add a stable, kebab-case id that should not
change after release.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* every exported concrete skill class and its factory has
been renamed. Consumers must update imports from `@proteinjs/conversation`,
e.g. `import { ConversationFsSkill } from '@proteinjs/conversation'`
(was `ConversationFsModule`).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>





# [3.3.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.2.0...@proteinjs/conversation@3.3.0) (2026-04-24)


### Features

* **conversation:** end-to-end file/image input across providers ([3faf1e2](https://github.com/proteinjs/conversation/commit/3faf1e22da046aad73b599add05b49c21827bf77))





# [3.2.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.9...@proteinjs/conversation@3.2.0) (2026-04-16)


### Features

* **anthropic:** add Opus 4.7 pricing and xhigh effort support ([de3ca31](https://github.com/proteinjs/conversation/commit/de3ca31b4b3424048c556d228a8dc7cfdd50edfb))





## [3.1.7](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.6...@proteinjs/conversation@3.1.7) (2026-04-09)


### Reverts

* remove tool-result and step-start from StreamPart ([f2b894c](https://github.com/proteinjs/conversation/commit/f2b894c073a8bba6b580e908d451973cbd5b09ab))





## [3.1.6](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.5...@proteinjs/conversation@3.1.6) (2026-04-09)


### Bug Fixes

* yield tool-result and step-start parts from mapFullStream ([1a91edd](https://github.com/proteinjs/conversation/commit/1a91edd47e91e4400914e9fcdb31bf276fcf9541))





## [3.1.4](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.3...@proteinjs/conversation@3.1.4) (2026-04-08)


### Bug Fixes

* add --passWithNoTests to jest test script ([6ec6d0b](https://github.com/proteinjs/conversation/commit/6ec6d0b58727aacaad0542d88af4ba10d2f2f6ea))





## [3.1.3](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.2...@proteinjs/conversation@3.1.3) (2026-04-05)


### Bug Fixes

* reorder system messages for Anthropic provider too ([c5a4f03](https://github.com/proteinjs/conversation/commit/c5a4f0374c0be7a0fb5cb7ebf08c91e29d65721a))





## [3.1.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.1...@proteinjs/conversation@3.1.2) (2026-04-05)


### Bug Fixes

* resolve provider-specific model issues ([ade5478](https://github.com/proteinjs/conversation/commit/ade547855905c87684f5923b271dbae8675e0662))





## [3.1.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.1.0...@proteinjs/conversation@3.1.1) (2026-03-31)


### Bug Fixes

* More reliable expectations from `generateList.test`. ([5124b92](https://github.com/proteinjs/conversation/commit/5124b92a5a44a3b57be669e0b405d65c1033d528))





# [3.1.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@3.0.0...@proteinjs/conversation@3.1.0) (2026-03-28)


### Bug Fixes

* skip web search tools for nano-class models ([963ebc2](https://github.com/proteinjs/conversation/commit/963ebc2d1e5d587ab70a2ab6ccd786f6ef9e9d95))


### Features

* auto web search, Haiku extended thinking, model pricing updates ([752314f](https://github.com/proteinjs/conversation/commit/752314fc0eaf87122a6e1f78cc455db633da7ad2))
* web search tools, interleaved fullStream, and lazy promise getters ([18e8c4d](https://github.com/proteinjs/conversation/commit/18e8c4d674d90559e0f3d90ed4929f711e3ac8e1))





# [3.0.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.7.3...@proteinjs/conversation@3.0.0) (2026-03-25)


### chore

* **conversation:** trigger major version bump ([f1598e6](https://github.com/proteinjs/conversation/commit/f1598e63e6d24d79b86c9bfcf9d39a94d67474e3))


### BREAKING CHANGES

* **conversation:** Conversation v2 rewrites the class API to use Vercel AI SDK v6





## [2.7.3](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.7.2...@proteinjs/conversation@2.7.3) (2026-03-25)

**Note:** Version bump only for package @proteinjs/conversation





# [2.7.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.6.0...@proteinjs/conversation@2.7.0) (2026-01-26)


### Bug Fixes

* `OpenAiResponses` enhanced error logging and added custom `OpenAiResponsesError` class. ([76eb085](https://github.com/proteinjs/conversation/commit/76eb0854eb49bc9e6ec8b9599347f86e5f889bd5))


### Features

* `FsFunctions` added `deleteFilesFunction`. ([0cb2722](https://github.com/proteinjs/conversation/commit/0cb272209aa60a4c1100efc84a511c5c3fe0c188))
* `OpenAiResponses` added `maxBackgroundWaitMs` param. ([a008e18](https://github.com/proteinjs/conversation/commit/a008e188be0f7071d5581e92ac0ede6471f3ac1e))
* Enhanced `UsageData` to track cost; considers service tier. ([8ff62f8](https://github.com/proteinjs/conversation/commit/8ff62f8e1d930e7f121724ffc7ded669eeeb7728))





# [2.6.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.5.0...@proteinjs/conversation@2.6.0) (2026-01-16)


### Features

* Added `OpenAiResponses` as a wrapper around the responses api. ([8fa7470](https://github.com/proteinjs/conversation/commit/8fa7470265cc5157e980600a910e96225ba8c43f))





# [2.5.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.4.2...@proteinjs/conversation@2.5.0) (2026-01-16)


### Bug Fixes

* `KeywordToFilesIndexFunctions` add additional details to the `searchFilesFunction` description regarding the expectation of search terms (case-insensitive and extensions ignored). ([aa14d9b](https://github.com/proteinjs/conversation/commit/aa14d9b723b2087217549f167f1cfa111f9891db))


### Features

* `Conversation` enable setting of `maxToolCalls`. ([c8c8398](https://github.com/proteinjs/conversation/commit/c8c8398fcf6247d9ed81c6a4b69fd041fc3dd1f2))





## [2.4.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.4.1...@proteinjs/conversation@2.4.2) (2026-01-13)


### Bug Fixes

* `ConversationFsModule` shut off `ConversationFsModerator`. Not currently being used and therefore is non-trivial complexity that runs and only has the change of causing failures (even if it's only made the open ai request logs look confusing). ([7cef54b](https://github.com/proteinjs/conversation/commit/7cef54bdf2923a3539624b8556c4d031093acd9a))





## [2.4.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.4.0...@proteinjs/conversation@2.4.1) (2026-01-07)

**Note:** Version bump only for package @proteinjs/conversation





# [2.4.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.3.1...@proteinjs/conversation@2.4.0) (2025-11-22)


### Features

* Added `ToolInvocationProgressEvent` type helpers. ([7781ba5](https://github.com/proteinjs/conversation/commit/7781ba54710845ea3d852d051b6b2bb643c915cc))





# [2.3.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.2.2...@proteinjs/conversation@2.3.0) (2025-11-03)


### Features

* `Conversation` add optional abortSignal to `generateResponse` and `generateObject`. ([6167107](https://github.com/proteinjs/conversation/commit/616710771832448f47e1d1e2d087b4ceccb1b3cc))





## [2.2.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.2.1...@proteinjs/conversation@2.2.2) (2025-11-02)


### Bug Fixes

* `KeyworkfToFilesIndexFunctions` should be clear that searchFiles keywords are file name matches. ([35468e6](https://github.com/proteinjs/conversation/commit/35468e61c969fea71368bb3d133758ba846ff25d))





## [2.2.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.2.0...@proteinjs/conversation@2.2.1) (2025-11-02)


### Bug Fixes

* `FsFunctions` wrap all code in read/write so exceptions are always caught and logged to the assistant. ([2f5511d](https://github.com/proteinjs/conversation/commit/2f5511d2d85d1a2cbc19ace128f1245f7c7f2395))





# [2.2.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.1.6...@proteinjs/conversation@2.2.0) (2025-10-21)


### Features

* Added reasoningEffort to `Conversation` and `OpenAi` APIs. ([9e43462](https://github.com/proteinjs/conversation/commit/9e434620f644d562bdf3f3b3f03dfa51e4f5f46e))





## [2.1.6](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.1.5...@proteinjs/conversation@2.1.6) (2025-10-18)


### Bug Fixes

* Ensure `FsFunctions` do not throw, but return the error message to the assistant. ([a9e8722](https://github.com/proteinjs/conversation/commit/a9e87227ed14f02733c3c9f5d329b7266f78121f))





## [2.1.5](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.1.4...@proteinjs/conversation@2.1.5) (2025-10-14)


### Bug Fixes

* `Conversation.generateObject` Better ensure oai models return objects by using strict mode. ([5177efc](https://github.com/proteinjs/conversation/commit/5177efc512f771105f86eb45348c1ae4c06d3053))





# [2.1.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.0.4...@proteinjs/conversation@2.1.0) (2025-04-21)


### Features

* `ConversationModule.getSystemMessages` may now optionally return a promise ([fa676b4](https://github.com/proteinjs/conversation/commit/fa676b4a09536552378c8282178f00a3b1a8d6f8))





## [2.0.3](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.0.2...@proteinjs/conversation@2.0.3) (2025-04-11)

**Note:** Version bump only for package @proteinjs/conversation





## [2.0.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@2.0.1...@proteinjs/conversation@2.0.2) (2024-09-24)


### Bug Fixes

* omit tools param in executeRequest if functions array doesn't exist or is empty ([9c0f15c](https://github.com/proteinjs/conversation/commit/9c0f15c3450edfeb62db3f7285f1469bcacd00aa))





# [2.0.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.7.5...@proteinjs/conversation@2.0.0) (2024-08-18)


### Features

* `OpenAi` now returns `UsageData` for `generateResponse` and `generateStreamingResponse` methods. ([c4f5488](https://github.com/proteinjs/conversation/commit/c4f54888949a3c64beda24a1735f6af2dbf7329d))


### BREAKING CHANGES

* converted `OpenAi` static methods to be instance methods. Also updated params to be a single object for most methods in `OpenAi` and `Conversation`.

Made these changes to simplify the maintenance of `OpenAi` and also to simplify the code calling the api to not need order args (and pass in undefined often) with such a large number of optional parameters.





## [1.7.4](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.7.3...@proteinjs/conversation@1.7.4) (2024-08-16)


### Bug Fixes

* refactored to implement new @proteinjs/logger/Logger api ([75f5744](https://github.com/proteinjs/conversation/commit/75f5744129c0798ef7a792b6bbe5463c4684e416))





## [1.7.3](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.7.2...@proteinjs/conversation@1.7.3) (2024-08-07)

**Note:** Version bump only for package @proteinjs/conversation





## [1.7.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.7.1...@proteinjs/conversation@1.7.2) (2024-08-06)


### Bug Fixes

* making sure all streams get destroyed in all input stream end scenarios ([ab2da7d](https://github.com/proteinjs/conversation/commit/ab2da7dfbc42a0bccae73db9ab49da8c1a01b61b))





# [1.7.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.6.1...@proteinjs/conversation@1.7.0) (2024-08-05)


### Features

* added optional `AbortSignal` to `generateStreamingResponse` ([a02eb64](https://github.com/proteinjs/conversation/commit/a02eb64444629bc4ec97f7336322fdfcec97d41b))





## [1.6.1](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.6.0...@proteinjs/conversation@1.6.1) (2024-07-28)


### Bug Fixes

* `OpenAiStreamProcessor.createControlStream` now ignores chunks with null content (scenarios we don't need to fail on) ([2e1ab6e](https://github.com/proteinjs/conversation/commit/2e1ab6ea1e2c38552ddd2035225bd0ba80a77fef))





# [1.6.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.5.2...@proteinjs/conversation@1.6.0) (2024-07-28)


### Features

* added `OpenAi.generateStreamingResponse` ([2b1ba30](https://github.com/proteinjs/conversation/commit/2b1ba30a7e27f84f4fe076be9d6e2ea46ac4df9d))





## [1.5.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.5.1...@proteinjs/conversation@1.5.2) (2024-07-17)

**Note:** Version bump only for package @proteinjs/conversation





# [1.5.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.4.0...@proteinjs/conversation@1.5.0) (2024-07-12)


### Features

* added `ChatCompletionMessageParamFactory` as a way to return files (like images) in response to function calls ([703ccbf](https://github.com/proteinjs/conversation/commit/703ccbfca2d644cd59d457bba57016e75cfc36a2))
* added `Conversation.addMessagesToHistory` to be able to add `ChatCompletionMessageParam`s to history ([7c1a98e](https://github.com/proteinjs/conversation/commit/7c1a98eb9acc57813aa7dd7ebd62893a6452dbca))





# [1.4.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.3.0...@proteinjs/conversation@1.4.0) (2024-07-11)


### Features

* updated `Function` and `OpenAi` to adopt the new `tools` api (replacing legacy function api) ([e77013f](https://github.com/proteinjs/conversation/commit/e77013f20af9e857fadbf9cb3709eb7325b601d3))





# [1.3.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.2.2...@proteinjs/conversation@1.3.0) (2024-07-11)


### Features

* implement max function calls, handle functions with void return type ([#2](https://github.com/proteinjs/conversation/issues/2)) ([36b26cf](https://github.com/proteinjs/conversation/commit/36b26cf31782c68ae230d7ae75c678d633340f44))





# [1.2.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.1.0...@proteinjs/conversation@1.2.0) (2024-06-25)


### Features

* `Conversation` and `OpenAi` now accept `ChatCompletionMessageParam`s as well as string messages. This enables the caller to send files and other data in with a message. ([63e9536](https://github.com/proteinjs/conversation/commit/63e9536fa39de09e85848b9658a30d1d4eb2face))





# [1.1.0](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.11...@proteinjs/conversation@1.1.0) (2024-05-20)


### Features

* updating tiktoken version ([a09e604](https://github.com/proteinjs/conversation/commit/a09e604c6174788b4a7c4cf757db6157acc8095f))





## [1.0.11](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.10...@proteinjs/conversation@1.0.11) (2024-05-12)


### Bug Fixes

* update tiktoken version ([0d02ba2](https://github.com/proteinjs/conversation/commit/0d02ba20ece095027c3ebb2c0de5c4e088b4d4e9))





## [1.0.10](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.9...@proteinjs/conversation@1.0.10) (2024-05-12)


### Bug Fixes

* update tiktoken version ([5bb285c](https://github.com/proteinjs/conversation/commit/5bb285ca4eafa499d844b25504fbc744bc2a181f))
* updating lerna config ([d1aa89f](https://github.com/proteinjs/conversation/commit/d1aa89f89dbe155a9a3b4f7d74cc860a08e720d9))





## [1.0.8](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.7...@proteinjs/conversation@1.0.8) (2024-05-10)


### Bug Fixes

* add .md to lint ignore files ([73034c8](https://github.com/proteinjs/conversation/commit/73034c883bdbd45ad098999258407d6396d6ed8c))





## [1.0.7](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.6...@proteinjs/conversation@1.0.7) (2024-05-10)


### Bug Fixes

* add linting and lint all files ([eae8f12](https://github.com/proteinjs/conversation/commit/eae8f128bb40ccc2a6656ec847ef4f39fc50c11b))





## [1.0.3](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.2...@proteinjs/conversation@1.0.3) (2024-04-19)

**Note:** Version bump only for package @proteinjs/conversation

## [1.0.2](https://github.com/proteinjs/conversation/compare/@proteinjs/conversation@1.0.1...@proteinjs/conversation@1.0.2) (2024-04-19)

**Note:** Version bump only for package @proteinjs/conversation

## 1.0.1 (2024-04-19)

**Note:** Version bump only for package @proteinjs/conversation
