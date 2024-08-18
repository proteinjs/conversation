# Change Log

All notable changes to this project will be documented in this file.
See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

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
