'use strict';
// THE ONE CI TEST PRACTICE (@proteinjs/jest-preset; its JestPresetBuilder states the practice whole): the house preset
// carries transpile-only ts-jest, the worker/memory knobs (JEST_WORKERS), the in-band declaration (package.json
// jestPreset.inBand) and the standard setup hooks (test/setup and the emulator lock, probed from the package) — this
// package has no seams of its own.
module.exports = {
  preset: '@proteinjs/jest-preset',
};
