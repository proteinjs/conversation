module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  transform: {
    '^.+\\.tsx?$': 'ts-jest',
  },
  testMatch: ["**/?(*.)+(spec|test).ts"],
  setupFiles: ['./test/setup'],
};