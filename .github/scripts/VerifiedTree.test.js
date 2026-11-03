'use strict';
/**
 * VerifiedTree.js — the publish-skip condition and its guard, pinned (node:test, no install):
 * `node --test .github/scripts/VerifiedTree.test.js`, in the build job before any minute is spent
 * on install/build.
 *
 * Why a test and not a review: the condition fails SILENTLY when wrong in the dangerous direction —
 * a tree released without its sweep. Each case names the outcome it guards: a dispatched run verifies
 * the sha in its NAME, never the ref tip it ran on; a departure commit is accepted only when it
 * changes .train/ alone; anything unreadable tests.
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { VerifiedTree } = require('./VerifiedTree');

const A = 'a'.repeat(40);
const B = 'b'.repeat(40);
const C = 'c'.repeat(40);
const run = (over = {}) => ({
  id: 1,
  head_sha: A,
  display_title: VerifiedTree.runName(A),
  event: 'push',
  status: 'completed',
  conclusion: 'success',
  html_url: 'https://example.invalid/runs/1',
  ...over,
});

test('the departure commit: HEAD^ verified by a green run named for it, HEAD changing only .train/ — verified; a change outside .train/ beside it — not', () => {
  const yes = VerifiedTree.judge({ head: B, parent: A, changed: ['.train/feat-a.json', '.train/feat-b.json'], runs: [run()] });
  assert.equal(yes.verified, true);
  assert.equal(yes.run.id, 1);
  assert.match(yes.why, /HEAD\^ a{8}.*only \.train\//);
  const no = VerifiedTree.judge({ head: B, parent: A, changed: ['.train/feat-a.json', 'packages/ui/src/x.ts'], runs: [run()] });
  assert.equal(no.verified, false);
  assert.match(no.why, /packages\/ui\/src\/x\.ts/);
});

test('HEAD itself verified by a run named for it (any event) or by a push run whose head it is; the newest green run wins', () => {
  assert.equal(VerifiedTree.judge({ head: A, parent: C, changed: ['packages/ui/src/x.ts'], runs: [run()] }).verified, true);
  assert.equal(VerifiedTree.judge({ head: A, parent: null, changed: [], runs: [run({ display_title: 'feat: something' })] }).verified, true, 'a push run with no run-name still verifies its head');
  assert.equal(VerifiedTree.judge({ head: C, parent: null, changed: [], runs: [run({ event: 'workflow_dispatch', head_sha: A, display_title: VerifiedTree.runName(C) })] }).verified, true, 'a dispatch verifies the sha in its name');
  const newest = VerifiedTree.judge({ head: A, parent: null, changed: [], runs: [run({ id: 9, html_url: 'u9' }), run({ id: 3, html_url: 'u3' })] });
  assert.equal(newest.run.id, 9);
});

test('the guard: a red or in-flight run, a dispatch NAMED for another sha even when its head_sha is HEAD, and no run at all — none verify', () => {
  assert.equal(VerifiedTree.judge({ head: A, parent: C, changed: [], runs: [run({ conclusion: 'failure' })] }).verified, false);
  assert.equal(VerifiedTree.judge({ head: A, parent: C, changed: [], runs: [run({ status: 'in_progress', conclusion: null })] }).verified, false);
  const dispatched = VerifiedTree.judge({ head: A, parent: C, changed: ['packages/ui/src/x.ts'], runs: [run({ event: 'workflow_dispatch', head_sha: A, display_title: VerifiedTree.runName(C) })] });
  assert.equal(dispatched.verified, false, 'the dispatch checked out C, not the ref tip A it was dispatched on');
  const none = VerifiedTree.judge({ head: A, parent: null, changed: [], runs: [] });
  assert.equal(none.verified, false);
  assert.match(none.why, /no green verify-train run/);
});

test('matches reads the REST shape and the gh CLI shape alike; runName is the verify-train.yml run-name', () => {
  assert.equal(VerifiedTree.matches({ headSha: A, displayTitle: 'x', event: 'push' }, A), true);
  assert.equal(VerifiedTree.matches({ headSha: A, displayTitle: 'x', event: 'workflow_dispatch' }, A), false);
  assert.equal(VerifiedTree.matches({ head_sha: B, display_title: VerifiedTree.runName(A), event: 'workflow_dispatch' }, A), true);
  assert.equal(VerifiedTree.runName(A), `verify-train ${A}`);
});

test('the door: verified=true|false and verify_run to GITHUB_OUTPUT; the API down, git without a parent, or VERIFY_WORKFLOW unset = verified=false with the reason — never a throw, never the token printed', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'verified-tree-'));
  const out = path.join(dir, 'output');
  const env = { GITHUB_REPOSITORY: 'example/repo', GITHUB_TOKEN: 'shh', GITHUB_API_URL: 'https://api.invalid', GITHUB_OUTPUT: out, VERIFY_WORKFLOW: 'verify-train.yml' };
  const git = (args) => {
    if (args[0] === 'rev-parse' && args[1] === 'HEAD') return B;
    if (args[0] === 'rev-parse' && args[1] === 'HEAD^') return A;
    if (args[0] === 'diff') return '.train/feat-a.json\n';
    throw new Error(`unexpected git ${args.join(' ')}`);
  };
  const asked = [];
  const fetchJson = async (url, headers) => (asked.push({ url, auth: headers.Authorization }), { workflow_runs: [run()] });
  const lines = [];
  let verdict = await VerifiedTree.main({ env, git, fetchJson, log: (l) => lines.push(l) });
  assert.equal(verdict.verified, true);
  assert.equal(asked[0].url, 'https://api.invalid/repos/example/repo/actions/workflows/verify-train.yml/runs?status=success&per_page=100');
  assert.equal(asked[0].auth, 'Bearer shh');
  assert.ok(lines.length && !lines.some((l) => /shh/.test(l)), 'the token never prints');
  assert.match(fs.readFileSync(out, 'utf8'), /^verified=true$/m);
  assert.match(fs.readFileSync(out, 'utf8'), /^verify_run=https:\/\/example\.invalid\/runs\/1$/m);
  fs.writeFileSync(out, '');
  verdict = await VerifiedTree.main({ env, git, fetchJson: async () => { throw new Error('HTTP 404'); }, log: () => {} });
  assert.equal(verdict.verified, false);
  assert.match(verdict.why, /HTTP 404/);
  assert.match(fs.readFileSync(out, 'utf8'), /^verified=false$/m);
  verdict = await VerifiedTree.main({ env, git: (args) => (args[1] === 'HEAD^' ? (() => { throw new Error('fatal: bad revision'); })() : git(args)), fetchJson, log: () => {} });
  assert.equal(verdict.verified, false);
  verdict = await VerifiedTree.main({ env: { ...env, VERIFY_WORKFLOW: '' }, git, fetchJson, log: () => {} });
  assert.equal(verdict.verified, false);
  assert.match(verdict.why, /VERIFY_WORKFLOW/);
  fs.rmSync(dir, { recursive: true, force: true });
});
