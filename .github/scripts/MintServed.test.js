'use strict';
/**
 * MintServed.js — the wait for the registry to serve a publish, pinned (node:test; needs git and a
 * shell on PATH, no install): `node --test .github/scripts/MintServed.test.js`, run by the workflow
 * before a minute is spent on install. The same file rides every repo's .github/scripts/ verbatim:
 * copy whole when it changes.
 *
 * What needs pinning is the wait itself — the poll keeps asking until the registry answers (the
 * lag that tore run 36295867264 answered on the third ask), the clock is one bound shared by the
 * whole mint, a mint past the deadline is a WARNING line and exit 1 never a throw, and the mints are
 * read from the tags lerna left on the release commit. The registry is a stub npm whose `view`
 * answers from the Nth call on; the clock is a counter the stub sleep advances, so "served after N s"
 * is exact.
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { MintServed, EXIT } = require('./MintServed');

const MINT = { name: '@proteinjs/fixture-common', version: '2.0.0' };

function tmp() {
  return fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'mint-served-')));
}

/**
 * The stub npm: `view <name>@<version> version` logs the ask and, from the SERVE_ON-th ask on, answers
 * the version on stdout (exit 0); before that it is the registry not yet serving the publish — E404 on
 * stderr, exit 1. Anything else is refused (this test never wants the real npm).
 */
function stubNpm(root, { serveOn = 1 } = {}) {
  const dir = path.join(root, '.stub');
  fs.mkdirSync(dir, { recursive: true });
  const log = path.join(dir, 'asks.log');
  fs.writeFileSync(log, '');
  const bin = path.join(dir, 'npm');
  fs.writeFileSync(
    bin,
    `#!/bin/sh
case "$1" in
  view)
    spec=""; for a in "$@"; do case "$a" in --*|view|version) ;; *) spec="$a" ;; esac; done
    echo "$spec" >> "$STUB_LOG"; n=$(wc -l < "$STUB_LOG" | tr -d ' ')
    if [ "$n" -ge "$STUB_SERVE_ON" ]; then echo "\${spec##*@}"; exit 0; fi
    echo "npm error code E404" >&2; echo "npm error 404 No match found for version $spec" >&2; exit 1 ;;
  *) echo "stub npm: refused $*" >&2; exit 99 ;;
esac
`
  );
  fs.chmodSync(bin, 0o755);
  const env = { ...process.env, STUB_LOG: log, STUB_SERVE_ON: String(serveOn) };
  return { bin, env, asks: () => fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) };
}

/** A fake clock: `now` reads it, `sleep` advances it — the wait's arithmetic exact, no real seconds spent. */
function clock() {
  let t = 1000;
  return { now: () => t, sleep: (ms) => (t += ms) };
}

/** A git repo with one commit and the given tags on it. */
function taggedRepo(tags) {
  const root = tmp();
  const git = (...args) =>
    execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  git('init', '-q');
  git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'chore(release): publish');
  for (const tag of tags) git('tag', tag);
  return root;
}

const lines = () => {
  const out = [];
  const log = (line) => out.push(line);
  return { out, log };
};

test('the registry serving the publish on the third ask: the wait keeps asking 10 s apart and says "served after 20 s" — served, nothing timed out', () => {
  const root = tmp();
  const stub = stubNpm(root, { serveOn: 3 });
  const c = clock();
  const { out, log } = lines();
  const wait = new MintServed({ repoRoot: root, npmBin: stub.bin, env: stub.env, log, sleep: c.sleep, now: c.now });
  const result = wait.wait([MINT], { timeoutMs: 300000, intervalMs: 10000 });
  assert.deepEqual(result.timedOut, []);
  assert.deepEqual(result.served, [{ ...MINT, afterMs: 20000 }]);
  assert.deepEqual(
    stub.asks(),
    ['@proteinjs/fixture-common@2.0.0', '@proteinjs/fixture-common@2.0.0', '@proteinjs/fixture-common@2.0.0'],
    'three asks, the third answered'
  );
  assert.deepEqual(out, ['@proteinjs/fixture-common@2.0.0: served after 20 s']);
});

test('a registry that never serves it: the asks stop at the deadline (one clock: 300 s at 10 s = 31 asks), one WARNING line naming the mint, timed out — exit 1 through run, never a throw', () => {
  const root = tmp();
  const stub = stubNpm(root, { serveOn: 99 });
  const c = clock();
  const { out, log } = lines();
  const wait = new MintServed({ repoRoot: root, npmBin: stub.bin, env: stub.env, log, sleep: c.sleep, now: c.now });
  const result = wait.wait([MINT], { timeoutMs: 300000, intervalMs: 10000 });
  assert.deepEqual(result.served, []);
  assert.deepEqual(result.timedOut, [MINT]);
  assert.equal(stub.asks().length, 31);
  assert.equal(out.length, 1);
  assert.match(out[0], /^::warning::@proteinjs\/fixture-common@2\.0\.0: the registry does not serve it after 300 s/);
  assert.equal(
    wait.run({ mints: ['@proteinjs/fixture-common@2.0.0'], timeoutMs: 300000, intervalMs: 10000 }),
    EXIT.TIMED_OUT
  );
});

test('two mints share the clock: the first served on the second ask (10 s), the second at once — "served after 10 s" for both; a mint past the deadline is asked once more, never waited for', () => {
  const root = tmp();
  const stub = stubNpm(root, { serveOn: 2 });
  const c = clock();
  const { out, log } = lines();
  const wait = new MintServed({ repoRoot: root, npmBin: stub.bin, env: stub.env, log, sleep: c.sleep, now: c.now });
  const other = { name: '@proteinjs/fixture-ui', version: '1.4.0' };
  const result = wait.wait([MINT, other], { timeoutMs: 300000, intervalMs: 10000 });
  assert.deepEqual(
    result.served.map((m) => [m.name, m.afterMs]),
    [
      ['@proteinjs/fixture-common', 10000],
      ['@proteinjs/fixture-ui', 10000],
    ]
  );
  assert.deepEqual(out, [
    '@proteinjs/fixture-common@2.0.0: served after 10 s',
    '@proteinjs/fixture-ui@1.4.0: served after 10 s',
  ]);

  const late = stubNpm(tmp(), { serveOn: 99 });
  const c2 = clock();
  const l2 = lines();
  const w2 = new MintServed({
    repoRoot: root,
    npmBin: late.bin,
    env: late.env,
    log: l2.log,
    sleep: c2.sleep,
    now: c2.now,
  });
  const r2 = w2.wait([MINT, other], { timeoutMs: 20000, intervalMs: 10000 });
  assert.deepEqual(r2.timedOut, [MINT, other]);
  assert.equal(
    late.asks().length,
    3 + 1,
    'the first mint used the clock (3 asks); the second was asked once at the deadline'
  );
});

test("the mints are lerna's tags on the release commit: every <name>@<version> tag at the ref, scoped names kept whole, other tags ignored, sorted; an untagged ref is nothing to wait for (exit 0)", () => {
  const root = taggedRepo([
    '@proteinjs/fixture-ui@1.4.0',
    '@proteinjs/fixture-common@2.0.0',
    'v9',
    'release-2026-09-27',
  ]);
  const stub = stubNpm(root, { serveOn: 1 });
  const { out, log } = lines();
  const wait = new MintServed({
    repoRoot: root,
    npmBin: stub.bin,
    env: stub.env,
    log,
    sleep: () => {},
    now: clock().now,
  });
  assert.deepEqual(wait.tagged('HEAD'), [
    { name: '@proteinjs/fixture-common', version: '2.0.0' },
    { name: '@proteinjs/fixture-ui', version: '1.4.0' },
  ]);
  assert.equal(wait.run({ ref: 'HEAD', timeoutMs: 0, intervalMs: 0 }), EXIT.OK);
  assert.deepEqual(stub.asks(), ['@proteinjs/fixture-common@2.0.0', '@proteinjs/fixture-ui@1.4.0']);
  assert.deepEqual(out, [
    '@proteinjs/fixture-common@2.0.0: served after 0 s',
    '@proteinjs/fixture-ui@1.4.0: served after 0 s',
  ]);

  const bare = taggedRepo([]);
  const l2 = lines();
  const w2 = new MintServed({ repoRoot: bare, npmBin: stub.bin, env: stub.env, log: l2.log, sleep: () => {} });
  assert.equal(w2.run({ ref: 'HEAD' }), EXIT.OK);
  assert.match(l2.out[0], /^no mint at HEAD/);
  assert.equal(stub.asks().length, 2, 'nothing asked for the untagged ref');
});

test('a registry lagging 450 s behind the accept (the stub answers on the 46th ask, 10 s apart — the lag a ten-package publish showed): the DEFAULT clock covers it — "served after 450 s", nothing timed out', () => {
  const root = tmp();
  const stub = stubNpm(root, { serveOn: 46 });
  const c = clock();
  const { out, log } = lines();
  const wait = new MintServed({ repoRoot: root, npmBin: stub.bin, env: stub.env, log, sleep: c.sleep, now: c.now });
  const result = wait.wait([MINT], { intervalMs: 10000 });
  assert.deepEqual(result.timedOut, [], 'the default clock outlasts a 450 s lag');
  assert.deepEqual(result.served, [{ ...MINT, afterMs: 450000 }]);
  assert.equal(stub.asks().length, 46, 'asked every 10 s until the 46th ask answered');
  assert.deepEqual(out, ['@proteinjs/fixture-common@2.0.0: served after 450 s']);
});

test('a mint that is not <name>@<version>, or a git the wait cannot read: refused (exit 2), named — never a poll', () => {
  const root = tmp();
  const stub = stubNpm(root, { serveOn: 1 });
  const { out, log } = lines();
  const wait = new MintServed({ repoRoot: root, npmBin: stub.bin, env: stub.env, log, sleep: () => {} });
  assert.equal(wait.run({ mints: ['fixture-common'] }), EXIT.FAIL);
  assert.match(out[0], /not a <name>@<version> mint: "fixture-common"/);
  assert.equal(wait.run({ ref: 'HEAD' }), EXIT.FAIL, 'not a git repo');
  assert.match(out[1], /git tag --points-at HEAD failed/);
  assert.equal(stub.asks().length, 0);
});
