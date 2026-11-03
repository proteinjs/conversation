'use strict';
/**
 * LockFloors.js — the lock-floor guard and its re-stamp, pinned (node:test; needs npm on PATH, no
 * install): `node --test .github/scripts/LockFloors.test.js`, run by the workflow before a minute is
 * spent on install. The same file rides every repo's .github/scripts/ verbatim: copy whole when it
 * changes.
 *
 * Why a test and not a review: the guard's verdict is npm's (`npm ci --dry-run`), so what needs
 * pinning is the glue around it — the trees it walks (lerna's `packages/**`), the diagnosis it prints
 * (declared range / lock installs), the re-stamp touching only torn trees, its retry while a registry
 * lags a publish, its refusal under the wrong npm major (a lock stamp is npm-major dependent), and
 * where CI's npm major is read from: the workflows the repo has, whichever shape — one publish
 * workflow, or a publish workflow beside the reusable build/test workflow it calls. The registry is a
 * stub npm whose `i --package-lock-only` stamps on the Nth call.
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { LockFloors, EXIT, NpmMajorMismatch } = require('./LockFloors');

const DEP = '@proteinjs/fixture-common';

/** A lerna repo: a root with nothing internal + packages/a (under `packages/**`) declaring DEP at `declared`, its lock installing `installed`. */
function repo({ declared = '^2.0.0', installed = '1.0.0' } = {}) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'lock-floors-'))); // realpath: macOS's /var -> /private/var, what a child's $PWD reports
  write(root, 'lerna.json', { version: 'independent', packages: ['packages/**'] });
  // Hermetic: a torn tree makes npm resolve the floor from a registry; this one refuses the connection at once, no retries.
  const npmrc =
    'registry=http://127.0.0.1:1/\nfetch-retries=0\nfetch-timeout=2000\nfetch-retry-mintimeout=1\nfetch-retry-maxtimeout=1\n';
  fs.writeFileSync(path.join(root, '.npmrc'), npmrc);
  write(root, 'package.json', { name: 'root', private: true, version: '0.0.0' });
  write(root, 'package-lock.json', lock('root', '0.0.0', {}));
  write(root, 'packages/a/package.json', {
    name: '@proteinjs/fixture-ui',
    version: '1.0.0',
    dependencies: { [DEP]: declared },
  });
  write(
    root,
    'packages/a/package-lock.json',
    lock('@proteinjs/fixture-ui', '1.0.0', { [DEP]: declared }, { [DEP]: installed })
  );
  fs.writeFileSync(path.join(root, 'packages/a/.npmrc'), npmrc);
  return root;
}

/**
 * The repo right after a sibling's publish: packages/common was released at 2.0.0 (lerna's release
 * commit raised packages/server's floor to ^2.0.0 and mirrored the range into its lock's root block),
 * while packages/server's lock still installs the 1.0.0 the pre-release stamp resolved — the tear.
 */
function publishedRepo() {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  fs.renameSync(path.join(root, 'packages/a'), path.join(root, 'packages/server'));
  write(root, 'packages/server/package.json', {
    name: '@proteinjs/fixture-server',
    version: '1.4.0',
    dependencies: { [DEP]: '^2.0.0' },
  });
  write(root, 'packages/common/package.json', { name: DEP, version: '2.0.0' });
  return root;
}

/** A workflow under .github/workflows/ setting up node `nodeVersion` (`publishes`: it runs `lerna publish`, the shape of the publish workflow). */
function workflow(root, file, { nodeVersion = '22', publishes = false } = {}) {
  const steps = [`      - uses: actions/setup-node@v4\n        with:\n          node-version: '${nodeVersion}'\n`];
  if (publishes) steps.push('      - run: npx lerna publish --yes --no-private\n');
  fs.mkdirSync(path.join(root, '.github/workflows'), { recursive: true });
  fs.writeFileSync(path.join(root, '.github/workflows', file), `jobs:\n  build:\n    steps:\n${steps.join('')}`);
}

/** An npm lockfile v3: the root block mirrors `deps`; each name in `installed` has a node_modules entry at that version. */
function lock(name, version, deps, installed = {}) {
  const packages = { '': { name, version, ...(Object.keys(deps).length ? { dependencies: deps } : {}) } };
  for (const [dep, v] of Object.entries(installed)) {
    packages[`node_modules/${dep}`] = {
      version: v,
      resolved: `https://registry.invalid/${dep}/-/${dep.split('/').pop()}-${v}.tgz`,
      integrity: `sha512-${'A'.repeat(86)}==`,
    };
  }
  return { name, version, lockfileVersion: 3, requires: true, packages };
}

function write(root, rel, json) {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(json, null, 2) + '\n');
}

function readLock(root, rel) {
  return JSON.parse(fs.readFileSync(path.join(root, rel, 'package-lock.json'), 'utf8'));
}

/**
 * The stub npm: `--version` answers STUB_NPM_VERSION; `i --package-lock-only` (the registry) logs its
 * cwd and, from the STUB_STAMP_ON-th call on, rewrites the lock's installed entry to STUB_STAMP_TO —
 * a registry that serves the publish only after a lag; everything else is the real npm on PATH.
 * `refuse: '<name>@<range>'`: before the STUB_STAMP_ON-th call the stamp is REFUSED the way npm
 * refuses a floor the registry does not serve yet — `code ETARGET` + `notarget No matching version
 * found for <name>@<range>` on stderr, exit 1, the lock untouched.
 */
function stubNpm(root, { version = '10.9.8', stampOn = 1, stampTo = '2.0.0', refuse = '' } = {}) {
  const dir = path.join(root, '.stub');
  fs.mkdirSync(dir, { recursive: true });
  const log = path.join(dir, 'calls.log');
  fs.writeFileSync(log, '');
  fs.writeFileSync(
    path.join(dir, 'stamp.js'),
    `const fs=require('fs');const f='package-lock.json';const l=JSON.parse(fs.readFileSync(f,'utf8'));
for (const [k,e] of Object.entries(l.packages)) if (k.startsWith('node_modules/')) e.version=process.env.STUB_STAMP_TO;
fs.writeFileSync(f, JSON.stringify(l,null,2)+'\\n');\n`
  );
  const bin = path.join(dir, 'npm');
  fs.writeFileSync(
    bin,
    `#!/bin/sh
case "$1" in
  --version) echo "$STUB_NPM_VERSION" ;;
  i) echo "$PWD" >> "$STUB_LOG"; n=$(wc -l < "$STUB_LOG" | tr -d ' ')
     if [ "$n" -ge "$STUB_STAMP_ON" ]; then node "$STUB_DIR/stamp.js"
     elif [ -n "$STUB_REFUSE" ]; then echo "npm error code ETARGET" >&2; echo "npm error notarget No matching version found for $STUB_REFUSE." >&2; exit 1; fi ;;
  *) exec npm "$@" ;;
esac
`
  );
  fs.chmodSync(bin, 0o755);
  const env = {
    ...process.env,
    STUB_NPM_VERSION: version,
    STUB_LOG: log,
    STUB_STAMP_ON: String(stampOn),
    STUB_STAMP_TO: stampTo,
    STUB_REFUSE: refuse,
    STUB_DIR: dir,
  };
  return { bin, env, calls: () => fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) };
}

/** A fake clock: `now` reads it, `sleep` advances it — the retry's arithmetic exact, no real seconds spent. */
function clock() {
  let t = 1000;
  return { now: () => t, sleep: (ms) => (t += ms), elapsed: () => t - 1000 };
}

const quiet = () => {};

test('torn: the lock installs 1.0.0 under a ^2.0.0 floor — the guard reds naming the tree, the floor and the installed version; exit 1', () => {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const floors = new LockFloors({ repoRoot: root, log: quiet });
  const result = floors.check();
  assert.equal(result.ok, false);
  assert.deepEqual(
    result.trees.map((t) => [t.label, t.ok]),
    [
      ['root', true],
      ['packages/a', false],
    ]
  );
  const a = result.trees[1];
  assert.deepEqual(a.floors, [{ name: DEP, section: 'dependencies', declared: '^2.0.0', installed: '1.0.0' }]);
  assert.ok(a.npmError.length > 0, 'npm said why');
  const report = LockFloors.report(result);
  assert.match(report, /TORN {2}packages\/a — @proteinjs\/fixture-common \^2\.0\.0 \(lock installs 1\.0\.0\)/);
  assert.match(report, /LOCK FLOORS: FAIL — packages\/a: package-lock\.json does not satisfy package\.json/);
  assert.match(report, /--restamp/);
  assert.equal(floors.run(), EXIT.TORN);
});

test('coherent: the lock installs 2.0.0 under ^2.0.0 — green with an empty cache, an unroutable registry and no node_modules; exit 0', () => {
  const root = repo({ declared: '^2.0.0', installed: '2.0.0' });
  const floors = new LockFloors({
    repoRoot: root,
    log: quiet,
    env: { ...process.env, npm_config_cache: path.join(root, '.empty-cache') },
  });
  const result = floors.check();
  assert.equal(result.ok, true);
  assert.equal(result.trees[1].floors[0].installed, '2.0.0');
  assert.match(
    LockFloors.report(result),
    /ok {4}packages\/a — @proteinjs\/fixture-common \^2\.0\.0\n.*LOCK FLOORS: PASS — 2 tree\(s\)/s
  );
  assert.equal(floors.run(), EXIT.OK);
});

test("the trees: lerna's packages/** finds every dir with a package.json below packages/ (nested too), never node_modules; a dir without a lock is skipped, not judged; the diagnosis lists the repo's own scope, read from its packages", () => {
  const root = repo({ declared: '^2.0.0', installed: '2.0.0' });
  write(root, 'packages/nested/b/package.json', { name: '@proteinjs/fixture-b', version: '1.0.0' });
  write(root, 'packages/a/node_modules/x/package.json', { name: 'x', version: '1.0.0' });
  fs.mkdirSync(path.join(root, 'packages/empty'), { recursive: true });
  const floors = new LockFloors({ repoRoot: root, log: quiet });
  const result = floors.check();
  assert.deepEqual(
    result.trees.map((t) => t.rel),
    ['.', 'packages/a', 'packages/nested/b']
  );
  assert.equal(result.trees[2].skipped, true);
  assert.equal(result.ok, true);
  assert.match(LockFloors.report(result), /-- {4}packages\/nested\/b: no package-lock\.json \(skipped\)/);
  assert.deepEqual(
    LockFloors.scopesOf(floors.loadTrees()),
    ['@proteinjs/'],
    "the trees' own scope; the unscoped root adds none"
  );
});

test('--restamp: only the torn tree is stamped, the registry serving the publish on the second round — green, stamped names packages/a, rounds 2, the root untouched', () => {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const stub = stubNpm(root, { version: '10.9.8', stampOn: 2, stampTo: '2.0.0' });
  const c = clock();
  const floors = new LockFloors({
    repoRoot: root,
    npmBin: stub.bin,
    ciNpmMajor: 10,
    env: stub.env,
    log: quiet,
    sleep: c.sleep,
    now: c.now,
  });
  const result = floors.restamp({ budgetMs: 60000, waitMs: 20000 });
  assert.equal(result.ok, true);
  assert.deepEqual(result.stamped, ['packages/a']);
  assert.equal(result.rounds, 2);
  assert.deepEqual(
    stub.calls(),
    [path.join(root, 'packages/a'), path.join(root, 'packages/a')],
    'two stamps, both in packages/a, none at the root'
  );
  assert.equal(readLock(root, 'packages/a').packages[`node_modules/${DEP}`].version, '2.0.0');
  assert.equal(new LockFloors({ repoRoot: root, log: quiet }).check().ok, true, 'the real npm agrees afterwards');
  assert.match(LockFloors.report(result), /stamped packages\/a \(2 rounds\)/);
});

test('--restamp: a registry that never serves the publish — the clock runs out (60 s at 20 s: rounds at 0, 20, 40 and a last one at 60), still torn, exit 1, nothing pretended', () => {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const stub = stubNpm(root, { version: '10.9.8', stampOn: 99 });
  const c = clock();
  const floors = new LockFloors({
    repoRoot: root,
    npmBin: stub.bin,
    ciNpmMajor: 10,
    env: stub.env,
    log: quiet,
    sleep: c.sleep,
    now: c.now,
  });
  const result = floors.restamp({ budgetMs: 60000, waitMs: 20000 });
  assert.equal(result.ok, false);
  assert.equal(result.rounds, 4);
  assert.equal(stub.calls().length, 4);
  assert.equal(c.elapsed(), 60000, 'rounds at 0, 20, 40 and 60 s: the last at the deadline, none past it');
  assert.equal(readLock(root, 'packages/a').packages[`node_modules/${DEP}`].version, '1.0.0');
  assert.equal(floors.run({ restamp: true, budgetMs: 60000, waitMs: 20000 }), EXIT.TORN);
});

test('--restamp: the registry refuses the first stamp (ETARGET: the publish not served yet) and serves on the second — the refusal is retried on the clock, never thrown: stamped, green, exit 0; a registry refusing until the clock runs out: still torn, exit 1, the report naming the floor it never served', () => {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const stub = stubNpm(root, { version: '10.9.8', stampOn: 2, stampTo: '2.0.0', refuse: `${DEP}@^2.0.0` });
  const c = clock();
  const floors = new LockFloors({
    repoRoot: root,
    npmBin: stub.bin,
    ciNpmMajor: 10,
    env: stub.env,
    log: quiet,
    sleep: c.sleep,
    now: c.now,
  });
  const result = floors.restamp({ budgetMs: 300000, waitMs: 20000 });
  assert.equal(result.ok, true);
  assert.deepEqual(result.stamped, ['packages/a']);
  assert.equal(result.rounds, 2);
  assert.equal(stub.calls().length, 2, 'the refused stamp, then the one the registry served');
  assert.equal(c.elapsed(), 20000, 'one wait between the two rounds');
  assert.equal(readLock(root, 'packages/a').packages[`node_modules/${DEP}`].version, '2.0.0');
  assert.equal(new LockFloors({ repoRoot: root, log: quiet }).check().ok, true, 'the real npm agrees afterwards');

  const never = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const late = stubNpm(never, { version: '10.9.8', stampOn: 99, refuse: `${DEP}@^2.0.0` });
  const c2 = clock();
  const f2 = new LockFloors({
    repoRoot: never,
    npmBin: late.bin,
    ciNpmMajor: 10,
    env: late.env,
    log: quiet,
    sleep: c2.sleep,
    now: c2.now,
  });
  const r2 = f2.restamp({ budgetMs: 60000, waitMs: 20000 });
  assert.equal(r2.ok, false);
  assert.equal(r2.rounds, 4, 'rounds at 0, 20, 40 and 60 s on the clock: the last at the deadline, none past it');
  assert.equal(late.calls().length, 4);
  assert.equal(c2.elapsed(), 60000, 'three waits; the clock never slept past its bound');
  assert.equal(readLock(never, 'packages/a').packages[`node_modules/${DEP}`].version, '1.0.0', 'nothing pretended');
  const report = LockFloors.report(r2);
  assert.match(report, /TORN {2}packages\/a — @proteinjs\/fixture-common \^2\.0\.0 \(lock installs 1\.0\.0\)/);
  assert.match(report, /last stamp: code ETARGET/);
  assert.match(report, /last stamp: notarget No matching version found for @proteinjs\/fixture-common@\^2\.0\.0/);
  assert.equal(f2.run({ restamp: true, budgetMs: 60000, waitMs: 20000 }), EXIT.TORN);
});

test('--restamp under the wrong npm major is refused before any stamp (64): lock stamps are npm-major dependent; the guard itself does not care', () => {
  const root = repo({ declared: '^2.0.0', installed: '1.0.0' });
  const stub = stubNpm(root, { version: '11.6.2' });
  const floors = new LockFloors({
    repoRoot: root,
    npmBin: stub.bin,
    ciNpmMajor: 10,
    env: stub.env,
    log: quiet,
    sleep: () => {},
  });
  assert.throws(() => floors.restamp({ budgetMs: 0, waitMs: 0 }), NpmMajorMismatch);
  assert.equal(stub.calls().length, 0, 'no stamp ran');
  assert.equal(floors.run({ restamp: true, budgetMs: 0, waitMs: 0 }), EXIT.NPM_MAJOR);
  assert.equal(floors.check().ok, false, 'the guard still judges under any npm');
});

test("the publish workflow alone (no reusable build/test workflow): after packages/common's publish tore packages/server (^2.0.0 declared, 1.0.0 installed), the workflow's --restamp reads CI's npm major from that one workflow's node-version, stamps the consumer, and its lock installs the published 2.0.0 — exit 0", () => {
  const root = publishedRepo();
  workflow(root, 'publish.js.yml', { nodeVersion: '22', publishes: true });
  assert.equal(readLock(root, 'packages/server').packages[`node_modules/${DEP}`].version, '1.0.0', 'torn before');
  const stub = stubNpm(root, { version: '10.9.8', stampOn: 1, stampTo: '2.0.0' });
  const floors = new LockFloors({ repoRoot: root, npmBin: stub.bin, env: stub.env, log: quiet, sleep: () => {} });
  assert.equal(floors.run({ restamp: true, budgetMs: 0, waitMs: 0 }), EXIT.OK);
  assert.deepEqual(
    stub.calls(),
    [path.join(root, 'packages/server')],
    'one stamp, in the torn consumer only — never the published package, never the root'
  );
  assert.equal(
    readLock(root, 'packages/server').packages[`node_modules/${DEP}`].version,
    '2.0.0',
    "the lock's installed version is the publish's"
  );
  assert.equal(new LockFloors({ repoRoot: root, log: quiet }).check().ok, true, 'the real npm agrees afterwards');
});

test("CI's npm major is read from every workflow under .github/workflows/: the publish workflow beside the reusable build/test workflow it calls agree on node 22 -> npm 10; a workflow on another major refuses naming both; no workflow refuses", () => {
  const root = repo();
  workflow(root, 'publish.js.yml', { nodeVersion: '22', publishes: true });
  workflow(root, '_test.yml', { nodeVersion: '22' });
  assert.equal(new LockFloors({ repoRoot: root, log: quiet }).readCiNpmMajor(), 10);
  workflow(root, 'nightly.yml', { nodeVersion: '24' });
  assert.throws(
    () => new LockFloors({ repoRoot: root, log: quiet }).readCiNpmMajor(),
    /more than one npm major — npm 10: _test\.yml \(node 22\), publish\.js\.yml \(node 22\); npm 11: nightly\.yml \(node 24\)/
  );
  fs.rmSync(path.join(root, '.github'), { recursive: true });
  assert.throws(
    () => new LockFloors({ repoRoot: root, log: quiet }).readCiNpmMajor(),
    /cannot read node-version from any workflow/
  );
});
