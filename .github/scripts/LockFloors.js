#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

/**
 * LockFloors — every lerna tree's package-lock.json agrees with its package.json, judged the way
 * `npm ci` judges it; `--restamp` makes the torn trees agree.
 *
 * THE TEAR: lerna's release commit raises each sibling consumer's floor when a package it depends on
 * is published from the same repo (packages/server and packages/ui: `@proteinjs/user` ^1.19.2 ->
 * ^1.20.0) and mirrors the range into the lock's root block, but never re-resolves the lock's
 * INSTALLED entry (`node_modules/@proteinjs/user`: version, resolved, integrity) — the tarball it
 * would resolve to does not exist until the publish that commit precedes (lerna 8.1.2's
 * update-lockfile-version.js is a JSON rewrite; its one `npm install --package-lock-only` runs on the
 * root lock, which declares no package). `npm ci` in those packages then refuses (EUSAGE: "lock
 * file's @proteinjs/user@1.19.2 does not satisfy @proteinjs/user@1.20.0") until someone re-stamps by
 * hand; the workflow's own per-package `npm i` tolerates the tear and its lock commit runs BEFORE the
 * publish, so main stays torn from each publish to the next.
 *
 * Two doors, one verdict:
 *   node .github/scripts/LockFloors.js            the guard — `npm ci --dry-run --prefer-offline`
 *                                                 per tree: npm's OWN package.json-vs-lock validation
 *                                                 (nothing installed, nothing written; the cache first,
 *                                                 the registry only for a packument the cache lacks —
 *                                                 `--offline` reds ENOTCACHED on a peer edge npm
 *                                                 re-resolves), ~0.3 s a tree; exit 1 naming each torn
 *                                                 tree with its own-scope floors (declared / lock
 *                                                 installs).
 *   node .github/scripts/LockFloors.js --restamp  the fix — `npm i --package-lock-only
 *                                                 --ignore-scripts --no-audit --no-fund` in the torn
 *                                                 trees only, judged again; a stamp the registry
 *                                                 refuses because it does not serve the floor yet
 *                                                 (ETARGET / E404 — the resolver lags the publish's
 *                                                 accept by minutes) is retried 20 s apart on one
 *                                                 5-minute clock (`--retry <s>`), then reported torn
 *                                                 naming the floor it never served; any other stamp
 *                                                 failure stops it (2). The publish workflow
 *                                                 runs it right after `lerna publish` and commits the
 *                                                 locks as the release's second commit. Refused (64)
 *                                                 under any npm major but CI's (read from the
 *                                                 workflows' node-version: node 22 -> npm 10): lock
 *                                                 stamps are npm-major dependent, so a node 24 / npm 11
 *                                                 shell shims — PATH an npm 10 first, or
 *                                                 LOCK_FLOORS_NPM=<an npm 10 binary>. A hand commit of
 *                                                 the locks never quotes the skip-ci token the
 *                                                 workflow's own lock commit carries: a push whose head
 *                                                 commit carries it runs no workflow.
 *   --root <dir>   another checkout / a fixture.
 * Exit 0 coherent · 1 torn (after --restamp: still torn) · 2 a tree unreadable, npm unrunnable, CI's
 * node unreadable · 64 --restamp under the wrong npm major.
 *
 * Judged by npm, never re-implemented: the guard IS `npm ci`'s validation — the failure a consumer
 * hits — so it cannot drift from it. The only reading of the lock here is the diagnosis (each
 * dependency in the repo's own scope: the declared range against the lock's installed version) so a
 * red names the numbers.
 */

const CHECK_ARGS = ['ci', '--dry-run', '--prefer-offline', '--ignore-scripts', '--no-audit', '--no-fund'];
const STAMP_ARGS = ['i', '--package-lock-only', '--ignore-scripts', '--no-audit', '--no-fund'];
const SECTIONS = ['dependencies', 'devDependencies', 'optionalDependencies'];
/** Where CI's node is declared: every workflow's `node-version` (the publish workflow; the reusable build/test workflow it calls, when the repo has one). */
const WORKFLOWS_DIR = path.join('.github', 'workflows');
/** actions/setup-node's major -> the npm major it ships. */
const NODE_TO_NPM_MAJOR = { 16: 8, 18: 10, 20: 10, 22: 10, 24: 11 };
/** --restamp's retry: one clock shared by every torn tree, the rounds this far apart. */
const DEFAULT_RETRY_MS = 5 * 60 * 1000;
const DEFAULT_RETRY_WAIT_MS = 20 * 1000;
/** npm's codes for "the registry serves no such version (yet)": ETARGET (the packument lacks the version), E404 (no packument — a package's first publish). A stamp refused with one is retried; any other failure is thrown. */
const REGISTRY_LAG_CODES = /^code (ETARGET|E404)$/;
const EXIT = { OK: 0, TORN: 1, FAIL: 2, NPM_MAJOR: 64 };

class NpmMajorMismatch extends Error {}

class LockFloors {
  constructor({ repoRoot, npmBin, ciNpmMajor, env = process.env, log = console.log, sleep, now } = {}) {
    this.repoRoot = repoRoot || path.resolve(__dirname, '..', '..');
    this.npmBin = npmBin || 'npm';
    this.ciNpmMajor = ciNpmMajor; // read from the workflows when --restamp needs it
    this.env = env;
    this.log = log;
    this.sleep = sleep || ((ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms));
    this.now = now || Date.now;
  }

  /** The CLI: the verdict printed, the exit code returned. */
  run({ restamp = false, budgetMs, waitMs } = {}) {
    try {
      const result = restamp ? this.restamp({ budgetMs, waitMs }) : this.check();
      this.log(LockFloors.report(result));
      return result.ok ? EXIT.OK : EXIT.TORN;
    } catch (err) {
      this.log(`LockFloors: ${err.message}`);
      return err instanceof NpmMajorMismatch ? EXIT.NPM_MAJOR : EXIT.FAIL;
    }
  }

  /** The guard: every tree judged. Returns { ok, trees: [{ rel, label, ok, skipped, floors, npmError }] }. */
  check() {
    const all = this.loadTrees();
    const scopes = LockFloors.scopesOf(all);
    const trees = all.map((tree) => this.judge(tree, scopes));
    return { ok: trees.every((t) => t.ok), trees, stamped: [], rounds: 0 };
  }

  /**
   * The fix: stamp the torn trees with CI's npm major, judge again; rounds `waitMs` apart while a
   * tree stays torn and the clock (`budgetMs`, one for every tree) holds. A stamp npm refuses with
   * ETARGET / E404 is the registry not yet serving the publish — the tree stays torn for the next
   * round and the refusal is kept for the report, never thrown; any other stamp failure throws.
   * Never touches a tree that agrees. Returns the final verdict + { stamped: [rel], rounds }, each
   * still-torn tree carrying `stampError` (npm's lines from its last refused stamp).
   */
  restamp({ budgetMs = DEFAULT_RETRY_MS, waitMs = DEFAULT_RETRY_WAIT_MS } = {}) {
    const npm = this.requireCiNpm();
    const stamped = new Set();
    const refused = new Map(); // rel -> npm's lines from the last stamp the registry refused
    const start = this.now();
    let result = this.check();
    let rounds = 0;
    while (!result.ok && (rounds === 0 || this.now() - start < budgetMs)) {
      rounds += 1;
      if (rounds > 1) {
        const left = budgetMs - (this.now() - start);
        const wait = Math.min(waitMs, left);
        this.log(
          `round ${rounds}: still torn — waiting ${Math.round(wait / 1000)} s for the registry (${Math.round(left / 1000)} s left on the ${Math.round(budgetMs / 1000)} s clock)`
        );
        this.sleep(wait);
      }
      for (const tree of result.trees.filter((t) => !t.ok)) {
        this.log(`stamping ${tree.label} (npm ${npm.version}: ${STAMP_ARGS.join(' ')})`);
        const out = this.npm(STAMP_ARGS, tree.dir);
        if (out.status === 0) {
          stamped.add(tree.rel);
          refused.delete(tree.rel);
          continue;
        }
        const lines = LockFloors.npmErrorLines(out);
        if (!lines.some((line) => REGISTRY_LAG_CODES.test(line)))
          throw new Error(`npm ${STAMP_ARGS.join(' ')} failed in ${tree.label}:\n${lines.join('\n')}`);
        refused.set(tree.rel, lines);
        this.log(
          `${tree.label}: the registry does not serve a floor yet (${lines.join('; ')}) — retried while the clock holds`
        );
      }
      result = this.check();
    }
    return {
      ...result,
      trees: result.trees.map((t) => ({ ...t, stampError: t.ok ? [] : refused.get(t.rel) || [] })),
      stamped: [...stamped],
      rounds,
    };
  }

  static report({ ok, trees, stamped = [], rounds = 0 }) {
    const lines = [];
    for (const t of trees) {
      if (t.skipped) {
        lines.push(`  --    ${t.label}: no package-lock.json (skipped)`);
        continue;
      }
      const floors = t.floors.map(
        (f) =>
          `${f.name} ${f.declared}${f.installed === f.declared.replace(/^[\^~]/, '') ? '' : ` (lock installs ${f.installed})`}`
      );
      lines.push(`  ${t.ok ? 'ok  ' : 'TORN'}  ${t.label}${floors.length ? ` — ${floors.join(', ')}` : ''}`);
      if (!t.ok) for (const l of t.npmError) lines.push(`          ${l}`);
      if (!t.ok) for (const l of t.stampError || []) lines.push(`          last stamp: ${l}`);
    }
    if (stamped.length) lines.push(`  stamped ${stamped.join(', ')} (${rounds} round${rounds === 1 ? '' : 's'})`);
    const torn = trees.filter((t) => !t.ok).map((t) => t.label);
    lines.push(
      ok
        ? `LOCK FLOORS: PASS — ${trees.filter((t) => !t.skipped).length} tree(s): every package-lock.json satisfies its package.json (npm ci would install)`
        : `LOCK FLOORS: FAIL — ${torn.join(', ')}: package-lock.json does not satisfy package.json (npm ci refuses, EUSAGE). Fix: \`node .github/scripts/LockFloors.js --restamp\` under CI's npm major (node 22 -> npm 10; on a node 24 / npm 11 shell PATH an npm 10 first or set LOCK_FLOORS_NPM), then commit the locks under a message that never quotes the skip-ci token — a push whose head commit carries it runs no workflow (the token belongs to the workflow's own lock commit alone).`
    );
    return lines.join('\n');
  }

  // ---- one tree ---------------------------------------------------------------

  judge(tree, scopes) {
    if (!fs.existsSync(path.join(tree.dir, 'package-lock.json')))
      return { ...tree, ok: true, skipped: true, floors: [], npmError: [] };
    const floors = this.floors(tree, scopes);
    const out = this.npm(CHECK_ARGS, tree.dir);
    if (out.error) throw new Error(`cannot run ${this.npmBin} in ${tree.label}: ${out.error.message}`);
    return {
      ...tree,
      ok: out.status === 0,
      skipped: false,
      floors,
      npmError: out.status === 0 ? [] : LockFloors.npmErrorLines(out),
    };
  }

  /** The diagnosis: each own-scope dependency's declared range beside what the lock installs for it. */
  floors(tree, scopes) {
    const lock = this.readJson(path.join(tree.dir, 'package-lock.json'));
    const packages = lock.packages || {};
    const floors = [];
    for (const section of SECTIONS) {
      for (const [name, declared] of Object.entries(tree.pkg[section] || {})) {
        if (!scopes.some((scope) => name.startsWith(scope))) continue;
        let entry = packages[`node_modules/${name}`];
        if (entry && entry.link && entry.resolved) entry = packages[entry.resolved];
        floors.push({ name, section, declared, installed: entry && entry.version ? entry.version : 'nothing' });
      }
    }
    return floors;
  }

  // ---- inputs: the trees, their scopes, npm, CI's npm major -------------------------

  /** Root + every package dir lerna.json names (`packages/**` here: each dir under packages/ with a package.json). */
  loadTrees() {
    const lerna = this.readJson(path.join(this.repoRoot, 'lerna.json'));
    const rels = ['.'];
    for (const pattern of lerna.packages || [])
      for (const rel of this.expand(pattern)) if (!rels.includes(rel)) rels.push(rel);
    return rels.map((rel) => {
      const dir = path.join(this.repoRoot, rel);
      return { rel, dir, label: rel === '.' ? 'root' : rel, pkg: this.readJson(path.join(dir, 'package.json')) };
    });
  }

  /** The repo's own scopes (`@proteinjs/` from `@proteinjs/user`), read from the trees' names: the dependencies the diagnosis lists. Sorted. */
  static scopesOf(trees) {
    const scopes = new Set();
    for (const tree of trees) {
      const m = /^(@[^/]+\/)/.exec(tree.pkg.name || '');
      if (m) scopes.add(m[1]);
    }
    return [...scopes].sort();
  }

  /** lerna's package globs: a literal dir, `<dir>/*` (its child dirs), `<dir>/**` (every dir below it, node_modules and dot dirs excluded) — kept when a package.json sits there. Sorted. */
  expand(pattern) {
    const segments = pattern.split('/').filter(Boolean);
    let dirs = [''];
    for (const segment of segments) {
      const next = [];
      for (const dir of dirs) {
        if (segment === '**') next.push(dir, ...this.walk(dir));
        else if (segment === '*') next.push(...this.children(dir));
        else next.push(path.join(dir, segment));
      }
      dirs = next;
    }
    return [...new Set(dirs)].filter((d) => d && fs.existsSync(path.join(this.repoRoot, d, 'package.json'))).sort();
  }

  children(dir) {
    const abs = path.join(this.repoRoot, dir);
    if (!fs.existsSync(abs)) return [];
    return fs
      .readdirSync(abs, { withFileTypes: true })
      .filter((e) => e.isDirectory() && e.name !== 'node_modules' && !e.name.startsWith('.'))
      .map((e) => path.join(dir, e.name));
  }

  walk(dir) {
    const out = [];
    for (const child of this.children(dir)) out.push(child, ...this.walk(child));
    return out;
  }

  npm(args, cwd) {
    return spawnSync(this.npmBin, args, { cwd, encoding: 'utf8', env: this.env });
  }

  /** --restamp's npm must be CI's major (the workflows' node-version -> NODE_TO_NPM_MAJOR): the stamp is npm-major dependent. */
  requireCiNpm() {
    const expected = this.ciNpmMajor === undefined ? this.readCiNpmMajor() : this.ciNpmMajor;
    const probe = this.npm(['--version'], this.repoRoot);
    if (probe.error || probe.status !== 0)
      throw new Error(
        `cannot run \`${this.npmBin} --version\`: ${(probe.error && probe.error.message) || (probe.stderr || '').trim()}`
      );
    const version = probe.stdout.trim().split('\n').pop();
    const major = Number(version.split('.')[0]);
    if (major !== expected) {
      throw new NpmMajorMismatch(
        `refusing to stamp with npm ${version} (${this.npmBin}): CI stamps with npm ${expected} and lock stamps are npm-major dependent — PATH an npm ${expected} first or set LOCK_FLOORS_NPM to one (\`printf '#!/bin/sh\\nexec npx -y npm@${expected} "$@"\\n' > /tmp/npm${expected}.sh && chmod +x /tmp/npm${expected}.sh\`)`
      );
    }
    return { version, major };
  }

  /**
   * CI's npm major: the `node-version` the workflows under .github/workflows/ set up — the publish
   * workflow alone, or the publish workflow beside the reusable build/test workflow it calls — mapped
   * through NODE_TO_NPM_MAJOR. Read from the tree, never declared here: one major, or a refusal naming
   * the files (a repo whose workflows run two npm majors has no single stamp to agree with).
   */
  readCiNpmMajor() {
    const dir = path.join(this.repoRoot, WORKFLOWS_DIR);
    const files = fs.existsSync(dir)
      ? fs
          .readdirSync(dir)
          .filter((f) => /\.ya?ml$/.test(f))
          .sort()
      : [];
    const majors = new Map(); // npm major -> ['<file> (node <n>)']
    for (const file of files) {
      const text = fs.readFileSync(path.join(dir, file), 'utf8');
      for (const m of text.matchAll(/node-version:\s*['"]?(\d+)/g)) {
        const node = Number(m[1]);
        const npmMajor = NODE_TO_NPM_MAJOR[node];
        if (!npmMajor)
          throw new Error(
            `no npm major known for node ${node} (${WORKFLOWS_DIR}/${file}) — extend NODE_TO_NPM_MAJOR in .github/scripts/LockFloors.js`
          );
        const at = `${file} (node ${node})`;
        if (!majors.has(npmMajor)) majors.set(npmMajor, []);
        if (!majors.get(npmMajor).includes(at)) majors.get(npmMajor).push(at);
      }
    }
    if (majors.size === 0)
      throw new Error(
        `cannot read node-version from any workflow under ${WORKFLOWS_DIR} (${files.join(', ') || 'no workflows'})`
      );
    if (majors.size > 1)
      throw new Error(
        `the workflows under ${WORKFLOWS_DIR} set up more than one npm major — ${[...majors].map(([npm, at]) => `npm ${npm}: ${at.join(', ')}`).join('; ')} — no single major to stamp with`
      );
    return [...majors.keys()][0];
  }

  /** npm's own lines, prefix stripped, the usage block dropped: the code + every Invalid/Missing line. */
  static npmErrorLines(out) {
    const lines = [];
    for (const raw of (out.stderr || '').split('\n')) {
      const m = raw.match(/^npm (?:error|ERR!) ?(.*)$/);
      if (!m) continue;
      const line = m[1].trim();
      if (/^(Clean install a project|Usage:|Options:|aliases:|Run "npm help|A complete log)/.test(line)) break;
      if (line) lines.push(line);
    }
    return lines.slice(0, 12);
  }

  readJson(file) {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  }
}

module.exports = {
  LockFloors,
  EXIT,
  CHECK_ARGS,
  STAMP_ARGS,
  NpmMajorMismatch,
  DEFAULT_RETRY_MS,
  DEFAULT_RETRY_WAIT_MS,
};

if (require.main === module) {
  const args = process.argv.slice(2);
  const opts = { restamp: false, root: undefined, budgetMs: undefined };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--restamp') opts.restamp = true;
    else if (arg === '--root') opts.root = path.resolve(args[(i += 1)] || '');
    else if (arg.startsWith('--root=')) opts.root = path.resolve(arg.slice('--root='.length));
    else if (arg === '--retry') {
      const seconds = Number(args[(i += 1)]);
      if (!Number.isFinite(seconds) || seconds < 0) {
        console.error(`LockFloors: --retry wants a number of seconds, got ${JSON.stringify(args[i])}`);
        process.exit(EXIT.FAIL);
      }
      opts.budgetMs = seconds * 1000;
    } else {
      console.error(`LockFloors: unknown argument ${arg}`);
      process.exit(EXIT.FAIL);
    }
  }
  process.exit(new LockFloors({ repoRoot: opts.root, npmBin: process.env.LOCK_FLOORS_NPM }).run(opts));
}
