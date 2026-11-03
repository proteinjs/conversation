#!/usr/bin/env node
const { spawnSync } = require('child_process');

/**
 * MintServed — the registry serves every package a publish just minted, or the wait says which one it
 * does not: the gate the consumer-lock re-stamp stands behind.
 *
 * THE RACE: `lerna publish` returns when the registry has ACCEPTED each tarball, not when it SERVES the
 * new version to a resolver — the packument a `npm i` reads lags the accept by seconds to minutes
 * (run 36295867264: the re-stamp asked for @proteinjs/db-file@^1.10.0 at 05:04:51Z and got ETARGET
 * "No matching version found"; the registry served 1.10.0 about 40 s later). LockFloors.js --restamp
 * then threw on that ETARGET, so the re-stamp step went red and main stayed torn until a hand stamp.
 *
 * THE WAIT: for every mint — the `<name>@<version>` tags lerna's release commit carries (`git tag
 * --points-at HEAD` right after `lerna publish`), or the mints named on the command line — poll
 * `npm view <name>@<version> version` until it answers the version, 10 s apart, one 15-minute clock
 * for the whole mint (a lag is the registry's, shared by every package of the publish; one bound
 * keeps the finalize job inside its own). One line per package ("served after N s"); a package the
 * registry still does not serve at the deadline is a WARNING line naming it, and the caller goes on
 * to the re-stamp anyway — LockFloors is the judge, this is the wait.
 *
 * THE CLOCK'S SIZE: the lag grows with the publish. A ten-package publish (run 36818699801) had its
 * first package served 63 s after the accept and its last about 450 s after it, past the 5-minute
 * clock the wait first shipped with; the re-stamp that followed hit ETARGET on that package and no
 * lock commit landed. Fifteen minutes covers the observed lag twice over; the finalize job's own
 * bound (timeout-minutes in the publish workflow) is sized to hold this wait plus the re-stamp's
 * retry clock plus the push. LockFloors --restamp retries a stamp the registry refuses (ETARGET /
 * E404) on its own clock, so a package served a minute after this wait expired still lands.
 *
 *   node .github/scripts/MintServed.js                        every tag at HEAD
 *   node .github/scripts/MintServed.js --ref <sha>            every tag at <sha>
 *   node .github/scripts/MintServed.js <name>@<version> …     these mints (a hand run at any ref)
 *   --timeout <s>  the shared clock (default 900) · --interval <s>  between polls (default 10)
 * Exit 0 every mint served · 1 a mint not served at the deadline (warned) · 2 git or npm unrunnable.
 * The publish workflow runs it under `|| [ $? -eq 1 ]`: a timeout goes on to the re-stamp, an
 * unrunnable wait stops the (best-effort) step.
 */

const VIEW_ARGS = ['view', '--prefer-online', '--no-audit', '--no-fund'];
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_INTERVAL_MS = 10 * 1000;
const EXIT = { OK: 0, TIMED_OUT: 1, FAIL: 2 };
/** lerna's independent-mode tag: `<name>@<version>` where the name may carry a scope (`@scope/name@1.2.3`). */
const TAG = /^(@?[^@]+)@(\d+\.\d+\.\d+[^@]*)$/;

class MintServed {
  constructor({ repoRoot, npmBin, env = process.env, log = console.log, sleep, now } = {}) {
    this.repoRoot = repoRoot || process.cwd();
    this.npmBin = npmBin || 'npm';
    this.env = env;
    this.log = log;
    this.sleep = sleep || ((ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms));
    this.now = now || Date.now;
  }

  /** The CLI: the mints named, else the tags at `ref`; the lines printed, the exit code returned. */
  run({ mints = [], ref = 'HEAD', timeoutMs, intervalMs } = {}) {
    try {
      const list = mints.length ? mints.map(MintServed.parse) : this.tagged(ref);
      if (!list.length) {
        this.log(`no mint at ${ref} (no <name>@<version> tag points at it) — nothing to wait for`);
        return EXIT.OK;
      }
      const result = this.wait(list, { timeoutMs, intervalMs });
      return result.timedOut.length ? EXIT.TIMED_OUT : EXIT.OK;
    } catch (err) {
      this.log(`MintServed: ${err.message}`);
      return EXIT.FAIL;
    }
  }

  /**
   * The wait: each mint polled until the registry answers its version, `intervalMs` apart, on one
   * clock of `timeoutMs` shared by the whole list. Returns { served: [{ name, version, afterMs }],
   * timedOut: [{ name, version }] } — a mint past the deadline is checked once more, never waited for.
   */
  wait(mints, { timeoutMs = DEFAULT_TIMEOUT_MS, intervalMs = DEFAULT_INTERVAL_MS } = {}) {
    const served = [];
    const timedOut = [];
    const start = this.now();
    for (const mint of mints) {
      let answered = this.served(mint);
      while (!answered && this.now() - start < timeoutMs) {
        this.sleep(Math.min(intervalMs, Math.max(0, timeoutMs - (this.now() - start))));
        answered = this.served(mint);
      }
      const afterMs = this.now() - start;
      if (answered) {
        served.push({ ...mint, afterMs });
        this.log(`${mint.name}@${mint.version}: served after ${Math.round(afterMs / 1000)} s`);
      } else {
        timedOut.push({ ...mint });
        this.log(
          `::warning::${mint.name}@${mint.version}: the registry does not serve it after ${Math.round(timeoutMs / 1000)} s — a consumer lock re-stamp resolving it fails (ETARGET) until it does`
        );
      }
    }
    return { served, timedOut };
  }

  /** The mints at `ref`: every `<name>@<version>` tag pointing at it (other tags ignored), sorted. */
  tagged(ref) {
    const out = spawnSync('git', ['tag', '--points-at', ref], { cwd: this.repoRoot, encoding: 'utf8', env: this.env });
    if (out.error) throw new Error(`cannot run git tag --points-at ${ref} in ${this.repoRoot}: ${out.error.message}`);
    if (out.status !== 0)
      throw new Error(`git tag --points-at ${ref} failed in ${this.repoRoot}: ${(out.stderr || '').trim()}`);
    return out.stdout
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => TAG.test(line))
      .sort()
      .map(MintServed.parse);
  }

  /** `<name>@<version>` → { name, version }; refused when it is not one. */
  static parse(tag) {
    const m = TAG.exec(tag);
    if (!m) throw new Error(`not a <name>@<version> mint: ${JSON.stringify(tag)}`);
    return { name: m[1], version: m[2] };
  }

  // ---- one poll ----------------------------------------------------------------

  /** True when `npm view <name>@<version> version` answers exactly the version (the registry serves it). */
  served({ name, version }) {
    const out = spawnSync(this.npmBin, [...VIEW_ARGS, `${name}@${version}`, 'version'], {
      cwd: this.repoRoot,
      encoding: 'utf8',
      env: this.env,
    });
    if (out.error) throw new Error(`cannot run ${this.npmBin} view: ${out.error.message}`);
    return out.status === 0 && out.stdout.trim().split('\n').pop() === version;
  }
}

module.exports = { MintServed, EXIT, VIEW_ARGS, DEFAULT_TIMEOUT_MS, DEFAULT_INTERVAL_MS };

if (require.main === module) {
  const args = process.argv.slice(2);
  const opts = { mints: [], ref: 'HEAD', timeoutMs: undefined, intervalMs: undefined };
  const seconds = (flag, value) => {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0) {
      console.error(`MintServed: ${flag} wants a number of seconds, got ${JSON.stringify(value)}`);
      process.exit(EXIT.FAIL);
    }
    return n * 1000;
  };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--ref') opts.ref = args[(i += 1)] || 'HEAD';
    else if (arg === '--timeout') opts.timeoutMs = seconds(arg, args[(i += 1)]);
    else if (arg === '--interval') opts.intervalMs = seconds(arg, args[(i += 1)]);
    else if (arg.startsWith('--')) {
      console.error(`MintServed: unknown argument ${arg}`);
      process.exit(EXIT.FAIL);
    } else opts.mints.push(arg);
  }
  process.exit(new MintServed({ npmBin: process.env.LOCK_FLOORS_NPM }).run(opts));
}
