const fs = require('fs');
const { execFileSync } = require('child_process');

/**
 * The publish-skip condition: whether this tree was already verified by a green run of the
 * verify-train workflow (`.github/workflows/verify-train.yml`), so the publish workflow may skip
 * its test job. The reusable `_test.yml` runs this in its build job under `skip-verified: true`
 * (the publish caller) to decide; the verify-train caller never does. A release tool departing
 * the `train` branch matches a verify-train run to the train tip with the same predicate, so the
 * predicate lives here whole — change it here and copy the file, never patch it in place.
 *
 * A tree is VERIFIED when a verify-train run concluded green for it:
 *   - a run NAMED for HEAD (`run-name: verify-train <sha>` — VerifiedTree.runName; a dispatch checks
 *     out the sha in its name, never the ref tip it was dispatched on, so the NAME is the truth of
 *     what ran; a push run's head_sha is what it ran, so a push run also matches by head_sha), or
 *   - a run named for HEAD^ while HEAD changes only files under `.train/` — the departure commit:
 *     the departure consumes the branch manifests on top of the verified tip and pushes THAT (the
 *     .train/ directory belongs to no package; the published trees are identical).
 * Anything else — no run, a red or in-flight one, a dispatch named for another sha, a change outside
 * .train/, the API unreadable, git unreadable, the workflow file name not staged — is NOT verified:
 * the test jobs run. The guard errs toward testing; it never skips on doubt.
 *
 * The door (`node .github/scripts/VerifiedTree.js`): reads GITHUB_REPOSITORY, GITHUB_TOKEN (or
 * GH_TOKEN; needs `actions: read`), GITHUB_API_URL, VERIFY_WORKFLOW (the verify-train file name),
 * HEAD and HEAD^ from git (a checkout of depth >= 2), the workflow's successful runs from the REST
 * API, and appends `verified=true|false` + `verify_run=<url>` to GITHUB_OUTPUT. Prints the verdict
 * and, when verified, a `::notice::` naming the run; never a token. Exit 0 always — a broken guard
 * is a run that tests, not a run that fails.
 */
class VerifiedTree {
  /** The run-name prefix every verify-train.yml declares: `run-name: verify-train ${{ inputs.sha || github.sha }}`. */
  static RUN_NAME = 'verify-train';

  static runName(sha) {
    return `${VerifiedTree.RUN_NAME} ${sha}`;
  }

  /**
   * The verdict for `head` (its `parent`, null for a root or a shallow clone; `changed` the paths
   * HEAD^..HEAD touches) given the workflow's `runs` (REST or gh CLI shape): { verified, run, why }.
   */
  static judge({ head, parent = null, changed = [], runs = [] }) {
    const green = runs
      .map((r) => VerifiedTree.normalize(r))
      .filter((r) => r.status === 'completed' && r.conclusion === 'success');
    const of = (sha) => green.filter((r) => VerifiedTree.matches(r, sha)).sort((a, b) => b.id - a.id)[0] || null;
    const exact = of(head);
    if (exact) return { verified: true, run: exact, why: `run ${exact.url} verified HEAD ${head.slice(0, 8)}` };
    if (parent) {
      const run = of(parent);
      if (run) {
        const outside = changed.filter((f) => !f.startsWith('.train/'));
        if (!outside.length)
          return {
            verified: true,
            run,
            why: `run ${run.url} verified HEAD^ ${parent.slice(0, 8)}; HEAD ${head.slice(0, 8)} changes only .train/ (${changed.length} file(s))`,
          };
        return {
          verified: false,
          run,
          why: `run ${run.url} verified HEAD^ ${parent.slice(0, 8)} but HEAD ${head.slice(0, 8)} changes ${outside.join(', ')} outside .train/`,
        };
      }
    }
    return {
      verified: false,
      run: null,
      why: `no green verify-train run named for HEAD ${head.slice(0, 8)}${parent ? ` or HEAD^ ${parent.slice(0, 8)}` : ''} (${green.length} green run(s) read)`,
    };
  }

  /** Whether `run` verified `sha`: named for it (any event), or a push run whose head is it. */
  static matches(run, sha) {
    const r = VerifiedTree.normalize(run);
    if (r.displayTitle === VerifiedTree.runName(sha)) return true;
    return r.event === 'push' && r.headSha === sha;
  }

  /** One shape from the REST API's (`id`, `head_sha`, `display_title`, `html_url`) and gh's (`databaseId`, `headSha`, `displayTitle`, `url`). */
  static normalize(run) {
    return {
      id: run.id !== undefined ? run.id : run.databaseId,
      headSha: run.headSha !== undefined ? run.headSha : run.head_sha,
      displayTitle: run.displayTitle !== undefined ? run.displayTitle : run.display_title,
      event: run.event,
      status: run.status,
      conclusion: run.conclusion,
      url: run.url !== undefined ? run.url : run.html_url,
    };
  }

  /** The door: never throws; writes the outputs; returns the verdict. */
  static async main({ env = process.env, git = VerifiedTree.git, fetchJson = VerifiedTree.fetchJson, log = console.log } = {}) {
    let verdict;
    try {
      verdict = await VerifiedTree.inspect({ env, git, fetchJson });
    } catch (err) {
      verdict = { verified: false, run: null, why: `${err.message.split('\n')[0]} — the test jobs run` };
    }
    log(`verified-tree: ${verdict.verified ? 'VERIFIED' : 'not verified'} — ${verdict.why}`);
    if (verdict.verified) log(`::notice::test jobs skipped — ${verdict.why}`);
    VerifiedTree.output(env, verdict);
    return verdict;
  }

  static async inspect({ env, git, fetchJson }) {
    const workflow = env.VERIFY_WORKFLOW;
    if (!workflow) throw new Error('VERIFY_WORKFLOW (the verify-train workflow file name) is not staged');
    const repo = env.GITHUB_REPOSITORY;
    if (!repo) throw new Error('GITHUB_REPOSITORY is not set');
    const token = env.GITHUB_TOKEN || env.GH_TOKEN;
    if (!token) throw new Error('GITHUB_TOKEN / GH_TOKEN is not set (the runs read needs actions: read)');
    const head = git(['rev-parse', 'HEAD']).trim();
    let parent = null;
    try {
      parent = git(['rev-parse', 'HEAD^']).trim() || null;
    } catch {
      parent = null; // a root commit, or a shallow clone: HEAD alone is judged
    }
    const changed = parent
      ? git(['diff', '--name-only', parent, head])
          .split('\n')
          .map((l) => l.trim())
          .filter(Boolean)
      : [];
    const api = (env.GITHUB_API_URL || 'https://api.github.com').replace(/\/$/, '');
    const url = `${api}/repos/${repo}/actions/workflows/${workflow}/runs?status=success&per_page=100`;
    const body = await fetchJson(url, {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    });
    return VerifiedTree.judge({ head, parent, changed, runs: (body && body.workflow_runs) || [] });
  }

  static output(env, verdict) {
    if (!env.GITHUB_OUTPUT) return;
    fs.appendFileSync(env.GITHUB_OUTPUT, `verified=${verdict.verified ? 'true' : 'false'}\nverify_run=${verdict.run ? verdict.run.url : ''}\n`);
  }

  static git(args) {
    return execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  }

  static async fetchJson(url, headers) {
    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`HTTP ${response.status} reading ${url}`);
    return response.json();
  }
}

module.exports = { VerifiedTree };

if (require.main === module) {
  VerifiedTree.main().then(
    () => process.exit(0),
    (err) => {
      console.log(`verified-tree: not verified — ${err.message} — the test jobs run`);
      process.exit(0);
    }
  );
}
