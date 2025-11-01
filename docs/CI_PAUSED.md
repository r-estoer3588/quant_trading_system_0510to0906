# CI auto-triggers paused (temporary)

To reduce iteration friction during active work, GitHub Actions auto triggers (push / pull_request / schedules) have been temporarily disabled for most workflows.

What changed:

- Workflows now run only via manual dispatch (workflow_dispatch) with a small comment header.
- Scheduled jobs (daily signals, dependency checks) are paused.

How to run manually:

- In GitHub, open Actions tab, pick a workflow, click "Run workflow".
- Optionally provide the "reason" input where available.

How to restore auto triggers:

- Edit each workflow under `.github/workflows/` and restore the `on: push`, `on: pull_request`, and/or `on: schedule` sections.
  - ci-unified.yml
  - controlled-tests.yml
  - coverage-report.yml
  - daily-signals.yml (re-add cron)
  - dependency-check.yml (re-add cron and/or PR trigger)
  - docs-auto-update.yml (re-add push w/ paths)
  - playwright.yml / playwright-docker.yml (re-add push/PR filters)

Notes:

- This pause is intended to be temporary. Please remove when CI stability is no longer a concern.
- Manual-only workflows still produce artifacts and summaries exactly as before.

## Staged reactivation plan

1. Restore the `on.push` trigger only for `ci-unified.yml` (targeting `main`) and confirm it completes without regressions.
2. Re-enable `controlled-tests.yml` and `playwright*.yml` for pull_request events after step 1, so feature branches regain fast feedback.
3. Reintroduce the cron schedules for `daily-signals.yml` and `dependency-check.yml` once push/PR coverage is healthy again.
4. Re-add any remaining docs or coverage triggers (`docs-auto-update.yml`, `coverage-report.yml`) last, after monitoring two clean days of runs.
