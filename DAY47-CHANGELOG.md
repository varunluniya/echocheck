# Day 47 changelog — EchoCheck

## Finding

Same portfolio-wide audit as the other 8 repos (Guide 5's Mistake 5: full
re-scan of `memory.decisions()` with no timestamp filter, on every
feedback call). EchoCheck shares the same `gen4/` kernel as the rest of
the portfolio, but doesn't have a `learn()` / `metrics()` /
`override_rates()`-style aggregation function of its own to rewrite.

## Changes

- `gen4/memory.py`, `gen4/feedback.py` — kernel-only update, copied
  byte-identical from the LendFair reference fix: added the `fact_history`
  table (so `add_fact()` preserves history instead of losing it on every
  upsert), the `decisions(outcome_since=...)` watermarking parameter, the
  `_select()` fix that stops silently dropping `outcome_ts` from returned
  rows, the `(system, ts DESC)` index (fixes a full table `SCAN` + sort on
  any `decisions(system=..., limit=...)` call, confirmed via `EXPLAIN QUERY
  PLAN`), and the `new_rows_since` / `mark_processed` / `accumulate` /
  `incremental_calibration` primitives.
- No changes to `service.py` — nothing in this repo was rewritten to use
  the new primitives, since there was no full-rescan aggregation function
  here to fix.

## Verification

- `pytest -q` → 18/18 passed, unchanged. The kernel update is purely
  additive (new table, new optional parameters, new index, new functions
  in `feedback.py`) — nothing existing was removed or had its signature
  changed, so this confirms the shared-kernel fix is backward-compatible
  across the portfolio.
