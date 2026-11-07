# Old results (first laptop)

Everything in this folder was produced on the first laptop, before the move to the new
Ubuntu laptop (Ryzen 5 PRO 4650U). Do not compare absolute numbers with the new laptop's
results: different CPU, thread count and clock behaviour. Compare shapes and relative effects.

- llama.cpp base: the snapshot that was in this repo until 2026-11-07 (upstream state around
  23 May 2026). Server flags in most runs: --ctx-size 2048 --kv-unified, otherwise defaults
  (default thread count, 8 threads in the logs we checked; prompt cache default 8192 MiB).
  Some runs add flags such as --cache-ram 1024 or -b/-ub; see each run's server log.
- root_files/: logs and json files that used to sit in the repo root
  (proxy_log_*, server_log_*, resume_*.json, proxy_results.json, results_smoke2.json).
- results/: the former results/ folder. capacity-search-new = the Step 1 re-run that the load
  sweeps used; load_sweep*, sweep, reserved-chat-slots, continuous_stress = Step 2 runs.
- Git tag pre-cleanup-old-laptop = repo state before this move.
