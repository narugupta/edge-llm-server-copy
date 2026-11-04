#!/usr/bin/env bash
# run_capacity_remeasure.sh -- re-run Step 1 (isolated capacity) for all 12
# (type x slots) combinations on the CURRENT machine state.
#
# Writes to a NEW folder (results/capacity-search-new/) so the old files in
# results/capacity-search/ are never overwritten. NOTE: your project root has
# symlinks results_capacity_*.json -> the OLD files; writing to those names
# would overwrite the old data through the symlink, so this script does not.
#
# Resumable: a run whose output file already exists is skipped, so after an
# interruption just start the script again.
#
# Usage:  bash run_capacity_remeasure.sh
#         SLOT_COUNTS="2 4" bash run_capacity_remeasure.sh      # subset
#         DRY_RUN=1 bash run_capacity_remeasure.sh              # print only
set -uo pipefail

MODEL="${MODEL:-$HOME/edge-llm-server/Qwen2.5-1.5B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build/bin/llama-server}"
SLOT_COUNTS=(${SLOT_COUNTS:-1 2 4 8})
TYPES=(${TYPES:-chat moderate batch})
OUTDIR="${OUTDIR:-results/capacity-search-new}"
MAX_N="${MAX_N:-64}"
REPEATS="${REPEATS:-3}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUTDIR"

server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

for slots in "${SLOT_COUNTS[@]}"; do
    echo "################ slots=$slots  ($(date '+%F %T')) ################"
    todo=0
    for t in "${TYPES[@]}"; do [[ -f "$OUTDIR/results_capacity_${t}_slots${slots}.json" ]] || todo=1; done
    if [[ $todo == 0 ]]; then echo "  all outputs exist for slots=$slots -- skipping"; continue; fi
    echo "  [server] --parallel $slots"
    if [[ "$DRY_RUN" != 1 ]]; then
        "$LLAMA_SERVER_BIN" -m "$MODEL" --port 8081 --parallel "$slots" \
            --ctx-size 2048 --kv-unified > "server_log_capacity_slots${slots}.txt" 2>&1 &
        server_pid=$!
        sleep 8
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "  [server] FAILED to start (see server_log_capacity_slots${slots}.txt)"; continue
        fi
    fi
    for t in "${TYPES[@]}"; do
        out="$OUTDIR/results_capacity_${t}_slots${slots}.json"
        if [[ -f "$out" ]]; then echo "  [$t] exists -- skipping"; continue; fi
        start_n=1   # start at 1 for every type so the uncontended (no-queue) service time is measured
        echo "  [$t] capacity_search.py --slots $slots --start-n $start_n --max-n $MAX_N  ($(date '+%T'))"
        [[ "$DRY_RUN" == 1 ]] && continue
        python3 capacity_search.py --client-type "$t" --slots "$slots" \
            --start-n "$start_n" --max-n "$MAX_N" --repeats "$REPEATS" \
            --label "capacity_${t}_slots${slots}" --output "$out" \
            2>&1 | tee "$OUTDIR/log_capacity_${t}_slots${slots}.txt"
        [[ -f "$out" ]] || echo "  [$t] FAILED -- no output written"
    done
    if [[ "$DRY_RUN" != 1 ]]; then kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null; server_pid=""; sleep 2; fi
done

echo; echo "=== Summary (max concurrency within SLO) ==="
python3 - "$OUTDIR" <<'PY'
import json, glob, sys, re
d = {}
for p in glob.glob(sys.argv[1] + "/results_capacity_*_slots*.json"):
    m = re.search(r"capacity_(\w+)_slots(\d+)", p)
    j = json.load(open(p))["derived"]
    d[(m[1], int(m[2]))] = (j["max_concurrency_within_slo"], j["capped_at_max_n"], j["non_monotonic_warning"])
print(f"{'type':9s}" + "".join(f"{s:>12d}" for s in (1, 2, 4, 8)))
for t in ("chat", "moderate", "batch"):
    row = ""
    for s in (1, 2, 4, 8):
        v = d.get((t, s))
        row += f"{'-' if not v else str(v[0]) + ('+' if v[1] else '') + ('*' if v[2] else ''):>12s}"
    print(f"{t:9s}{row}")
print("('+' = capped at max-n, '*' = non-monotonic warning)")
PY