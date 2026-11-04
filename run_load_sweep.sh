#!/usr/bin/env bash
# run_load_sweep.sh -- seeded, matched offered-load sweep for Scenarios 1-3.
#
# For every (slots, seed, rho): S1, S2 and S3 are run with the SAME --seed, rates
# and duration, so they receive IDENTICAL arrival schedules (check: the schedule
# digests in the result files, verified by summarize_load_sweep.py).
# rho is the NOMINAL offered load (sum of rate*service_time/slots, service time =
# uncontended, from the Step 1 files). Realized load is recorded per run.
#
# Loop order is slots > seed > rho > scenario, so an interrupted sweep always
# leaves complete matched cells (all three scenarios) behind. Resumable: a run
# whose output file exists is skipped.
#
# Needs in the current folder: proxy.py, continuous_stress.py (the v2 with
# --offered-rho/--seed/--service-time-source), stress_test.py, and
# results_capacity_{chat,moderate,batch}_slots{N}.json (the NEW capacity files).
#
# Usage:   bash run_load_sweep.sh
#          RHOS="0.7 1.0" SEEDS="1" SLOT_COUNTS="2" bash run_load_sweep.sh
#          DRY_RUN=1 bash run_load_sweep.sh
set -uo pipefail

MODEL="${MODEL:-$HOME/edge-llm-server/Qwen2.5-1.5B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build/bin/llama-server}"
RHOS=(${RHOS:-0.4 0.7 1.0})
SLOT_COUNTS=(${SLOT_COUNTS:-2})
SCENARIOS=(${SCENARIOS:-1 2 3})
SEEDS=(${SEEDS:-1 2 3})
DURATION="${DURATION:-600}"
MIX="${MIX:-8:2:1}"
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-180}"
OUTDIR="${OUTDIR:-results/load_sweep}"
DRY_RUN="${DRY_RUN:-0}"
# Extra llama-server flags, e.g. SERVER_EXTRA="--cache-ram 0" to disable the prompt cache
# (default: none, i.e. same server settings as the first sweep).
SERVER_EXTRA="${SERVER_EXTRA:-}"
# 1 = restart llama-server before EVERY run so no state (queued/abandoned requests,
# prompt-cache memory) carries over from one run to the next. 0 = once per slot count.
SERVER_PER_RUN="${SERVER_PER_RUN:-1}"
mkdir -p "$OUTDIR"

server_pid=""; proxy_pid=""
cleanup() {
    [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

start_server() {   # $1=slots $2=log suffix
    [[ "$DRY_RUN" == 1 ]] && return 0
    "$LLAMA_SERVER_BIN" -m "$MODEL" --port 8081 --parallel "$1" \
        --ctx-size 2048 --kv-unified $SERVER_EXTRA > "server_log_loadsweep_$2.txt" 2>&1 &
    server_pid=$!
    sleep 8
    kill -0 "$server_pid" 2>/dev/null || { echo "server failed to start (server_log_loadsweep_$2.txt)"; server_pid=""; return 1; }
}
stop_server() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null; server_pid=""; sleep 2
}

# HOLD>0 turns Scenario 3 into S3b: smart preemption + keep the Chat slot reserved for HOLD
# seconds after the last Chat activity (proxy.py --chat-hold-seconds). Use a separate OUTDIR
# (e.g. OUTDIR=results/load_sweep_s3b_T5) so the S3 files are not skipped as "exists".
HOLD="${HOLD:-0}"
# Scenario 5 = abort-and-requeue preemption of Batch (proxy.py --preempt-batch). Extra proxy
# flags, e.g. PREEMPT_EXTRA="--preempt-for-moderate --max-preemptions 3".
PREEMPT_EXTRA="${PREEMPT_EXTRA:-}"
# Extra proxy.py flags for EVERY scenario, e.g. PROXY_EXTRA="--pin-slots".
PROXY_EXTRA="${PROXY_EXTRA:-}"
proxy_flags() { case "$1" in 1) echo "--reserved-chat-slots 0";; 2) echo "--reserved-chat-slots 1";; 3) echo "--smart-preemption --chat-hold-seconds $HOLD";; 5) echo "--reserved-chat-slots 0 --preempt-batch $PREEMPT_EXTRA";; esac; }
stress_flags() { case "$1" in 1) echo "--reserved-chat-slots 0";; 2) echo "--reserved-chat-slots 1";; 3) echo "--reserved-chat-slots 1 --smart-preemption";; 5) echo "--reserved-chat-slots 0";; esac; }

n_runs=$(( ${#SLOT_COUNTS[@]} * ${#SEEDS[@]} * ${#RHOS[@]} * ${#SCENARIOS[@]} ))
per=$(( DURATION + 95 ))
echo "=== Seeded matched offered-load sweep ==="
echo "slots: ${SLOT_COUNTS[*]} | rho: ${RHOS[*]} | seeds: ${SEEDS[*]} | scenarios: ${SCENARIOS[*]}"
echo "hold(S3b)=${HOLD}s (0 = plain S3)"
echo "duration=${DURATION}s mix=${MIX} -> ${n_runs} runs, ~$(python3 -c "print(round($n_runs*$per/3600,1))") h (incl. ~95 s overhead/run)"
echo

for slots in "${SLOT_COUNTS[@]}"; do
    echo "################ slots=$slots ($(date '+%F %T')) ################"
    for t in chat moderate batch; do
        [[ -f "results_capacity_${t}_slots${slots}.json" ]] || { echo "MISSING results_capacity_${t}_slots${slots}.json -- skipping slots=$slots"; continue 2; }
    done
    if [[ "$SERVER_PER_RUN" != 1 ]]; then start_server "$slots" "slots${slots}" || continue; fi
    for seed in "${SEEDS[@]}"; do for rho in "${RHOS[@]}"; do for sc in "${SCENARIOS[@]}"; do
        label="sweep_rho${rho}_scenario${sc}_slots${slots}_seed${seed}"
        out="${OUTDIR}/results_${label}.json"
        if [[ -f "$out" ]]; then echo "  [skip] $label (exists)"; continue; fi
        echo " -- $label  ($(date '+%T'))"
        [[ "$DRY_RUN" == 1 ]] && continue
        if [[ "$SERVER_PER_RUN" == 1 ]]; then start_server "$slots" "${label}" || continue; fi
        python3 proxy.py --slots "$slots" $(proxy_flags "$sc") $PROXY_EXTRA > "proxy_log_${label}.txt" 2>&1 &
        proxy_pid=$!
        sleep 2
        if ! kill -0 "$proxy_pid" 2>/dev/null; then echo "  proxy failed to start"; proxy_pid=""; continue; fi
        python3 continuous_stress.py --scenario "$sc" --slots "$slots" $(stress_flags "$sc") \
            --duration "$DURATION" --offered-rho "$rho" --mix "$MIX" --seed "$seed" \
            --drain-timeout "$DRAIN_TIMEOUT" \
            --chat-capacity-json "results_capacity_chat_slots${slots}.json" \
            --moderate-capacity-json "results_capacity_moderate_slots${slots}.json" \
            --batch-capacity-json "results_capacity_batch_slots${slots}.json" \
            --label "$label" --output "$out" 2>&1 | tee "${OUTDIR}/log_${label}.txt"
        [[ -f "$out" ]] || echo "  FAILED -- no output for $label"
        [[ "$sc" == 5 ]] && curl -s localhost:8082/proxy/stats > "${OUTDIR}/preempt_stats_${label}.json" 2>/dev/null
        kill "$proxy_pid" 2>/dev/null; wait "$proxy_pid" 2>/dev/null; proxy_pid=""; sleep 2
        if [[ "$SERVER_PER_RUN" == 1 ]]; then stop_server; fi
    done; done; done
    if [[ "$SERVER_PER_RUN" != 1 ]]; then stop_server; fi
done
echo "=== Done. Next: python3 summarize_load_sweep.py --dir ${OUTDIR} --plots ; python3 analyze_sweep_pooled.py ${OUTDIR} ==="