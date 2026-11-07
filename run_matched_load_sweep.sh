#!/usr/bin/env bash
# run_matched_load_sweep.sh -- matched offered-load sweep for Scenarios 1-3.
#
# Idea: for each LOAD LEVEL L (= combined utilization of Chat+Moderate+Batch
# on the shared slots), run S1, S2 and S3 at IDENTICAL arrival rates, then
# compare the scenarios across the whole L range (mean AND p95/max).
#
# How L is set (uses continuous_stress.py --target-samples mode, unchanged):
#   - rate per type = TARGET_SAMPLES / duration (same for all three types)
#   - combined utilization = rate * sum(mean_service_time) / slots
#   - DURATION_BY_SLOTS below is the duration that gives L = 0.7 with
#     TARGET_SAMPLES=15 (same numbers as run_step2_s1_s3_matrix.sh). For level
#     L we use duration = base * 0.7 / L, so every level keeps ~15 samples per
#     type, and LOWER load = LONGER run. --combined-budget is set to L as a cap.
#   Rates are therefore identical across S1/S2/S3 at a given (slots, L).
#
# Requires in the working directory: proxy.py, continuous_stress.py,
# stress_test.py, and results_capacity_{chat,moderate,batch}_slots{1,2,4,8}.json
#
# Usage examples:
#   ./run_matched_load_sweep.sh                       # defaults below
#   LEVELS="0.4 0.7" SLOT_COUNTS="2 4" ./run_matched_load_sweep.sh
#   DRY_RUN=1 ./run_matched_load_sweep.sh             # print plan, run nothing
#
# Results: results/sweep/L<level>/results_sweep_L<level>_scenarioN_slotsS_ts15.json
#
# RUNTIME (approx., at the defaults): ~1.0 h of arrivals at L=0.7 for slots
# 2/4/8 x 3 scenarios, scaling by 0.7/L per level -> roughly 5 h for three
# levels (0.3, 0.5, 0.7), plus ~1 min/run of server+proxy start/stop and drain.
# slots=1 is excluded by default (already settled: S2 starves Moderate/Batch).

set -uo pipefail   # NOT -e: one failed run must not kill the whole sweep

MODEL="${MODEL:-$HOME/Qwen2.5-1.5B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build/bin/llama-server}"
SERVER_PORT=8081
PROXY_PORT=8082
TARGET_SAMPLES="${TARGET_SAMPLES:-15}"
LEVELS=(${LEVELS:-0.3 0.5 0.7})
SLOT_COUNTS=(${SLOT_COUNTS:-2 4 8})
SCENARIOS=(${SCENARIOS:-1 2 3})
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-180}"
WARMUP_SECONDS=5
DRY_RUN="${DRY_RUN:-0}"
BASE_LEVEL=0.7
declare -A BASE_DURATION=( [1]=1250 [2]=620 [4]=380 [8]=210 )   # for L=0.7, ts=15

server_pid=""; proxy_pid=""
cleanup() {
    [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

duration_for() {   # slots level -> integer seconds
    python3 -c "print(round(${BASE_DURATION[$1]} * $BASE_LEVEL / $2))"
}

start_server() {
    local slots=$1
    echo "  [server] llama-server --parallel $slots"
    [[ "$DRY_RUN" == 1 ]] && return 0
    "$LLAMA_SERVER_BIN" -m "$MODEL" --port "$SERVER_PORT" \
        --parallel "$slots" --ctx-size 2048 --kv-unified ${SERVER_EXTRA:-} \
        > "server_log_sweep_slots${slots}.txt" 2>&1 &
    server_pid=$!
    sleep "$WARMUP_SECONDS"
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "  [server] FAILED to start (see server_log_sweep_slots${slots}.txt)"
        return 1
    fi
}
stop_server() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null; server_pid=""; sleep 1
}

proxy_flags() {   # scenario -> flags that DEFINE it in proxy.py
    case "$1" in
        1) echo "--reserved-chat-slots 0" ;;
        2) echo "--reserved-chat-slots 1" ;;
        3) echo "--smart-preemption" ;;
    esac
}
start_proxy() {
    local scenario=$1 slots=$2 level=$3
    local flags; flags=$(proxy_flags "$scenario")
    echo "  [proxy] python3 proxy.py --slots $slots $flags"
    [[ "$DRY_RUN" == 1 ]] && return 0
    python3 proxy.py --slots "$slots" $flags \
        > "proxy_log_sweep_L${level}_s${scenario}_slots${slots}.txt" 2>&1 &
    proxy_pid=$!
    sleep 2
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
        echo "  [proxy] FAILED to start"; return 1
    fi
}
stop_proxy() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null
    wait "$proxy_pid" 2>/dev/null; proxy_pid=""; sleep 1
}

run_stress() {
    local scenario=$1 slots=$2 level=$3
    local duration; duration=$(duration_for "$slots" "$level")
    local outdir="results/sweep/L${level}"
    local label="sweep_L${level}_scenario${scenario}_slots${slots}_ts${TARGET_SAMPLES}"
    local output="${outdir}/results_${label}.json"
    mkdir -p "$outdir"

    local cap=()
    for t in chat moderate batch; do
        local f="results_capacity_${t}_slots${slots}.json"
        [[ -f "$f" ]] || { echo "  [stress] MISSING $f -- skipping $label"; return 1; }
        cap+=(--"${t}"-capacity-json "$f")
    done
    local extra=()
    case "$scenario" in
        1) extra=(--reserved-chat-slots 0) ;;
        2) extra=(--reserved-chat-slots 1) ;;
        3) extra=(--reserved-chat-slots 1 --smart-preemption) ;;
    esac

    echo "  [stress] $label  duration=${duration}s (~$((duration/60)) min), budget=${level}"
    [[ "$DRY_RUN" == 1 ]] && return 0
    python3 continuous_stress.py \
        --scenario "$scenario" --slots "$slots" --duration "$duration" \
        "${cap[@]}" \
        --target-samples "$TARGET_SAMPLES" --combined-budget "$level" \
        --drain-timeout "$DRAIN_TIMEOUT" "${extra[@]}" \
        --label "$label" --output "$output" 2>&1 | tee "${outdir}/log_${label}.txt"
    [[ -f "$output" ]] && echo "  [stress] OK -> $output" \
                       || echo "  [stress] FAILED -- no output for $label"
}

echo "=== Matched offered-load sweep ==="
echo "levels: ${LEVELS[*]} | slots: ${SLOT_COUNTS[*]} | scenarios: ${SCENARIOS[*]} | ts=$TARGET_SAMPLES"
total=0
for level in "${LEVELS[@]}"; do for slots in "${SLOT_COUNTS[@]}"; do
    d=$(duration_for "$slots" "$level"); total=$(( total + d * ${#SCENARIOS[@]} ))
done; done
echo "estimated arrival time: $(( total / 60 )) min (~$(python3 -c "print(round($total/3600,1))") h) + per-run overhead"
echo

# Loop order: slots outermost so llama-server restarts only once per slot count.
for slots in "${SLOT_COUNTS[@]}"; do
    echo "### slots=$slots ###"
    start_server "$slots" || continue
    for level in "${LEVELS[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
            echo " -- L=$level scenario=$scenario slots=$slots --"
            if start_proxy "$scenario" "$slots" "$level"; then
                run_stress "$scenario" "$slots" "$level"
                stop_proxy
            fi
            sleep 1
        done
    done
    stop_server
done
echo "=== Done. Next: python3 summarize_sweep.py results/sweep ==="