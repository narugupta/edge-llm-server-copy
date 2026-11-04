#!/usr/bin/env python3
"""
continuous_stress.py — Continuous mixed-workload stress test (Step 2)
======================================================================

Step 2 evaluates Chat, Moderate, and Batch under continuous mixed load
through the proxy. Unlike stress_test.py, requests are generated
continuously using independent Poisson arrival processes rather than
discrete waves.

Step 1 measured the capacity of each workload in isolation. Step 2 uses
those measurements to evaluate the scheduling policies under mixed load.

Scenarios currently supported by proxy.py:

    Scenario 1 — Priority Proxy
        --reserved-chat-slots 0

    Scenario 2 — Fixed Reservation
        --reserved-chat-slots N
        N > 0, without --smart-preemption

    Scenario 3 — Smart Preemption
        --reserved-chat-slots N --smart-preemption

        The current implementation does not interrupt requests that are
        already running. When Chat is waiting, the scheduler can reclaim
        a slot when a running request finishes and give it to Chat.

Scenarios 4 and 5 are not currently implemented in proxy.py:

    Scenario 4 — EDF (Earliest Deadline First)
    Scenario 5 — WFQ + Preemption

The test does not start or configure llama-server or proxy.py. Start
both separately with the required configuration before running this
script.

Server configuration:
    --parallel should match the --slots value used for the test.

Proxy configuration:
    Set --reserved-chat-slots and --smart-preemption according to the
    scenario being evaluated.

Example:

    Terminal 1 — llama-server

    ./build/bin/llama-server \
        -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \
        --port 8081 \
        --parallel 2 \
        --ctx-size 2048 \
        --kv-unified

    Terminal 2 — proxy

    python3 proxy.py --reserved-chat-slots 1

    Terminal 3 — continuous mixed load

    python3 continuous_stress.py \
        --scenario 2 \
        --slots 2 \
        --duration 90 \
        --chat-capacity-json results_capacity_chat_slots2.json \
        --moderate-capacity-json results_capacity_moderate_slots2.json \
        --batch-capacity-json results_capacity_batch_slots2.json \
        --rate-frac 0.5 \
        --reserved-chat-slots 1 \
        --output results_scenario2_slots2.json


Arrival rates
-------------

Arrival rates can be provided directly:

    --chat-rate
    --moderate-rate
    --batch-rate

They can also be derived from Step 1 capacity results using Little's
Law:

    rate = capacity / mean_service_time_at_capacity

The derived rate is multiplied by --rate-frac.

For each workload type, an explicitly supplied rate takes precedence
over the corresponding capacity JSON file.

If neither an explicit rate nor a capacity JSON file is provided for a
workload type, no requests of that type are generated.


Step 2 test goals
-----------------

The continuous mixed-load tests are used to evaluate:

1. The effect of fixed slot reservation on Moderate and Batch.

2. Scaling behaviour at different slot counts, particularly in light
   of the non-linear scaling observed during Step 1.

3. Scheduler behaviour when requests arrive at different times.

4. SLO compliance under sustained mixed workload.

5. Whether lower-priority workloads continue to make progress under
   higher-priority load.


Output
------

The output JSON contains:

    meta
        Test configuration, scenario, slot count, duration, arrival
        pattern, resolved arrival rates, and rate source.

    per_type
        Results for Chat, Moderate, and Batch:
        - n_arrived
        - n_completed
        - n_errors
        - completion_rate
        - mean SLO metric
        - p99 SLO metric
        - SLO violation rate
        - front-half mean
        - back-half mean

    raw
        Individual request results from stress_test.py:
        - client_id
        - client_type
        - priority
        - t_start
        - wave_arrival_s
        - ttft
        - total_time
        - tokens_out
        - tokens_per_s
        - error

Note:
    There are no discrete waves in this test. The field
    'wave_arrival_s' is retained for compatibility with the existing
    stress_test.py result format and represents the time since the
    corresponding workload generator started.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
from datetime import datetime

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    sys.exit("requests not installed — run:  pip install requests")

try:
    from stress_test import run_one_client, health_check, PROMPTS, PRIORITY_LABEL, SLO_CHECK
except ImportError:
    sys.exit(
        "Could not import from stress_test.py -- this script must be run "
        "from the same directory as stress_test.py (it reuses its client "
        "request/TTFT-measurement logic and SLO definitions directly "
        "rather than duplicating them, the same way capacity_search.py does)."
    )

SCENARIO_LABELS = {
    1: "Priority Proxy",
    2: "Fixed Reservation",
    3: "Smart Preemption (v1 reactive, non-interrupting)",
    4: "EDF (not yet implemented)",
    5: "WFQ + Preemption (not yet implemented)",
}


# ── Rate resolution ──────────────────────────────────────────────────────────

# Service-time source for rate derivation (set from --service-time-source in main()).
#   "uncontended": mean total_time at the LARGEST tested concurrency <= min(capacity, slots),
#                  i.e. the heaviest level that still has no queueing and still met the SLO. This is closest to the true
#                  time a request occupies a slot.
#   "capacity":    mean total_time at the capacity level (previous behaviour). When
#                  capacity > slots that level includes queueing, so it OVERSTATES service
#                  time (2.4-4x in the Oct-2026 re-measurement) and the realized load comes
#                  out far below the nominal one.
_SVC = {"source": "uncontended", "slots": None}


def _mean_total_time(level):
    vals = [r["total_time"] for r in level.get("raw", [])
            if r.get("total_time") is not None and not r.get("error")]
    return statistics.mean(vals) if vals else None


def _capacity_and_mean_service_time(path: str) -> tuple:
    """Shared by the derive path and the explicit-rate sanity check: reads a
    capacity_search.py result JSON and returns (capacity, mean_service_time).

    "Service time" must be slot-occupancy time (total_time), not the SLO metric (TTFT
    for chat/moderate). Which concurrency level it is read from is controlled by
    _SVC["source"] (see above)."""
    with open(path) as f:
        d = json.load(f)
    capacity = d["derived"]["max_concurrency_within_slo"]
    if capacity is None or capacity <= 0:
        raise ValueError(f"{path}: no usable capacity (max_concurrency_within_slo="
                          f"{capacity!r}) to derive a rate from")
    level = None
    if _SVC["source"] == "uncontended" and _SVC["slots"]:
        limit = min(capacity, _SVC["slots"])   # same effective_servers as Little's Law
        usable = [l for l in d["levels"]
                  if l["concurrency"] <= limit and _mean_total_time(l) is not None]
        if usable:
            level = max(usable, key=lambda l: l["concurrency"])
    if level is None:   # "capacity" source, or no level <= slots was tested
        level = next((l for l in d["levels"] if l["concurrency"] == capacity), None)
    if level is None:
        raise ValueError(f"{path}: capacity={capacity} but no matching level in "
                          f"'levels' to read service time from")
    mean_service_time = _mean_total_time(level)
    if mean_service_time is None:
        raise ValueError(f"{path}: no usable total_time values in level "
                          f"{level['concurrency']}'s raw records")
    return capacity, mean_service_time


def derive_rate_from_capacity_json(path: str, rate_frac: float, slots: int) -> tuple:
    """Little's Law: rate = effective_servers / mean_service_time_at_capacity,
    scaled by rate_frac.

    FIX: effective_servers is min(capacity, slots), NOT capacity alone.
    `capacity` (max_concurrency_within_slo) is how many concurrent requests
    capacity_search.py found the system could absorb while still meeting the
    SLO metric -- it can legitimately exceed the physical slot count (e.g. a
    burst that queues briefly but still clears its SLO window). Treating
    capacity itself as the server count in Little's Law overstates sustained
    throughput: you cannot actually run more than `slots` requests
    concurrently for a *sustained* duration, no matter how much SLO
    headroom a short burst has. Capping at the true physical slot count is
    what makes the derived rate something continuous_stress.py's steady
    Poisson arrivals over `--duration` seconds can actually sustain.

    Returns (rate, detail_str, mean_service_time) -- mean_service_time is
    returned too so the caller can run the same utilization check used for
    explicit rates, without re-reading the file."""
    capacity, mean_service_time = _capacity_and_mean_service_time(path)
    effective_servers = min(capacity, slots)
    base_rate = effective_servers / mean_service_time
    rate = base_rate * rate_frac
    detail = (f"derived from {path}: capacity={capacity}, slots={slots}, "
              f"effective_servers=min(capacity,slots)={effective_servers}, "
              f"mean_total_time={mean_service_time:.3f}s (from raw records, "
              f"NOT pooled.mean which may be ttft), "
              f"base_rate={base_rate:.3f}/s, rate_frac={rate_frac} -> {rate:.3f}/s")
    return rate, detail, mean_service_time


def warn_if_unsustainable(client_type, rate, mean_service_time, slots, source, log):
    """Catches the sanity-test class of problem for ANY rate -- explicit or
    derived -- not just ones that went through Little's Law.

    Utilization = (rate * mean_service_time) / slots = the number of
    physical slots this rate would need, on average, to keep up, divided by
    the slots actually available. >1.0 means the rate is unsustainable in
    isolation: queue length and latency grow without bound over a long
    enough run, independent of scheduling policy (--reserved-chat-slots,
    --smart-preemption) and independent of the other two client types'
    load -- this is a floor, not a worst case."""
    if slots is None or slots <= 0 or mean_service_time is None:
        return
    needed_servers = rate * mean_service_time
    utilization = needed_servers / slots
    if utilization > 1.0:
        log(f"  WARNING: {client_type} rate {rate:.3f}/s needs ~{needed_servers:.2f} "
            f"concurrent slot(s) on average (rate x mean_service_time "
            f"{mean_service_time:.3f}s) but only {slots} physical slot(s) exist "
            f"(utilization={utilization:.2f}x, {source}). This rate is "
            f"UNSUSTAINABLE IN ISOLATION -- queue length and latency will grow "
            f"without bound the longer --duration runs, regardless of scheduling "
            f"policy or the other client types' load.")


def resolve_rate(client_type: str, explicit_rate, capacity_json, slots, rate_frac, log) -> tuple:
    """Returns (rate, detail, mean_service_time_or_None). mean_service_time is
    None only when no capacity_json was given at all (explicit rate, no
    sustainability data available) -- callers that need it for a combined
    cross-type utilization check should treat None as "unknown, can't check"."""
    if explicit_rate is not None:
        log(f"  {client_type:<9s} rate = {explicit_rate:.3f}/s (explicit)")
        if capacity_json:
            try:
                _, mean_service_time = _capacity_and_mean_service_time(capacity_json)
                warn_if_unsustainable(
                    client_type, explicit_rate, mean_service_time, slots,
                    source=f"explicit rate checked against {capacity_json}", log=log)
                return explicit_rate, "explicit", mean_service_time
            except (OSError, KeyError, ValueError) as exc:
                log(f"  {client_type:<9s} (could not sanity-check explicit rate "
                    f"against {capacity_json}: {exc})")
        return explicit_rate, "explicit", None
    if capacity_json:
        rate, detail, mean_service_time = derive_rate_from_capacity_json(
            capacity_json, rate_frac, slots)
        log(f"  {client_type:<9s} rate = {rate:.3f}/s ({detail})")
        warn_if_unsustainable(
            client_type, rate, mean_service_time, slots,
            source=f"derived from {capacity_json}", log=log)
        return rate, detail, mean_service_time
    log(f"  {client_type:<9s} rate = 0/s (no --{client_type}-rate or "
        f"--{client_type}-capacity-json given -- this type sends no load)")
    return 0.0, "none (no load)", None


def warn_if_combined_unsustainable(rates_and_mst: dict, slots, log):
    """Catches EXACTLY the bug the smart_preemption sanity test and the full
    S1-S3 x slots{1,2,4,8} matrix both hit: each type's rate can look
    individually sustainable (utilization <= 1.0 against `slots` on its
    own) while the combined demand still overloads the system, because
    chat/moderate/batch all draw from the SAME physical slot pool
    simultaneously rather than each having it to themselves.

    Per-type utilization must SUM to <=1.0 across all active types, not
    just individually. rates_and_mst: {client_type: (rate, mean_service_time)}
    -- types with mean_service_time=None (explicit rate, no capacity_json)
    are skipped in the sum with a note, since their true slot-time demand
    is unknown."""
    if slots is None or slots <= 0:
        return
    total_util = 0.0
    parts = []
    skipped = []
    for ct, (rate, mst) in rates_and_mst.items():
        if rate <= 0:
            continue
        if mst is None:
            skipped.append(ct)
            continue
        u = (rate * mst) / slots
        total_util += u
        parts.append(f"{ct}={u:.3f}")
    if skipped:
        log(f"  (combined-utilization check: skipping {', '.join(skipped)} -- "
            f"explicit rate with no capacity_json, true service time unknown)")
    if total_util > 1.0:
        log(f"  WARNING: COMBINED utilization across all active types = "
            f"{total_util:.3f}x ({', '.join(parts)}) on {slots} shared slot(s). "
            f"Each type's rate may look sustainable ALONE, but chat/moderate/"
            f"batch all draw from the SAME {slots} physical slot(s) "
            f"simultaneously -- their utilizations must SUM to <=1.0, not "
            f"each individually. This combined load is UNSUSTAINABLE: queues "
            f"will grow without bound across all three types as --duration "
            f"runs longer, regardless of scheduling policy. Lower --rate-frac "
            f"(or split it across types) until this sum is comfortably <1.0.")
    elif parts:
        log(f"  Combined utilization across active types = {total_util:.3f}x "
            f"({', '.join(parts)}) on {slots} shared slot(s) -- sustainable.")


def resolve_rates_target_samples(target_samples, duration, slots, capacity_jsons,
                                  explicit_rates, budget, log):
    """Alternative to the equal-utilization split in resolve_rate(): aims for
    ~target_samples completed requests PER TYPE over `duration`, then scales
    ALL types down together (preserving their ratio, i.e. still roughly equal
    sample counts, just fewer) only if the naive equal-count rates would
    exceed `budget` combined utilization.

    Why this exists: equal utilization (old approach) means
    utilization = rate * mean_service_time is the same for every type, but
    since batch's mean_service_time is 30-80x chat's, equal utilization
    forces batch's rate (and therefore its sample count) to be 30-80x
    smaller than chat's -- a 45s run can get 30+ chat samples and 0 batch
    samples. Equal SAMPLE COUNT is what you actually need for a fair
    scenario comparison, so this derives rate = target_samples / duration
    for every type first, then checks the combined utilization that implies
    and scales everything down together (never just one type) if it's over
    budget -- so if scaling is needed, every type falls equally short of
    target_samples rather than batch being singled out again.

    explicit_rates: {client_type: rate_or_None} -- types with an explicit
    rate are left untouched; their utilization (if a matching capacity_json
    is available) is subtracted from `budget` before splitting the rest.

    Returns {client_type: (rate, detail, mean_service_time_or_None)}."""
    types = ("chat", "moderate", "batch")
    mst = {}
    naive_rate = {}
    naive_util = {}
    fixed_util = 0.0

    for ct in types:
        if explicit_rates.get(ct) is not None:
            cj = capacity_jsons.get(ct)
            if cj:
                try:
                    _, m = _capacity_and_mean_service_time(cj)
                    fixed_util += (explicit_rates[ct] * m) / slots
                except (OSError, KeyError, ValueError):
                    pass
            continue
        cj = capacity_jsons.get(ct)
        if not cj:
            continue
        try:
            _, m = _capacity_and_mean_service_time(cj)
        except (OSError, KeyError, ValueError) as exc:
            log(f"  {ct:<9s} (could not compute target-samples rate: {exc})")
            continue
        mst[ct] = m
        naive_rate[ct] = target_samples / duration
        naive_util[ct] = (naive_rate[ct] * m) / slots

    total_naive = sum(naive_util.values())
    remaining_budget = max(0.0, budget - fixed_util)
    scale = 1.0
    if total_naive > 0 and total_naive > remaining_budget:
        scale = remaining_budget / total_naive

    results = {}
    for ct in types:
        if explicit_rates.get(ct) is not None:
            _, m = (None, None)
            cj = capacity_jsons.get(ct)
            if cj:
                try:
                    _, m = _capacity_and_mean_service_time(cj)
                except (OSError, KeyError, ValueError):
                    pass
            log(f"  {ct:<9s} rate = {explicit_rates[ct]:.3f}/s (explicit, "
                f"target-samples mode leaves it untouched)")
            results[ct] = (explicit_rates[ct], "explicit", m)
            continue
        if ct not in naive_rate:
            log(f"  {ct:<9s} rate = 0/s (no --{ct}-rate or --{ct}-capacity-json "
                f"given -- this type sends no load)")
            results[ct] = (0.0, "none (no load)", None)
            continue
        r = naive_rate[ct] * scale
        expected_n = r * duration
        detail = (f"target-samples: aiming for {target_samples} over {duration:.0f}s "
                  f"(naive_rate={naive_rate[ct]:.4f}/s), scaled {scale:.3f}x for "
                  f"combined budget {budget} -> rate={r:.4f}/s, expected_n~={expected_n:.1f}")
        log(f"  {ct:<9s} rate = {r:.4f}/s ({detail})")
        results[ct] = (r, detail, mst[ct])

    if scale < 1.0:
        log(f"  NOTE: naive equal-{target_samples}-sample rates implied "
            f"{total_naive:.3f}x combined utilization, over the "
            f"{remaining_budget:.3f}x budget remaining after explicit rates -- "
            f"ALL target-samples types scaled down together by {scale:.3f}x. "
            f"Expected samples per type will be below {target_samples}; "
            f"increase --duration to recover the full target without raising "
            f"utilization.")
    return results


# ── Arrival generator ─────────────────────────────────────────────────────────

def arrival_generator(client_type, rate, duration, pattern, session, host, port,
                       timeout, results, results_lock, in_flight_counter, log):
    """Fire requests for `client_type` for `duration` seconds at `rate` req/s,
    fire-and-forget (each request runs in its own thread so the generator
    never blocks waiting for a response). Returns the list of spawned
    threads so the caller can join them during drain.

    Uses stress_test.py's run_one_client with barrier=None -- the exact
    calling convention it already supports for fire-immediately, caller-
    controlled timing, confirmed compatible before this refactor."""
    threads = []
    if rate <= 0:
        return threads

    t0 = time.perf_counter()
    n_fired = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break
        if pattern == "poisson":
            gap = random.expovariate(rate)
        else:  # fixed
            gap = 1.0 / rate
        time.sleep(min(gap, max(0.0, duration - elapsed)))
        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break

        with in_flight_counter["lock"]:
            in_flight_counter[client_type] += 1
        t = threading.Thread(
            target=_send_and_decrement,
            args=(client_type, n_fired, host, port, results, results_lock,
                  timeout, session, t0, in_flight_counter),
            daemon=True,
        )
        threads.append(t)
        t.start()
        n_fired += 1

    log(f"  [{client_type}] generator done: fired {n_fired} arrivals over "
        f"{duration}s (target rate {rate:.3f}/s)")
    return threads


def _send_and_decrement(client_type, instance_id, host, port, results,
                         results_lock, timeout, session, t0, in_flight_counter):
    try:
        run_one_client(client_type, instance_id, wave_idx=0, host=host, port=port,
                        barrier=None, results=results, results_lock=results_lock,
                        wave_t0=t0, timeout=timeout, session=session)
    finally:
        with in_flight_counter["lock"]:
            in_flight_counter[client_type] -= 1


# ── Evaluation ────────────────────────────────────────────────────────────────

# ── Matched offered-load mode ────────────────────────────────────────────────

def parse_mix(text):
    """'8:2:1' -> {'chat': 0.727, 'moderate': 0.182, 'batch': 0.091} (request-count shares)."""
    parts = [float(x) for x in text.split(":")]
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError(f"--mix must be chat:moderate:batch non-negative numbers, got {text!r}")
    tot = sum(parts)
    return {"chat": parts[0] / tot, "moderate": parts[1] / tot, "batch": parts[2] / tot}


def resolve_rates_offered_load(rho, mix_text, slots, duration, capacity_jsons, log):
    """Offered load as a single NOMINAL utilization rho, split across types by a
    fixed request-count mix.

        mean_service  = sum_t share_t * service_time_t        (Step 1 service times)
        lambda_total  = rho * slots / mean_service
        lambda_t      = share_t * lambda_total

    Why this form: the load is described by ONE number (rho) plus a fixed mix, so
    it can be swept (rho = 0.5, 0.8, 1.0, 1.2 ...) and each point is identical
    for every scheduling scenario. rho is NOMINAL: the Step 1 service times were
    measured in isolation, so the real saturation point can differ -- the sweep
    shows where it actually is.

    Returns {client_type: (rate, detail, mean_service_time)}."""
    shares = parse_mix(mix_text)
    mst = {}
    for ct in ("chat", "moderate", "batch"):
        cj = capacity_jsons.get(ct)
        if not cj:
            raise ValueError(f"--offered-rho needs --{ct}-capacity-json (service time source)")
        _, mst[ct] = _capacity_and_mean_service_time(cj)
    mean_service = sum(shares[ct] * mst[ct] for ct in shares)
    lam_total = rho * slots / mean_service
    out = {}
    for ct in ("chat", "moderate", "batch"):
        rate = shares[ct] * lam_total
        exp_n = rate * duration
        detail = (f"offered-load: rho={rho} slots={slots} mix-share={shares[ct]:.3f} "
                  f"mean_service={mean_service:.3f}s -> rate={rate:.4f}/s, expected_n~={exp_n:.1f}")
        log(f"  {ct:<9s} rate = {rate:.4f}/s  (share {shares[ct]:.3f}, service {mst[ct]:.2f}s, "
            f"expected n ~ {exp_n:.0f})")
        out[ct] = (rate, detail, mst[ct])
    if shares["chat"] * lam_total * duration < 100:
        log("  NOTE: expected Chat samples < 100 -- p99 will not be reliable "
            "(p95 needs >= 20). Increase --duration for tail metrics.")
    return out


# ── Seeded, replayable arrival schedules ─────────────────────────────────────

def build_schedule(client_type, rate, duration, pattern, seed):
    """Pre-generate arrival offsets (seconds from run start) for one type.

    Identical (seed, type, rate, duration, pattern) -> identical schedule, so
    S1/S2/S3 can be fed EXACTLY the same arrivals (matched comparison). Gaps are
    unit-rate exponentials divided by `rate`, so a higher-rate schedule with the
    same seed is the lower-rate one compressed in time (common random numbers
    across load levels too)."""
    if rate <= 0:
        return []
    rng = random.Random(f"{seed}:{client_type}")
    offsets, t = [], 0.0
    while True:
        gap = (rng.expovariate(1.0) / rate) if pattern == "poisson" else (1.0 / rate)
        t += gap
        if t >= duration:
            break
        offsets.append(round(t, 6))
    return offsets


def schedule_digest(offsets):
    return hashlib.sha1(json.dumps(offsets).encode()).hexdigest()[:12]


def scheduled_arrival_generator(client_type, offsets, t0, session, host, port, timeout,
                                results, results_lock, in_flight_counter, log):
    """Fire one request per pre-generated offset (absolute times from the shared
    t0, so sleeping error does not accumulate). Fire-and-forget, as before."""
    threads = []
    for n_fired, off in enumerate(offsets):
        delay = (t0 + off) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        with in_flight_counter["lock"]:
            in_flight_counter[client_type] += 1
        t = threading.Thread(
            target=_send_and_decrement,
            args=(client_type, n_fired, host, port, results, results_lock,
                  timeout, session, t0, in_flight_counter),
            daemon=True,
        )
        threads.append(t)
        t.start()
    log(f"  [{client_type}] generator done: fired {len(threads)} scheduled arrivals")
    return threads


# ── Evaluation (tail-aware) ──────────────────────────────────────────────────

def percentile(sorted_vals, p):
    """Linear-interpolated percentile (numpy's default definition)."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (sane for small n)."""
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def summarise(values):
    """Stats over COMPLETED requests only. NOTE: p99 here is linearly
    interpolated; result files written before this change used a nearest-rank
    lower index, so p99 is not comparable across the two."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    sv = sorted(valid)
    return {"n": len(valid), "mean": round(statistics.mean(valid), 4),
            "p50": round(percentile(sv, 50), 4), "p95": round(percentile(sv, 95), 4),
            "p99": round(percentile(sv, 99), 4), "max": round(max(valid), 4)}


def evaluate_type(client_type, results, n_scheduled=None, censored_floors=None, timeout=150.0):
    """Per-type evaluation.

    n_scheduled / censored_floors handle requests still in flight when the drain
    timeout expired. Dropping them (the old behaviour) silently removes the
    SLOWEST requests, biasing every tail metric low exactly when the system is
    overloaded. Here they are CENSORED: counted as arrivals, counted as SLO
    violations, and entered into the tail percentiles at a lower-bound latency
    (time from scheduled arrival to the end of the drain). Errored requests
    (e.g. HTTP 504) are likewise counted as violations and enter the tail at
    `timeout`. A percentile that lands on such a bound is flagged
    `*_is_lower_bound` (true value >= reported)."""
    metric = SLO_CHECK[client_type]["metric"]
    threshold = SLO_CHECK[client_type]["threshold"]
    typed = [r for r in results if r["client_type"] == client_type]
    typed_sorted = sorted(typed, key=lambda r: r["wave_arrival_s"])
    censored_floors = list(censored_floors or [])
    n_arrived = n_scheduled if n_scheduled is not None else len(typed_sorted)
    errors = [r for r in typed_sorted if r["error"]]
    ok = [r for r in typed_sorted if not r["error"] and r.get(metric) is not None]
    values = [r[metric] for r in ok]
    stats = summarise(values)

    n_over = sum(1 for v in values if v > threshold)
    violations = n_over + len(errors) + len(censored_floors)
    violation_rate = round(violations / n_arrived, 4) if n_arrived else None

    pairs = sorted([(v, False) for v in values]
                   + [(timeout, True)] * len(errors)
                   + [(f, True) for f in censored_floors])
    tvals = [p[0] for p in pairs]
    n_tail = len(pairs)

    def tail_pct(p):
        if n_tail == 0:
            return None, False
        k = (n_tail - 1) * p / 100.0
        lo = int(math.floor(k))
        hi = min(lo + 1, n_tail - 1)
        return round(percentile(tvals, p), 4), (pairs[lo][1] or pairs[hi][1])

    p50, _ = tail_pct(50)
    p95, p95_lb = tail_pct(95)
    p99, p99_lb = tail_pct(99)
    tail = {
        "n": n_tail,
        "p50": p50, "p95": p95, "p99": p99,
        "max": round(tvals[-1], 4) if n_tail else None,
        "p95_is_lower_bound": p95_lb, "p99_is_lower_bound": p99_lb,
        "max_is_lower_bound": bool(n_tail and pairs[-1][1]),
        "p95_reliable": n_tail >= 20, "p99_reliable": n_tail >= 100,
    }

    half = len(typed_sorted) // 2
    front = [r[metric] for r in typed_sorted[:half] if not r["error"] and r.get(metric) is not None]
    back = [r[metric] for r in typed_sorted[half:] if not r["error"] and r.get(metric) is not None]

    return {
        "metric": metric,
        "threshold": threshold,
        "n_arrived": n_arrived,
        "n_completed": len(ok),
        "n_errors": len(errors),
        "n_censored": len(censored_floors),
        "error_samples": sorted({e["error"] for e in errors})[:5],
        "completion_rate": round(len(ok) / n_arrived, 4) if n_arrived else None,
        "stats": stats,
        "tail": tail,
        "n_violations": violations,
        "slo_violation_rate": violation_rate,
        "slo_violation_ci95": wilson_interval(violations, n_arrived),
        "front_half_mean": round(statistics.mean(front), 4) if front else None,
        "back_half_mean": round(statistics.mean(back), 4) if back else None,
        "degrading_over_run": (
            front and back and statistics.mean(back) > statistics.mean(front) * 1.2
        ),
    }


def compute_aggregate_metrics(results, generation_start, timeout, n_scheduled_total=None,
                              n_censored_total=0, horizon_elapsed=None, slots=None, duration=None):
    """Overall completion % and total wall-clock completion time, pooled over
    all types. Errored requests are assumed to finish at t_start + timeout
    (upper bound). If any request was still in flight at the drain deadline,
    total_completion_time is only a LOWER bound (the run was cut off)."""
    total_arrived = n_scheduled_total if n_scheduled_total is not None else len(results)
    total_completed = sum(1 for r in results if not r["error"])
    finish_times = []
    for r in results:
        if r["error"]:
            finish_times.append(r["t_start"] + timeout)
        elif r.get("total_time") is not None:
            finish_times.append(r["t_start"] + r["total_time"])
    tct = round(max(finish_times) - generation_start, 3) if finish_times else None
    # REALIZED load: nominal rho comes from Step 1 service times measured on another
    # day/machine state, so it does not transfer. This measures what actually
    # happened: decode-busy slot-seconds / (slots * duration). Excludes prefill and
    # queue wait, so it slightly UNDERestimates; use it to compare runs, not as an
    # exact utilization.
    realized = None
    if slots and duration:
        busy = sum((r["total_time"] - r["ttft"]) for r in results
                   if not r["error"] and r.get("total_time") is not None and r.get("ttft") is not None)
        realized = round(busy / (slots * duration), 4)
    lower_bound = False
    if n_censored_total > 0 and horizon_elapsed is not None:
        tct = round(max(tct or 0.0, horizon_elapsed), 3)
        lower_bound = True
    return {
        "total_arrived": total_arrived,
        "total_completed": total_completed,
        "n_censored": n_censored_total,
        "overall_completion_rate": round(total_completed / total_arrived, 4) if total_arrived else None,
        "total_completion_time": tct,
        "total_completion_time_is_lower_bound": lower_bound,
        "realized_busy_fraction": realized,
    }


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed-workload continuous-arrival stress test (Step 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="Informational label only -- does not configure the proxy. "
                         "See module docstring for the Scenario -> proxy.py flag mapping.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8082,
                    help="Default 8082 -- the PROXY, not the raw server (unlike "
                         "capacity_search.py). Scenario comparisons are only "
                         "meaningful through the scheduling layer.")
    p.add_argument("--slots", type=int, required=True,
                    help="Informational only -- must match what you started "
                         "llama-server with via --parallel.")
    p.add_argument("--reserved-chat-slots", type=int, default=None,
                    help="Informational only -- must match what you started proxy.py "
                         "with. 0 = Scenario 1 (Priority Proxy), >0 = Scenario 2 "
                         "(Fixed Reservation) UNLESS --smart-preemption is also set.")
    p.add_argument("--smart-preemption", action="store_true",
                    help="Informational only -- set this if proxy.py was started "
                         "WITH --smart-preemption (Scenario 3). Recorded in metadata "
                         "so a run can be correctly attributed to Scenario 2 vs 3 "
                         "later, since both use --reserved-chat-slots.")
    p.add_argument("--duration", type=float, default=90.0,
                    help="Seconds of continuous arrivals per type (default 90).")
    p.add_argument("--arrival-pattern", default="poisson", choices=["poisson", "fixed"])
    p.add_argument("--drain-timeout", type=float, default=180.0,
                    help="Max seconds to wait for in-flight requests to finish "
                         "after arrivals stop (default 180).")
    p.add_argument("--timeout", type=float, default=150.0, help="Per-request HTTP timeout.")

    for ct in ("chat", "moderate", "batch"):
        p.add_argument(f"--{ct}-rate", type=float, default=None,
                        help=f"{ct} arrivals/second, explicit. Overrides --{ct}-capacity-json.")
        p.add_argument(f"--{ct}-capacity-json", type=str, default=None,
                        help=f"Path to a capacity_search.py result JSON for {ct}; "
                             f"rate derived via Little's Law if --{ct}-rate not given.")
    p.add_argument("--rate-frac", type=float, default=1.0,
                    help="Multiplier applied to every rate derived from a "
                         "--*-capacity-json (default 1.0 = 100%% of implied "
                         "capacity throughput). Has no effect on explicit --*-rate. "
                         "IGNORED if --target-samples is set (use --combined-budget "
                         "instead in that mode).")
    p.add_argument("--target-samples", type=int, default=None,
                    help="If set, switches rate derivation to aim for roughly this "
                         "many completed samples PER TYPE over --duration, instead of "
                         "an equal utilization split. Equal utilization (--rate-frac "
                         "applied uniformly) gives wildly unequal sample counts when "
                         "service times differ by orders of magnitude (e.g. batch's "
                         "~50s vs chat's ~1s) -- a 45s run can get 0 batch samples "
                         "while chat gets 30+. This mode fixes that by targeting equal "
                         "counts, then scaling ALL types down together (preserving "
                         "their ratio) only if that would exceed --combined-budget. "
                         "Types with an explicit --*-rate are left alone and their "
                         "utilization (if a matching --*-capacity-json is given) is "
                         "reserved out of the budget before splitting the rest.")
    p.add_argument("--combined-budget", type=float, default=0.7,
                    help="Target combined utilization ceiling (summed across all "
                         "active types, see warn_if_combined_unsustainable) used only "
                         "in --target-samples mode. Default 0.7.")

    p.add_argument("--offered-rho", type=float, default=None,
                    help="Matched offered-load mode: total NOMINAL utilization "
                         "rho = sum(rate_t * service_time_t) / slots, split across "
                         "types by --mix. Needs all three --*-capacity-json. "
                         "Mutually exclusive with --target-samples.")
    p.add_argument("--mix", default="8:2:1",
                    help="chat:moderate:batch request-COUNT ratio for --offered-rho "
                         "(default 8:2:1).")
    p.add_argument("--seed", type=int, default=None,
                    help="Seed for the pre-generated arrival schedule. Same seed + "
                         "same rates/duration => identical arrivals in every "
                         "scenario (matched comparison). Default: time-based, "
                         "recorded in the result file.")
    p.add_argument("--service-time-source", choices=["uncontended", "capacity"],
                    default="uncontended",
                    help="Which Step 1 level supplies the mean service time used to turn "
                         "capacity/rho into arrival rates. 'uncontended' (default) = the "
                         "largest tested concurrency <= min(capacity, --slots) (no queueing, SLO met); 'capacity' = "
                         "the capacity level (old behaviour, overstates service time when "
                         "capacity > slots).")
    p.add_argument("--label", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def health_check(host, port) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        print(f"  Reachable — status: {r.json().get('status', '')!r}")
        return True
    except Exception as exc:
        print(f"  Health check failed: {exc}")
        return False


def main():
    args = parse_args()
    label = args.label or f"scenario{args.scenario}_slots{args.slots}"
    output_path = args.output or f"results_{label}.json"

    def log(msg):
        print(msg)

    print(f"\nedge-llm-server — Continuous Mixed-Workload Stress Test (Step 2)")
    print(f"  Label          : {label}")
    print(f"  Scenario       : {args.scenario} ({SCENARIO_LABELS[args.scenario]})")
    print(f"  Target         : {args.host}:{args.port} "
          f"{'(proxy)' if args.port != 8081 else '(WARNING: looks like the raw server, not the proxy)'}")
    print(f"  Slots (info)   : {args.slots}")
    print(f"  Reservation    : reserved_slots={args.reserved_chat_slots}  "
          f"smart_preemption={args.smart_preemption}  (informational -- must match "
          f"what proxy.py was actually started with)")
    print(f"  Duration       : {args.duration}s per type, {args.arrival_pattern} arrivals")
    print(f"  Started        : {datetime.now().isoformat(timespec='seconds')}\n")

    _SVC["source"] = args.service_time_source
    _SVC["slots"] = args.slots
    print(f"  Service-time source: {_SVC['source']} (slots={args.slots})")
    print("  Resolving arrival rates …")
    rates, rate_details = {}, {}
    rates_and_mst = {}
    if args.offered_rho is not None:
        if args.target_samples is not None:
            sys.exit("--offered-rho and --target-samples are mutually exclusive.")
        print(f"  (offered-load mode: rho={args.offered_rho}, mix={args.mix})")
        capacity_jsons = {ct: getattr(args, f"{ct}_capacity_json") for ct in ("chat", "moderate", "batch")}
        resolved = resolve_rates_offered_load(
            args.offered_rho, args.mix, args.slots, args.duration, capacity_jsons, log)
        for ct, (rate, detail, mst) in resolved.items():
            rates[ct] = rate
            rate_details[ct] = detail
            rates_and_mst[ct] = (rate, mst)
    elif args.target_samples is not None:
        print(f"  (target-samples mode: aiming for {args.target_samples} samples/type, "
              f"combined-budget={args.combined_budget})")
        capacity_jsons = {ct: getattr(args, f"{ct}_capacity_json") for ct in ("chat", "moderate", "batch")}
        explicit_rates = {ct: getattr(args, f"{ct}_rate") for ct in ("chat", "moderate", "batch")}
        resolved = resolve_rates_target_samples(
            args.target_samples, args.duration, args.slots, capacity_jsons,
            explicit_rates, args.combined_budget, log)
        for ct, (rate, detail, mst) in resolved.items():
            rates[ct] = rate
            rate_details[ct] = detail
            rates_and_mst[ct] = (rate, mst)
    else:
        for ct in ("chat", "moderate", "batch"):
            explicit = getattr(args, f"{ct}_rate")
            cap_json = getattr(args, f"{ct}_capacity_json")
            rate, detail, mst = resolve_rate(ct, explicit, cap_json, args.slots, args.rate_frac, log)
            rates[ct] = rate
            rate_details[ct] = detail
            rates_and_mst[ct] = (rate, mst)

    warn_if_combined_unsustainable(rates_and_mst, args.slots, log)

    print("\n  Checking health …")
    if not health_check(args.host, args.port):
        sys.exit("  Proxy/server not reachable -- aborting.")

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=256, pool_maxsize=256)
    session.mount("http://", adapter)

    results = []
    results_lock = threading.Lock()
    in_flight = {"chat": 0, "moderate": 0, "batch": 0, "lock": threading.Lock()}

    seed = args.seed if args.seed is not None else int(time.time())
    types = ("chat", "moderate", "batch")
    schedules = {ct: build_schedule(ct, rates[ct], args.duration, args.arrival_pattern, seed)
                 for ct in types}
    digests = {ct: schedule_digest(schedules[ct]) for ct in types}
    print(f"\n  Arrival schedule: seed={seed} (same seed + rates + duration => "
          f"identical arrivals in every scenario)")
    for ct in types:
        print(f"    {ct:<9s} scheduled={len(schedules[ct]):5d}  digest={digests[ct]}")

    print(f"\n  Generating arrivals for {args.duration}s …")
    t0 = time.perf_counter()
    generation_start = t0
    thread_lists = {}

    def _run_generator(ct):
        thread_lists[ct] = scheduled_arrival_generator(
            ct, schedules[ct], t0, session, args.host, args.port, args.timeout,
            results, results_lock, in_flight, log)

    gen_threads = [threading.Thread(target=_run_generator, args=(ct,), daemon=True) for ct in types]
    for gt in gen_threads:
        gt.start()
    for gt in gen_threads:
        gt.join()
    all_request_threads = [t for ct in types for t in thread_lists.get(ct, [])]

    print(f"\n  Arrivals complete. Draining in-flight requests "
          f"(up to {args.drain_timeout}s) …")
    drain_deadline = time.perf_counter() + args.drain_timeout
    for t in all_request_threads:
        remaining = drain_deadline - time.perf_counter()
        if remaining <= 0:
            break
        t.join(timeout=remaining)
    still_running = sum(1 for t in all_request_threads if t.is_alive())
    drain_end_elapsed = time.perf_counter() - t0
    if still_running:
        print(f"  WARNING: {still_running} request(s) still in flight after the drain "
              f"timeout -- recorded as CENSORED (counted as SLO violations; tail "
              f"percentiles use a lower bound). Increase --drain-timeout to observe them.")

    with results_lock:
        recorded = list(results)
    recorded_ids = {r["client_id"] for r in recorded}
    censored_floors = {
        ct: [round(max(0.0, drain_end_elapsed - off), 3)
             for i, off in enumerate(schedules[ct]) if f"{ct}-{i}" not in recorded_ids]
        for ct in types
    }
    per_type = {ct: evaluate_type(ct, recorded, n_scheduled=len(schedules[ct]),
                                   censored_floors=censored_floors[ct], timeout=args.timeout)
                for ct in types}
    aggregate = compute_aggregate_metrics(
        recorded, generation_start, args.timeout,
        n_scheduled_total=sum(len(s) for s in schedules.values()),
        n_censored_total=sum(len(v) for v in censored_floors.values()),
        horizon_elapsed=drain_end_elapsed, slots=args.slots, duration=args.duration)

    print(f"\n{'=' * 78}")
    print(f"  CONTINUOUS STRESS SUMMARY — {label}")
    print(f"{'=' * 78}")
    for ct in types:
        s = per_type[ct]
        tl = s["tail"]
        mean_str = f"{s['stats']['mean']:.3f}s" if s['stats']['mean'] is not None else "n/a"
        p95_str = (f"{tl['p95']:.3f}s" + ("+" if tl["p95_is_lower_bound"] else "")) if tl["p95"] is not None else "n/a"
        max_str = (f"{tl['max']:.2f}s" + ("+" if tl["max_is_lower_bound"] else "")) if tl["max"] is not None else "n/a"
        drift = " [DEGRADING OVER RUN]" if s["degrading_over_run"] else ""
        print(f"  {ct:<9s} arrived={s['n_arrived']:4d} completed={s['n_completed']:4d} "
              f"censored={s['n_censored']:3d}  {s['metric']} mean={mean_str:>8s} "
              f"p95={p95_str:>9s} max={max_str:>8s}  "
              f"violation_rate={s['slo_violation_rate']} ci95={s['slo_violation_ci95']}{drift}")
    tct = aggregate["total_completion_time"]
    tct_str = (f"{tct:.1f}s" + ("+" if aggregate["total_completion_time_is_lower_bound"] else "")) if tct is not None else "n/a"
    print(f"\n  OVERALL   arrived={aggregate['total_arrived']:4d}  "
          f"completed={aggregate['total_completed']:4d}  "
          f"completion={aggregate['overall_completion_rate']}  total_completion_time={tct_str}")
    print("  ('+' = lower bound: some requests were still unfinished at the drain deadline)")

    output = {
        "meta": {
            "label": label,
            "scenario": args.scenario,
            "scenario_label": SCENARIO_LABELS[args.scenario],
            "host": args.host,
            "port": args.port,
            "slots": args.slots,
            "reserved_chat_slots": args.reserved_chat_slots,
            "smart_preemption": args.smart_preemption,
            "duration": args.duration,
            "arrival_pattern": args.arrival_pattern,
            "drain_timeout": args.drain_timeout,
            "timeout": args.timeout,
            "rates": rates,
            "rate_resolution": rate_details,
            "rate_frac": args.rate_frac,
            "service_time_source": args.service_time_source,
            "service_times": {ct: rates_and_mst[ct][1] for ct in rates_and_mst},
            "offered_rho": args.offered_rho,
            "mix": args.mix if args.offered_rho is not None else None,
            "seed": seed,
            "scheduled_arrivals": {ct: len(schedules[ct]) for ct in types},
            "schedule_digest": digests,
            "still_in_flight_after_drain": still_running,
            "timestamp": datetime.now().isoformat(),
        },
        "per_type": per_type,
        "aggregate": aggregate,
        "raw": sorted(recorded, key=lambda r: r["wave_arrival_s"]),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved -> {output_path}\n")


if __name__ == "__main__":
    main()