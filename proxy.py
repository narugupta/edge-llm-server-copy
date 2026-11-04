#!/usr/bin/env python3
"""
proxy.py — Priority Queue Proxy for edge-llm-server
=====================================================
Sits between clients and llama-server on port 8082.
Forwards to llama-server on port 8081.

What it does
  - Accepts requests tagged with X-Priority: high / medium / low
    (or inferred from X-Client-Id: chat / moderate / batch)
  - Holds requests in a priority queue: high > medium > low
  - Only forwards when the server has a free slot (semaphore-based,
    no polling gap)
  - Records queue_wait, ttft_at_proxy, token_generation separately
    (ttft_at_proxy = wall-clock from proxy-forward to first token
     received at proxy; NOT equivalent to server-side prompt_eval)

Usage
  # Terminal 1 — llama-server
  ./build/bin/llama-server \\
      -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \\
      --port 8081 --parallel 2 --ctx-size 2048 --kv-unified

  # Terminal 2 — proxy (defaults)
  python3 proxy.py

  # Terminal 2 — proxy (tuned for stress tests)
  python3 proxy.py --arrival-window 0.010 --dispatch-gap 0.005

  # Terminal 2 — proxy (reserve 1 of 2 slots for Chat)
  python3 proxy.py --reserved-chat-slots 1

  # Terminal 3 — evaluation
  python3 eval.py --port 8082 --server-port 8081 \\
      --label proxy_qos --output results_proxy.json

Design notes
  - SLOTS (= --parallel on server) set via --slots CLI arg (default 2).
  - ARRIVAL_WINDOW: dispatcher waits this many seconds after acquiring
    the semaphore before pulling from the queue, so simultaneous
    arrivals in the same burst are all enqueued before the priority
    decision is made.  Under heavy load you may want to increase this
    (e.g. 0.010–0.020s) to capture slower-arriving clients; under
    low load keep it small to avoid adding unnecessary latency.
    Tune with --arrival-window.
  - RESERVED_CHAT_SLOTS: number of physical slots the dispatcher will
    refuse to hand to Moderate/Batch requests, keeping them free for
    Chat even when Chat has nothing queued right now.  This addresses
    the no-preemption gap seen under staggered arrivals, where Chat
    could land behind an already-running long Batch/Moderate job with
    no way to interrupt it.  Configurable via --reserved-chat-slots;
    0 disables reservation and reproduces pre-reservation dispatch
    behaviour exactly.  There is currently no timeout that releases an
    unused reserved slot back to Batch — that policy question is still
    open.
  - Streaming is passed through chunk-by-chunk so TTFT measurement
    in the client remains accurate.
"""

import argparse
import json
import logging
import queue
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── CLI arguments ─────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Priority-queue proxy for edge-llm-server"
    )
    parser.add_argument(
        "--proxy-host", default="localhost",
        help="Interface to bind proxy on (default: localhost)",
    )
    parser.add_argument(
        "--proxy-port", default=8082, type=int,
        help="Port proxy listens on (default: 8082)",
    )
    parser.add_argument(
        "--server-host", default="localhost",
        help="llama-server host (default: localhost)",
    )
    parser.add_argument(
        "--server-port", default=8081, type=int,
        help="llama-server port (default: 8081)",
    )
    parser.add_argument(
        "--slots", default=2, type=int,
        help="Number of inference slots — must match --parallel on server (default: 2)",
    )
    parser.add_argument(
        "--arrival-window", default=0.005, type=float,
        metavar="SECONDS",
        help=(
            "Seconds the dispatcher waits after acquiring a slot before "
            "pulling from the priority queue.  This allows simultaneous "
            "arrivals to all enqueue before the dispatch decision is made. "
            "Increase for stress tests with many concurrent clients. "
            "(default: 0.005)"
        ),
    )
    parser.add_argument(
        "--dispatch-gap", default=0.005, type=float,
        metavar="SECONDS",
        help=(
            "Seconds the dispatcher sleeps after each pop before looping back "
            "to acquire the next slot.  This staggers successive dispatch "
            "decisions so that when multiple slots are free simultaneously, "
            "each slot's arrival window starts AFTER the previous slot's pop, "
            "ensuring all burst arrivals are queued before every priority "
            "decision.  Without this, two concurrent slot acquisitions sleep "
            "their arrival windows in parallel and both pop from the same "
            "queue snapshot, which can let a lower-priority client (that "
            "arrived first) claim a slot ahead of a higher-priority one that "
            "arrived 1-2ms later.  Set to 0 to disable (reverts to v5 "
            "behaviour).  (default: 0.005)"
        ),
    )
    parser.add_argument(
        "--reserved-chat-slots", default=1, type=int,
        metavar="N",
        help=(
            "Number of physical slots to keep reserved for Chat (priority "
            "0) requests. The dispatcher will not forward a Moderate or "
            "Batch request if doing so would leave fewer than N slots free "
            "for Chat -- even if Chat has nothing queued at that moment. "
            "This removes the wait Chat otherwise experiences behind an "
            "already-running Moderate/Batch request under staggered "
            "arrivals (see July 2 staggered-arrival results). There is no "
            "timeout yet that releases an unused reserved slot back to "
            "Batch; it stays reserved indefinitely. Must satisfy "
            "0 <= N <= --slots. Set to 0 to disable reservation and "
            "reproduce pre-reservation dispatch behaviour exactly. "
            "Ignored if --smart-preemption is set. (default: 1)"
        ),
    )
    parser.add_argument(
        "--smart-preemption", action="store_true",
        help=(
            "Adaptive alternative to --reserved-chat-slots. Instead of a "
            "fixed N, holds at most 1 slot open for Chat, and ONLY while "
            "Chat is demonstrably active right now (queued or currently "
            "being served) -- not indefinitely regardless of demand. "
            "Fixes the measured case where a statically-reserved slot sat "
            "idle for ~31s in a wave with no Chat activity at all, while "
            "Moderate waited behind Batch on the single remaining shared "
            "slot. Takes precedence over --reserved-chat-slots if both "
            "are given."
        ),
    )
    parser.add_argument(
        "--chat-hold-seconds", default=0.0, type=float, metavar="T",
        help=(
            "S3b: only with --smart-preemption. Keep the 1-slot Chat "
            "reservation for T seconds after the LAST Chat activity (a Chat "
            "arrival, dispatch or completion), instead of releasing it the "
            "instant Chat is neither queued nor in flight. T=0 reproduces "
            "plain --smart-preemption exactly; very large T approaches the "
            "static --reserved-chat-slots 1 behaviour. Like smart-preemption "
            "this never interrupts a running request. (default: 0)"
        ),
    )
    parser.add_argument(
        "--preempt-batch", action="store_true",
        help=(
            "S5: real preemption by abort-and-requeue. When a Chat request "
            "is waiting and no slot is free, abort the running Batch request "
            "that has done the LEAST work (close its connection to "
            "llama-server, which frees the slot) and put it back in the "
            "queue to restart from scratch. Batch responses are buffered by "
            "the proxy until complete so the client never sees a partial "
            "stream twice. Only streaming (stream=true) Batch requests are "
            "ever aborted. Use with --reserved-chat-slots 0."
        ),
    )
    parser.add_argument(
        "--preempt-for-moderate", action="store_true",
        help="With --preempt-batch: Moderate requests may also abort Batch.",
    )
    parser.add_argument(
        "--max-preemptions", default=2, type=int, metavar="N",
        help=(
            "With --preempt-batch: a Batch request is aborted at most N "
            "times, then it is no longer a victim (bounds wasted work and "
            "livelock). (default: 2)"
        ),
    )
    parser.add_argument(
        "--pin-slots", action="store_true",
        help=(
            "Assign each request an explicit llama-server slot (id_slot): Chat gets "
            "the LOWEST free slot id, Moderate/Batch the HIGHEST. Measured: llama-server "
            "fills each iteration's prompt tokens in slot-index order, so a Chat request "
            "sitting in a higher-numbered slot than a request that is prefilling waits for "
            "the whole prefill (about 3.6-4.3 s for 400 tokens), while Chat in the lower "
            "slot waits about 0.6 s (with -b 32 -ub 32)."
        ),
    )
    args = parser.parse_args()

    if args.max_preemptions < 0:
        parser.error("--max-preemptions must be >= 0")
    if args.chat_hold_seconds < 0:
        parser.error("--chat-hold-seconds must be >= 0")
    if args.chat_hold_seconds > 0 and not args.smart_preemption:
        parser.error("--chat-hold-seconds requires --smart-preemption")
    if args.reserved_chat_slots < 0:
        parser.error("--reserved-chat-slots must be >= 0")
    if args.reserved_chat_slots > args.slots:
        parser.error(
            f"--reserved-chat-slots ({args.reserved_chat_slots}) cannot "
            f"exceed --slots ({args.slots})"
        )
    return args


# ── Configuration (populated at startup from CLI args) ───────────────────────

_cfg: argparse.Namespace  # set in main()

PRIORITY_MAP = {
    "high":   0,
    "medium": 1,
    "low":    2,
    "chat":     0,
    "moderate": 1,
    "batch":    2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [proxy] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proxy")

# ── Shared state ─────────────────────────────────────────────────────────────

_pq: queue.PriorityQueue = queue.PriorityQueue()
_seq_lock    = threading.Lock()
_seq_counter = 0

# Semaphore starts at SLOTS (all free).
# Dispatcher acquire()s before forwarding; _forward release()s when done.
_slot_sem: threading.Semaphore  # initialised in main()

# Tracks how many *physical* slots are currently occupied, broken down by
# priority class (0=chat, 1=moderate, 2=batch).  Used only for the
# --reserved-chat-slots policy; the semaphore above remains the sole source
# of truth for "is a physical slot free at all".
_active_by_priority = {0: 0, 1: 0, 2: 0}
_active_lock = threading.Lock()

# perf_counter() time of the most recent Chat activity (arrival, dispatch or
# completion); None until the first Chat request is seen. Only read when
# --chat-hold-seconds > 0.
_last_chat_activity: Optional[float] = None


def _touch_chat():
    """Record 'Chat was active just now' (used by --chat-hold-seconds)."""
    global _last_chat_activity
    _last_chat_activity = time.perf_counter()


# S5 (--preempt-batch) state
_free_slot_ids: set = set()      # slot ids not currently assigned (under _active_lock); --pin-slots
_running: set = set()            # Batch requests currently holding a slot (under _active_lock)
_dispatcher_claim = False        # dispatcher holds an acquired slot it has not yet counted
_preempt_stats = {"preemptions": 0, "wasted_s": 0.0, "completed_after_preempt": 0,
                  "max_preempts_one_request": 0}
_preempt_lock = threading.Lock()

_results: list[dict] = []
_results_lock = threading.Lock()


def _release_slot_id(req):
    """Give the request's slot id back (call under _active_lock)."""
    if req.slot_id is not None:
        _free_slot_ids.add(req.slot_id)
        req.slot_id = None


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ── Pending request ───────────────────────────────────────────────────────────

class PendingRequest:
    def __init__(self, method: str, path: str, headers: dict,
                 body: bytes, priority: int, client_id: str, req_id: str):
        self.method     = method
        self.path       = path
        self.headers    = headers
        self.body       = body
        self.priority   = priority
        self.client_id  = client_id
        self.req_id     = req_id
        self.t_arrived  = time.perf_counter()
        self.ready_event: threading.Event = threading.Event()
        self.response_status: Optional[int] = None
        self.response_headers: Optional[list] = None
        self.response_body_iter = None

        self.timing = {}

        self.t_queue_enter = None
        # Set True by the client-facing handler if ready_event.wait() times
        # out (130s) before dispatch. The dispatcher checks this and drops
        # the request instead of spending a physical slot on a response
        # nobody is listening for anymore.
        self.abandoned = False
        self.t_dispatch = None
        self.t_first_token = None
        self.t_complete = None
        # S5 (--preempt-batch)
        self.slot_id = None
        self.seq = None
        self.is_stream = False
        self.upstream = None
        self.preempt_requested = False
        self.preempt_count = 0
        self.t_start = None
        self.wasted_s = 0.0


# ── Slot dispatcher ───────────────────────────────────────────────────────────

def _chat_present() -> bool:
    """
    True if a Chat request is currently queued (arrived, not yet dispatched)
    OR currently being served (in-flight, mid-generation).

    Used by --smart-preemption to decide whether the "reserved" slot should
    be held open right now. Checking ONLY "queued" would be nearly a no-op:
    by the time the dispatcher is evaluating a non-Chat candidate, that
    candidate was already popped as the head of the priority queue, which
    -- given the heap's own ordering -- is only possible if no Chat request
    was waiting at that exact instant (Chat would have been the head
    instead). So "queued" alone almost never fires when it matters.

    "Currently being served" is the signal that actually targets the
    measured problem: static reservation held a slot open for ~31 seconds
    in one scale-sweep wave where Chat neither arrived nor ran at all
    during that time. Checking in-flight activity too means the
    reservation naturally releases the moment Chat goes idle, rather than
    holding a slot open indefinitely regardless of whether Chat needs it.
    """
    with _active_lock:
        if _active_by_priority[0] > 0:
            return True
    with _pq.mutex:
        if any(item[0] == 0 for item in _pq.queue):
            return True
    # S3b hold timer: Chat was active within the last T seconds.
    hold = _cfg.chat_hold_seconds
    if hold > 0:
        last = _last_chat_activity
        if last is not None and (time.perf_counter() - last) <= hold:
            return True
    return False


def _reservation_blocks(priority: int) -> bool:
    """
    True if dispatching a request of this priority right now would violate
    the active reservation policy -- either the static --reserved-chat-slots
    constant, or, if --smart-preemption is set, a live 0-or-1 reservation
    that only holds a slot open while Chat is demonstrably active (queued
    or in-flight) right now, per _chat_present() above.

    Chat (priority 0) is never blocked by its own reservation. With
    --reserved-chat-slots 0 and --smart-preemption unset, this always
    returns False, reproducing pre-reservation dispatch behaviour exactly.

    --smart-preemption takes precedence over --reserved-chat-slots when
    both are set (logged once at startup so this isn't silently surprising).
    """
    if priority == 0:
        return False

    if _cfg.smart_preemption:
        effective_reserved = 1 if _chat_present() else 0
    elif _cfg.reserved_chat_slots > 0:
        effective_reserved = _cfg.reserved_chat_slots
    else:
        return False

    with _active_lock:
        non_chat_active = _active_by_priority[1] + _active_by_priority[2]
    return non_chat_active >= (_cfg.slots - effective_reserved)


def _dispatcher():
    while True:
        # WAIT for something in the queue that is actually dispatchable
        # under the reservation policy (no slot held yet). If the
        # highest-priority item is non-chat and reservation forbids it,
        # put it back and keep polling -- a chat arrival will naturally
        # become queue-head next (priority queue ordering), or the
        # reservation will clear as an already-running non-chat request
        # completes and decrements _active_by_priority.
        req = None
        while req is None:
            try:
                priority, seq, candidate = _pq.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.01)
                continue

            if candidate.abandoned:
                # Client already gave up (504, 130s) while this sat queued.
                # Drop it here rather than let it consume a physical slot
                # for a response nobody will read -- this is what caused
                # queue backlog to compound under sustained high load.
                waited = time.perf_counter() - candidate.t_queue_enter
                log.info(f"[{candidate.client_id:8s}] dropped (abandoned, "
                         f"waited={waited:.1f}s)")
                continue

            if _reservation_blocks(candidate.priority):
                _pq.put((priority, seq, candidate))
                time.sleep(0.01)
                continue

            req = candidate

        # Put it back; now acquire the semaphore and re-collect with arrival window
        _pq.put((priority, seq, req))

        _slot_sem.acquire()
        global _dispatcher_claim
        _dispatcher_claim = True
        time.sleep(_cfg.arrival_window)   # now all burst clients can arrive

        # Pop the highest-priority request
        priority, seq, req = _pq.get()

        if req.abandoned:
            waited = time.perf_counter() - req.t_queue_enter
            log.info(f"[{req.client_id:8s}] dropped (abandoned, "
                     f"waited={waited:.1f}s)")
            _dispatcher_claim = False
            _slot_sem.release()
            continue

        # Re-check: a higher-priority (still non-chat) request could have
        # taken queue-head during the arrival window, or another dispatch
        # could -- in principle -- have tightened the reservation in the
        # meantime. If this pop is no longer dispatchable, release the
        # slot we're not using and retry rather than forwarding it.
        if _reservation_blocks(req.priority):
            _pq.put((priority, seq, req))
            _dispatcher_claim = False
            _slot_sem.release()
            continue

        req.timing["queue_wait"] = time.perf_counter() - req.t_arrived

        req.t_start = time.perf_counter()
        with _active_lock:
            if _cfg.pin_slots and _free_slot_ids:
                req.slot_id = (min(_free_slot_ids) if req.priority == 0
                               else max(_free_slot_ids))
                _free_slot_ids.discard(req.slot_id)
            _active_by_priority[req.priority] += 1
            if req.priority == 2:
                _running.add(req)
            _dispatcher_claim = False
        if req.priority == 0:
            _touch_chat()

        t = threading.Thread(target=_forward, args=(req,), daemon=True)
        t.start()

        if _cfg.dispatch_gap > 0:
            time.sleep(_cfg.dispatch_gap)


def _forward(req: PendingRequest):
    """
    Forward one request to llama-server, stream the response back,
    record split timings, then release the semaphore.
    """
    url = f"http://{_cfg.server_host}:{_cfg.server_port}{req.path}"
    body = req.body
    if _cfg.pin_slots and req.slot_id is not None and body:
        try:
            j = json.loads(body)
            if isinstance(j, dict):
                j["id_slot"] = req.slot_id
                body = json.dumps(j).encode()
        except Exception:
            pass
    http_req = Request(url, data=body, method=req.method)
    for k, v in req.headers.items():
        if k.lower() in ("host", "content-length", "transfer-encoding"):
            continue
        http_req.add_header(k, v)
    http_req.add_header("Content-Length", str(len(body)))

    t_forward = time.perf_counter()
    req.timing["t_forward"] = t_forward

    try:
        resp = urlopen(http_req, timeout=120)
    except URLError as e:
        req.response_status = 502
        req.response_headers = []
        req.response_body_iter = iter([f"Proxy error: {e}".encode()])
        req.timing["prompt_eval"]      = 0.0
        req.timing["token_generation"] = 0.0
        req.ready_event.set()
        if req.priority == 0:
            _touch_chat()
        with _active_lock:
            _active_by_priority[req.priority] -= 1
            _running.discard(req)
            _release_slot_id(req)
        _slot_sem.release()
        return

    if _cfg.preempt_batch and req.priority == 2 and req.is_stream:
        _forward_buffered(req, resp, t_forward)
        return

    req.response_status  = resp.status
    req.response_headers = list(resp.headers.items())

    first_token_seen = False
    t_first_token    = None

    def _stream():
        nonlocal first_token_seen, t_first_token
        try:
            for raw_line in resp:
                if not first_token_seen:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                chunk = json.loads(data_str)
                                delta = (chunk.get("choices", [{}])[0]
                                             .get("delta", {})
                                             .get("content", ""))
                                if delta:
                                    t_first_token = time.perf_counter()
                                    req.t_first_token = t_first_token
                                    first_token_seen = True
                            except (json.JSONDecodeError, IndexError):
                                pass
                yield raw_line
        finally:
            t_done = time.perf_counter()
            req.t_complete = t_done
            req.timing["prompt_eval"] = (
                (t_first_token - t_forward) if t_first_token else 0.0
            )
            req.timing["token_generation"] = (
                (t_done - t_first_token) if t_first_token else 0.0
            )
            req.timing["total_service_time"] = t_done - t_forward

            if req.priority == 0:
                _touch_chat()
            with _active_lock:
                _active_by_priority[req.priority] -= 1
                _release_slot_id(req)
            _slot_sem.release()

            record = {
                "dispatch_to_first_token": round(
                    (req.t_first_token - req.t_dispatch)
                    if req.t_first_token and req.t_dispatch
                    else 0,
                    4,
                ),
                "req_id":           req.req_id,
                "client_id":        req.client_id,
                "priority":         req.priority,
                "queue_wait":       round(req.timing.get("queue_wait", 0), 4),
                "ttft_at_proxy":    round(req.timing.get("prompt_eval", 0), 4),
                "token_generation": round(req.timing.get("token_generation", 0), 4),
                "total_service":    round(req.timing.get("total_service_time", 0), 4),
                "total_e2e":        round(
                    req.timing.get("queue_wait", 0) +
                    req.timing.get("total_service_time", 0), 4
                ),
            }
            with _results_lock:
                _results.append(record)

            log.info(
                f"[{req.client_id:8s}] "
                f"qw={record['queue_wait']:.3f}s "
                f"bw={record['dispatch_to_first_token']:.3f}s "
                f"ttft={record['ttft_at_proxy']:.3f}s "
                f"tg={record['token_generation']:.3f}s "
                f"e2e={record['total_e2e']:.3f}s"
            )

    req.response_body_iter = _stream()
    req.ready_event.set()


def _abort_upstream(req: PendingRequest):
    """Kill the proxy->llama-server connection of a running request. The
    server notices the disconnect and frees the slot; our reader thread sees
    EOF/an error and handles the requeue in _forward_buffered."""
    resp = req.upstream
    if resp is None:
        return
    try:
        resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        resp.close()
    except Exception:
        pass


def _forward_buffered(req: PendingRequest, resp, t_forward: float):
    """Batch path under --preempt-batch: read the whole upstream stream into
    memory, then hand it to the client in one go. If we are asked to abort
    while it is still running (and it has not finished), discard the partial
    output, requeue the request at its original queue position, free the slot."""
    req.upstream = resp
    lines, saw_done, t_first = [], False, None
    try:
        for raw in resp:
            lines.append(raw)
            if t_first is None:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    d = line[5:].strip()
                    if d and d != "[DONE]":
                        try:
                            delta = (json.loads(d).get("choices", [{}])[0]
                                     .get("delta", {}).get("content", ""))
                            if delta:
                                t_first = time.perf_counter()
                        except (json.JSONDecodeError, IndexError):
                            pass
            if raw.strip() == b"data: [DONE]":
                saw_done = True
    except Exception:
        pass
    finally:
        try:
            resp.close()
        except Exception:
            pass
    t_done = time.perf_counter()
    req.upstream = None

    with _active_lock:
        _active_by_priority[2] -= 1
        _running.discard(req)
        _release_slot_id(req)

    if req.preempt_requested and not saw_done:
        elapsed = t_done - t_forward
        req.wasted_s += elapsed
        req.preempt_count += 1
        with _preempt_lock:
            _preempt_stats["preemptions"] += 1
            _preempt_stats["wasted_s"] += elapsed
            _preempt_stats["max_preempts_one_request"] = max(
                _preempt_stats["max_preempts_one_request"], req.preempt_count)
        log.info(f"[{req.client_id:8s}] PREEMPTED after {elapsed:.1f}s "
                 f"(#{req.preempt_count}) -> requeued")
        req.preempt_requested = False
        req.timing = {}
        req.t_first_token = None
        _pq.put((req.priority, req.seq, req))   # requeue BEFORE freeing the slot
        _slot_sem.release()
        return

    req.t_first_token = t_first
    req.t_complete = t_done
    req.timing["prompt_eval"] = (t_first - t_forward) if t_first else 0.0
    req.timing["token_generation"] = (t_done - t_first) if t_first else 0.0
    req.timing["total_service_time"] = t_done - t_forward
    if req.preempt_count:
        with _preempt_lock:
            _preempt_stats["completed_after_preempt"] += 1
    record = {
        "dispatch_to_first_token": 0,
        "req_id": req.req_id, "client_id": req.client_id, "priority": req.priority,
        "queue_wait": round(req.timing.get("queue_wait", 0), 4),
        "ttft_at_proxy": round(req.timing["prompt_eval"], 4),
        "token_generation": round(req.timing["token_generation"], 4),
        "total_service": round(req.timing["total_service_time"], 4),
        "total_e2e": round(req.timing.get("queue_wait", 0) + req.timing["total_service_time"], 4),
        "preemptions": req.preempt_count,
        "wasted_s": round(req.wasted_s, 4),
    }
    with _results_lock:
        _results.append(record)
    log.info(f"[{req.client_id:8s}] qw={record['queue_wait']:.3f}s "
             f"svc={record['total_service']:.3f}s e2e={record['total_e2e']:.3f}s "
             f"preempts={req.preempt_count}")
    req.response_status = resp.status
    req.response_headers = list(resp.headers.items())
    req.response_body_iter = iter(lines)
    req.ready_event.set()
    _slot_sem.release()


def _preemptor():
    """S5: when requests that are allowed to preempt (Chat, optionally
    Moderate) are waiting and no slot is, or is about to be, free, abort the
    youngest running streaming Batch request(s)."""
    thr = 1 if _cfg.preempt_for_moderate else 0
    while True:
        time.sleep(0.02)
        with _pq.mutex:
            n_wait = sum(1 for it in _pq.queue
                         if it[0] <= thr and not it[2].abandoned)
        if not n_wait:
            continue
        with _active_lock:
            total_active = sum(_active_by_priority.values())
            free = _cfg.slots - total_active - (1 if _dispatcher_claim else 0)
            pending = sum(1 for r in _running if r.preempt_requested)
            victims = [r for r in _running
                       if r.is_stream and r.upstream is not None
                       and not r.preempt_requested
                       and r.preempt_count < _cfg.max_preemptions]
        need = n_wait - max(free, 0) - pending
        if need <= 0 or not victims:
            continue
        victims.sort(key=lambda r: r.t_start or 0, reverse=True)   # least work done first
        for v in victims[:need]:
            v.preempt_requested = True
            _abort_upstream(v)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _infer_priority(self, headers: dict) -> tuple[int, str]:
        p_header   = headers.get("x-priority", "").lower().strip()
        cid_header = headers.get("x-client-id", "").lower().strip()
        client_id  = cid_header or "unknown"

        if p_header in PRIORITY_MAP:
            return PRIORITY_MAP[p_header], client_id
        if cid_header in PRIORITY_MAP:
            return PRIORITY_MAP[cid_header], client_id
        return 1, client_id

    def do_POST(self):
        self._handle()

    def do_GET(self):
        if self.path == "/health":
            self._passthrough_get()
            return
        self._handle()

    def _passthrough_get(self):
        url = f"http://{_cfg.server_host}:{_cfg.server_port}{self.path}"
        try:
            resp = urlopen(url, timeout=5)
            body = resp.read()
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() in ("transfer-encoding",):
                    continue
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _handle(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b""
        hdrs   = {k.lower(): v for k, v in self.headers.items()}

        priority, client_id = self._infer_priority(hdrs)
        req_id = str(uuid.uuid4())[:8]

        req = PendingRequest(
            method    = self.command,
            path      = self.path,
            headers   = dict(self.headers),
            body      = body,
            priority  = priority,
            client_id = client_id,
            req_id    = req_id,
        )

        if priority == 0:
            _touch_chat()
        seq = _next_seq()
        req.seq = seq
        try:
            req.is_stream = bool(json.loads(body or b"{}").get("stream"))
        except Exception:
            req.is_stream = False
        log.info(f"[{client_id:8s}] arrived  priority={priority}  seq={seq}")
        req.t_queue_enter = time.perf_counter()
        _pq.put((priority, seq, req))

        req.ready_event.wait(timeout=130)

        if req.response_status is None:
            req.abandoned = True
            self.send_response(504)
            self.end_headers()
            self.wfile.write(b"Proxy timeout")
            return

        self.send_response(req.response_status)
        sent_headers = set()
        for k, v in (req.response_headers or []):
            if k.lower() in ("transfer-encoding",):
                continue
            if k.lower() not in sent_headers:
                self.send_header(k, v)
                sent_headers.add(k.lower())
        self.end_headers()

        try:
            for chunk in req.response_body_iter:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class ProxyHandlerWithResults(ProxyHandler):
    def do_GET(self):
        if self.path == "/proxy/results":
            with _results_lock:
                body = json.dumps(_results, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/proxy/stats":
            with _preempt_lock:
                body = json.dumps(dict(_preempt_stats,
                                       preempt_batch=_cfg.preempt_batch,
                                       max_preemptions=_cfg.max_preemptions),
                                  indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _cfg, _slot_sem
    _cfg = _parse_args()
    _slot_sem = threading.Semaphore(_cfg.slots)
    _free_slot_ids.update(range(_cfg.slots))

    dt = threading.Thread(target=_dispatcher, daemon=True, name="dispatcher")
    dt.start()
    if _cfg.preempt_batch:
        threading.Thread(target=_preemptor, daemon=True, name="preemptor").start()

    server = ThreadingHTTPServer(
        (_cfg.proxy_host, _cfg.proxy_port), ProxyHandlerWithResults
    )
    log.info(f"Priority proxy listening on {_cfg.proxy_host}:{_cfg.proxy_port}")
    log.info(f"Forwarding to llama-server at {_cfg.server_host}:{_cfg.server_port}")
    log.info(f"Slot capacity  : {_cfg.slots}")
    log.info(f"Arrival window : {_cfg.arrival_window * 1000:.1f} ms")
    log.info(f"Dispatch gap   : {_cfg.dispatch_gap * 1000:.1f} ms  (set 0 to disable)")
    if _cfg.smart_preemption:
        log.info(
            "Reservation    : SMART-PREEMPTION (adaptive) -- holds at most "
            "1 slot for Chat, only while Chat is queued or in-flight right "
            "now; releases automatically when Chat goes idle"
            + (f" (+ S3b hold: keeps it {_cfg.chat_hold_seconds:g}s after the "
               f"last Chat activity)" if _cfg.chat_hold_seconds > 0 else "")
        )
        if _cfg.reserved_chat_slots != 1:
            log.info(
                f"               (--reserved-chat-slots {_cfg.reserved_chat_slots} "
                f"was also given but is ignored -- smart-preemption takes precedence)"
            )
    elif _cfg.reserved_chat_slots > 0:
        log.info(
            f"Reserved slots : {_cfg.reserved_chat_slots} of {_cfg.slots} "
            f"held for Chat (Moderate/Batch capped at "
            f"{_cfg.slots - _cfg.reserved_chat_slots} concurrent), static, "
            f"never released"
        )
    else:
        log.info("Reserved slots : 0 (reservation disabled)")
    if _cfg.pin_slots:
        log.info("Slot pinning   : ON (Chat -> lowest free slot id, Moderate/Batch -> highest)")
    if _cfg.preempt_batch:
        log.info(
            f"Preemption     : ABORT-AND-REQUEUE Batch for "
            f"{'Chat+Moderate' if _cfg.preempt_for_moderate else 'Chat'}; "
            f"max {_cfg.max_preemptions} per request; Batch responses buffered "
            f"(stats at /proxy/stats)"
        )
    log.info("Columns: queue_wait | ttft_at_proxy | token_gen | e2e")
    log.info("─" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down proxy.")
        if _results:
            out = "proxy_results.json"
            with open(out, "w") as f:
                json.dump(_results, f, indent=2)
            log.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()