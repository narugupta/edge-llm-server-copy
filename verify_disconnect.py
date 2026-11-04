#!/usr/bin/env python3
"""verify_disconnect.py -- does the REAL llama-server free a slot when the client disconnects?
Needed before trusting --preempt-batch. Start the server with ONE slot:
  ./build/bin/llama-server -m Qwen2.5-1.5B-Q4_K_M.gguf --port 8081 --parallel 1 --ctx-size 2048 --kv-unified
then:  python3 verify_disconnect.py
Test A: start a long generation, kill its socket after 2 s, immediately send a short request,
        measure that request's time-to-first-token. PASS if it is < 3 s.
Test B (control): start a long generation, do NOT kill it, send the short request:
        it should wait for the long one (TTFT of many seconds).  This shows A's PASS is real."""
import json, socket, sys, threading, time, urllib.request
HOST, PORT = "localhost", int(sys.argv[1]) if len(sys.argv) > 1 else 8081
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
def body(prompt, n):
    return json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                       "stream": True, "temperature": 0.7,
                       "ignore_eos": n > 100}).encode()   # long requests must not stop early
def short_ttft():
    r = urllib.request.Request(URL, data=body("Say hi.", 8), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        for l in resp:
            if l.startswith(b"data:") and b"content" in l:
                return time.time() - t0
    return float("nan")
def long_req(holder, n=600):
    r = urllib.request.Request(URL, data=body("Write a very long story about a dragon.", n),
                               headers={"Content-Type": "application/json"})
    holder["resp"] = urllib.request.urlopen(r, timeout=300)
    holder["t0"] = time.time()
    try:
        for l in holder["resp"]:
            pass
    except Exception:
        pass
    holder["end"] = time.time()
def run(kill):
    h = {}
    th = threading.Thread(target=long_req, args=(h,)); th.start()
    time.sleep(2.0)
    if not th.is_alive():
        sys.exit("INVALID: the long request finished before 2 s (generation too short). "
                 "Rerun; if it repeats, raise max_tokens in long_req().")
    if kill:
        try: h["resp"].fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except Exception as e: print("shutdown error:", e)
        try: h["resp"].close()
        except Exception: pass
    if "resp" not in h:
        sys.exit("long request failed to start -- test invalid, rerun")
    t = short_ttft()
    th.join(timeout=300)
    return t, h.get("end", time.time()) - h.get("t0", time.time())
def wait_ready(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                if r.status == 200: return
        except Exception:
            pass
        time.sleep(1)
    sys.exit("server not ready (is llama-server running on this port?)")
wait_ready()
print("Test B (control, no abort) ...")
tb, lb = run(False)
print(f"  short request TTFT = {tb:.2f}s   (long request lasted {lb:.1f}s)")
time.sleep(3)
print("Test A (abort the long request after 2 s) ...")
ta, la = run(True)
print(f"  short request TTFT = {ta:.2f}s   (long request ended after {la:.1f}s)")
print()
if ta < 3.0 and tb > ta + 3:
    print("PASS: disconnect frees the slot; --preempt-batch is valid on this server.")
else:
    print("FAIL/INCONCLUSIVE: the slot was not freed promptly (or the control did not queue).")
    print("  -> do NOT trust --preempt-batch results; check llama-server version/flags.")