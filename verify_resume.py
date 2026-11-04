#!/usr/bin/env python3
"""verify_resume.py -- what does it cost to ABORT a Batch generation and RESUME it later
(by re-sending prompt + the text generated so far)?  Also: does a long prefill in one slot
stall Chat TTFT in the other slot (the --ubatch-size question)?

Start the server FIRST, with 2 slots, e.g. (run this script once per server setting):
  A) default ubatch:   ./build/bin/llama-server -m Qwen2.5-1.5B-Q4_K_M.gguf --port 8081 --parallel 2 --ctx-size 2048 --kv-unified --cache-ram 1024
  B) small ubatch:     ./build/bin/llama-server -m Qwen2.5-1.5B-Q4_K_M.gguf --port 8081 --parallel 2 --ctx-size 2048 --kv-unified --cache-ram 1024 --ubatch-size 64
then:  python3 verify_resume.py --label default_ub      (and again with --label ub64)
Takes about 5-8 minutes. Writes resume_<label>.json. Uses the raw /completion endpoint with the
Qwen ChatML template written by hand (so the proxy could do the same).
Part 1: prefill speed at 100/300/600 tokens (no cache).
Part 2: resume cost: (a) immediately after the abort, (b) after 3 other short requests ran,
        (c) reference: same text with NO cache available (cold prefill).
Part 3: Chat TTFT alone / while the other slot decodes / while the other slot prefills ~400 tokens."""
import argparse, http.client, json, random, socket, statistics as st, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--label", default="run")
ap.add_argument("--prompt-tokens", type=int, default=350, help="Batch prompt size (tokens)")
ap.add_argument("--gen", type=int, default=300, help="Batch generation length (tokens)")
ap.add_argument("--abort-after", type=int, default=150, help="abort after this many generated tokens")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--parts", default="123", help="which parts to run, e.g. 3 for the Chat-interference test only")
args = ap.parse_args()

def post(path, body, timeout=300):
    c = http.client.HTTPConnection(args.host, args.port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    if r.status != 200: sys.exit(f"{path} -> HTTP {r.status}: {data[:200]!r}")
    return json.loads(data)

def ntok(text):
    return len(post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"])

def nonce(): return "ref%08d" % random.randrange(10**8)
SENT = ("The harbour town woke slowly as fishing boats returned, their nets heavy with silver, "
        "and merchants unrolled striped awnings over crates of citrus, salt, and tangled rope. ")
def chatml(user): return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                          f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
def prompt_of(n_tokens, nc=None):
    """ChatML prompt of about n_tokens tokens; a unique nonce at the START prevents any cache reuse."""
    nc = nc or nonce(); body = ""
    while ntok(chatml(f"{nc} Summarise and continue: " + body)) < n_tokens: body += SENT
    return chatml(f"{nc} Summarise and continue: " + body), nc

def stream(prompt, n_predict, cache=True, abort_after=None, id_slot=None):
    """POST /completion (stream). Returns dict(ttft,total,text,chunks,timings,aborted)."""
    body = {"prompt": prompt, "n_predict": n_predict, "stream": True, "cache_prompt": cache,
            "temperature": 0.7, "ignore_eos": True}
    if id_slot is not None: body["id_slot"] = id_slot     # pin the request to one server slot
    c = http.client.HTTPConnection(args.host, args.port, timeout=300)
    t0 = time.perf_counter()
    c.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200: sys.exit(f"/completion HTTP {r.status}")
    out = {"ttft": None, "text": "", "chunks": 0, "timings": None, "aborted": False}
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if not d or d == "[DONE]": continue
        j = json.loads(d)
        if j.get("content"):
            if out["ttft"] is None: out["ttft"] = time.perf_counter() - t0
            out["text"] += j["content"]; out["chunks"] += 1
        if j.get("timings"): out["timings"] = j["timings"]
        if abort_after and out["chunks"] >= abort_after:
            try: c.sock.shutdown(socket.SHUT_RDWR)
            except Exception: pass
            out["aborted"] = True; break
    out["total"] = time.perf_counter() - t0
    try: c.close()
    except Exception: pass
    return out

def med(v): return st.median(v) if v else float("nan")
def tm(o, k): return (o["timings"] or {}).get(k)
R = {"label": args.label, "args": vars(args)}

# --- wait for the server
for _ in range(120):
    try:
        c = http.client.HTTPConnection(args.host, args.port, timeout=2); c.request("GET", "/health")
        if c.getresponse().status == 200: break
    except Exception: pass
    time.sleep(1)
else: sys.exit("server not ready")
stream(prompt_of(50)[0], 4)   # warm-up

# --- Part 1: prefill speed
print("Part 1: prefill speed (idle server, no cache)")
R["prefill"] = {}
for n in ((100, 300, 600) if "1" in args.parts else ()):
    v = []
    for _ in range(args.reps):
        p, _nc = prompt_of(n); o = stream(p, 1, cache=False); v.append(o["ttft"])
    R["prefill"][n] = med(v)
    print(f"  ~{n:3d} prompt tokens: first-token {med(v)*1000:6.0f} ms  ({n/med(v):5.0f} tokens/s incl. 1 decode step)")

# --- Part 2: resume cost
print(f"\nPart 2: resume cost (prompt ~{args.prompt_tokens} tok, aborted after {args.abort_after} generated tokens)")
res = {"immediate": [], "after_other_requests": [], "cold_reference": []}
info = {"immediate": [], "after_other_requests": [], "cold_reference": []}
for rep in (range(args.reps) if "2" in args.parts else ()):
    for variant in ("immediate", "after_other_requests", "cold_reference"):
        p, nc = prompt_of(args.prompt_tokens)
        a = stream(p, args.gen, abort_after=args.abort_after)
        if not a["aborted"]: print("  (generation ended before the abort point; skip)"); continue
        time.sleep(0.2)
        if variant == "after_other_requests":
            for _ in range(3): stream(prompt_of(40)[0], 8)
        resume_prompt = p + a["text"]
        if variant == "cold_reference":   # same length, but nothing cacheable
            resume_prompt = resume_prompt.replace(nc, nonce(), 1)
        b = stream(resume_prompt, args.gen - a["chunks"], cache=(variant != "cold_reference"))
        res[variant].append(b["ttft"])
        info[variant].append((tm(b, "prompt_n"), tm(b, "cache_n")))
R["resume_ttft_s"] = res; R["resume_prompt_n_cache_n"] = info
for k in res:
    pn = [x[0] for x in info[k] if x[0] is not None]; cn = [x[1] for x in info[k] if x[1] is not None]
    print(f"  {k:22s}: resume first-token {med(res[k])*1000:6.0f} ms (all: {[round(x*1000) for x in res[k]]})"
          f"   server re-processed prompt_n={med(pn) if pn else '?'} tokens, reused cache_n={med(cn) if cn else '?'}")

# --- Part 3: Chat TTFT under interference
print("\nPart 3: Chat TTFT (short prompt) in three situations")
def chat_ttft(slot=None):
    try: return stream(chatml(nonce() + " Say hi in one short sentence."), 7, id_slot=slot)["ttft"]
    except Exception as e:
        print("   !! chat request failed:", repr(e), "-> did the server crash? check its log"); return None
def bg(fn, *a):
    def run():
        try: fn(*a)
        except Exception as e: print("   !! background request failed:", repr(e))
    th = threading.Thread(target=run, daemon=True); th.start(); return th
c0, c1, c2, c3, c4 = [], [], [], [], []
if "3" in args.parts:
    for _ in range(6): time.sleep(1.0); c0.append(chat_ttft())
    for _ in range(6):
        th = bg(stream, prompt_of(60)[0], 120)
        time.sleep(1.2); c1.append(chat_ttft()); th.join(); time.sleep(0.5)
    for _ in range(6):
        long_p, _nc = prompt_of(400)
        th = bg(stream, long_p, 1, False)
        time.sleep(random.uniform(0.05, 0.3)); c2.append(chat_ttft()); th.join(); time.sleep(0.5)
    # controlled slot assignment: does it matter WHICH slot Chat lands in?
    for _ in range(6):
        long_p, _nc = prompt_of(400)
        th = bg(stream, long_p, 1, False, None, 1)
        time.sleep(random.uniform(0.05, 0.3)); c3.append(chat_ttft(0)); th.join(); time.sleep(0.5)
    for _ in range(6):
        long_p, _nc = prompt_of(400)
        th = bg(stream, long_p, 1, False, None, 0)
        time.sleep(random.uniform(0.05, 0.3)); c4.append(chat_ttft(1)); th.join(); time.sleep(0.5)
R["chat_ttft_s"] = {"alone": c0, "other_slot_decoding": c1, "other_slot_prefilling_400tok": c2,
                    "prefilling__chat_pinned_slot0__long_pinned_slot1": c3,
                    "prefilling__chat_pinned_slot1__long_pinned_slot0": c4}
for k, v in R["chat_ttft_s"].items():
    v = [x for x in v if x is not None]
    if v: print(f"  {k:50s}: median {med(v)*1000:5.0f} ms   max {max(v)*1000:5.0f} ms   ({len(v)} ok)")

# --- summary
imm, oth, cold = med(res["immediate"]), med(res["after_other_requests"]), med(res["cold_reference"])
if "2" in args.parts: print("\nSUMMARY (resume overhead per interruption, vs. an uninterrupted job):")
for k, v in ((("immediate resume", imm), ("resume after other requests", oth), ("no cache at all", cold)) if "2" in args.parts else ()):
    print(f"  {k:28s}: {v:5.2f} s   -> 8 interruptions cost about {8*v:5.1f} s on a ~25 s Batch job")
R["summary"] = {"immediate_s": imm, "after_other_s": oth, "cold_s": cold}
json.dump(R, open(f"resume_{args.label}.json", "w"), indent=1)
print(f"\nWrote resume_{args.label}.json")