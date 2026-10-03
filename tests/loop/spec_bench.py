#!/usr/bin/env python3
"""Decode-speed and speculative-acceptance benchmark against a llama-server OpenAI endpoint.

Runs a few chat prompts (RU, EN reasoning, code) at temp 0 and temp 1.0 (top_p 0.95, top_k 40, the GLM-4.6
recommended sampling) and reads llama-server's `timings` block: prompt/predicted tokens per second and, when
speculative decoding is on, draft_n / draft_n_accepted. Optional long-context case (LONG_FILE, LONG_TOKENS):
a long document followed by a question, to measure prefill rate and decode rate at depth.

Env: BENCH_BASE (http://127.0.0.1:8011/v1), BENCH_MODEL (glm46), BENCH_OUT (jsonl path), BENCH_MAX (768),
     BENCH_SHORT (1 = run the short cases, 0 = long case only),
     LONG_FILE (text file, optional), LONG_TOKENS (approximate prompt size in tokens, default 0 = skip).
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8011/v1")
MODEL = os.environ.get("BENCH_MODEL", "glm46")
OUT = os.environ.get("BENCH_OUT", "spec_bench.jsonl")
MAXTOK = int(os.environ.get("BENCH_MAX", "768"))
LONG_FILE = os.environ.get("LONG_FILE", "")
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "0"))
SHORT = os.environ.get("BENCH_SHORT", "1") == "1"

CASES = [
    ("ru", "Объясни, как работает алгоритм Дейкстры, и приведи пример на графе из пяти вершин."),
    ("reason", "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 25 minutes, then travels "
               "another 96 km at 64 km/h. When does it arrive? Reason step by step."),
    ("code", "Write a Python function that parses an nginx access log line into a dict and a short test for it."),
]


def call(messages, temp, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temp}
    if temp > 0:
        body.update({"top_p": 0.95, "top_k": 40})
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        j = json.load(r)
    return j, time.time() - t0


def row(name, temp, j, dt):
    t = j.get("timings", {})
    msg = j["choices"][0]["message"]
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    acc = (da / dn) if dn else None
    r = {"case": name, "temp": temp, "wall_s": round(dt, 1),
         "prompt_n": t.get("prompt_n"), "prompt_tps": round(t.get("prompt_per_second", 0), 1),
         "predicted_n": t.get("predicted_n"), "predicted_tps": round(t.get("predicted_per_second", 0), 2),
         "draft_n": dn, "draft_accepted": da, "accept_rate": round(acc, 3) if acc is not None else None,
         "finish": j["choices"][0].get("finish_reason"),
         "head": ((msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or ""))[:160]}
    print(f"[{name:6s} t={temp}] prompt {r['prompt_n']} tok @ {r['prompt_tps']} t/s | gen {r['predicted_n']} tok @ "
          f"{r['predicted_tps']} t/s | draft {dn}/{da} acc={r['accept_rate']} | {r['finish']} | {dt:.0f}s", flush=True)
    return r


def main():
    rows = []
    with open(OUT, "w") as f:
        for name, prompt in (CASES if SHORT else []):
            for temp in (0.0, 1.0):
                try:
                    j, dt = call([{"role": "user", "content": prompt}], temp, MAXTOK)
                except Exception as e:
                    print(f"[{name} t={temp}] ERROR {e}", flush=True)
                    continue
                r = row(name, temp, j, dt); rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if LONG_FILE and LONG_TOKENS > 0:
            text = open(LONG_FILE, encoding="utf-8", errors="ignore").read()
            doc = text[: LONG_TOKENS * 4]          # ~4 chars per token for English markdown
            q = ("The text above is a set of llama.cpp documentation files. List the speculative decoding types "
                 "it describes and give one sentence on each.")
            try:
                j, dt = call([{"role": "user", "content": doc + "\n\n" + q}], 0.0, 256)
                r = row(f"long{LONG_TOKENS // 1000}k", 0.0, j, dt); rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[long] ERROR {e}", flush=True)
    gen = [r["predicted_tps"] for r in rows if r["case"] in dict(CASES)]
    if gen:
        print(f"[bench] mean decode {sum(gen) / len(gen):.2f} t/s over {len(gen)} runs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
