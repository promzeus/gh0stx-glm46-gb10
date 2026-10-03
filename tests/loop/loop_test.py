#!/usr/bin/env python3
"""Loop test for a served GLM-4.6: 6 prompts x temp {0.0, 1.0} against an OpenAI-compatible endpoint
(llama-server on gx10). Two long reasoning prompts stress the long-<think> path where pruned checkpoints looped.
Flags repetition (same word 8x in a row, a 30-char chunk repeated 6+ times, 4-gram uniqueness < 0.5) so a bad
model is caught without reading every output. A step-by-step enumeration can trip the chunk rule: read the flagged
case before calling it a loop.

Env: LOOP_BASE (http://127.0.0.1:8000/v1), LOOP_MODEL (glm46), LOOP_OUT (output jsonl).
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("LOOP_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("LOOP_MODEL", "glm46")
OUT = os.environ.get("LOOP_OUT", "loop.jsonl")

# 4 short prompts + 2 long-reasoning stressors.
# Caps sized so the model closes </think> and emits a final answer; GLM-4.6
# thinks even on trivial prompts, so a 64-token cap never reaches content.
CASES = [
    ("short",  "Скажи одним словом: работаешь?", 512),
    ("explain","Explain in a few sentences what a hash table is and why it is useful.", 1024),
    ("math",   "What is 15 * 23? Show the steps.", 1024),
    ("code",   "Write a Python function that reverses a string.", 1024),
    ("reason", "A snail climbs a 10 m well. Each day it climbs 3 m, each night it "
               "slips back 2 m. On which day does it reach the top? Reason step by step.", 3000),
    ("longcode","Implement a rate limiter in Go using a token bucket. Explain the design, "
                "then give the full code with comments.", 4000),
]


def rep_flags(s: str):
    """Return repetition signals: max n-gram repeat and longest immediate run."""
    flags = []
    # immediate token/word run: same word >=8x in a row
    words = s.split()
    run = maxrun = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            run += 1; maxrun = max(maxrun, run)
        else:
            run = 1
    if maxrun >= 8:
        flags.append(f"word_run={maxrun}")
    # repeated 30-char chunk
    for chunk in (s[i:i+30] for i in range(0, max(0, len(s)-30), 30)):
        if chunk.strip() and s.count(chunk) >= 6:
            flags.append(f"chunk_x{s.count(chunk)}:{chunk!r}")
            break
    # 4-gram repetition ratio
    grams = [tuple(words[i:i+4]) for i in range(len(words)-4)]
    if grams:
        uniq = len(set(grams)) / len(grams)
        if uniq < 0.5:
            flags.append(f"4gram_uniq={uniq:.2f}")
    return flags


def call(prompt, temp, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0]["message"]
    content = msg.get("content") or ""
    # llama-server puts GLM thinking in "reasoning_content"; some servers use "reasoning". Read both.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    finish = j["choices"][0].get("finish_reason")
    usage = j.get("usage", {})
    usage = dict(usage or {}, finish=finish)
    return reasoning, content, dt, usage


def main():
    results = []
    bad = 0
    with open(OUT, "w") as f:
        for name, prompt, mx in CASES:
            for temp in (0.0, 1.0):
                try:
                    reasoning, content, dt, usage = call(prompt, temp, mx)
                except Exception as e:
                    print(f"[{name} t={temp}] ERROR {e}", flush=True)
                    f.write(json.dumps({"case": name, "temp": temp, "error": str(e)}) + "\n")
                    bad += 1
                    continue
                whole = (reasoning + "\n" + content).strip()
                flags = rep_flags(whole)
                verdict = "LOOP" if flags else "ok"
                if flags:
                    bad += 1
                ct = usage.get("completion_tokens", "?")
                fin = usage.get("finish", "?")
                print(f"[{name} t={temp}] {verdict} {dt:.1f}s tok={ct} fin={fin} "
                      f"think={len(reasoning)}c out={len(content)}c {flags}", flush=True)
                tail = (content or reasoning)[:200].replace("\n", " ")
                print("  " + ("OUT" if content else "THINK") + ": " + tail, flush=True)
                f.write(json.dumps({"case": name, "temp": temp, "verdict": verdict,
                                    "flags": flags, "dt": dt, "usage": usage,
                                    "reasoning": reasoning, "content": content}) + "\n")
    print(f"[test] DONE bad={bad}/{len(CASES)*2}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
