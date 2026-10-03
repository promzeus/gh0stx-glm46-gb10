#!/usr/bin/env python3
"""Mixed calibration corpus for GLM-4.6: reasoning traces plus plain chat in RU, EN and code.

Writes {"domain": ..., "text": ...} per line. Every row is one GLM chat exchange rendered through
chat_template.jinja, truncated to CALIB_MAXLEN tokens. Consumer: the imatrix text of the full-model GGUF build
(gguf/glm-gguf-full.sh), which takes a per-domain token quota from it.

  domain      source                                   fields                       row share
  reasoning   glaiveai/reasoning-v1-20m                prompt/response (<think>)    0.20
  math        open-r1/OpenR1-Math-220k                 problem/generations[0]       0.10
  code_cot    open-r1/codeforces-cots (solutions_py)   prompt/generation            0.10
  ru          d0rj/OpenHermes-2.5-ru                   conversations (sharegpt)     0.30
  general     teknium/OpenHermes-2.5                   conversations (sharegpt)     0.20
  code        ise-uiuc/Magicoder-OSS-Instruct-75K      problem/solution             0.10

Row shares favour the short plain domains: a reasoning row is ~1.6-4k tokens, a plain row ~0.3-0.6k, so
the token mix stays reasoning-heavy and the imatrix builder can still fill a per-domain token quota.
IlyaGusev/ru_turbo_alpaca is not usable: it ships a loading script, which current `datasets` rejects.
teknium/OpenHermes-2.5 is one 1.9 GB JSON array; streaming it parses the whole file (OOMKilled at 12Gi), so it is
read from the Hub's parquet conversion (revision refs/convert/parquet).

Reasoning rows must contain </think> and at least CALIB_MIN tokens; plain rows need CALIB_MIN_PLAIN.
Env: CALIB_TOK (tokenizer dir), CALIB_OUT, CALIB_N (1200), CALIB_MAXLEN (4096), CALIB_MIN (768),
     CALIB_MIN_PLAIN (128), CALIB_SEED (0).
"""
import json
import os
import random
import sys

TOK       = os.environ.get("CALIB_TOK", "/model/tok")
OUT       = os.environ.get("CALIB_OUT", "/work/calib_mix.jsonl")
N         = int(os.environ.get("CALIB_N", "1200"))
MAXLEN    = int(os.environ.get("CALIB_MAXLEN", "4096"))
MINLEN    = int(os.environ.get("CALIB_MIN", "768"))
MIN_PLAIN = int(os.environ.get("CALIB_MIN_PLAIN", "128"))
SEED      = int(os.environ.get("CALIB_SEED", "0"))

ROLE = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def sharegpt(conv):
    msgs = []
    for t in conv:
        r = ROLE.get(t.get("from") or t.get("role"))
        v = t.get("value") or t.get("content")
        if r and v:
            msgs.append({"role": r, "content": v})
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in msgs):
        return None
    return msgs


def pair(u, a):
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}] if u and a else None


# (domain, dataset, config, share, extract -> messages, needs_think)
PLAN = [
    ("reasoning", "glaiveai/reasoning-v1-20m", None, 0.20, lambda ex: pair(ex["prompt"], ex["response"]), True),
    ("math", "open-r1/OpenR1-Math-220k", None, 0.10, lambda ex: pair(ex["problem"], ex["generations"][0]), True),
    ("code_cot", "open-r1/codeforces-cots", "solutions_py", 0.10, lambda ex: pair(ex["prompt"], ex["generation"]), True),
    ("ru", "d0rj/OpenHermes-2.5-ru", None, 0.30, lambda ex: sharegpt(ex["conversations"]), False),
    ("general", "teknium/OpenHermes-2.5", None, 0.20, lambda ex: sharegpt(ex["conversations"]), False),
    ("code", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, 0.10, lambda ex: pair(ex["problem"], ex["solution"]), False),
]


REVISION = {"teknium/OpenHermes-2.5": "refs/convert/parquet"}


def log(msg):
    print(f"[calib] {msg}", flush=True)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    rows = []
    for domain, name, config, share, extract, needs_think in PLAN:
        want = max(1, round(N * share))
        kw = {"split": "train", "streaming": True}
        if config:
            kw["name"] = config
        if name in REVISION:
            kw["revision"] = REVISION[name]
        try:
            it = load_dataset(name, **kw).shuffle(seed=SEED, buffer_size=2000)
        except Exception as e:
            log(f"{domain}: load failed ({e!r}), skipping domain")
            continue
        got, bad, cut, toks = [], 0, 0, 0
        for ex in it:
            try:
                msgs = extract(ex)
            except Exception:
                bad += 1
                continue
            if not msgs or (needs_think and "</think>" not in msgs[-1]["content"]):
                bad += 1
                continue
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) < (MINLEN if needs_think else MIN_PLAIN):
                bad += 1
                continue
            if len(ids) > MAXLEN:
                cut += 1
                text = tok.decode(ids[:MAXLEN], skip_special_tokens=False)
            n = min(len(ids), MAXLEN)
            toks += n
            got.append({"domain": domain, "text": text, "tokens": n})
            if len(got) >= want:
                break
        log(f"{domain} <- {name}{'/' + config if config else ''}: {len(got)} rows (wanted {want}, skipped {bad}, "
            f"truncated at {MAXLEN}: {cut}), {toks} tokens")
        rows += got
    random.Random(SEED).shuffle(rows)
    rows = rows[:N]
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps({"domain": r["domain"], "text": r["text"]}, ensure_ascii=False) + "\n")
    lens = sorted(r["tokens"] for r in rows) or [0]
    log(f"wrote {len(rows)} rows -> {OUT}; tokens min/median/max = {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}; "
        f"total {sum(lens)} tokens; {os.path.getsize(OUT) / 1e6:.1f} MB")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
