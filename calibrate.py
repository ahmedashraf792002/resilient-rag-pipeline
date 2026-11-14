"""Fit the stand-in's latency parameters from raw/probe_real.jsonl and write them into standin_config.json."""
import json, numpy as np
rows = [json.loads(l) for l in open("raw/probe_real.jsonl")]
ok = [r for r in rows if r["status"] == 200 and r["ttft_ms"] and r["tag"].startswith("lat_c")]
c1 = [r for r in ok if r["tag"] == "lat_c1"]
tin = np.array([(r["usage"] or {}).get("prompt_tokens", r["words"] * 1.5) for r in c1]); tt = np.array([r["ttft_ms"] for r in c1])
b, a = np.polyfit(tin, tt, 1); a = max(a, 20); b = max(b, 0.0)
res = np.log(tt / np.maximum(a + b * tin, 1)); sigma = float(np.std(res))
def vis(u): return (u or {}).get("completion_tokens", 100) - ((u or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
dec = np.array([max(vis(r["usage"]), 1) / max((r["total_ms"] - r["ttft_ms"]) / 1000, 1e-3) for r in c1])
dsig = float(np.std(np.log(dec / np.median(dec))))
med = {c: np.median([r["ttft_ms"] for r in ok if r["tag"] == f"lat_c{c}"]) for c in (1, 2, 4, 8) if any(r["tag"] == f"lat_c{c}" for r in ok)}
cs = np.array(sorted(med)); ratio = np.array([med[c] / med[1] - 1 for c in cs]); alpha = float(max(0, np.polyfit(cs, ratio, 1)[0])) if len(cs) > 1 else 0.0
pred = a + b * tin; slow = tt / pred; sp = float(np.mean(slow > 3)); sm = float(np.median(slow[slow > 3])) if (slow > 3).any() else 1.0
fail = float(np.mean([isinstance(r["status"], int) and r["status"] >= 500 for r in rows]))
cfg = json.load(open("standin_config.json"))
cfg.update(ttft_base_ms=float(a), prefill_ms_per_tok=float(b), ttft_sigma=sigma, decode_tok_s=float(np.median(dec)), decode_sigma=dsig,
           knee=0, alpha=alpha, slow_prob=sp, slow_mult=sm, fail_prob=fail)
json.dump(cfg, open("standin_config.json", "w"), indent=2); print(json.dumps(cfg, indent=2))
