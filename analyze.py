"""Compute Tables 2a, 2b, 3 from results.csv (+ the slowest-request breakdown from raw/trace.jsonl). Everything derives from the CSV.
Definitions: latency percentiles use status=ok rows only; tokens/answer and cost use status=ok rows only;
peak in system = peak number of requests concurrently started-and-not-finished, computed from start_time and total_ms."""
import json, sys, numpy as np, pandas as pd
import config as C

df = pd.read_csv("results.csv")
df["start"] = pd.to_datetime(df["start_time"], utc=True)
df["t0"] = (df["start"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()

def pct(s, p):
    s = s.dropna()
    return "n/a" if s.empty else f"{np.percentile(s, p):.1f}"
def trio(s): return " / ".join(pct(s, p) for p in (50, 95, 99))
def peak_inflight(g):
    ev = sorted([(t, 1) for t in g.t0] + [(t + m / 1000, -1) for t, m in zip(g.t0, g.total_ms)])
    c = m = 0
    for _, d in ev:
        c += d; m = max(m, c)
    return m

print("## Table 2a (latency over status=ok rows)")
print("| Scenario | Source | Sent | Duration s | Offered/min | TTFT p50/p95/p99 ms | Total p50/p95/p99 ms |")
print("|---|---|---|---|---|---|---|")
for s, g in df[~df.scenario.str.startswith("quality_")].groupby("scenario"):
    ok = g[g.status == "ok"]
    dur = (g.t0 + g.total_ms / 1000).max() - g.t0.min()
    win = max(g.t0.max() - g.t0.min(), 1.0)
    print(f"| {s} | {','.join(sorted(g.source.unique()))} | {len(g)} | {dur:.1f} | {len(g)/win*60:.1f} | {trio(ok.first_token_ms)} | {trio(ok.total_ms)} |")

print("\n## Table 2b (tokens/answer and cost over status=ok rows; peak in system from the CSV)")
print("| Scenario | Succeeded % | Rate-limited % | Timed out/given up % | Retries/req | Peak in system | Tokens/answer | Cost per 1,000 answers (USD) |")
print("|---|---|---|---|---|---|---|---|")
for s, g in df.groupby("scenario"):
    f = lambda st: 100 * (g.status == st).mean()
    okg = g[g.status == "ok"]
    tok = (okg.tokens_in + okg.tokens_out).mean() if len(okg) else float("nan")
    cost = ((okg.tokens_in * C.PRICE_IN_PER_M + okg.tokens_out * C.PRICE_OUT_PER_M) / 1e6).mean() * 1000 if len(okg) else float("nan")
    print(f"| {s} | {f('ok'):.1f} | {f('rate_limited'):.1f} | {f('timeout'):.1f} | {g.retries.mean():.2f} | {peak_inflight(g)} | {tok:.0f} | {cost:.4f} |")

q = df[df.scenario.str.startswith("quality_")]
if len(q):
    print("\n## Table 3 (tokens/answer over status=ok rows)")
    print("| Setting | Asked | Correct % | Wrong % | Refused correctly % | Refused wrongly % | Tokens/answer | TTFT p50/p95 | Total p50/p95 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s, g in q.groupby("retrieval_setting"):
        gr = lambda x: 100 * (g.answer_grade == x).mean()
        ok = g[g.status == "ok"]
        tk = (ok.tokens_in + ok.tokens_out).mean() if len(ok) else float("nan")
        print(f"| {s} {C.SETTINGS[s]} | {len(g)} | {gr('correct'):.1f} | {gr('wrong'):.1f} | {gr('refused_correctly'):.1f} | {gr('refused_wrongly'):.1f} | {tk:.0f} | {pct(ok.first_token_ms,50)} / {pct(ok.first_token_ms,95)} | {pct(ok.total_ms,50)} / {pct(ok.total_ms,95)} |")

if "--slowest" in sys.argv:
    i = sys.argv.index("--slowest")
    src = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
    tr = [json.loads(l) for l in open("raw/trace.jsonl")]
    tr = [t for t in tr if src in (None, t["source"])]
    ok = {r.request_id: r for r in df[df.status == "ok"].itertuples()}
    t = max((t for t in tr if t["request_id"] in ok), key=lambda t: ok[t["request_id"]].total_ms)
    print("\n## Slowest ok request:", t["request_id"], "total", ok[t["request_id"]].total_ms, "ms\n", json.dumps(t["trace"], indent=1))