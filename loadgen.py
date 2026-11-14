"""Load generator + quality runner. Appends one row per request to results.csv, full traces to raw/trace.jsonl.
  python loadgen.py load --scenario burst --source standin
  python loadgen.py quality --source real --settings S1 S2 S3 S4"""
import argparse, asyncio, csv, datetime, json, os, random, re, httpx
import config as C
from pipeline import Runner

COLS = "request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade".split(",")
Q = json.load(open("questions.json"))
TITLE = {"att": "Attention Is All You Need", "moshi": "Moshi", "dc": "Dynamic Chunking"}

# name -> (shape, n, rate_per_min, stand-in overrides, client limit overrides)
SCENARIOS = {
  "steady":        dict(shape="poisson", n=300, rate=20,  standin={}, limits={}),
  "burst":         dict(shape="burst",   n=150, rate=0,   standin={}, limits={}),
  "overload":      dict(shape="poisson", n=500, rate=150, standin={}, limits={}),
  "rpm_halved":    dict(shape="poisson", n=200, rate=20,  standin={"rpm": "HALF"}, limits={"rpm": "HALF"}),
  "daily_exhaust": dict(shape="poisson", n=200, rate=20,  standin={"rpd": 150}, limits={"rpd": 150}),
}

def grade(q, r):
    if r["status"] != "ok": return ""
    if r["answer_type"] == "refused": return "refused_wrongly" if q["answerable"] else "refused_correctly"
    if not q["answerable"]: return "wrong"
    ok = all(re.search(p, r["text"], re.I) for p in q["must"]) and TITLE[q["paper"]] in r["text"]
    return "correct" if ok else "wrong"

def append(row):
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new: w.writeheader()
        w.writerow(row)

async def one(runner, scen, source, i, q, setting, t_start):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    r = await runner.run_request(q["q"], setting)
    g = grade(q, r) if source == "real" else ""       # stand-in text is canned: not graded
    rid = f"{scen}-{i:05d}"
    fmt = lambda x: "" if x is None else round(x, 1)
    append(dict(request_id=rid, scenario=scen, source=source, start_time=start, first_token_ms=fmt(r["first_token_ms"]),
                total_ms=fmt(r["total_ms"]), status=r["status"], retries=r["retries"], tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"], retrieval_setting=setting, answer_grade=g))
    with open("raw/trace.jsonl", "a") as f:
        f.write(json.dumps(dict(request_id=rid, scenario=scen, source=source, question=q["id"], answer_type=r["answer_type"],
                                answer=r["text"], trace=r["trace"], peak_queue_runner=runner.peak_queue)) + "\n")

async def admin(path, payload=None):
    async with httpx.AsyncClient() as c:
        return (await c.post(f"{C.STANDIN_ADMIN}/{path}", json=payload or {})).json()

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["load", "quality"])
    ap.add_argument("--scenario"); ap.add_argument("--source", default="standin", choices=["real", "standin"])
    ap.add_argument("--n", type=int); ap.add_argument("--rate", type=float); ap.add_argument("--setting", default=C.DEFAULT_SETTING)
    ap.add_argument("--settings", nargs="*", default=["S1", "S2", "S3"]); ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--no-retry-budget", action="store_true")
    ap.add_argument("--policy", default="{}", help="JSON overrides of POLICY, e.g. '{\"max_retries\":6}'")
    a = ap.parse_args(); rnd = random.Random(a.seed)
    policy = json.loads(a.policy)
    if a.no_retry_budget: policy["retry_budget_ratio"] = None

    if a.mode == "quality":
        for s in a.settings:
            runner = Runner(a.source, persist_budget=".budget.json", policy=policy)
            for i, q in enumerate(Q): await one(runner, f"quality_{s}", a.source, i, q, s, 0); await asyncio.sleep(a.gap)
            print("done", s)
        return

    sc = dict(SCENARIOS[a.scenario]); n = a.n or sc["n"]; rate = a.rate if a.rate is not None else sc["rate"]
    limits, standin = {}, {}
    for k, v in sc["limits"].items(): limits[k] = C.LIMITS[k] // 2 if v == "HALF" else v
    for k, v in sc["standin"].items(): standin[k] = (json.load(open("standin_config.json"))[k] // 2) if v == "HALF" else v
    if a.source == "standin":
        await admin("reset");
        base = json.load(open("standin_config.json")); await admin("config", dict(base, **standin))
    runner = Runner(a.source, limits=limits, policy=policy, persist_budget=".budget.json")
    if sc["shape"] == "burst": offs = [0.0] * n
    else:
        t = 0.0; offs = []
        for _ in range(n): t += rnd.expovariate(rate / 60.0); offs.append(t)
    loop = asyncio.get_event_loop(); t0 = loop.time(); tasks = []
    async def fire(i, off):
        await asyncio.sleep(max(0, off - (loop.time() - t0))); await one(runner, a.scenario, a.source, i, rnd.choice(Q), a.setting, off)
    tasks = [asyncio.create_task(fire(i, o)) for i, o in enumerate(offs)]
    await asyncio.gather(*tasks)
    print(f"{a.scenario}: {n} requests in {loop.time() - t0:.1f}s, peak queue {runner.peak_queue}")

asyncio.run(main())
