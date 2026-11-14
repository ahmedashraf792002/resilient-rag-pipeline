"""Measure the REAL service. Appends to raw/probe_real.jsonl (one JSON line per request, with rate-limit headers).
  python probe.py latency --n 4      # latency vs input size and concurrency (paced to stay under the RPM/TPM limits)
  python probe.py limits  --burst 40 # burst of tiny requests: shows the per-minute limit and the 429 body/headers"""
import argparse, asyncio, json, os, time, httpx
import config as C

os.makedirs("raw", exist_ok=True)
def prompt(words): return "Summarise in two sentences: " + " ".join(["transformer"] * words)

async def req(c, words, max_tokens, tag):
    body = dict(model=C.MODEL, messages=[{"role": "user", "content": prompt(words)}], max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={"include_usage": True})
    if "gpt-oss" in C.MODEL:
        body["reasoning_effort"] = "low"; body["max_tokens"] = max_tokens + 400   # hidden reasoning eats tokens first
    t0 = time.perf_counter(); ttft = None; usage = None; status = None; hd = {}; text = ""
    try:
        async with c.stream("POST", C.GROQ_URL, json=body, headers={"Authorization": f"Bearer {C.GROQ_KEY}"}, timeout=60) as r:
            status = r.status_code
            hd = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}
            if status != 200: text = (await r.aread()).decode(errors="ignore")[:300]
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.endswith("[DONE]"): continue
                    j = json.loads(line[5:]); usage = j.get("usage") or (j.get("x_groq") or {}).get("usage") or usage
                    for ch in j.get("choices", []):
                        if (ch.get("delta") or {}).get("content") and ttft is None: ttft = (time.perf_counter() - t0) * 1000
    except Exception as e: status = f"exc:{e!r}"
    row = dict(tag=tag, ts=time.time(), words=words, max_tokens=max_tokens, status=status, ttft_ms=ttft,
               total_ms=(time.perf_counter() - t0) * 1000, usage=usage, headers=hd, body=text)
    with open("raw/probe_real.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    return row

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["latency", "limits"])
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--burst", type=int, default=40)
    a = ap.parse_args()
    if not C.GROQ_KEY: raise SystemExit("GROQ_API_KEY is not set")
    async with httpx.AsyncClient() as c:
        if a.mode == "limits":
            rows = await asyncio.gather(*[req(c, 20, 1, "limits") for _ in range(a.burst)])
            for r in rows: print(r["status"], r["headers"], r["body"][:120])
            return
        for conc in (1, 2, 4):
            for words in (50, 400, 800):
                rows = []
                for _ in range(0, a.n, conc):
                    rows += await asyncio.gather(*[req(c, words, 300, f"lat_c{conc}") for _ in range(conc)])
                    await asyncio.sleep(12)          # stay under RPM and TPM so we measure latency, not rate limiting
                good = sorted(round(r["ttft_ms"]) for r in rows if r["ttft_ms"])
                codes = sorted({str(r["status"]) for r in rows})
                print(f"conc={conc} words={words} ok={len(good)}/{len(rows)} status={codes} median_ttft_ms={good[len(good)//2] if good else 'none'}")
asyncio.run(main())