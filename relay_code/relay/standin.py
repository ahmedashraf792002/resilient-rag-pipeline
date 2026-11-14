"""Stand-in model service (OpenAI-compatible /v1/chat/completions). Controls: rate limits (RPM/TPM/RPD),
latency (lognormal TTFT + prefill + decode rate), load-dependent slowdown, random 5xx, 'succeeds slowly', hangs, SSE streaming.
Content is canned: it is for TIMING and failure behaviour only, never for answer quality."""
import asyncio, json, math, random, re, time, datetime
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CFG = json.load(open("standin_config.json"))
app = FastAPI()
S = dict(reqs=deque(), toks=deque(), day=0, inflight=0, peak_inflight=0, n=0, n429=0, n5xx=0)

def tok_est(s): return max(1, len(s) // 4)

def make_reply(text):
    if "[[TASK:ANALYZE]]" in text:
        q = text.split("Rewrite", 1)[-1].split("\n")[-1]
        return json.dumps({"query": " ".join(q.split()[:8])})
    if "[[TASK:VERIFY]]" in text: return '{"supported": true}'
    if "[[TASK:DRAFT]]" in text:
        m = re.search(r"\[([a-z]+:p\d+:c\d+)\] ([^\[]{0,400})", text)
        if m: return " ".join(m.group(2).split()[:30]) + f" [{m.group(1)}]"
        return "NOT_IN_PAPERS"
    return "ok"

def limit_headers(now):
    return {"x-ratelimit-limit-requests": str(CFG["rpd"]), "x-ratelimit-remaining-requests": str(max(0, CFG["rpd"] - S["day"])),
            "x-ratelimit-limit-tokens": str(CFG["tpm"]),
            "x-ratelimit-remaining-tokens": str(max(0, CFG["tpm"] - sum(t for _, t in S["toks"])))}

def reject(retry_after, hdr, msg):
    S["n429"] += 1
    return JSONResponse({"error": {"message": msg, "type": "rate_limit_exceeded"}}, status_code=429,
                        headers=dict(hdr, **{"retry-after": str(int(math.ceil(retry_after)))}))

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json(); now = time.time(); S["n"] += 1
    prompt = " ".join(m["content"] for m in body["messages"]); tin = tok_est(prompt)
    while S["reqs"] and now - S["reqs"][0] > 60: S["reqs"].popleft()
    while S["toks"] and now - S["toks"][0][0] > 60: S["toks"].popleft()
    hdr = limit_headers(now)
    if S["day"] >= CFG["rpd"]: return reject(CFG["retry_after_daily_s"], hdr, "daily request limit reached")
    if len(S["reqs"]) >= CFG["rpm"]: return reject(60 - (now - S["reqs"][0]) + 0.5, hdr, "requests per minute limit reached")
    if sum(t for _, t in S["toks"]) + tin > CFG["tpm"]: return reject(60 - (now - S["toks"][0][0]) + 0.5, hdr, "tokens per minute limit reached")
    S["reqs"].append(now); S["day"] += 1
    maxtok = body.get("max_tokens", 256); S["toks"].append((now, tin + maxtok))
    S["inflight"] += 1; S["peak_inflight"] = max(S["peak_inflight"], S["inflight"])
    done = False
    def finish():
        nonlocal done
        if not done: done = True; S["inflight"] -= 1
    try:
        slow = 1 + CFG["alpha"] * max(0, S["inflight"] - CFG["knee"])
        if random.random() < CFG["fail_prob"]:
            await asyncio.sleep(random.uniform(0.05, 0.4)); S["n5xx"] += 1; finish()
            return JSONResponse({"error": {"message": "internal error"}}, status_code=503, headers=hdr)
        if random.random() < CFG["hang_prob"]: await asyncio.sleep(CFG["hang_s"])
        mult = CFG["slow_mult"] if random.random() < CFG["slow_prob"] else 1.0
        ttft = (CFG["ttft_base_ms"] * random.lognormvariate(0, CFG["ttft_sigma"]) + CFG["prefill_ms_per_tok"] * tin) * slow * mult / 1000
        content = make_reply(prompt); words = content.split(" "); ntok = max(1, min(maxtok, int(len(words) * 1.3)))
        rate = CFG["decode_tok_s"] * random.lognormvariate(0, CFG["decode_sigma"]) / slow
        usage = dict(prompt_tokens=tin, completion_tokens=ntok, total_tokens=tin + ntok)
        if not body.get("stream"):
            await asyncio.sleep(ttft + ntok / rate); finish()
            return JSONResponse(dict(choices=[dict(message=dict(role="assistant", content=content))], usage=usage), headers=hdr)
        async def gen():
            try:
                await asyncio.sleep(ttft)
                for i, w in enumerate(words):
                    yield "data: " + json.dumps(dict(choices=[dict(delta=dict(content=w + (" " if i < len(words) - 1 else "")))])) + "\n\n"
                    await asyncio.sleep((ntok / len(words)) / rate)
                yield "data: " + json.dumps(dict(choices=[], usage=usage)) + "\n\n"; yield "data: [DONE]\n\n"
            finally: finish()
        return StreamingResponse(gen(), media_type="text/event-stream", headers=hdr)
    except BaseException:
        finish(); raise

@app.post("/admin/config")
async def set_cfg(req: Request): CFG.update(await req.json()); return CFG
@app.post("/admin/reset")
async def reset():
    S.update(reqs=deque(), toks=deque(), day=0, peak_inflight=0, n=0, n429=0, n5xx=0); return S["n"]
@app.get("/admin/stats")
async def stats(): return dict(inflight=S["inflight"], peak_inflight=S["peak_inflight"], n=S["n"], n429=S["n429"], n5xx=S["n5xx"], day=S["day"])

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
