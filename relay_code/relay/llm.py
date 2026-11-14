import asyncio, json, time, httpx

class RateLimited(Exception):
    def __init__(self, retry_after, headers, body=""):
        super().__init__("429"); self.retry_after, self.headers, self.body = retry_after, headers, body
class Transient(Exception): pass
class CallTimeout(Exception): pass
class Fatal(Exception): pass

def _hdrs(r):
    return {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}

def est_tokens(s): return max(1, len(s) // 4)

REASONING_HEADROOM = 400   # gpt-oss spends tokens on hidden reasoning before any content (measured: 398 of 400 tokens)

async def chat(client, url, key, model, messages, *, max_tokens, timeout, stream=False):
    """One model call. Returns dict(text, ttft_ms, total_ms, tokens_in, tokens_out, headers)."""
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, stream=stream)
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
        body["max_tokens"] = max_tokens + REASONING_HEADROOM
    if stream: body["stream_options"] = {"include_usage": True}
    h = {"Authorization": f"Bearer {key}"} if key else {}
    t0 = time.perf_counter(); ttft = None; usage = None; parts = []
    try:
        async with asyncio.timeout(timeout):
            if not stream:
                r = await client.post(url, json=body, headers=h)
                _check(r, r.text)
                j = r.json(); hd = _hdrs(r)
                parts = [j["choices"][0]["message"].get("content") or ""]; usage = j.get("usage")
                ttft = (time.perf_counter() - t0) * 1000
            else:
                async with client.stream("POST", url, json=body, headers=h) as r:
                    if r.status_code != 200:
                        txt = (await r.aread()).decode(errors="ignore"); _check(r, txt)
                    hd = _hdrs(r)
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"): continue
                        d = line[5:].strip()
                        if d == "[DONE]": break
                        j = json.loads(d)
                        u = j.get("usage") or (j.get("x_groq") or {}).get("usage")
                        if u: usage = u
                        for ch in j.get("choices", []):
                            c = (ch.get("delta") or {}).get("content")
                            if c:
                                if ttft is None: ttft = (time.perf_counter() - t0) * 1000
                                parts.append(c)
    except asyncio.TimeoutError:
        raise CallTimeout()
    except httpx.HTTPError as e:
        raise Transient(repr(e))
    text = "".join(parts)
    tin = (usage or {}).get("prompt_tokens") or sum(est_tokens(m["content"]) for m in messages)
    tout = (usage or {}).get("completion_tokens") or est_tokens(text)
    return dict(text=text, ttft_ms=ttft, total_ms=(time.perf_counter() - t0) * 1000, tokens_in=tin, tokens_out=tout, headers=hd)

def _check(r, text):
    if r.status_code == 200: return
    if r.status_code == 429:
        ra = r.headers.get("retry-after")
        raise RateLimited(float(ra) if ra else 1.0, _hdrs(r), text[:200])
    if r.status_code >= 500: raise Transient(f"HTTP {r.status_code}")
    raise Fatal(f"HTTP {r.status_code}: {text[:200]}")