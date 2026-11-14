"""The pipeline: analyze -> retrieve -> (tool: neighbour chunk) -> draft -> verify -> release.
Branches: refuse early when retrieval confidence is low; one redraft loop when verification fails;
degrade (smaller k, no rerank, no LLM verify, no analyze) as the queue grows; extractive fallback when the daily budget is gone."""
import asyncio, json, random, re, time, datetime, os
from collections import deque
import config as C
from llm import chat, RateLimited, Transient, CallTimeout, Fatal
from retrieval import Index

TITLES = {"att": "Attention Is All You Need", "moshi": "Moshi", "dc": "Dynamic Chunking"}
CITE = re.compile(r"\[([a-z]+:p\d+:c\d+)\]")
REFUSAL = "The three papers I have do not say this, so I won't guess."

class Failed(Exception):
    def __init__(self, kind): super().__init__(kind); self.kind = kind   # timeout | rate_limited | error
class DailyExhausted(Exception): pass

class Trace:
    def __init__(self):
        self.t0 = time.perf_counter(); self.wait = self.work = self.retry = 0.0
        self.retries = 0; self.tin = self.tout = 0; self.events = []; self.draft_ttft = None
    def now_ms(self): return (time.perf_counter() - self.t0) * 1000
    def ev(self, **kw): self.events.append(dict(kw, at_ms=round(self.now_ms(), 1)))

class Gate:
    """Client-side limiter: sliding-window RPM and TPM plus a cap on calls in flight."""
    def __init__(self, rpm, tpm, max_inflight):
        self.rpm, self.tpm = rpm, tpm
        self.sem = asyncio.Semaphore(max_inflight)
        self.reqs = deque(); self.toks = deque(); self.waiting = 0
    def _prune(self, now):
        while self.reqs and now - self.reqs[0] > 60: self.reqs.popleft()
        while self.toks and now - self.toks[0][0] > 60: self.toks.popleft()
    async def acquire(self, est, deadline_at, tr):
        t = time.perf_counter(); self.waiting += 1
        est = min(est, self.tpm)
        try:
            while True:
                now = time.perf_counter(); self._prune(now)
                if len(self.reqs) < self.rpm and sum(x[1] for x in self.toks) + est <= self.tpm: break
                if now + 0.1 > deadline_at: raise Failed("timeout")
                await asyncio.sleep(0.1)
            try:
                await asyncio.wait_for(self.sem.acquire(), timeout=max(0.01, deadline_at - time.perf_counter()))
            except asyncio.TimeoutError:
                raise Failed("timeout")
            now = time.perf_counter(); self.reqs.append(now); self.toks.append((now, est))
        finally:
            self.waiting -= 1; tr.wait += time.perf_counter() - t
    def release(self): self.sem.release()

class Budget:
    """Daily request budget. Real runs persist to disk so restarts do not forget what was spent."""
    def __init__(self, rpd, reserve, persist_path=None):
        self.limit = int(rpd * (1 - reserve)); self.path = persist_path; self.used = 0; self.flag = False
        today = datetime.datetime.utcnow().date().isoformat()
        if persist_path and os.path.exists(persist_path):
            d = json.load(open(persist_path))
            if d.get("date") == today: self.used = d["used"]
        self.today = today
    def count(self):
        self.used += 1
        if self.path: json.dump(dict(date=self.today, used=self.used), open(self.path, "w"))
    @property
    def exhausted(self): return self.flag or self.used >= self.limit

class RetryBudget:
    def __init__(self, ratio): self.ratio, self.tokens = ratio, 10.0
    def credit(self):
        if self.ratio is not None: self.tokens = min(50.0, self.tokens + self.ratio)
    def allow(self):
        if self.ratio is None: return True
        if self.tokens >= 1: self.tokens -= 1; return True
        return False

class Runner:
    def __init__(self, source, pages_path=os.getenv("PAGES_PATH", "corpus/pages.json"), limits=None, policy=None, persist_budget=None):
        self.source = source
        self.url = C.GROQ_URL if source == "real" else C.STANDIN_URL
        self.key = C.GROQ_KEY if source == "real" else ""
        self.L = dict(C.LIMITS, **(limits or {})); self.P = dict(C.POLICY, **(policy or {}))
        self.gate = Gate(self.L["rpm"], self.L["tpm"], self.L["max_inflight"])
        self.max_active = self.L["max_inflight"] * self.P["max_active_factor"]
        self.active = asyncio.Semaphore(self.max_active); self.admit_waiting = 0; self.peak_queue = 0
        self.budget = Budget(self.L["rpd"], self.P["daily_reserve"], persist_budget if source == "real" else None)
        self.rb = RetryBudget(self.P["retry_budget_ratio"])
        self.pages_path = pages_path; self._idx = {}
        import httpx
        self.client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=500))

    def index(self, S):
        key = (S["chunk_words"], S["overlap"])
        if key not in self._idx: self._idx[key] = Index(self.pages_path, *key)
        return self._idx[key]

    def level(self):
        p = (self.admit_waiting + self.gate.waiting) / max(1, self.max_active)
        return 0 if p < 0.5 else (1 if p < 1.0 else 2)

    # ---------- model call with the retry policy ----------
    def backoff(self, attempt):
        cap = min(self.P["backoff_cap_s"], self.P["backoff_base_s"] * 2 ** attempt)
        return random.uniform(0, cap)   # full jitter

    async def call(self, step, messages, max_tokens, stream, tr, deadline_at):
        attempt = 0; last = "error"
        while True:
            remaining = deadline_at - time.perf_counter()
            if remaining <= 0.3: raise Failed("timeout")
            est = sum(len(m["content"]) for m in messages) // 4 + max_tokens
            await self.gate.acquire(est, deadline_at, tr)
            t = time.perf_counter(); delay = None
            try:
                res = await chat(self.client, self.url, self.key, C.MODEL, messages, max_tokens=max_tokens,
                                 timeout=min(self.P["per_call_timeout_s"], deadline_at - time.perf_counter()), stream=stream)
                dt = time.perf_counter() - t; tr.work += dt; self.budget.count()
                tr.tin += res["tokens_in"]; tr.tout += res["tokens_out"]
                tr.ev(step=step, attempt=attempt, start_ms=round((t - tr.t0) * 1000, 1), dur_ms=round(dt * 1000, 1), outcome="ok",
                      ttft_ms=None if res["ttft_ms"] is None else round(res["ttft_ms"], 1))
                return res
            except RateLimited as e:
                dt = time.perf_counter() - t; tr.retry += dt; last = "rate_limited"
                tr.ev(step=step, attempt=attempt, start_ms=round((t - tr.t0) * 1000, 1), dur_ms=round(dt * 1000, 1), outcome="429", retry_after=e.retry_after)
                if e.retry_after > self.P["long_wait_s"]:
                    self.budget.flag = True; raise DailyExhausted()
                delay = max(e.retry_after, self.backoff(attempt))
            except (Transient, CallTimeout) as e:
                dt = time.perf_counter() - t; tr.retry += dt
                last = "timeout" if isinstance(e, CallTimeout) else "error"
                tr.ev(step=step, attempt=attempt, start_ms=round((t - tr.t0) * 1000, 1), dur_ms=round(dt * 1000, 1), outcome=last)
                delay = self.backoff(attempt)
            except Fatal:
                raise Failed("error")
            finally:
                self.gate.release()
            attempt += 1; tr.retries += 1
            if attempt > self.P["max_retries"] or not self.rb.allow(): raise Failed(last)
            if time.perf_counter() + delay > deadline_at: raise Failed("timeout")
            await asyncio.sleep(delay); tr.retry += delay

    # ---------- steps ----------
    @staticmethod
    def passages_block(hits): return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)

    def render(self, text, hits):
        pg = {h["id"]: h for h in hits}
        def sub(m):
            h = pg.get(m.group(1))
            if not h: return ""
            p = f"p.{h['page']}" if h["page"] == h["page_end"] else f"pp.{h['page']}-{h['page_end']}"
            return f"[{TITLES[h['paper']]}, {p}]"
        return CITE.sub(sub, text).strip()

    @staticmethod
    def lexical_ok(text, hits):
        """Cheap deterministic check: has a citation, only cites retrieved chunks, every number appears in a cited chunk."""
        ids = CITE.findall(text)
        by = {h["id"]: h["text"] for h in hits}
        if not ids or any(i not in by for i in ids): return False
        cited = " ".join(by[i] for i in set(ids)).lower()
        body = CITE.sub("", text)
        return all(n in cited for n in re.findall(r"\d+(?:\.\d+)?", body))

    def fallback(self, q, S, tr):
        hits = self.index(S).search(q, 2, False)
        lines = []
        for h in hits:
            words = h["text"].split()[:60]
            lines.append(f"{' '.join(words)} [{h['id']}]")
        txt = "Model capacity is used up for today, so here are the most relevant passages instead:\n" + "\n".join(lines)
        return dict(answer_type="fallback", text=self.render(txt, hits))

    async def pipeline(self, q, S, tr, deadline_at):
        P = self.P; lvl = self.level(); idx = self.index(S)
        k, rerank = S["k"], S["rerank"]
        if lvl >= 1: k, rerank = min(k, 2), False
        tr.ev(step="admit", level=lvl)
        if self.budget.exhausted: return self.fallback(q, S, tr)
        try:
            query = q
            if lvl < 2:
                r = await self.call("analyze", [
                    {"role": "system", "content": '[[TASK:ANALYZE]] Rewrite the question as a short keyword search query for machine-learning papers. Reply as JSON: {"query": "..."}'},
                    {"role": "user", "content": q}], 300, False, tr, deadline_at)
                try: query = json.loads(re.search(r"\{.*\}", r["text"], re.S).group(0))["query"] or q
                except Exception: query = q
            t = time.perf_counter(); hits = idx.search(query + " " + q, k, rerank); tr.work += time.perf_counter() - t
            top = hits[0]["conf"] if hits else 0.0
            tr.ev(step="retrieve", top_conf=round(top, 3), k=k, rerank=rerank)
            if top < P["refuse_conf"]:
                return dict(answer_type="refused", text=REFUSAL)
            if top < P["expand_conf"]:    # tool call: fetch the neighbouring chunk, with its own timeout
                t = time.perf_counter()
                try:
                    nb = await asyncio.wait_for(self.tool_neighbour(idx, hits[0]["id"]), P["tool_timeout_s"])
                    if nb and nb["id"] not in {h["id"] for h in hits}: hits.append(nb)
                    tr.ev(step="tool_neighbour", outcome="ok")
                except asyncio.TimeoutError:
                    tr.ev(step="tool_neighbour", outcome="timeout")
                tr.work += time.perf_counter() - t
            strict = ""
            for attempt in range(2):      # loop: one redraft if verification fails
                d = await self.call("draft", [
                    {"role": "system", "content": "[[TASK:DRAFT]] Answer ONLY from the passages. Cite every claim with the passage id in square brackets, e.g. [att:p3:c12]. "
                     "Copy numbers exactly. If the passages do not contain the answer, reply exactly NOT_IN_PAPERS." + strict},
                    {"role": "user", "content": f"Passages:\n{self.passages_block(hits)}\n\nQuestion: {q}"}], 600, True, tr, deadline_at)
                tr.draft_ttft = d["ttft_ms"] if tr.draft_ttft is None else tr.draft_ttft
                text = d["text"].strip()
                if "NOT_IN_PAPERS" in text: return dict(answer_type="refused", text=REFUSAL)
                ok = self.lexical_ok(text, hits)
                if ok and lvl < 1 and P["llm_verify"]:
                    v = await self.call("verify", [
                        {"role": "system", "content": '[[TASK:VERIFY]] Is every claim in the answer supported by the cited passages? Reply JSON: {"supported": true|false}'},
                        {"role": "user", "content": f"Passages:\n{self.passages_block(hits)}\n\nAnswer: {text}"}], 300, False, tr, deadline_at)
                    ok = '"supported": true' in v["text"].lower().replace("true ", "true")
                if ok: return dict(answer_type="answer", text=self.render(text, hits))
                strict = " Your previous answer failed a fact check; cite only what the passages literally say."
            return dict(answer_type="refused", text=REFUSAL + " (I could not verify a draft.)")
        except DailyExhausted:
            return self.fallback(q, S, tr)

    async def tool_neighbour(self, idx, cid):
        if random.random() < float(os.getenv("TOOL_SLOW_P", "0")): await asyncio.sleep(5)   # fault injection for tests
        return idx.neighbour(cid)

    # ---------- one request ----------
    async def run_request(self, q, setting):
        S = C.SETTINGS[setting]; tr = Trace(); deadline_at = tr.t0 + self.P["deadline_s"]; self.rb.credit()
        status, out = "ok", dict(answer_type="failed", text="")
        t = time.perf_counter(); self.admit_waiting += 1
        self.peak_queue = max(self.peak_queue, self.admit_waiting + self.gate.waiting)
        try:
            try:
                await asyncio.wait_for(self.active.acquire(), timeout=max(0.01, deadline_at - time.perf_counter()))
            except asyncio.TimeoutError:
                raise Failed("timeout")
            finally:
                self.admit_waiting -= 1; tr.wait += time.perf_counter() - t
            try: out = await self.pipeline(q, S, tr, deadline_at)
            finally: self.active.release()
        except Failed as e: status = e.kind
        total = tr.now_ms()
        return dict(status=status, first_token_ms=total if status == "ok" else None, total_ms=total, retries=tr.retries,
                    tokens_in=tr.tin, tokens_out=tr.tout, answer_type=out["answer_type"], text=out["text"],
                    trace=dict(wait_ms=round(tr.wait * 1000, 1), work_ms=round(tr.work * 1000, 1), retry_ms=round(tr.retry * 1000, 1),
                               draft_first_token_ms=tr.draft_ttft, events=tr.events))
