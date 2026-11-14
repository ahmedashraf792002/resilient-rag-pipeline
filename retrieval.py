import json, math, re, collections

STOP = set("a an the of in on for to and or is are was were be by with as at from that this it its what which how does do did why who when where than then into using use used paper papers".split())
TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
def toks(s): return TOK.findall(s.lower())
def qterms(q): return [t for t in toks(q) if t not in STOP]

class Index:
    def __init__(self, pages_path, chunk_words, overlap):
        pages = json.load(open(pages_path))
        by = collections.OrderedDict()
        for p in pages: by.setdefault(p["paper"], []).append(p)
        self.chunks = []
        for paper, ps in by.items():
            words, wpage = [], []
            for p in ps:
                for w in p["text"].split():
                    words.append(w); wpage.append(p["page"])
            step = max(1, chunk_words - overlap); i = n = 0
            while i < len(words):
                j = min(i + chunk_words, len(words))
                self.chunks.append(dict(id=f"{paper}:p{wpage[i]}:c{n}", paper=paper, page=wpage[i],
                                        page_end=wpage[j - 1], text=" ".join(words[i:j])))
                n += 1; i += step
        self.pos = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.tf = [collections.Counter(toks(c["text"])) for c in self.chunks]
        self.len = [sum(t.values()) or 1 for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = collections.Counter()
        for t in self.tf: df.update(t.keys())
        N = len(self.chunks)
        self.idf = {w: math.log(1 + (N - d + .5) / (d + .5)) for w, d in df.items()}
        self.max_idf = max(self.idf.values())

    def _bm25(self, terms, k1=1.5, b=.75):
        sc = [0.0] * len(self.chunks)
        for t in set(terms):
            idf = self.idf.get(t)
            if idf is None: continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f: sc[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
        return sc

    def coverage(self, terms, i):
        """idf-weighted share of query terms present in chunk i; unknown words count as rare (penalises unanswerable questions)."""
        ts = set(terms)
        if not ts: return 0.0
        w = lambda t: self.idf.get(t, self.max_idf)
        return sum(w(t) for t in ts if t in self.tf[i]) / sum(w(t) for t in ts)

    def search(self, query, k, rerank):
        terms = qterms(query)
        sc = self._bm25(terms)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        pool = order[: k * 3] if rerank else order[:k]
        if rerank:
            raw = toks(query); bigr = set(zip(raw, raw[1:]))
            def rr(i):
                tl = toks(self.chunks[i]["text"])
                return self.coverage(terms, i) + 0.5 * len(bigr & set(zip(tl, tl[1:]))) / max(1, len(bigr))
            pool = sorted(pool, key=lambda i: -rr(i))[:k]
        return [dict(self.chunks[i], bm25=sc[i], conf=self.coverage(terms, i)) for i in pool]

    def neighbour(self, chunk_id):
        i = self.pos[chunk_id] + 1
        if i < len(self.chunks) and self.chunks[i]["paper"] == self.chunks[i - 1]["paper"]:
            return self.chunks[i]
        return None
