"""Download the three papers at the exact versions and extract text per page -> corpus/pages.json.
Run on your own machine (arxiv.org must be reachable)."""
import json, httpx, io
from pypdf import PdfReader

PAPERS = {"att": "1706.03762v7", "moshi": "2410.00037v1", "dc": "2507.07955v2"}

pages = []
for key, ver in PAPERS.items():
    r = httpx.get(f"https://arxiv.org/pdf/{ver}", follow_redirects=True, timeout=60)
    r.raise_for_status()
    open(f"corpus/{key}_{ver}.pdf", "wb").write(r.content)
    reader = PdfReader(io.BytesIO(r.content))
    for i, p in enumerate(reader.pages, 1):
        pages.append(dict(paper=key, version=ver, page=i, text=(p.extract_text() or "")))
    print(key, ver, len(reader.pages), "pages")
json.dump(pages, open("corpus/pages.json", "w"))
