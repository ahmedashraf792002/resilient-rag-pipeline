# The Relay - code skeleton

Python 3.11+. `pip install -r requirements.txt`. Set `GROQ_API_KEY` in your shell (never commit it).

## Run order
1. `python fetch_papers.py` - downloads the 3 papers at the exact versions, extracts text per page into corpus/pages.json. **Open the JSON and read some pages**: two-column PDFs, tables and equations extract badly, and that is material for the "how this library will mislead a naive retriever" section.
2. `python probe.py limits --burst 40`, then `python probe.py latency --n 12` - measures the real service (raw/probe_real.jsonl). Read Groq's rate-limit docs and compare with the `x-ratelimit-*` / `retry-after` headers; fill Table 1 from both. The daily limit is shared with everything you run: do not burn it on purpose, read it from headers/docs and mark confidence medium.
3. Put your Table 1 numbers into `config.LIMITS` and `standin_config.json`, then `python calibrate.py`.
4. `python standin.py` (separate terminal), then `python loadgen.py load --scenario steady|burst|overload|rpm_halved|daily_exhaust --source standin`. A small real run: `--source real --scenario steady --n 40 --rate 6`.
5. `python loadgen.py quality --source real --settings S1 S2 S3 S4` - 18 questions x settings (real model; grades only computed for real rows).
6. `python analyze.py --slowest real` prints Tables 2a/2b/3 and the wait/work/retry split of the slowest request (from raw/trace.jsonl).

## Things you must check or decide yourself (the reviewers will ask)
- Every number in config.py / standin_config.json is a PLACEHOLDER until your probe replaces it. Model name and prices too.
- questions.json: gold answers for Moshi and Dynamic Chunking are from my memory and marked "verify" - check each against the PDF and fix the regexes. Add questions of your own, including ones whose answer sits in a table or figure caption.
- Design decision to own: the draft is buffered and verified BEFORE anything is released, so first_token_ms is the time of the first VERIFIED token (draft time-to-first-token is in the trace). That trades speed for never showing an unchecked answer. If you disagree, change `pipeline.py` and say why.
- Degradation order (lvl 1: k<=2, no rerank, no LLM verify; lvl 2: also skip analyze), the retry budget, refuse_conf/expand_conf thresholds: tune them from your runs; do not keep defaults you cannot justify.
- Peak queue length in Table 2b is computed from the CSV as peak requests concurrently in the system; the runner's true queue length is in trace.jsonl. State your definition.

## Known weaknesses
- Stand-in text is canned, so it says nothing about answer quality; it only models timing, limits and faults. Stand-in rows have an empty answer_grade.
- Grading is regex + citation-title matching: brittle. Spot-check by hand and report any disagreement.
- Retrieval is BM25 with an optional lexical rerank; sizes are in words, not model tokens.
- Client-side limiter assumes one process; the real provider may count tokens differently.
