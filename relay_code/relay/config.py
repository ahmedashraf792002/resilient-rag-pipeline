"""All tunables in one place. Values marked PLACEHOLDER must be replaced by YOUR measurements."""
import os

if os.path.exists(".env"):
    for _l in open(".env", encoding="utf-8"):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
STANDIN_URL = os.getenv("STANDIN_URL", "http://127.0.0.1:8100/v1/chat/completions")
STANDIN_ADMIN = os.getenv("STANDIN_ADMIN", "http://127.0.0.1:8100/admin")
MODEL = MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b") # check the Groq console for current models/limits
GROQ_KEY = os.getenv("GROQ_API_KEY", "")                  # never commit this

# Price per million tokens, used for cost per 1,000 answers. PLACEHOLDER: copy from Groq's pricing page.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.05"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0.08"))

# What the CLIENT believes the limits are. PLACEHOLDER: fill from Table 1 (probe.py + docs).
LIMITS = dict(rpm=30, tpm=8000, rpd=1000, max_inflight=8)

POLICY = dict(
    deadline_s=30.0,          # a person gives up after this; request is counted as timeout
    per_call_timeout_s=12.0,  # one model call; protects against "succeeds slowly"
    max_retries=3,
    backoff_base_s=0.5,
    backoff_cap_s=8.0,
    retry_budget_ratio=0.2,   # retries allowed per new request (None = unlimited, for the retry-storm experiment)
    long_wait_s=120.0,        # a 429 with retry-after above this is treated as "daily budget gone"
    daily_reserve=0.05,       # stop using the model when this fraction of RPD is left
    refuse_conf=0.35,         # top-chunk coverage below this -> "papers do not say" without calling the model
    expand_conf=0.60,         # below this (but above refuse) -> call the neighbour-chunk tool
    llm_verify=True,          # second model call that checks the draft against the passages
    max_active_factor=2,      # admitted requests = factor * max_inflight; the rest wait in the admission queue
    tool_timeout_s=1.0,
)

# Retrieval settings. Sizes are in WORDS (not model tokens).
SETTINGS = {
    "S1": dict(chunk_words=120, overlap=20, k=4, rerank=False),
    "S2": dict(chunk_words=250, overlap=40, k=3, rerank=True),
    "S3": dict(chunk_words=500, overlap=80, k=2, rerank=False),
    "S4": dict(chunk_words=250, overlap=40, k=6, rerank=True),
}
DEFAULT_SETTING = "S2"
