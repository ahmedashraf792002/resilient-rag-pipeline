import json, collections
rows = [json.loads(l) for l in open("raw/probe_real.jsonl")]
lat = [r for r in rows if str(r["tag"]).startswith("lat_c")]
print("total latency rows:", len(lat))
print(collections.Counter(str(r["status"]) for r in lat))
for r in lat[:3]:
    print(r["status"], r["usage"], r["ttft_ms"], r["body"][:300])