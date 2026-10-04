"""Threshold sweep on BANKING77 (offline, no Jev, no key). Teacher = gold label (best case).
For each setting: share of calls saved and accuracy of local answers, warm and cold start.
Embeddings are computed once and reused, so the sweep runs in a few minutes on CPU."""
import csv, io, sys, time, urllib.request
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from jev_saver import JevSaver

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def load(split):
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(URL.format(split)).read().decode())))
    return [r["text"] for r in rows], [r["category"] for r in rows]


trX, trY = load("train"); teX, teY = load("test")
labels = {c: c.replace("_", " ") for c in sorted(set(trY))}
gold = dict(zip(teX, teY))
base = JevSaver(labels, "q", ask_jev=lambda t: (gold[t], 1.0, []))
t = time.time()
TR = base.embed(trX); TE = base.embed(teX)
print(f"embedded {len(trX)+len(teX)} messages in {time.time()-t:.0f}s", flush=True)
cache = {x: v for x, v in zip(teX, TE)}
order = np.random.default_rng(0).permutation(len(teX))


def run(warm, **kw):
    s = JevSaver.__new__(JevSaver); s.__dict__.update(base.__dict__)
    for k, v in kw.items(): setattr(s, k, v)
    s.vecs = base.vecs[:0].copy(); s.labs = base.labs[:0].copy(); s.counts = base.counts[:0].copy()
    s.stats = {"local": 0, "jev": 0}
    s.embed = lambda texts: np.stack([cache[x] for x in texts])
    if warm:
        for x, y in zip(TR, trY): s.learn(x, y)
    ok = n = 0
    for i in order:
        d = s.decide(teX[i])
        if d.source == "local": n += 1; ok += d.choice == teY[i]
    return s.saved_fraction(), ok / max(n, 1), n - ok


print(f"{'start':5} {'min_sim':>7} {'margin':>6} {'saved':>6} {'local acc':>9} {'wrong':>6}")
for warm in (True, False):
    for ms in (0.85, 0.90, 0.94):
        for mg in (0.1, 0.3, 0.6):
            sv, acc, wrong = run(warm, min_similarity=ms, min_margin=mg)
            print(f"{'warm' if warm else 'cold':5} {ms:>7.2f} {mg:>6.1f} {sv:>6.0%} {acc:>9.1%} {wrong:>6}", flush=True)
