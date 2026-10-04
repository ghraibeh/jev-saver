"""Offline benchmark on BANKING77 (public dataset, 77 intents). No Jev calls, no key needed.

The 'teacher' here is the dataset's gold label, standing in for Jev. That is a BEST CASE:
real Jev is sometimes wrong, and the saver would store those mistakes.

Setup:
  - warm start: memory pre-filled from the 10,003 BANKING77 training messages (no teacher calls)
  - stream: the 3,080 test messages in random order; the saver answers locally when sure,
    otherwise asks the teacher and learns the answer.
Reports per quarter: share sent to the teacher, accuracy of local answers.
Also a cold-start run (empty memory).
"""
import csv, io, sys, time, urllib.request
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from jev_saver import JevSaver

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def load(split):
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(URL.format(split)).read().decode())))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def run(warm: bool):
    trX, trY = load("train"); teX, teY = load("test")
    labels = {c: c.replace("_", " ") for c in sorted(set(trY))}
    gold = dict(zip(teX, teY))
    teacher = lambda text: (gold[text], 1.0, [])          # stands in for Jev
    s = JevSaver(labels, "Which banking request type is this?", ask_jev=teacher)
    if warm:
        t = time.time(); s.warm_start(trX, trY); print(f"warm start: {len(s.labs)} memories from {len(trX)} messages ({time.time()-t:.0f}s)")
    order = np.random.default_rng(0).permutation(len(teX))
    local_ok = local_n = 0
    print(f"{'quarter':8} {'to teacher':>11} {'local answers':>14} {'local correct':>14}")
    for q, part in enumerate(np.array_split(order, 4), 1):
        before = dict(s.stats); ok = n = 0
        for i in part:
            d = s.decide(teX[i])
            if d.source == "local":
                n += 1; ok += d.choice == teY[i]
        asked = s.stats["jev"] - before["jev"]
        local_ok += ok; local_n += n
        print(f"Q{q:<7} {asked/len(part):>10.0%} {n/len(part):>14.0%} {ok/max(n,1):>14.1%}")
    print(f"ALL: teacher calls saved {s.saved_fraction():.0%}, local accuracy {local_ok/max(local_n,1):.1%}, "
          f"local mistakes {local_n-local_ok}, memories {len(s.labs)}\n")


if __name__ == "__main__":
    print("== warm start (memory pre-filled from training data)"); run(True)
    print("== cold start (empty memory)"); run(False)
