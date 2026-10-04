# jev-saver

A small local memory layer in front of [TypeSafe Jev](https://typesafe.ai) for repeated **Choice** decisions.

- If a new input looks like inputs it has already seen, and those inputs agree, it **answers locally**. No Jev call is made.
- Otherwise it **asks Jev**, returns Jev's answer, and **stores it**, so the same or a very similar input is answered locally next time.

It runs on CPU and keeps a small, saveable memory file.

> **What this is, honestly:** semantic caching plus nearest-neighbour voting. It does **not** understand anything new. It only re-uses answers for inputs that look like inputs it has already seen. A rephrasing with different words often still goes to Jev. If your traffic rarely repeats, it will save little.

## Why

Jev is already cheap (about $0.042 per 1M input tokens). For most people, money is **not** the main reason to use this. The main reasons are:

- **Latency:** in our runs, a local answer took about 15–60 ms on CPU, compared with roughly 250–400 ms for a Jev round trip.
- **Rate limits:** fewer requests count toward Jev's per-minute limits.
- **Offline / privacy:** repeated inputs never leave the machine.

## Install

```bash
pip install git+https://github.com/ghraibeh/jev-saver
export TYPESAFE_API_KEY=...
```

## Use

```python
from jev_saver import JevSaver

saver = JevSaver(
    labels={"billing": "Payment issues", "technical": "Bugs or errors", "account": "Login or profile"},
    instructions="Which team should handle this ticket?",
    model="jev-1.13.0",          # pin the version so stored answers stay consistent
)

d = saver.decide("I was charged twice this month")
print(d.source, d.choice, d.confidence)   # "jev" the first time, "local" for repeats

saver.warm_start(texts, labels)  # optional: pre-fill from labelled data you already have
saver.save("memory.npz")         # reuse later: saver.load("memory.npz")
```

Main knobs:

- `min_similarity`: how close a stored input must be before the saver answers locally.
- `min_margin`: how clearly the local vote must win.
- `exact_similarity`: above this, the closest stored input is treated as the same input.
- `min_jev_confidence`: do not store Jev answers that are below this confidence.

## Benchmark (offline, reproducible)

Run `python benchmarks/banking77.py`. It uses BANKING77 (77 intents) and needs no key.

**The teacher in this benchmark is the dataset's gold label, standing in for Jev.** That makes it a **best case**: real Jev is sometimes wrong, and the saver would store those mistakes. A live run against Jev has **not** been done yet.

Setup: 3,080 test messages in random order. The saver answers locally when it is sure; otherwise it asks the teacher and learns the answer. Default settings.

| setup | teacher calls saved | local answers correct | local mistakes |
|---|---|---|---|
| warm start (memory pre-filled from 10,003 training messages) | **87%** | **97.6%** | 65 |
| cold start (empty memory, learns only from the stream) | **53%** | **97.4%** | 42 |

Cold start, by quarter of the stream: 31% → 54% → 65% → 64% answered locally.

## Limits we measured

- **No generalisation.** In a separate test, the teacher labelled 548 new messages. Accuracy on *other* unseen messages stayed at 93.7%, the same as before. The saver remembers; it does not learn the concept.
- **Wrong but sure.** About 2–3% of local answers are confidently wrong, and those are never sent to Jev.
- **English embedder.** The default `bge-small-en-v1.5` is weak on other languages. Swap the `embedder=` for a multilingual model if you need one.
- **Shifting answers.** Stored answers are tied to the Jev version and to your labels. Clear the memory when either changes.

## Related work

- [GPTCache](https://github.com/zilliztech/GPTCache): semantic cache for LLMs.
- [jevcache](https://github.com/hyperspaceai/jevcache): exact-match cache for Jev decisions.
- FrugalGPT, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176): cascades that defer to a larger model only when needed.

## License

MIT
