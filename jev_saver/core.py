"""jev-saver: a local memory layer in front of Jev that saves calls on repeated decisions.

How it works
------------
1. Every input text is turned into a vector by a small local embedding model (CPU, no GPU).
2. The saver looks for the most similar inputs it has already seen.
3. If they agree and are close enough, it answers locally (no Jev call).
4. Otherwise it asks Jev, returns Jev's answer, and stores it, so the same (or a very
   similar) input is answered locally next time.

This is semantic caching + nearest-neighbour voting. It does NOT understand anything:
it only re-uses answers for inputs that look like inputs it has seen before.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

JEV_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass
class Decision:
    choice: str
    confidence: float
    source: str                      # "local" or "jev"
    top: list = field(default_factory=list)
    closest_similarity: float = 0.0
    ms: float = 0.0


class JevSaver:
    """Local memory in front of one Jev Choice question.

    Parameters
    ----------
    labels : dict[str, str]
        Choice options -> description (passed to Jev as `criteria`).
    instructions : str
        The Choice question sent to Jev.
    api_key : str | None
        TypeSafe key. Defaults to env TYPESAFE_API_KEY.
    model : str
        Jev version. Pin it (e.g. "jev-1.13.0") so stored answers stay consistent.
    embedder : str
        sentence-transformers model name used locally.
    k : int
        Neighbours used for the local vote.
    min_similarity : float
        The closest stored input must be at least this similar to answer locally.
    min_margin : float
        Top local vote must beat the second by this much to answer locally.
    exact_similarity : float
        Above this, the closest stored input is treated as "the same input" and its answer is used alone.
    merge_similarity : float
        When learning, a new example this close to a stored one with the same label is merged into it
        (keeps memory small). Set > 1 to never merge.
    min_jev_confidence : float
        Jev answers below this confidence are returned but NOT stored (avoid learning Jev's doubts).
    """

    def __init__(self, labels: dict, instructions: str, api_key: str | None = None,
                 model: str = "jev-latest", embedder: str = "BAAI/bge-small-en-v1.5",
                 k: int = 5, min_similarity: float = 0.90, min_margin: float = 0.30,
                 exact_similarity: float = 0.98, merge_similarity: float = 0.95,
                 min_jev_confidence: float = 0.0, ask_jev=None):
        from sentence_transformers import SentenceTransformer
        self.labels = dict(labels)
        self.names = list(self.labels)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.instructions = instructions
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.enc = SentenceTransformer(embedder, device="cpu")
        self.k, self.min_similarity, self.min_margin = k, min_similarity, min_margin
        self.exact_similarity, self.merge_similarity = exact_similarity, merge_similarity
        self.min_jev_confidence = min_jev_confidence
        self._ask_jev = ask_jev or self._jev_http   # injectable (tests / offline benchmarks)
        dim = len(self.enc.encode(["x"], normalize_embeddings=True)[0])
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.labs = np.zeros(0, dtype=np.int64)
        self.counts = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.stats = {"local": 0, "jev": 0}

    # ---------- embedding / memory ----------
    def embed(self, texts):
        return self.enc.encode(list(texts), batch_size=128, normalize_embeddings=True).astype(np.float32)

    def learn(self, text_or_vec, label: str) -> str:
        """Store one example. Returns 'new' or 'merged'."""
        x = self.embed([text_or_vec])[0] if isinstance(text_or_vec, str) else np.asarray(text_or_vec, np.float32)
        lab = self.index[label]
        with self.lock:
            if len(self.labs):
                sims = np.where(self.labs == lab, self.vecs @ x, -2.0)
                j = int(sims.argmax())
                if sims[j] > self.merge_similarity:
                    self.counts[j] += 1
                    v = self.vecs[j] + (x - self.vecs[j]) / self.counts[j]
                    self.vecs[j] = v / np.linalg.norm(v)
                    return "merged"
            self.vecs = np.vstack([self.vecs, x[None]])
            self.labs = np.append(self.labs, lab)
            self.counts = np.append(self.counts, 1.0)
            return "new"

    def warm_start(self, texts, labels):
        """Pre-fill memory from labelled examples you already have (no Jev calls)."""
        X = self.embed(texts)
        for x, y in zip(X, labels):
            self.learn(x, y)

    # ---------- local vote ----------
    def _local(self, x):
        n = len(self.labs)
        if n == 0:
            return None, 0.0, [], 0.0, False
        with self.lock:
            sims = self.vecs @ x
            kk = min(self.k, n)
            idx = np.argpartition(-sims, kk - 1)[:kk]
            idx = idx[np.argsort(-sims[idx])]
            v = sims[idx]
            if v[0] >= self.exact_similarity:          # same input seen before -> trust it alone
                score = np.zeros(len(self.names)); score[self.labs[idx[0]]] = 1.0
            else:
                w = np.exp((v - 1.0) * 20.0)
                score = np.bincount(self.labs[idx], weights=w, minlength=len(self.names))
                score = score / score.sum()
        order = np.argsort(-score)[:3]
        top = [(self.names[i], float(score[i])) for i in order]
        margin = top[0][1] - (top[1][1] if len(top) > 1 else 0.0)
        sure = bool(v[0] >= self.min_similarity and margin > self.min_margin)
        return top[0][0], top[0][1], top, float(v[0]), sure

    # ---------- Jev ----------
    def _jev_http(self, text: str):
        if not self.api_key:
            raise RuntimeError("no Jev key: set TYPESAFE_API_KEY or pass api_key=")
        body = {"state": text, "model": self.model, "questions": {"q": {
            "type": "choice", "instructions": self.instructions, "criteria": self.labels}}}
        req = urllib.request.Request(JEV_URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        a = json.loads(urllib.request.urlopen(req, timeout=60).read())["answers"]["q"]
        probs = sorted((a.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:3]
        return a["choice"], float(a.get("confidence") or 0.0), probs

    # ---------- public ----------
    def decide(self, text: str, learn: bool = True) -> Decision:
        t = time.perf_counter()
        x = self.embed([text])[0]
        choice, conf, top, sim, sure = self._local(x)
        if sure:
            self.stats["local"] += 1
            return Decision(choice, conf, "local", top, sim, (time.perf_counter() - t) * 1000)
        jc, jconf, jtop = self._ask_jev(text)
        self.stats["jev"] += 1
        if learn and jc in self.index and jconf >= self.min_jev_confidence:
            self.learn(x, jc)
        return Decision(jc, jconf, "jev", jtop, sim, (time.perf_counter() - t) * 1000)

    def saved_fraction(self) -> float:
        n = self.stats["local"] + self.stats["jev"]
        return self.stats["local"] / n if n else 0.0

    # ---------- persistence ----------
    def save(self, path: str):
        np.savez(path, vecs=self.vecs, labs=self.labs, counts=self.counts,
                 names=np.array(self.names), model=np.array(self.model))

    def load(self, path: str):
        z = np.load(path, allow_pickle=False)
        if list(z["names"]) != self.names:
            raise ValueError("stored memory was built for different labels")
        self.vecs, self.labs, self.counts = z["vecs"], z["labs"], z["counts"]
        return self
