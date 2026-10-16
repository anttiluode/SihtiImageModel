"""Gates GG1-GG3 from PREREG_GLOBAL.md.   python global_gates.py [mnist fashion]

Needs the caches from gates.py (whole-image model, judge), patch_gates.py (P3) and codes2_gates.py (Q).
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from patch_gates import shift
from sihti_model.data import load
from sihti_model.globalpatch import GlobalPatchMemory
from sihti_model.patch import NB, bands_of

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")
ARMS = ["P3", "Q", "G", "GP", "GR"]


def psnr(a, b):
    mse = ((np.clip(a, 0, 1) - b) ** 2).reshape(len(a), -1).mean(1)
    return float(np.mean(10 * np.log10(1 / np.maximum(mse, 1e-12))))


def run(ds):
    R = {"dataset": ds}
    Xtr, ytr = load(ds, "train"); Xtr = Xtr[:, 0]
    Xte, yte = load(ds, "test"); Xte = Xte[:, 0]
    pm = pickle.load(open(os.path.join(CACHE, f"patch_{ds}.pkl"), "rb"))
    wm = pickle.load(open(os.path.join(CACHE, f"model_{ds}.pkl"), "rb"))
    q = pickle.load(open(os.path.join(CACHE, f"codes2_{ds}.pkl"), "rb"))
    judge = pickle.load(open(os.path.join(CACHE, f"judge_{ds}.pkl"), "rb"))
    batches = lambda: (Xtr[i:i + 5000] for i in range(0, len(Xtr), 5000))
    models = {}
    for mode in ("G", "GP", "GR"):
        p = os.path.join(CACHE, f"global_{mode}_{ds}.pkl")
        if os.path.exists(p):
            models[mode] = pickle.load(open(p, "rb"))
        else:
            t = time.time()
            models[mode] = GlobalPatchMemory(pm, wm, mode).fit(batches)
            print(f"{ds} {mode} fit {time.time() - t:.0f}s")
            pickle.dump(models[mode], open(p, "wb"))
    R["size"] = {"P3": pm.size()["P3"], "Q": q.size(), **{m: models[m].size() for m in models}}

    def complete(arm, B, keep):
        if arm == "P3":
            return pm.complete(B, keep, "P3")
        if arm == "Q":
            img, _, out = q.complete(B, keep)
            return img, out
        return models[arm].complete(B, keep)

    # fidelity, 2000 images
    idx = np.random.default_rng(4).choice(len(Xte), 2000, replace=False)
    X = Xte[idx]; B = bands_of(X)
    for keep in (1, 2):
        R[f"fill_keep{keep}"] = {a: psnr(complete(a, B, keep)[0], X) for a in ARMS}
        print(f"{ds} fill keep{keep}: " + "  ".join(f"{a} {v:.2f}" for a, v in R[f"fill_keep{keep}"].items()))
    Xs = shift(X, np.random.default_rng(11)); Bs = bands_of(Xs)
    R["shift_keep2"] = {a: psnr(complete(a, Bs, 2)[0], Xs) for a in ARMS}
    R["shift_cost_dB"] = {a: R["fill_keep2"][a] - R["shift_keep2"][a] for a in ARMS}
    print(f"{ds} shift cost: " + "  ".join(f"{a} {v:.2f}" for a, v in R["shift_cost_dB"].items()))

    # identity, individuality, band errors, 1000 images
    idx = np.random.default_rng(4).choice(len(Xte), 1000, replace=False)
    X, y = Xte[idx], yte[idx]; B = bands_of(X)
    flat = lambda Z: np.clip(Z, 0, 1).reshape(len(Z), -1)
    ok = judge.predict(flat(X)) == y
    rng = np.random.default_rng(21)
    pools = []
    for i, j in enumerate(idx):
        same = np.flatnonzero((yte == y[i]) & (np.arange(len(yte)) != j))
        pools.append(np.concatenate([[j], rng.choice(same, 99, replace=False)]))
    pools = np.stack(pools)                                       # (1000, 100), column 0 = truth
    cand = Xte[pools].reshape(1000, 100, -1)
    R["flips_keep1"], R["individuality_top1"], R["band_error_keep1"] = {}, {}, {}
    for a in ARMS:
        img, out = complete(a, B, 1)
        f = flat(img)
        R["flips_keep1"][a] = int(((judge.predict(f) != y) & ok).sum())
        d = ((cand - f[:, None]) ** 2).sum(-1)
        R["individuality_top1"][a] = float((d.argmin(1) == 0).mean())
        R["band_error_keep1"][a] = {f"r{b}": float(((out[b] - B[b]) ** 2).mean()) for b in range(NB - 2, -1, -1)}
    R["n_ok"] = int(ok.sum())
    print(f"{ds} flips keep1 (of {ok.sum()}): {R['flips_keep1']}")
    print(f"{ds} individuality top-1: " + "  ".join(f"{a} {v:.3f}" for a, v in R["individuality_top1"].items()))
    for a in ARMS:
        print(f"{ds} band error {a}: " + "  ".join(f"{k} {v:.2e}" for k, v in R["band_error_keep1"][a].items()))

    # information ceiling on the same 1000 images
    G = models["G"]
    gtr = np.concatenate([G.raw_g(bands_of(Xtr[i:i + 10000])[4]) for i in range(0, len(Xtr), 10000)])
    gte = G.raw_g(B[4])
    mu, sd = gtr.mean(0), gtr.std(0)
    gtr, gte = (gtr - mu) / sd, (gte - mu) / sd
    R["readers_same_1000"] = {
        "kNN15": float(KNeighborsClassifier(15).fit(gtr, ytr).score(gte, y)),
        "MLP": float(MLPClassifier((256, 256), max_iter=60, random_state=0).fit(gtr, ytr).score(gte, y)),
        **{f"judge_on_{a}_fill": None for a in []}}
    R["readers_same_1000"]["judge_on_real"] = float(ok.mean())
    print(f"{ds} readers on the same 1000: {R['readers_same_1000']}")

    f1, fl, ind = R["fill_keep1"], R["flips_keep1"], R["individuality_top1"]
    R["gates"] = {a: {"GG1": bool(f1[a] >= f1["P3"] + 0.3), "GG2": bool(fl[a] <= 0.8 * fl["P3"]),
                      "GG3": bool(ind[a] >= ind["P3"] - 0.02)} for a in ("G", "GP", "GR")}
    print(f"{ds} gates: {R['gates']}")
    json.dump(R, open(os.path.join(OUT, f"global_results_{ds}.json"), "w"), indent=1)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["mnist", "fashion"]:
        run(ds)
