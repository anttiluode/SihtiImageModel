"""Gates CG1-CG5 from PREREG_CODES2.md.

    python codes2_gates.py mnist      # CG1-CG4  (needs patch_gates.py + gates.py caches: P3 model, judge)
    python codes2_gates.py fashion
    python codes2_gates.py bsds       # reported comparison on photos
    python codes2_gates.py cifar10    # CG5, fits P3 and Q per colour channel (run where CIFAR is on disk)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from patch_gates import bsds_images, by_shape, shift
from sihti_model.codes2 import ContextPatchMemory
from sihti_model.data import load
from sihti_model.patch import NB, PatchMemory, bands_of
from sihti_model.uncertainty import code_variances, fill_with_uncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")


def psnr(a, b):
    mse = ((np.clip(a, 0, 1) - b) ** 2).reshape(len(a), -1).mean(1)
    return float(np.mean(10 * np.log10(1 / np.maximum(mse, 1e-12))))


def fit_q(name, batches_fn, per_image):
    p = os.path.join(CACHE, f"codes2_{name}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    t = time.time()
    q = ContextPatchMemory(per_image=per_image).fit(batches_fn)
    print(f"Q fit {time.time() - t:.0f}s")
    pickle.dump(q, open(p, "wb"))
    return q


def run_small(ds):
    R = {"dataset": ds}
    Xtr, ytr = load(ds, "train"); Xtr = Xtr[:, 0]
    Xte, yte = load(ds, "test")
    batches = lambda: (Xtr[i:i + 5000] for i in range(0, len(Xtr), 5000))
    q = fit_q(ds, batches, 10)
    pm = pickle.load(open(os.path.join(CACHE, f"patch_{ds}.pkl"), "rb"))
    judge = pickle.load(open(os.path.join(CACHE, f"judge_{ds}.pkl"), "rb"))
    R["size"] = {"Q": q.size(), "P3": pm.size()["P3"]}

    # CG1: fill quality, 2000 test images
    idx = np.random.default_rng(4).choice(len(Xte), 2000, replace=False)
    X = Xte[idx, 0]
    B = bands_of(X)
    for keep in (1, 2):
        R[f"fill_keep{keep}"] = {"none": psnr(sum(B[j] for j in range(NB - keep, NB)), X),
                                 "P3": psnr(pm.complete(B, keep, "P3")[0], X),
                                 "Q": psnr(q.complete(B, keep)[0], X)}
        print(f"{ds} fill keep{keep}: {R[f'fill_keep{keep}']}")
    f1 = R["fill_keep1"]
    R["CG1_gain_keep1_dB"] = float(f1["Q"] - f1["P3"])
    R["CG1_pass"] = bool(R["CG1_gain_keep1_dB"] >= 0.3)

    # CG4: shift, keep 2
    Xs = shift(X, np.random.default_rng(11))
    Bs = bands_of(Xs)
    R["shift_keep2"] = {"P3": psnr(pm.complete(Bs, 2, "P3")[0], Xs), "Q": psnr(q.complete(Bs, 2)[0], Xs)}
    R["CG4_drop_dB"] = float(R["fill_keep2"]["Q"] - R["shift_keep2"]["Q"])
    R["CG4_pass"] = bool(R["CG4_drop_dB"] < 0.5)
    print(f"{ds} shift keep2: {R['shift_keep2']}  Q drop {R['CG4_drop_dB']:.2f} dB")

    # CG2 / CG3: flips on the UG3 set
    idx = np.random.default_rng(4).choice(len(Xte), 1000, replace=False)
    X, y = Xte[idx, 0], yte[idx]
    B = bands_of(X)
    flat = lambda Z: np.clip(Z, 0, 1).reshape(len(Z), -1)
    ok = judge.predict(flat(X)) == y
    fp3 = pm.complete(B, 1, "P3")[0]
    fq, uq, _ = q.complete(B, 1)
    flip_p3 = (judge.predict(flat(fp3)) != y) & ok
    flip_q = (judge.predict(flat(fq)) != y) & ok
    dis = ((np.clip(fq, 0, 1) - np.clip(fp3, 0, 1)) ** 2).reshape(len(X), -1).mean(1)
    R["flips_keep1"] = {"n_ok": int(ok.sum()), "P3": int(flip_p3.sum()), "Q": int(flip_q.sum())}
    R["CG2_pass"] = bool(flip_q.sum() <= 0.8 * flip_p3.sum())
    R["CG3"] = {"disagreement_auroc": float(roc_auc_score(flip_q[ok], dis[ok])),
                "Q_uncertainty_auroc": float(roc_auc_score(flip_q[ok], uq.reshape(len(X), -1).mean(1)[ok]))}
    R["CG3_pass"] = bool(R["CG3"]["disagreement_auroc"] >= 0.75)
    print(f"{ds} flips keep1: {R['flips_keep1']}  CG3 {R['CG3']}")
    print(f"{ds} CG1 {R['CG1_pass']} ({R['CG1_gain_keep1_dB']:+.2f} dB)  CG2 {R['CG2_pass']}  "
          f"CG3 {R['CG3_pass']}  CG4 {R['CG4_pass']}")
    json.dump(R, open(os.path.join(OUT, f"codes2_results_{ds}.json"), "w"), indent=1)


def run_bsds():
    train = bsds_images("train"); groups = by_shape(train)
    q = fit_q("bsds", lambda: (np.stack([train[k] for k in ks[i:i + 25]]) for ks in groups.values()
                               for i in range(0, len(ks), 25)), 1500)
    pm = pickle.load(open(os.path.join(CACHE, "patch_bsds.pkl"), "rb"))
    test = {**bsds_images("val"), **bsds_images("test")}
    ks = list(by_shape(test).values())[0][:60]
    X = np.stack([test[k] for k in ks]); B = bands_of(X)
    R = {"dataset": "bsds500", "fill_keep2": {"none": psnr(B[3] + B[4], X), "P0": psnr(pm.complete(B, 2, "P0")[0], X),
                                              "P3": psnr(pm.complete(B, 2, "P3")[0], X), "Q": psnr(q.complete(B, 2)[0], X)}}
    print("bsds fill keep2:", R["fill_keep2"])
    json.dump(R, open(os.path.join(OUT, "codes2_results_bsds.json"), "w"), indent=1)


def run_cifar():
    """Per colour channel: every channel of every training image is a grey training image."""
    Xtr, _ = load("cifar10", "train")
    Xte, _ = load("cifar10", "test")
    G = Xtr.reshape(-1, 32, 32)                       # 150 000 grey images
    batches = lambda: (G[i:i + 5000] for i in range(0, len(G), 5000))
    p = os.path.join(CACHE, "patch_cifar10.pkl")
    if os.path.exists(p):
        pm = pickle.load(open(p, "rb"))
    else:
        t = time.time(); pm = PatchMemory(per_image=4).fit(batches); print(f"P3 fit {time.time() - t:.0f}s")
        pickle.dump(pm, open(p, "wb"))
    q = fit_q("cifar10", batches, 4)
    idx = np.random.default_rng(4).choice(len(Xte), 2000, replace=False)   # same images as G4
    X = Xte[idx]
    Xg = X.reshape(-1, 32, 32)
    B = bands_of(Xg)
    R = {"dataset": "cifar10", "whole_image_linear": {"keep1": 17.56, "keep2": 19.29}}
    back = lambda Z: Z.reshape(X.shape)
    for keep in (1, 2):
        given = sum(B[j] for j in range(NB - keep, NB))
        R[f"fill_keep{keep}"] = {"none": psnr(back(given), X),
                                 "P0": psnr(back(pm.complete(B, keep, "P0")[0]), X),
                                 "P3": psnr(back(pm.complete(B, keep, "P3")[0]), X),
                                 "Q": psnr(back(q.complete(B, keep)[0]), X)}
        print(f"cifar10 fill keep{keep}: {R[f'fill_keep{keep}']}")
    R["CG5_pass"] = bool(R["fill_keep1"]["Q"] > 17.56 and R["fill_keep2"]["Q"] > 19.29)
    print("CG5", R["CG5_pass"])
    json.dump(R, open(os.path.join(OUT, "codes2_results_cifar10.json"), "w"), indent=1)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["mnist", "fashion", "bsds"]:
        {"bsds": run_bsds, "cifar10": run_cifar}.get(ds, lambda: run_small(ds))()
