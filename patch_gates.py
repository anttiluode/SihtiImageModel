"""Pre-registered gates for the patch residue memory (PREREG_PATCH.md).

    python patch_gates.py mnist      # PG1, PG2, PG4   (needs gates.py mnist run first: whole-image model + judge)
    python patch_gates.py fashion    # PG1, PG2, PG4, PG5
    python patch_gates.py bsds       # PG3 on real photos (BSDS500 images in data/bsds, see README)
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
from PIL import Image

from sihti_model.data import load
from sihti_model.patch import NB, PatchMemory, bands_of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ARMS = ["P0", "P2", "P3"]


def psnr_each(a, b):
    mse = ((np.clip(a, 0, 1) - b) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def shift(X, rng, m=4):
    out = np.zeros_like(X)
    for i in range(len(X)):
        while True:
            dx, dy = rng.integers(-m, m + 1, 2)
            if dx or dy:
                break
        src = X[i]
        H, W = src.shape
        ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
        xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
        out[i][yd, xd] = src[ys, xs]
    return out


def box3(M):
    p = np.pad(M, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(p[:, i:i + M.shape[1], j:j + M.shape[2]] for i in range(3) for j in range(3)) / 9


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label.ravel(), score.ravel()))


def fit_small(ds, X):
    path = os.path.join(OUT, "cache", f"patch_{ds}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    t = time.time()
    pm = PatchMemory(per_image=10).fit(lambda: (X[i:i + 5000] for i in range(0, len(X), 5000)))
    print(f"patch memory fit {time.time() - t:.0f}s")
    pickle.dump(pm, open(path, "wb"))
    return pm


def eval_set(name, Xc, yc, m, judge, pm, R, p1_n=300):
    flat = lambda X: X.reshape(len(X), -1)
    Bs = bands_of(Xc)
    codes = m.encode(Xc[:, None])
    res = {"judge_on_truth": float((judge.predict(flat(Xc)) == yc).mean())}
    for keep in (1, 2):
        outs = {"zero": sum(Bs[j] for j in range(NB - keep, NB)),
                "whole_linear": m.decode(m.complete(codes, keep, "gauss"))[:, 0],
                "whole_memory": m.decode(m.complete(codes, keep, "memory"))[:, 0]}
        for arm in ARMS:
            outs[arm] = pm.complete(Bs, keep, arm)[0]
        r = {k: {"psnr": float(psnr_each(o, Xc).mean()),
                 "judge_acc": float((judge.predict(flat(np.clip(o, 0, 1))) == yc).mean())}
             for k, o in outs.items()}
        sub = [b[:p1_n] for b in Bs]
        p1 = pm.complete(sub, keep, "P1")[0]
        r["P1_on_first_%d" % p1_n] = float(psnr_each(p1, Xc[:p1_n]).mean())
        r["P3_on_first_%d" % p1_n] = float(psnr_each(outs["P3"][:p1_n], Xc[:p1_n]).mean())
        r["judge_on_real_images"] = res["judge_on_truth"]
        R[f"{name}_keep{keep}"] = r
        print(f"{name:8s} keep={keep}: " + "  ".join(f"{k} {v['psnr']:.2f}/{v['judge_acc']:.2f}"
                                                for k, v in r.items() if isinstance(v, dict))
              + f"   [P1 {r['P1_on_first_%d' % p1_n]:.2f} vs P3 {r['P3_on_first_%d' % p1_n]:.2f} on {p1_n}]")
        if keep == 2:
            R.setdefault("_figs", {})[name] = {k: o[:8] for k, o in outs.items()}


def defects(Xc, rng):
    X = Xc.copy()
    lab = np.zeros(len(X), bool)
    mask = np.zeros(X.shape, bool)
    for i in range(len(X)):
        if rng.random() < 0.5:
            continue
        lab[i] = True
        ys, xs = np.nonzero(X[i] > 0.2)
        if len(ys) == 0:
            ys, xs = np.array([16]), np.array([16])
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
        ang, L = rng.uniform(0, np.pi), rng.integers(6, 15)
        d = 0.5 if rng.random() < 0.5 else -0.5
        before = X[i].copy()
        for t in np.arange(-L / 2, L / 2, 0.5):
            y, x = int(round(cy + t * np.sin(ang))), int(round(cx + t * np.cos(ang)))
            if 0 <= y < 32 and 0 <= x < 32:
                X[i, y, x] = np.clip(before[y, x] + d, 0, 1)
        mask[i] = np.abs(X[i] - before) > 0.05
    return X, lab, mask


def run_small(ds):
    R = {"dataset": ds}
    Xtr, ytr = load(ds, "train")
    Xte, yte = load(ds, "test")
    m = pickle.load(open(os.path.join(OUT, "cache", f"model_{ds}.pkl"), "rb"))
    judge = pickle.load(open(os.path.join(OUT, "cache", f"judge_{ds}.pkl"), "rb"))
    pm = fit_small(ds, Xtr[:, 0])
    R["PG4_size"] = pm.size()
    R["PG4_whole_image"] = {"fixed": 1.43e6, "memory_bank": 10 * 2000 * sum(m.dims)}
    print("sizes", R["PG4_size"])
    idx = np.random.default_rng(4).choice(len(Xte), 2000, replace=False)
    Xc, yc = Xte[idx, 0], yte[idx]
    Xs = shift(Xc, np.random.default_rng(11))
    eval_set("standard", Xc, yc, m, judge, pm, R)
    eval_set("shifted", Xs, yc, m, judge, pm, R)
    s2, h2 = R["standard_keep2"], R["shifted_keep2"]
    R["PG1_pass"] = s2["P3"]["psnr"] >= s2["whole_linear"]["psnr"]
    R["PG2a_drop_dB"] = s2["P3"]["psnr"] - h2["P3"]["psnr"]
    R["PG2a_pass"] = R["PG2a_drop_dB"] < 0.5
    R["PG2b_margin_dB"] = h2["P3"]["psnr"] - h2["whole_memory"]["psnr"]
    R["PG2b_pass"] = R["PG2b_margin_dB"] >= 1.0
    R["PG4_pass"] = R["PG4_size"]["P3"] <= 1e6
    print(f"PG1 {R['PG1_pass']}  PG2a drop {R['PG2a_drop_dB']:.2f} dB {R['PG2a_pass']}  "
          f"PG2b margin {R['PG2b_margin_dB']:.2f} dB {R['PG2b_pass']}  PG4 {R['PG4_pass']}")

    if ds == "fashion":
        rng = np.random.default_rng(9)
        Xd, lab, mask = defects(Xte[rng.choice(len(Xte), 1000, replace=False), 0], rng)
        Bs = bands_of(Xd)
        fine = Bs[0] + Bs[1]
        _, pb = pm.complete(Bs, 3, "P3")
        codes = m.encode(Xd[:, None])
        wb = m.complete(codes, 3, "gauss")
        wfine = m.decode_band(wb, 0)[:, 0] + m.decode_band(wb, 1)[:, 0]
        maps = {"highpass": box3(np.abs(fine)),
                "whole_linear": box3(np.abs(fine - wfine)),
                "P3": box3(np.abs(fine - pb[0] - pb[1]))}
        g5 = {}
        for k, M in maps.items():
            g5[k] = {"pixel_auroc": auroc(M[lab], mask[lab]),
                     "image_auroc": auroc(M.reshape(len(M), -1).max(1), lab)}
        g5["PG5_pass"] = g5["P3"]["pixel_auroc"] - g5["highpass"]["pixel_auroc"] >= 0.05
        R["PG5"] = g5
        print("PG5", json.dumps(g5))
        R["_defect_fig"] = (Xd[lab][:6], mask[lab][:6], {k: v[lab][:6] for k, v in maps.items()})

    figs = R.pop("_figs")
    dfig = R.pop("_defect_fig", None)
    json.dump(R, open(os.path.join(OUT, f"patch_results_{ds}.json"), "w"), indent=1)
    pickle.dump({"figs": figs, "truth_std": Xc[:8], "truth_shift": Xs[:8], "defect": dfig},
                open(os.path.join(OUT, "cache", f"patch_figs_{ds}.pkl"), "wb"))


# ---------------- BSDS500 ---------------------------------------------------------------
BSDS = os.path.join(HERE, "data", "bsds", "BSDS500", "data", "images")


def bsds_images(split, colour=False):
    out = {}
    for p in sorted(glob.glob(os.path.join(BSDS, split, "*.jpg"))):
        im = Image.open(p).convert("RGB")
        im = im.resize((im.width // 2, im.height // 2), Image.BICUBIC)
        a = np.asarray(im, np.float32) / 255
        out[os.path.basename(p)[:-4]] = a if colour else a @ np.array([0.299, 0.587, 0.114], np.float32)
    return out


def by_shape(d):
    g = {}
    for k, v in d.items():
        g.setdefault(v.shape[:2], []).append(k)
    return g


def run_bsds():
    train = bsds_images("train")
    test = {**bsds_images("val"), **bsds_images("test")}
    path = os.path.join(OUT, "cache", "patch_bsds.pkl")
    if os.path.exists(path):
        pm = pickle.load(open(path, "rb"))
    else:
        t = time.time()
        groups = by_shape(train)
        pm = PatchMemory(per_image=1500).fit(
            lambda: (np.stack([train[k] for k in ks[i:i + 25]]) for ks in groups.values()
                     for i in range(0, len(ks), 25)))
        print(f"BSDS patch memory fit {time.time() - t:.0f}s")
        pickle.dump(pm, open(path, "wb"))
    R = {"dataset": "bsds500", "n_train": len(train), "n_test": len(test), "size": pm.size()}
    groups = by_shape(test)
    for keep in (1, 2):
        scores = {k: [] for k in ["zero"] + ARMS}
        for shape, ks in groups.items():
            for i in range(0, len(ks), 20):
                X = np.stack([test[k] for k in ks[i:i + 20]])
                Bs = bands_of(X)
                scores["zero"] += list(psnr_each(sum(Bs[j] for j in range(NB - keep, NB)), X))
                for arm in ARMS:
                    scores[arm] += list(psnr_each(pm.complete(Bs, keep, arm)[0], X))
        r = {k: float(np.mean(v)) for k, v in scores.items()}
        r["P3_beats_P0_on_fraction_of_images"] = float(np.mean(np.array(scores["P3"]) > np.array(scores["P0"])))
        # P1 on 30 images
        ks = list(groups.values())[0][:30]
        X = np.stack([test[k] for k in ks])
        Bs = bands_of(X)
        r["P1_on_30"] = float(psnr_each(pm.complete(Bs, keep, "P1")[0], X).mean())
        r["P3_on_30"] = float(psnr_each(pm.complete(Bs, keep, "P3")[0], X).mean())
        R[f"keep{keep}"] = r
        print(f"BSDS keep={keep}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
    R["PG3_pass"] = R["keep2"]["P3"] > R["keep2"]["P0"]
    print("PG3 pass:", R["PG3_pass"])
    json.dump(R, open(os.path.join(OUT, "patch_results_bsds.json"), "w"), indent=1)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    run_bsds() if ds == "bsds" else run_small(ds)
