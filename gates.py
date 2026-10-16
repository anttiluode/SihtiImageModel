"""Run the pre-registered gates (see PREREG.md).

    python gates.py mnist
    python gates.py fashion
    python gates.py cifar10      # needs internet once; not run in the sandbox

Writes results/results_<ds>.json and figures in results/.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

from sihti_model.data import CLASSES, load
from sihti_model.model import SihtiImageModel, ridge_fit
from sihti_model.sieve import sieve, unsieve
from sihti_model import figures

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAMS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def onehot(y, n):
    Y = -np.ones((len(y), n))
    Y[np.arange(len(y)), y] = 1
    return Y


def add1(F):
    return np.concatenate([F, np.ones((len(F), 1), F.dtype)], 1)


def ridge_readout(Ftr, ytr, Fte, yte, n, rng):
    """lambda picked on a 10% split of the training set, then refit on all of it."""
    Ftr, Fte = add1(Ftr.astype(np.float64)), add1(Fte.astype(np.float64))
    idx = rng.permutation(len(Ftr))
    v, t = idx[: len(idx) // 10], idx[len(idx) // 10:]
    best = max(LAMS, key=lambda l: (np.argmax(Ftr[v] @ ridge_fit(Ftr[t], onehot(ytr[t], n), l), 1)
                                    == ytr[v]).mean())
    W = ridge_fit(Ftr, onehot(ytr, n), best)
    return float((np.argmax(Fte @ W, 1) == yte).mean()), best, W


def loo_ridge_fewshot(Ftr, ytr, Fte, yte, n):
    """Dual ridge, lambda by exact leave-one-out on the few-shot set itself."""
    Ftr, Fte = add1(Ftr.astype(np.float64)), add1(Fte.astype(np.float64))
    K = Ftr @ Ftr.T
    Y = onehot(ytr, n)
    best, bacc = None, -1
    for l in LAMS:
        lam = l * np.trace(K) / len(K)
        H = K @ np.linalg.inv(K + lam * np.eye(len(K)))
        loo = (Y - H @ Y) / (1 - np.diag(H))[:, None]
        acc = (np.argmax(Y - loo, 1) == ytr).mean()
        if acc > bacc:
            best, bacc = lam, acc
    alpha = np.linalg.solve(K + best * np.eye(len(K)), Y)
    return float((np.argmax(Fte @ Ftr.T @ alpha, 1) == yte).mean())


class RandomNet:
    """Fixed random ReLU layer.  Nothing in it is ever trained."""

    def __init__(self, d, h=2000, seed=1):
        r = np.random.default_rng(seed)
        self.W = (r.standard_normal((d, h)) / np.sqrt(d)).astype(np.float32)
        self.b = r.uniform(-1, 1, h).astype(np.float32)

    def fit_scale(self, F):
        self.mu, self.sd = F.mean(0), F.std() + 1e-8
        return self

    def __call__(self, F):
        return np.maximum(((F - self.mu) / self.sd) @ self.W + self.b, 0)


def nn_dist(A, B):
    b2 = (B ** 2).sum(1)
    out = np.empty(len(A))
    for s in range(0, len(A), 500):
        q = A[s:s + 500]
        d2 = (q ** 2).sum(1)[:, None] + b2[None] - 2 * q @ B.T
        out[s:s + 500] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def psnr(a, b):
    mse = ((a - b) ** 2).reshape(len(a), -1).mean(1)
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))


def run_g4(m, judge, Xte, yte, Zte, Ztr, ytr, n):
    """Completion: keep the coarse bands of a test image, fill the finer residues back in."""
    flat = lambda X: X.reshape(len(X), -1)
    rng = np.random.default_rng(4)
    g4 = {}
    idx = rng.choice(len(Xte), 2000, replace=False)
    Xc, yc = Xte[idx], yte[idx]
    Zc = [z[idx] for z in Zte]
    comp_figs = {}
    for keep in (1, 2):
        bands = list(range(m.nb - keep, m.nb))
        acc, _, W = ridge_readout(m.features(Ztr, bands), ytr, m.features(Zc, bands), yc, n, rng)
        cls = np.argmax(add1(m.features(Zc, bands).astype(np.float64)) @ W, 1)
        res = {"coarse_readout_acc": float((cls == yc).mean())}
        outs = {}
        # baseline: ONLY the given coarse bands (the blurred image itself)
        outs["zero"] = sum(m.decode_band(Zc, b) for b in bands)
        outs["gauss_agnostic"] = m.decode(m.complete(Zc, keep, "gauss"))
        gc = np.zeros_like(outs["zero"])
        mm = np.zeros_like(outs["zero"])
        for c in range(n):
            s = cls == c
            if s.any():
                sub = [z[s] for z in Zc]
                gc[s] = m.decode(m.complete(sub, keep, "gauss", c=c))
                mm[s] = m.decode(m.complete(sub, keep, "memory", c=c))
        outs["gauss_classread"] = gc
        outs["memory_classread"] = mm
        outs["memory_agnostic"] = m.decode(m.complete(Zc, keep, "memory"))
        for k, o in outs.items():
            o = np.clip(o, 0, 1)
            res[k] = {"psnr": psnr(o, Xc), "judge_acc": float((judge.predict(flat(o)) == yc).mean())}
        res["vocab_ceiling"] = {"psnr": psnr(np.clip(m.decode(Zc), 0, 1), Xc)}
        g4[f"keep{keep}"] = res
        comp_figs[keep] = outs
        print(f"G4 keep={keep} (class read from coarse {res['coarse_readout_acc']:.3f}): "
              + "  ".join(f"{k} {res[k]['psnr']:.2f}dB/{res[k]['judge_acc']:.3f}" for k in outs))
    g4["G4a_pass"] = g4["keep1"]["gauss_agnostic"]["psnr"] - g4["keep1"]["zero"]["psnr"] >= 1.0
    print("G4a pass:", g4["G4a_pass"])
    return g4, comp_figs, Xc


def rerun_g4(ds):
    """Re-run only G4 from the cached model and judge (used after the baseline fix)."""
    m = pickle.load(open(os.path.join(OUT, "cache", f"model_{ds}.pkl"), "rb"))
    judge = pickle.load(open(os.path.join(OUT, "cache", f"judge_{ds}.pkl"), "rb"))
    Xtr, ytr = load(ds, "train")
    Xte, yte = load(ds, "test")
    n = len(CLASSES[ds])
    Ztr, Zte = m.encode(Xtr), m.encode(Xte)
    g4, comp_figs, Xc = run_g4(m, judge, Xte, yte, Zte, Ztr, ytr, n)
    p = os.path.join(OUT, f"results_{ds}.json")
    R = json.load(open(p))
    R["G4"] = g4
    json.dump(R, open(p, "w"), indent=1)
    figures.completion_figure(Xc[:8], comp_figs[1], os.path.join(OUT, f"fig_completion_{ds}.png"))


def main(ds):
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    rng = np.random.default_rng(0)
    R = {"dataset": ds}
    names = CLASSES[ds]
    n = len(names)
    Xtr, ytr = load(ds, "train")
    Xte, yte = load(ds, "test")
    flat = lambda X: X.reshape(len(X), -1)

    # ---- G0 -------------------------------------------------------------------
    t = time.time()
    err = max(float(np.abs(unsieve(sieve(Xte[s:s + 2000])) - Xte[s:s + 2000]).max())
              for s in range(0, len(Xte), 2000))
    R["G0"] = {"max_abs_err": err, "pass": err < 1e-12}
    print(f"G0 lossless: max err {err:.2e}  {'PASS' if err < 1e-12 else 'FAIL'}  ({time.time()-t:.0f}s)")

    # ---- fit the model ----------------------------------------------------------
    t = time.time()
    m = SihtiImageModel()
    Ztr = m.fit(Xtr, ytr, n)
    Zte = m.encode(Xte)
    rec = m.decode(Zte)
    R["vocab"] = {"dims": list(m.dims), "recon_psnr_test": psnr(rec, Xte)}
    print(f"fit {time.time()-t:.0f}s; vocabulary reconstruction PSNR {R['vocab']['recon_psnr_test']:.2f} dB")

    # ---- G1: where does the class live? ---------------------------------------
    nb = m.nb
    g1 = {"single": {}, "cumulative": {}}
    for b in range(nb):
        acc, _, _ = ridge_readout(m.features(Ztr, [b]), ytr, m.features(Zte, [b]), yte, n, rng)
        g1["single"][str(b)] = acc
    for k in range(1, nb + 1):
        bands = list(range(nb - k, nb))
        acc, lam, W = ridge_readout(m.features(Ztr, bands), ytr, m.features(Zte, bands), yte, n, rng)
        g1["cumulative"][str(k)] = acc
    R["G1"] = g1
    print("G1 single bands (r0..core):", [round(g1['single'][str(b)], 4) for b in range(nb)])
    print("G1 cumulative coarse->fine:", [round(g1['cumulative'][str(k)], 4) for k in range(1, nb + 1)])

    # ---- G2: readouts ------------------------------------------------------------
    g2 = {}
    Ptr, Pte = flat(Xtr), flat(Xte)
    Str, Ste = m.features(Ztr), m.features(Zte)
    g2["A_raw_ridge"] = ridge_readout(Ptr, ytr, Pte, yte, n, rng)[0]
    rnP = RandomNet(Ptr.shape[1]).fit_scale(Ptr)
    g2["B_randomnet_raw_ridge"] = ridge_readout(rnP(Ptr), ytr, rnP(Pte), yte, n, rng)[0]
    g2["C_sihti_ridge"] = ridge_readout(Str, ytr, Ste, yte, n, rng)[0]
    rnS = RandomNet(Str.shape[1]).fit_scale(Str)
    g2["D_randomnet_sihti_ridge"] = ridge_readout(rnS(Str), ytr, rnS(Ste), yte, n, rng)[0]
    jpath = os.path.join(OUT, "cache", f"judge_{ds}.pkl")
    if os.path.exists(jpath):
        judge = pickle.load(open(jpath, "rb"))
    else:
        from sklearn.neural_network import MLPClassifier
        t = time.time()
        judge = MLPClassifier(hidden_layer_sizes=(512,), max_iter=20, random_state=0)
        judge.fit(Ptr, ytr)
        pickle.dump(judge, open(jpath, "wb"))
        print(f"  backprop judge trained in {time.time()-t:.0f}s")
    g2["E_backprop_mlp"] = float(judge.score(Pte, yte))
    few = {"raw": [], "sihti": []}
    for d in range(5):
        r = np.random.default_rng(100 + d)
        idx = np.concatenate([r.choice(np.flatnonzero(ytr == c), 10, replace=False) for c in range(n)])
        few["raw"].append(loo_ridge_fewshot(Ptr[idx], ytr[idx], Pte, yte, n))
        few["sihti"].append(loo_ridge_fewshot(Str[idx], ytr[idx], Ste, yte, n))
    g2["fewshot10_raw"] = float(np.mean(few["raw"]))
    g2["fewshot10_sihti"] = float(np.mean(few["sihti"]))
    g2["fewshot10_draws"] = few
    g2["G2a_pass"] = abs(g2["C_sihti_ridge"] - g2["A_raw_ridge"]) <= 0.01
    g2["G2b_pass"] = g2["D_randomnet_sihti_ridge"] >= g2["B_randomnet_raw_ridge"]
    g2["G2c_pass"] = g2["E_backprop_mlp"] > g2["D_randomnet_sihti_ridge"]
    g2["G2d_pass"] = g2["fewshot10_sihti"] - g2["fewshot10_raw"] >= 0.02
    R["G2"] = g2
    for k in ["A_raw_ridge", "B_randomnet_raw_ridge", "C_sihti_ridge", "D_randomnet_sihti_ridge",
              "E_backprop_mlp", "fewshot10_raw", "fewshot10_sihti"]:
        print(f"G2 {k:26s} {g2[k]:.4f}")
    print("G2 a/b/c/d pass:", g2["G2a_pass"], g2["G2b_pass"], g2["G2c_pass"], g2["G2d_pass"])

    # ---- G3: generation ----------------------------------------------------------
    g3 = {}
    te_nn = np.concatenate([nn_dist(flat(Xte[yte == c][:300]), Ptr[ytr == c]) for c in range(n)])
    te_med = float(np.median(te_nn))
    g3["test_nn_median"] = te_med
    samples = {}
    for mode in ["gauss", "memory"]:
        r = np.random.default_rng(7)
        imgs, labs = [], []
        for c in range(n):
            imgs.append(m.decode(m.generate(c, 100, mode=mode, rng=r)))
            labs.append(np.full(100, c))
        S, L = np.clip(np.concatenate(imgs), 0, 1), np.concatenate(labs)
        samples[mode] = S
        rec_ = float((judge.predict(flat(S)) == L).mean())
        d = np.concatenate([nn_dist(flat(S[L == c]), Ptr[ytr == c]) for c in range(n)])
        ratio = float(np.median(d) / te_med)
        copy = float((d < 0.1 * te_med).mean())
        g3[mode] = {"recognisable": rec_, "novelty_ratio": ratio, "copy_rate": copy,
                    "G3a_pass": rec_ >= 0.8, "G3b_pass": ratio >= 0.5}
        print(f"G3 {mode:6s} recognisable {rec_:.3f}  novelty ratio {ratio:.2f}  copy rate {copy:.3f}")
    R["G3"] = g3

    # ---- G4: completion ----------------------------------------------------------
    g4, comp_figs, Xc = run_g4(m, judge, Xte, yte, Zte, Ztr, ytr, n)
    R["G4"] = g4

    json.dump(R, open(os.path.join(OUT, f"results_{ds}.json"), "w"), indent=1)
    pickle.dump(m, open(os.path.join(OUT, "cache", f"model_{ds}.pkl"), "wb"))

    # ---- figures -------------------------------------------------------------------
    figures.sieve_figure(Xte[:6], os.path.join(OUT, f"fig_sieve_{ds}.png"))
    figures.samples_figure(samples, Xtr, ytr, names, os.path.join(OUT, f"fig_samples_{ds}.png"))
    figures.stages_figure(m, names, os.path.join(OUT, f"fig_stages_{ds}.png"))
    figures.completion_figure(Xc[:8], comp_figs[1], os.path.join(OUT, f"fig_completion_{ds}.png"))
    figures.where_class_lives(R, os.path.join(OUT, f"fig_where_class_lives_{ds}.png"))
    return R


if __name__ == "__main__":
    if "--g4" in sys.argv:
        rerun_g4(sys.argv[1])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "mnist")
