"""Gates UG1-UG3 from PREREG_UNCERTAINTY.md.   python uncertainty_gates.py [mnist fashion bsds]

Needs the cached patch memories and judges from gates.py / patch_gates.py.
"""
import json
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from patch_gates import bsds_images, by_shape
from sihti_model.data import CLASSES, load
from sihti_model.patch import NB, bands_of
from sihti_model.uncertainty import ause, code_variances, edge_energy, fill_with_uncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def pixel_gates(name, U, e2, scale, R):
    r = {}
    for k, u in U.items():
        a, cu, co = ause(u, e2)
        r[k] = {"ause": a}
        if k in scale:
            sd = np.sqrt(scale[k] * u)
            err = np.sqrt(e2)
            r[k]["cover_1sd"] = float((err <= sd).mean())
            r[k]["cover_2sd"] = float((err <= 2 * sd).mean())
    r["UG1_pass"] = r["U1"]["ause"] <= 0.9 * r["U3"]["ause"]
    r["UG2_pass"] = 0.90 <= r["U1"]["cover_2sd"] <= 0.98
    R[name] = r
    print(f"  {name}: " + "  ".join(f"{k} AUSE {v['ause']:.3f}" + (f" cover1 {v['cover_1sd']:.2f} cover2 {v['cover_2sd']:.2f}" if "cover_2sd" in v else "")
                                for k, v in r.items() if isinstance(v, dict)) + f"   UG1 {r['UG1_pass']} UG2 {r['UG2_pass']}")


def show(ax, im, cmap="gray", vmax=None):
    ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax if vmax is not None else 1)
    ax.set_xticks([]); ax.set_yticks([])


def run_small(ds):
    R = {"dataset": ds}
    pm = pickle.load(open(os.path.join(OUT, "cache", f"patch_{ds}.pkl"), "rb"))
    judge = pickle.load(open(os.path.join(OUT, "cache", f"judge_{ds}.pkl"), "rb"))
    Xtr, ytr = load(ds, "train"); Xtr = Xtr[:, 0]
    Xte, yte = load(ds, "test")
    t = time.time()
    var = code_variances(pm, lambda: (Xtr[i:i + 5000] for i in range(0, 20000, 5000)))
    print(f"{ds}: code variances {time.time() - t:.0f}s")
    # calibration on 1000 training images
    Xcal = Xtr[-1000:]
    Bc = bands_of(Xcal)
    fc, U1c, U2c = fill_with_uncertainty(pm, Bc, 2, var, neighbours=True)
    e2c = (np.clip(fc, 0, 1) - Xcal) ** 2
    scale = {"U1": float(e2c.mean() / U1c.mean()), "U2": float(e2c.mean() / U2c.mean())}
    R["scale"] = scale
    idx = np.random.default_rng(4).choice(len(Xte), 1000, replace=False)
    X, y = Xte[idx, 0], yte[idx]
    B = bands_of(X)
    real_ok = judge.predict(X.reshape(len(X), -1)) == y
    figs = {}
    for keep in (2, 1):
        f, U1, U2 = fill_with_uncertainty(pm, B, keep, var, neighbours=(keep == 2))
        f = np.clip(f, 0, 1)
        e2 = (f - X) ** 2
        given = sum(B[j] for j in range(NB - keep, NB))
        U3 = edge_energy(given)
        U = {"U1": U1, "U3": U3}
        if U2 is not None:
            U["U2"] = U2
        pixel_gates(f"keep{keep}", U, e2, scale, R)
        flip = judge.predict(f.reshape(len(f), -1)) != y
        m = real_ok
        fr = {"n_flips": int((flip & m).sum()), "n_images": int(m.sum())}
        for k, u in U.items():
            fr[k] = float(roc_auc_score(flip[m], u.reshape(len(u), -1).mean(1)[m])) if 0 < (flip & m).sum() < m.sum() else None
        R[f"flips_keep{keep}"] = fr
        print(f"  flips keep{keep}: {fr}")
        figs[keep] = (f, U1, flip & m, e2)
    R["UG3_pass"] = (R["flips_keep1"]["U1"] or 0) >= 0.75
    print("  UG3", R["UG3_pass"])
    json.dump(R, open(os.path.join(OUT, f"uncertainty_results_{ds}.json"), "w"), indent=1)

    # figure: keep 1, three flipped and three correct images
    f, U1, flip, e2 = figs[1]
    sel = list(np.flatnonzero(flip)[:3]) + list(np.flatnonzero(~flip & real_ok)[:3])
    cols = ["truth", "given: core", "patch fill", "actual error", "predicted uncertainty"]
    fig, axs = plt.subplots(len(sel), 5, figsize=(7.5, 1.6 * len(sel)))
    Bg = B[4]
    for r, i in enumerate(sel):
        show(axs[r, 0], X[i]); show(axs[r, 1], np.clip(Bg[i], 0, 1)); show(axs[r, 2], f[i])
        show(axs[r, 3], np.sqrt(e2[i]), "magma", 0.8)
        show(axs[r, 4], np.sqrt(scale["U1"] * U1[i]), "magma", 0.8)
        lab = CLASSES[ds][judge.predict(f[i].reshape(1, -1))[0]]
        axs[r, 2].set_xlabel(("FLIPPED to " if flip[i] else "read as ") + lab, fontsize=7)
    for j, c in enumerate(cols):
        axs[0, j].set_title(c, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_uncertainty_{ds}.png"), dpi=110); plt.close(fig)


def run_bsds():
    R = {"dataset": "bsds500"}
    pm = pickle.load(open(os.path.join(OUT, "cache", "patch_bsds.pkl"), "rb"))
    train = bsds_images("train"); groups = by_shape(train)
    t = time.time()
    var = code_variances(pm, lambda: (np.stack([train[k] for k in ks[i:i + 25]]) for ks in groups.values()
                                      for i in range(0, len(ks), 25)))
    print(f"bsds: code variances {time.time() - t:.0f}s")
    cal_keys = list(groups.values())[0][:30]
    Xcal = np.stack([train[k] for k in cal_keys])
    fc, U1c, _ = fill_with_uncertainty(pm, bands_of(Xcal), 2, var)
    scale = {"U1": float(((np.clip(fc, 0, 1) - Xcal) ** 2).mean() / U1c.mean())}
    R["scale"] = scale
    test = {**bsds_images("val"), **bsds_images("test")}
    ks = list(by_shape(test).values())[0][:60]
    X = np.stack([test[k] for k in ks])
    B = bands_of(X)
    f, U1, _ = fill_with_uncertainty(pm, B, 2, var)
    f = np.clip(f, 0, 1)
    e2 = (f - X) ** 2
    pixel_gates("keep2", {"U1": U1, "U3": edge_energy(B[3] + B[4])}, e2, scale, R)
    json.dump(R, open(os.path.join(OUT, "uncertainty_results_bsds.json"), "w"), indent=1)
    # figure on the hawk if present, else first image
    col = bsds_images("val", True)
    n = "42049"
    Xc = np.transpose(col[n], (2, 0, 1)); Bc = bands_of(Xc)
    fcol, Ucol, _ = fill_with_uncertainty(pm, Bc, 2, var)
    fig, axs = plt.subplots(1, 4, figsize=(14, 2.8))
    axs[0].imshow(col[n]); axs[1].imshow(np.clip(np.transpose(fcol, (1, 2, 0)), 0, 1))
    err = np.sqrt(((np.clip(fcol, 0, 1) - Xc) ** 2).mean(0)); unc = np.sqrt(scale["U1"] * Ucol.mean(0))
    v = np.percentile(err, 99.5)
    axs[2].imshow(err, cmap="magma", vmin=0, vmax=v); axs[3].imshow(unc, cmap="magma", vmin=0, vmax=v)
    for a, t in zip(axs, ["photo (held out)", "patch fill from core + r3", "actual error", "predicted uncertainty"]):
        a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_uncertainty_bsds.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["mnist", "fashion", "bsds"]:
        run_bsds() if ds == "bsds" else run_small(ds)
