"""Export a fitted model (results/cache/model_<ds>.pkl) to web/model_<ds>.json for the browser page.

Matrices are stored as base64 int16 (int8 for the memory bank) with one scale per array.
"""
import base64
import json
import os
import pickle
import sys

import numpy as np

from sihti_model.data import CLASSES, load

HERE = os.path.dirname(os.path.abspath(__file__))


def q16(a):
    a = np.asarray(a, np.float64)
    s = float(np.abs(a).max()) / 32767 or 1.0
    return {"shape": list(a.shape), "scale": s,
            "b64": base64.b64encode(np.round(a / s).astype("<i2").tobytes()).decode()}


def q8(a):
    a = np.asarray(a, np.float64)
    s = float(np.abs(a).max()) / 127 or 1.0
    return {"shape": list(a.shape), "scale": s,
            "b64": base64.b64encode(np.round(a / s).astype(np.int8).tobytes()).decode()}


def main(ds, bank_n=200, test_n=12):
    m = pickle.load(open(os.path.join(HERE, "results", "cache", f"model_{ds}.pkl"), "rb"))
    out = {"dataset": ds, "classes": CLASSES[ds], "dims": list(m.dims), "depths": list(m.depths),
           "shape": list(m.shape), "K": m.K,
           "mean": [q16(x) for x in m.mean], "U": [q16(x) for x in m.U],
           "gen": [], "bank": []}
    for c in range(m.n_classes):
        g = m.gen[c]
        w, mu, L = g["core"]
        e = {"w": [float(x) for x in w], "mu": q16(mu), "L": q16(L), "A": {}, "Lb": {}}
        for b in range(m.nb - 1):
            e["A"][b] = q16(g[b][0])
            e["Lb"][b] = q16(g[b][1])
        out["gen"].append(e)
        out["bank"].append([q8(z[:bank_n]) for z in m.bank[c]])
    g = m.gen_all
    out["gen_all"] = {"A": {b: q16(g[b][0]) for b in range(m.nb - 1)}}
    X, y = load(ds, "test")
    rng = np.random.default_rng(5)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), test_n, replace=False)
                          for c in range(m.n_classes)])
    out["test"] = {"labels": [int(v) for v in y[idx]],
                   "b64": base64.b64encode((X[idx] * 255).round().astype(np.uint8).tobytes()).decode()}
    os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
    p = os.path.join(HERE, "web", f"model_{ds}.json")
    json.dump(out, open(p, "w"))
    print(p, f"{os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["mnist", "fashion"]:
        main(ds)
