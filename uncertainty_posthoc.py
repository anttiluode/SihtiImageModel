"""POST-HOC (not pre-registered) ablations: is U1 more than 'window contrast' or 'where the fill draws'?"""
import sys, pickle, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
from sihti_model.data import load
from sihti_model.patch import bands_of, NB
from sihti_model.uncertainty import code_variances, fill_with_uncertainty, edge_energy, ause
from patch_gates import bsds_images, by_shape

def report(label, pm, var, X, keep):
    B = bands_of(X)
    f, U1, _ = fill_with_uncertainty(pm, B, keep, var)
    flat = {b: np.tile((v * 1).mean(0, keepdims=True), (len(v), 1)) for b, v in var.items()}   # contrast only
    _, U5, _ = fill_with_uncertainty(pm, B, keep, flat)
    e2 = (np.clip(f, 0, 1) - X) ** 2
    U4 = edge_energy(np.clip(f, 0, 1))                     # where the fill itself draws detail
    print(f"{label} keep{keep}: U1 {ause(U1,e2)[0]:.3f}  contrast-only {ause(U5,e2)[0]:.3f}  fill-edges {ause(U4,e2)[0]:.3f}")

for ds in ["mnist", "fashion"]:
    pm = pickle.load(open(f"results/cache/patch_{ds}.pkl", "rb"))
    Xtr, _ = load(ds, "train"); Xtr = Xtr[:, 0]
    var = code_variances(pm, lambda: (Xtr[i:i+5000] for i in range(0, 20000, 5000)))
    Xte, _ = load(ds, "test"); X = Xte[np.random.default_rng(4).choice(len(Xte), 1000, replace=False), 0]
    for keep in (2, 1): report(ds, pm, var, X, keep)
pm = pickle.load(open("results/cache/patch_bsds.pkl", "rb"))
train = bsds_images("train"); g = by_shape(train)
var = code_variances(pm, lambda: (np.stack([train[k] for k in ks[i:i+25]]) for ks in g.values() for i in range(0, len(ks), 25)))
test = {**bsds_images("val"), **bsds_images("test")}; ks = list(by_shape(test).values())[0][:60]
report("photos", pm, var, np.stack([test[k] for k in ks]), 2)
