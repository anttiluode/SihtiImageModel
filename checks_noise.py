"""Fair version: fit the memories on noisy coarse input (sigma=0.01), test on noisy coarse input."""
import sys, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
import sihti_model.patch as P
from sihti_model.data import load
from patch_gates import bsds_images, by_shape
psnr_e = lambda a, b: 10*np.log10(1/((np.clip(a,0,1)-b)**2).reshape(len(a),-1).mean(1))
SIG = 0.01
clean_bands = P.bands_of
rng = np.random.default_rng(1)
def noisy_bands(X):
    Bs = clean_bands(X)
    Bs[4] = Bs[4] + rng.normal(0, SIG, Bs[4].shape).astype(np.float32)   # noise on the given part only
    return Bs

def go(label, train_fn, Xt):
    P.bands_of = noisy_bands
    pm = P.PatchMemory(per_image=train_fn[1]).fit(train_fn[0], verbose=False)
    P.bands_of = clean_bands
    Bs = clean_bands(Xt)
    g = Bs[3] + Bs[4] + np.random.default_rng(2).normal(0, SIG, Bs[4].shape).astype(np.float32)
    B = [None, None, None, np.zeros_like(g), g]
    row = {"none": psnr_e(g, Xt).mean()}
    for arm in ("P0", "P2", "P3"):
        row[arm] = psnr_e(pm.complete(B, 2, arm)[0], Xt).mean()
    print(f"{label} trained+tested at noise {SIG}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))

Xtr, _ = load('mnist', 'train'); Xtr = Xtr[:, 0]
Xte, _ = load('mnist', 'test'); Xt = Xte[np.random.default_rng(4).choice(10000, 2000, replace=False), 0]
go("digits", (lambda: (Xtr[i:i+5000] for i in range(0, 60000, 5000)), 10), Xt)
train = bsds_images("train"); groups = by_shape(train)
test = {**bsds_images("val"), **bsds_images("test")}
ks = list(by_shape(test).values())[0][:60]
go("photos", (lambda: (np.stack([train[k] for k in ks2[i:i+25]]) for ks2 in groups.values() for i in range(0, len(ks2), 25)), 1500),
   np.stack([test[k] for k in ks]))
