"""Does the coarse state leak fine detail through the undamped checkerboard mode?"""
import sys, time, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
import sihti_model.sieve as S
import sihti_model.patch as P
from sihti_model.data import load

lam = lambda wx, wy: 4 - 2*np.cos(wx) - 2*np.cos(wy)
for dt, d16, d64 in [(0.25, 16, 64), (0.2, 20, 80)]:
    print(f"dt={dt}: gain at checkerboard after {d16} steps = {abs(1-dt*8)**d16:.3g}, after {d64} = {abs(1-dt*8)**d64:.3g}")

Xtr, _ = load('mnist', 'train'); Xtr = Xtr[:, 0]
Xte, _ = load('mnist', 'test')
idx = np.random.default_rng(4).choice(len(Xte), 2000, replace=False); Xc = Xte[idx, 0]

def energy_split(X, dt, depth):
    S.purify.__defaults__ = (dt,)
    st = X[:, None].astype(np.float64)
    for _ in range(depth): st = S.purify(st)
    F = np.fft.fft2(st[:, 0]); w = np.fft.fftfreq(32) * 2 * np.pi
    L = lam(w[:, None], w[None, :])
    corner = L > 6.0                      # near-Nyquist region, invisible to a real blur
    E = np.abs(F) ** 2
    return E[:, corner].sum() / E.sum()

psnr = lambda a, b: float(np.mean(10*np.log10(1/((np.clip(a,0,1)-b)**2).mean((1,2)))))
for dt, depths in [(0.25, (0, 1, 4, 16, 64)), (0.2, (0, 1, 5, 20, 80))]:
    S.purify.__defaults__ = (dt,); P.DEPTHS = depths
    print(f"\n== sieve dt={dt} depths={depths}")
    print(f"  share of energy of (core+r3) in the near-Nyquist corner: {energy_split(Xc, dt, depths[3]):.2e}")
    t = time.time()
    pm = P.PatchMemory(per_image=10).fit(lambda: (Xtr[i:i+5000] for i in range(0, 60000, 5000)), verbose=False)
    print(f"  fit {time.time()-t:.0f}s")
    Bs = P.bands_of(Xc)
    for keep in (1, 2):
        given = sum(Bs[j] for j in range(5-keep, 5))
        row = {"none": psnr(given, Xc)}
        for arm in ("P0", "P3"):
            row[arm] = psnr(pm.complete(Bs, keep, arm)[0], Xc)
        print(f"  keep={keep}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))
