"""Uncertainty maps for the patch memory (see PREREG_UNCERTAINTY.md).

U1 code spread:          each code's own leftover training variance, scaled by window contrast^2.
U2 neighbour disagreement: spread of the 8 nearest stored examples' patches.
U3 edge energy:          baseline, 5x5 mean of the squared gradient of the given coarse image.
"""
import numpy as np

from .patch import G, NB, P, STRIDES, nearest, normalise, overlap_add, windows


def _p3(An, a, k, chunk=40000):
    V = np.empty((len(An), P * P), np.float32)
    for s in range(0, len(An), chunk):
        x = np.concatenate([An[s:s + chunk], np.ones((len(An[s:s + chunk]), 1), np.float32)], 1)
        V[s:s + chunk] = np.einsum("nd,ndv->nv", x, a["W"][k[s:s + chunk]])
    return V


def code_variances(pm, batches_fn):
    """Per band: (K, 16) mean squared residual each code leaves on a fresh draw of training windows."""
    S = pm._samples(batches_fn())
    out = {}
    for b, (A, V) in S.items():
        a = pm.arms[b]
        An, sc = normalise(A)
        Vn = V / sc
        k = nearest(An, a["C"])
        r2 = (Vn - _p3(An, a, k)) ** 2
        var = np.zeros((pm.K, P * P))
        np.add.at(var, k, r2)
        cnt = np.bincount(k, minlength=pm.K)[:, None]
        glob = r2.mean(0)
        out[b] = np.where(cnt > 0, var / np.maximum(cnt, 1), glob).astype(np.float32)
    return out


def fill_with_uncertainty(pm, Bs, keep, var, neighbours=False, topk=8):
    N, H, W = Bs[-1].shape
    out = [None] * NB
    for j in range(NB - keep, NB):
        out[j] = Bs[j]
    U1 = np.zeros((N, H, W), np.float32)
    U2 = np.zeros((N, H, W), np.float32) if neighbours else None
    for b in range(NB - keep - 1, -1, -1):
        C = sum(out[j] for j in range(b + 1, NB))
        A, ii, jj = windows(C, STRIDES[b])
        A = A.reshape(-1, G * G)
        a = pm.arms[b]
        An, sc = normalise(A)
        k = nearest(An, a["C"])
        shape = (N, len(ii), len(jj), P * P)
        out[b] = overlap_add((_p3(An, a, k) * sc).reshape(shape), ii, jj, H, W)
        U1 += overlap_add((var[b][k] * sc ** 2).reshape(shape), ii, jj, H, W)
        if neighbours:
            sp = np.empty((len(An), P * P), np.float32)
            B2 = (a["bankA"] ** 2).sum(1)
            for s in range(0, len(An), 4000):
                q = An[s:s + 4000]
                d2 = (q ** 2).sum(1)[:, None] + B2[None] - 2 * q @ a["bankA"].T
                nn = np.argpartition(d2, topk - 1, 1)[:, :topk]
                dn = np.take_along_axis(d2, nn, 1)
                w = np.exp(-(dn - dn.min(1, keepdims=True)) / (dn.mean(1, keepdims=True) + 1e-9))
                w /= w.sum(1, keepdims=True)
                nv = a["bankV"][nn]                                   # (n, k, 16)
                mu = np.einsum("nk,nkd->nd", w, nv)
                sp[s:s + 4000] = np.einsum("nk,nkd->nd", w, (nv - mu[:, None]) ** 2)
            U2 += overlap_add((sp * sc ** 2).reshape(shape), ii, jj, H, W)
    return sum(out), U1, U2


def edge_energy(given):
    gy, gx = np.gradient(given, axis=(1, 2))
    e = gx ** 2 + gy ** 2
    p = np.pad(e, ((0, 0), (2, 2), (2, 2)), mode="edge")
    return sum(p[:, i:i + e.shape[1], j:j + e.shape[2]] for i in range(5) for j in range(5)) / 25


def ause(U, e2):
    """Area between the sparsification curve of U and the oracle, over full MSE (lower is better)."""
    U, e2 = U.ravel(), e2.ravel().astype(np.float64)
    n = len(e2)
    fr = np.linspace(0, 0.99, 100)

    def curve(order):
        s = e2[order][::-1].cumsum()[::-1]                # suffix sums: MSE of what remains
        return np.array([s[int(f * n)] / (n - int(f * n)) for f in fr])
    cu = curve(np.argsort(-U, kind="stable"))
    co = curve(np.argsort(-e2, kind="stable"))
    return float((cu - co).mean() / e2.mean()), cu / e2.mean(), co / e2.mean()
