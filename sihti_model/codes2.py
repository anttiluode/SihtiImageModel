"""Codes of codes: a nonlinear coarse scale on top of the patch memory (PREREG_CODES2.md).

Per band b, on the coarser reconstruction C_b:
  level 1   each 5x5 window -> one of K1 = 64 codes          (turns the coarse image into a code map)
  context   one-hot codes of the 3x3 neighbouring places, 4 px apart (576) -> PCA to 32 numbers z
  level 2   k-means (K = 512) on [n ; z]; each region a linear map [n ; z ; 1] -> 4x4 residue patch

The expert is chosen by a wider neighbourhood's codes, so the choice is a nonlinear function of a
region about three windows wide. Everything is closed form.
"""
import numpy as np

from .patch import G, NB, P, STRIDES, _kmeans, bands_of, nearest, normalise, overlap_add, patches, windows

OFF = [(a, b) for a in (-2, 0, 2) for b in (-2, 0, 2)]


def neighbour_codes(L, K1):
    """L (N, ni, nj) code map -> (N, ni, nj, 9) ids in 0..9*K1 (slot-major)."""
    N, ni, nj = L.shape
    Lp = np.pad(L, ((0, 0), (2, 2), (2, 2)), mode="edge")
    return np.stack([Lp[:, 2 + di:2 + di + ni, 2 + dj:2 + dj + nj] + t * K1 for t, (di, dj) in enumerate(OFF)], -1)


class ContextPatchMemory:
    def __init__(self, K=512, K1=64, zdim=32, per_image=10, seed=0):
        self.K, self.K1, self.zdim, self.per_image = K, K1, zdim, per_image
        self.rng = np.random.default_rng(seed)
        self.arms = {}

    # ---- features -------------------------------------------------------------------
    def _features(self, C, b):
        N = len(C)
        A, ii, jj = windows(C, STRIDES[b])
        An, sc = normalise(A.reshape(-1, G * G))
        a = self.arms[b]
        L = nearest(An, a["C1"]).reshape(N, len(ii), len(jj))
        I = neighbour_codes(L, self.K1).reshape(-1, 9)
        z = a["Pz"][I].sum(1) - a["zmu"]
        return An, sc, z.astype(np.float32), ii, jj

    # ---- fitting --------------------------------------------------------------------
    def fit(self, batches_fn, verbose=True):
        samp = {b: [] for b in (3, 2, 1, 0)}
        for X in batches_fn():                                   # pass 1: level-1 codebooks
            Bs = bands_of(X)
            for b in samp:
                C = sum(Bs[j] for j in range(b + 1, NB))
                A, _, _ = windows(C, STRIDES[b])
                A = A.reshape(len(X), -1, G * G)
                sel = self.rng.integers(0, A.shape[1], (len(X), self.per_image))
                samp[b].append(np.take_along_axis(A, sel[..., None], 1).reshape(-1, G * G))
        for b in samp:
            An, _ = normalise(np.concatenate(samp[b]))
            self.arms[b] = {"C1": _kmeans(An, self.K1, self.rng)}
        acc = {b: ([], [], [], []) for b in samp}               # pass 2: windows, contexts, targets
        for X in batches_fn():
            Bs = bands_of(X)
            N = len(X)
            for b in samp:
                C = sum(Bs[j] for j in range(b + 1, NB))
                A, ii, jj = windows(C, STRIDES[b])
                An, sc = normalise(A.reshape(-1, G * G))
                L = nearest(An, self.arms[b]["C1"]).reshape(N, len(ii), len(jj))
                I = neighbour_codes(L, self.K1).reshape(N, -1, 9)
                V = patches(Bs[b], ii, jj).reshape(N, -1, P * P)
                n_pos = I.shape[1]
                sel = self.rng.integers(0, n_pos, (N, self.per_image))
                take = lambda Z, d: np.take_along_axis(Z.reshape(N, n_pos, d), sel[..., None], 1).reshape(-1, d)
                acc[b][0].append(take(An, G * G)); acc[b][1].append(take(I, 9).astype(np.int32))
                acc[b][2].append(take(sc, 1)); acc[b][3].append(take(V, P * P))
        D = 9 * self.K1
        for b in samp:
            An, I, sc, V = (np.concatenate(x) for x in acc[b])
            M = len(I)
            # PCA of the one-hot context vectors, from co-occurrence counts
            m = np.bincount(I.ravel(), minlength=D) / M
            pairs = (I[:, :, None] * D + I[:, None, :]).ravel()
            S = np.bincount(pairs, minlength=D * D).reshape(D, D) / M
            ev, U = np.linalg.eigh(S - np.outer(m, m))
            U = U[:, ::-1][:, : self.zdim].astype(np.float32)
            z = np.concatenate([U[I[s:s + 100000]].sum(1) for s in range(0, M, 100000)]) - m @ U
            lam = np.sqrt(An.var(0).sum() / max(z.var(0).sum(), 1e-12))
            a = self.arms[b]
            a["Pz"], a["zmu"] = (U * lam).astype(np.float32), (m @ U * lam).astype(np.float32)
            z = (z * lam).astype(np.float32)
            F = np.concatenate([An, z], 1)
            a["C2"] = _kmeans(F, self.K, self.rng)
            k = nearest(F, a["C2"])
            Vn = V / sc
            X1 = np.concatenate([An, z, np.ones((M, 1), np.float32)], 1).astype(np.float64)
            d = X1.shape[1]
            W = np.zeros((self.K, d, P * P), np.float32)
            var = np.zeros((self.K, P * P), np.float32)
            glob = Vn.var(0)
            for c in range(self.K):
                s = k == c
                if s.sum() >= 2 * d:
                    Xc = X1[s]
                    Gm = Xc.T @ Xc
                    Gm[np.diag_indices(d)] += 1e-2 * np.trace(Gm) / d
                    W[c] = np.linalg.solve(Gm, Xc.T @ Vn[s])
                    var[c] = ((Vn[s] - Xc @ W[c]) ** 2).mean(0)
                elif s.sum() > 0:
                    W[c, -1] = Vn[s].mean(0)
                    var[c] = Vn[s].var(0) if s.sum() > 1 else glob
                else:
                    var[c] = glob
            a["W"], a["var"] = W, var
            if verbose:
                print(f"  codes2 band r{b}: {M} samples, context explains "
                      f"{100 * ev[::-1][:self.zdim].sum() / ev.sum():.0f}% of code-neighbourhood variance")
        return self

    def size(self):
        return sum(a["C1"].size + a["Pz"].size + a["C2"].size + a["W"].size for a in self.arms.values())

    # ---- prediction ------------------------------------------------------------------
    def predict_band(self, C, b, chunk=250):
        outs, uncs = [], []
        a = self.arms[b]
        for s0 in range(0, len(C), chunk):
            Cc = C[s0:s0 + chunk]
            N, H, W_ = Cc.shape
            An, sc, z, ii, jj = self._features(Cc, b)
            k = nearest(np.concatenate([An, z], 1), a["C2"])
            x = np.concatenate([An, z, np.ones((len(An), 1), np.float32)], 1)
            V = np.einsum("nd,ndv->nv", x, a["W"][k]) * sc
            shape = (N, len(ii), len(jj), P * P)
            outs.append(overlap_add(V.reshape(shape), ii, jj, H, W_))
            uncs.append(overlap_add((a["var"][k] * sc ** 2).reshape(shape), ii, jj, H, W_))
        return np.concatenate(outs), np.concatenate(uncs)

    def complete(self, Bs, keep=2):
        """Returns (image, uncertainty map, bands)."""
        out = [None] * NB
        for j in range(NB - keep, NB):
            out[j] = Bs[j]
        U = np.zeros_like(Bs[-1])
        for b in range(NB - keep - 1, -1, -1):
            C = sum(out[j] for j in range(b + 1, NB))
            out[b], u = self.predict_band(C, b)
            U += u
        return sum(out), U, out
