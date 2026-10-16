"""Patch memory with access to a global summary of the given coarse observation (PREREG_GLOBAL.md).

g = the 16 PCA coefficients of the given core band, the same for every place in the image.
  G   P3's routing and codebook unchanged; experts linear in [n ; g ; 1]
  GP  as G plus position and position x g:  [n ; g ; pos ; g*pos_y ; g*pos_x ; 1]
  GR  routing by k-means on [n ; g]; experts linear in [n ; g ; 1]
"""
import numpy as np

from .patch import G as GW, NB, P, STRIDES, _kmeans, bands_of, nearest, normalise, overlap_add, patches, windows


class GlobalPatchMemory:
    def __init__(self, pm, wm, mode="G", K=512, per_image=10, seed=0):
        self.pm, self.mode, self.per_image = pm, mode, per_image
        self.K = K if mode == "GR" else pm.K          # G and GP reuse P3's codebook
        self.cmean, self.cU = wm.mean[4].astype(np.float32), wm.U[4].astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self.arms = {}

    def raw_g(self, core):
        return (core.reshape(len(core), -1) - self.cmean) @ self.cU

    def _g(self, core):
        return ((self.raw_g(core) - self.gmu) / self.gsd * self.lam).astype(np.float32)

    def _inputs(self, An, g_rows, pos_rows):
        parts = [An, g_rows]
        if self.mode == "GP":
            parts += [pos_rows, g_rows * pos_rows[:, :1], g_rows * pos_rows[:, 1:2]]
        parts.append(np.ones((len(An), 1), np.float32))
        return np.concatenate(parts, 1)

    def _route(self, An, g_rows, b):
        a = self.arms[b]
        if self.mode == "GR":
            return nearest(np.concatenate([An, g_rows], 1), a["C"])
        return nearest(An, self.pm.arms[b]["C"])

    @staticmethod
    def _pos(ii, jj, H, W):
        py = (ii + 2) / (H - 1) * 2 - 1
        px = (jj + 2) / (W - 1) * 2 - 1
        return np.stack(np.meshgrid(py, px, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)

    def fit(self, batches_fn, verbose=True):
        acc = {b: ([], [], [], [], []) for b in (3, 2, 1, 0)}
        graw = []
        for X in batches_fn():
            Bs = bands_of(X)
            N, H, W = X.shape
            gr = self.raw_g(Bs[4])
            graw.append(gr)
            for b in acc:
                C = sum(Bs[j] for j in range(b + 1, NB))
                A, ii, jj = windows(C, STRIDES[b])
                V = patches(Bs[b], ii, jj).reshape(N, -1, P * P)
                A = A.reshape(N, -1, GW * GW)
                npos = A.shape[1]
                pos = self._pos(ii, jj, H, W)
                sel = self.rng.integers(0, npos, (N, self.per_image))
                acc[b][0].append(np.take_along_axis(A, sel[..., None], 1).reshape(-1, GW * GW))
                acc[b][1].append(np.take_along_axis(V, sel[..., None], 1).reshape(-1, P * P))
                acc[b][2].append(np.repeat(gr, self.per_image, 0))
                acc[b][3].append(pos[sel.ravel()])
        graw = np.concatenate(graw)
        self.gmu, self.gsd = graw.mean(0), graw.std(0) + 1e-9
        self.lam = None
        for b in (3, 2, 1, 0):
            A, V, gr, pos = (np.concatenate(acc[b][i]) for i in range(4))
            An, sc = normalise(A)
            gs = (gr - self.gmu) / self.gsd
            if self.lam is None:                       # same scaling rule as Q: g carries n's total variance
                self.lam = float(np.sqrt(An.var(0).sum() / gs.var(0).sum()))
            g = (gs * self.lam).astype(np.float32)
            a = {}
            self.arms[b] = a
            if self.mode == "GR":
                a["C"] = _kmeans(np.concatenate([An, g], 1), self.K, self.rng)
            k = self._route(An, g, b)
            X1 = self._inputs(An, g, pos).astype(np.float64)
            Vn = V / sc
            d = X1.shape[1]
            Wk = np.zeros((self.K, d, P * P), np.float32)
            for c in range(self.K):
                s = k == c
                if s.sum() >= 2 * d:
                    Xc = X1[s]
                    Gm = Xc.T @ Xc
                    Gm[np.diag_indices(d)] += 1e-2 * np.trace(Gm) / d
                    Wk[c] = np.linalg.solve(Gm, Xc.T @ Vn[s])
                elif s.sum() > 0:
                    Wk[c, -1] = Vn[s].mean(0)
            a["W"] = Wk
            if verbose:
                print(f"  {self.mode} band r{b}: {len(A)} samples, expert inputs {d}")
        return self

    def size(self):
        n = sum(a["W"].size + a.get("C", np.zeros(0)).size for a in self.arms.values())
        if self.mode != "GR":
            n += sum(self.pm.arms[b]["C"].size for b in self.arms)
        return int(n + self.cU.size + self.cmean.size)

    def predict_band(self, C, b, g, chunk=250):
        outs = []
        for s0 in range(0, len(C), chunk):
            Cc, gc = C[s0:s0 + chunk], g[s0:s0 + chunk]
            N, H, W = Cc.shape
            A, ii, jj = windows(Cc, STRIDES[b])
            An, sc = normalise(A.reshape(-1, GW * GW))
            npos = len(ii) * len(jj)
            g_rows = np.repeat(gc, npos, 0)
            pos_rows = np.tile(self._pos(ii, jj, H, W), (N, 1))
            k = self._route(An, g_rows, b)
            x = self._inputs(An, g_rows, pos_rows)
            V = np.einsum("nd,ndv->nv", x, self.arms[b]["W"][k]) * sc
            outs.append(overlap_add(V.reshape(N, len(ii), len(jj), P * P), ii, jj, H, W))
        return np.concatenate(outs)

    def complete(self, Bs, keep=2):
        g = self._g(Bs[4])
        out = [None] * NB
        for j in range(NB - keep, NB):
            out[j] = Bs[j]
        for b in range(NB - keep - 1, -1, -1):
            C = sum(out[j] for j in range(b + 1, NB))
            out[b] = self.predict_band(C, b, g)
        return sum(out), out
