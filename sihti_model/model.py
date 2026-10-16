"""Sihti image model: a fixed sieve + a residue memory.  No backpropagation anywhere.

Learned parts (all closed form, one pass over the data):
  1. vocabulary : per-band PCA basis U_b  (unsupervised; "what shapes live in band b")
  2. readout    : ridge regression from band codes to class   (Hinton's "train only the last layer")
  3. generator  : per class, coarse -> fine
                    core   ~ mixture of K Gaussians          (imagine the coarse)
                    band_b = A_b [coarser codes, 1] + L_b eps (each finer residue predicted
                                                              from everything coarser)
  4. memory     : the stored residue codes of training images; a finer band is READ by
                  addressing with the coarser state (nearest stored coarse states, blend
                  their residues).  This is "write the bird, keep its residue" literally.

An image is always  x = sum_b (mean_b + U_b z_b): the soma is addition.
"""
import numpy as np

from .sieve import DEPTHS, sieve

DEFAULT_DIMS = (160, 128, 64, 24, 16)    # r0, r1, r2, r3, core  (fine -> coarse)


def _chol(S, eps):
    S = 0.5 * (S + S.T) + eps * np.eye(len(S)) * max(np.trace(S) / len(S), 1e-12)
    return np.linalg.cholesky(S)


def ridge_fit(F, Y, lam):
    d = F.shape[1]
    G = F.T @ F
    G[np.diag_indices(d)] += lam * np.trace(G) / d
    return np.linalg.solve(G, F.T @ Y)


def kmeans(X, K, rng, iters=40):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in range(K):
            m = a == k
            C[k] = X[m].mean(0) if m.sum() > 0 else X[rng.integers(len(X))]
    return a


class SihtiImageModel:
    def __init__(self, dims=DEFAULT_DIMS, depths=DEPTHS, K=8, seed=0):
        self.dims, self.depths, self.K = tuple(dims), tuple(depths), K
        self.nb = len(dims)
        self.rng = np.random.default_rng(seed)

    # ---------- sieve + vocabulary -------------------------------------------------
    def bands(self, X):
        return [b.reshape(len(X), -1).astype(np.float32) for b in sieve(X, self.depths, np.float32)]

    def encode(self, X=None, bands=None):
        if bands is None:
            bands = self.bands(X)
        return [(bands[b] - self.mean[b]) @ self.U[b] for b in range(self.nb)]

    def decode(self, codes, upto=None):
        """codes: list fine->coarse. upto=j decodes only bands j..core (coarse part)."""
        lo = 0 if upto is None else upto
        n = len(codes[-1])
        x = np.zeros((n, self.mean[0].size), np.float32)
        for b in range(lo, self.nb):
            x += self.mean[b] + codes[b] @ self.U[b].T
        return x.reshape(n, *self.shape)

    def decode_band(self, codes, b):
        return (self.mean[b] + codes[b] @ self.U[b].T).reshape(len(codes[b]), *self.shape)

    # ---------- fitting --------------------------------------------------------------
    def fit(self, X, y, n_classes=10, bank_per_class=2000, verbose=True):
        self.shape = X.shape[1:]
        self.n_classes = n_classes
        B = self.bands(X)
        self.mean, self.U, self.band_var = [], [], []
        for b in range(self.nb):
            m = B[b].mean(0)
            F = B[b] - m
            S = (F.T @ F) / len(F)
            ev, V = np.linalg.eigh(S.astype(np.float64))
            V = V[:, ::-1][:, : self.dims[b]].astype(np.float32)
            self.mean.append(m)
            self.U.append(V)
            self.band_var.append(float(ev.sum()))
            if verbose:
                print(f"  band {b}: kept {100 * ev[::-1][: self.dims[b]].sum() / ev.sum():.1f}% var "
                      f"with {self.dims[b]} comps")
        Z = self.encode(bands=B)
        del B
        self.gen = [self._fit_generator(Z, y == c) for c in range(n_classes)]
        self.gen_all = self._fit_generator(Z, np.ones(len(y), bool))
        # memory bank (stored residue codes)
        self.bank = []
        for c in range(n_classes):
            idx = np.flatnonzero(y == c)
            idx = self.rng.permutation(idx)[:bank_per_class]
            self.bank.append([z[idx] for z in Z])
        return Z

    def _prev(self, Zs, b):
        """coarser codes than band b, concatenated (coarse first)."""
        return np.concatenate([Zs[j] for j in range(self.nb - 1, b, -1)], 1)

    def _fit_generator(self, Z, mask, lam=1e-3):
        Zs = [z[mask].astype(np.float64) for z in Z]
        g = {}
        zc = Zs[-1]
        lab = kmeans(zc, self.K, self.rng)
        comps = []
        for k in range(self.K):
            m = lab == k
            if m.sum() < 3:
                continue
            S = np.cov(zc[m].T) if m.sum() > 1 else np.eye(zc.shape[1]) * 1e-4
            comps.append((m.mean(), zc[m].mean(0), _chol(S, 1e-3)))
        w = np.array([c[0] for c in comps])
        g["core"] = (w / w.sum(), np.stack([c[1] for c in comps]), np.stack([c[2] for c in comps]))
        for b in range(self.nb - 2, -1, -1):
            P = np.concatenate([self._prev(Zs, b), np.ones((len(zc), 1))], 1)
            A = ridge_fit(P, Zs[b], lam)
            E = Zs[b] - P @ A
            g[b] = (A, _chol(np.cov(E.T), 1e-3))
        return g

    # ---------- generation -----------------------------------------------------------
    def generate(self, c, n=16, tau=0.7, mode="gauss", rng=None, topk=8):
        """Coarse -> fine.  mode 'gauss': each band = linear prediction + noise.
        mode 'memory': core imagined from the Gaussian mixture, every finer band READ
        from the stored residues of the training images whose coarser state is closest."""
        rng = self.rng if rng is None else rng
        g = self.gen[c] if c is not None else self.gen_all
        w, mu, L = g["core"]
        ks = rng.choice(len(w), n, p=w)
        eps = rng.standard_normal((n, mu.shape[1]))
        zc = mu[ks] + tau * np.einsum("nij,nj->ni", L[ks], eps)
        Zs = [None] * (self.nb - 1) + [zc]
        for b in range(self.nb - 2, -1, -1):
            prev = self._prev(Zs, b)
            if mode == "gauss":
                A, Lb = g[b]
                P = np.concatenate([prev, np.ones((n, 1))], 1)
                Zs[b] = P @ A + tau * rng.standard_normal((n, Lb.shape[0])) @ Lb.T
            else:
                Zs[b] = self.read_memory(prev, b, c, topk)
        return [z.astype(np.float32) for z in Zs]

    def read_memory(self, prev, b, c, topk=8):
        if c is None:
            bank = [np.concatenate([self.bank[k][j] for k in range(self.n_classes)])
                    for j in range(self.nb)]
        else:
            bank = self.bank[c]
        Pb = self._prev(bank, b)
        pb2 = (Pb ** 2).sum(1)[None]
        topk = min(topk, len(Pb))
        nn = np.empty((len(prev), topk), np.int64)
        dn = np.empty((len(prev), topk))
        for s in range(0, len(prev), 512):
            q = prev[s:s + 512]
            d2 = (q ** 2).sum(1)[:, None] + pb2 - 2 * q @ Pb.T
            nn[s:s + 512] = np.argpartition(d2, topk - 1, 1)[:, :topk]
            dn[s:s + 512] = np.take_along_axis(d2, nn[s:s + 512], 1)
        h = dn.mean(1, keepdims=True) + 1e-9
        wt = np.exp(-(dn - dn.min(1, keepdims=True)) / h)
        wt /= wt.sum(1, keepdims=True)
        return np.einsum("nk,nkd->nd", wt, bank[b][nn])

    # ---------- completion: keep the coarse bands, fill in the residues ----------------
    def complete(self, codes, keep=1, mode="gauss", c=None, topk=8):
        """keep = how many coarse bands are given (1 = core only).  Fills finer bands with
        the conditional MEAN (gauss) or a memory read.  c=None: class-agnostic."""
        g = self.gen[c] if c is not None else self.gen_all
        Zs = [None] * self.nb
        for j in range(self.nb - keep, self.nb):
            Zs[j] = codes[j]
        n = len(codes[-1])
        for b in range(self.nb - keep - 1, -1, -1):
            prev = self._prev(Zs, b)
            if mode == "gauss":
                A, _ = g[b]
                Zs[b] = np.concatenate([prev, np.ones((n, 1))], 1) @ A
            elif mode == "zero":
                Zs[b] = np.zeros((n, self.dims[b]))
            else:
                Zs[b] = self.read_memory(prev, b, c, topk)
        return [z.astype(np.float32) for z in Zs]

    # ---------- readout features ---------------------------------------------------
    def features(self, codes, bands=None):
        """Band codes, each band scaled to unit total variance (the 'Sihti metric')."""
        bands = range(self.nb) if bands is None else bands
        return np.concatenate([codes[b] / np.sqrt(self.band_var[b]) for b in bands], 1)
