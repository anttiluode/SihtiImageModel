import numpy as np

from sihti_model.model import SihtiImageModel
from sihti_model.sieve import sieve, unsieve


def _toy(n=400, seed=0):
    r = np.random.default_rng(seed)
    X = np.zeros((n, 1, 32, 32), np.float32)
    y = r.integers(0, 2, n)
    for i in range(n):
        c = r.integers(8, 24, 2)
        if y[i] == 0:
            X[i, 0, c[0] - 4:c[0] + 4, c[1] - 4:c[1] + 4] = 1
        else:
            X[i, 0, c[0] - 1:c[0] + 1, 4:28] = 1
    return X, y


def test_sieve_lossless():
    X = np.random.default_rng(1).random((5, 3, 32, 32))
    assert np.abs(unsieve(sieve(X)) - X).max() < 1e-12


def test_first_band_is_centre_surround():
    x = np.zeros((1, 1, 32, 32)); x[0, 0, 16, 16] = 1
    r0 = sieve(x)[0][0, 0]
    assert np.isclose(r0[16, 16], 1.0) and np.isclose(r0[15, 16], -0.25)


def test_fit_generate_complete_shapes():
    X, y = _toy()
    m = SihtiImageModel(dims=(32, 32, 16, 8, 4), K=2)
    m.fit(X, y, 2, verbose=False)
    for mode in ("gauss", "memory"):
        Z = m.generate(1, 3, mode=mode)
        assert [z.shape for z in Z] == [(3, d) for d in m.dims]
        assert m.decode(Z).shape == (3, 1, 32, 32)
    Zt = m.encode(X[:5])
    for mode in ("zero", "gauss", "memory"):
        out = m.complete(Zt, 1, mode)
        assert all(np.allclose(out[j], Zt[j]) for j in (4,))


def test_full_vocabulary_is_lossless():
    X, y = _toy(100)
    m = SihtiImageModel(dims=(1024,) * 5, K=2)
    m.fit(X, y, 2, verbose=False)
    assert np.abs(m.decode(m.encode(X)) - X).max() < 1e-4


def test_patch_overlap_add_is_exact():
    from sihti_model.patch import overlap_add, patches, windows
    B = np.random.default_rng(2).random((3, 21, 30)).astype(np.float32)
    _, ii, jj = windows(B, 2)
    assert np.abs(overlap_add(patches(B, ii, jj), ii, jj, 21, 30) - B).max() < 1e-6


def test_patch_memory_fits_and_completes():
    from sihti_model.patch import PatchMemory, bands_of
    X, _ = _toy(200)
    X = X[:, 0]
    pm = PatchMemory(K=16, bank=500, per_image=20).fit(lambda: [X[:100], X[100:]], verbose=False)
    Bs = bands_of(X[:4])
    for arm in ("P0", "P1", "P2", "P3"):
        im, out = pm.complete(Bs, 2, arm)
        assert im.shape == X[:4].shape and np.allclose(out[4], Bs[4])
