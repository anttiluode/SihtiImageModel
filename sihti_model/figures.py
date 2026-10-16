import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sieve import BAND_NAMES, sieve


def _hwc(im):
    im = np.asarray(im)
    return im[0] if im.shape[0] == 1 else np.transpose(im, (1, 2, 0))


def _show(ax, im, signed=False):
    im = _hwc(im)
    if signed:
        v = np.abs(im).max() + 1e-9
        ax.imshow(im if im.ndim == 2 else 0.5 + 0.5 * im / v, cmap="RdBu_r", vmin=-v, vmax=v)
    else:
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def sieve_figure(X, path):
    B = sieve(X)
    cols = ["image"] + list(BAND_NAMES)
    fig, axs = plt.subplots(len(X), len(cols), figsize=(1.6 * len(cols), 1.6 * len(X)))
    for i in range(len(X)):
        _show(axs[i, 0], X[i])
        for b in range(len(B)):
            _show(axs[i, b + 1], B[b][i], signed=b < len(B) - 1)
    for j, c in enumerate(cols):
        axs[0, j].set_title(c, fontsize=9)
    fig.suptitle("The sieve: image = r0 + r1 + r2 + r3 + core  (exact)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def samples_figure(samples, Xtr, ytr, names, path, per=8):
    """gauss | memory | nearest training image to each memory sample."""
    n = len(names)
    fig, axs = plt.subplots(n, 3 * per + 2, figsize=(0.75 * (3 * per + 2), 0.8 * n))
    flat = Xtr.reshape(len(Xtr), -1)
    for c in range(n):
        g = samples["gauss"][c * 100: c * 100 + per]
        mm = samples["memory"][c * 100: c * 100 + per]
        tr = flat[ytr == c]
        d = ((mm.reshape(per, -1)[:, None] - tr[None]) ** 2).sum(-1)
        nn = Xtr[ytr == c][d.argmin(1)]
        row = list(g) + [None] + list(mm) + [None] + list(nn)
        for j, im in enumerate(row):
            ax = axs[c, j]
            if im is None:
                ax.axis("off"); continue
            _show(ax, im)
        axs[c, 0].set_ylabel(names[c], fontsize=7, rotation=0, ha="right", va="center")
    axs[0, per // 2].set_title("gauss: every band predicted + noise", fontsize=9)
    axs[0, per + 1 + per // 2].set_title("memory: core imagined, residues read", fontsize=9)
    axs[0, 2 * per + 2 + per // 2].set_title("nearest TRAINING image to each memory sample", fontsize=9)
    fig.tight_layout(pad=0.2); fig.savefig(path, dpi=110); plt.close(fig)


def stages_figure(m, names, path, mode="memory"):
    rng = np.random.default_rng(3)
    n = len(names)
    cols = ["core", "+ r3", "+ r2", "+ r1", "+ r0 = image"]
    fig, axs = plt.subplots(n, 5, figsize=(6.5, 1.3 * n))
    for c in range(n):
        Z = m.generate(c, 1, mode=mode, rng=rng)
        for j in range(5):
            _show(axs[c, j], m.decode(Z, upto=m.nb - 1 - j)[0])
        axs[c, 0].set_ylabel(names[c], fontsize=7, rotation=0, ha="right", va="center")
    for j, t in enumerate(cols):
        axs[0, j].set_title(t, fontsize=9)
    fig.suptitle("Generation, coarse to fine (memory mode)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def completion_figure(X, outs, path):
    keys = ["zero", "gauss_agnostic", "gauss_classread", "memory_classread"]
    titles = ["truth", "given: core only", "+ residues\n(linear, no class)",
              "+ residues\n(class read)", "+ residues\n(memory read)"]
    fig, axs = plt.subplots(len(X), 5, figsize=(7, 1.4 * len(X)))
    for i in range(len(X)):
        _show(axs[i, 0], X[i])
        for j, k in enumerate(keys):
            _show(axs[i, j + 1], outs[k][i])
    for j, t in enumerate(titles):
        axs[0, j].set_title(t, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def where_class_lives(R, path):
    g1 = R["G1"]
    nb = len(g1["single"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    labels = ["r0", "r1", "r2", "r3", "core"]
    ax[0].bar(labels, [g1["single"][str(b)] for b in range(nb)], color="#4a6fa5")
    ax[0].set_title("readout from ONE band alone"); ax[0].set_ylim(0, 1)
    cum = ["core", "+r3", "+r2", "+r1", "+r0"]
    ax[1].plot(cum, [g1["cumulative"][str(k)] for k in range(1, nb + 1)], "o-", color="#c0504d")
    ax[1].set_title("readout, adding bands coarse -> fine"); ax[1].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.3); a.set_ylabel("test accuracy")
    fig.suptitle(f"Where does the class live? ({R['dataset']})")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
