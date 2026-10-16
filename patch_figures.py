"""Figures for the patch residue memory (run after patch_gates.py)."""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from patch_gates import BSDS, OUT, bsds_images
from sihti_model.patch import NB, bands_of


def gray(ax, im, title=None):
    ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def shift_figure(ds):
    F = pickle.load(open(os.path.join(OUT, "cache", f"patch_figs_{ds}.pkl"), "rb"))
    cols = [("truth", None), ("given: core + r3", "zero"), ("whole-image\nmemory", "whole_memory"),
            ("patch memory\n(P3)", "P3")]
    fig, axs = plt.subplots(8, 8, figsize=(9, 9.5))
    for half, (name, truth) in enumerate([("standard", F["truth_std"]), ("shifted", F["truth_shift"])]):
        for i in range(8):
            for j, (t, k) in enumerate(cols):
                ax = axs[i, half * 4 + j]
                gray(ax, truth[i] if k is None else F["figs"][name][k][i], t if i == 0 else None)
    fig.text(0.27, 0.985, "centred test images", ha="center", fontsize=10)
    fig.text(0.75, 0.985, "same images shifted by up to 4 px", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(OUT, f"fig_patch_shift_{ds}.png"), dpi=110)
    plt.close(fig)


def defect_figure():
    F = pickle.load(open(os.path.join(OUT, "cache", "patch_figs_fashion.pkl"), "rb"))
    X, mask, maps = F["defect"]
    cols = ["image with scratch", "where the scratch is", "plain high-pass", "patch memory surprise"]
    fig, axs = plt.subplots(len(X), 4, figsize=(6.5, 1.6 * len(X)))
    for i in range(len(X)):
        gray(axs[i, 0], X[i]); gray(axs[i, 1], mask[i].astype(float))
        for j, k in enumerate(["highpass", "P3"]):
            M = maps[k][i]
            axs[i, 2 + j].imshow(M / (M.max() + 1e-9), cmap="magma"); axs[i, 2 + j].set_xticks([]); axs[i, 2 + j].set_yticks([])
    for j, t in enumerate(cols):
        axs[0, j].set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_patch_defects_fashion.png"), dpi=110)
    plt.close(fig)


def bird_figure(names=("42049", "8023", "135069")):
    pm = pickle.load(open(os.path.join(OUT, "cache", "patch_bsds.pkl"), "rb"))
    col = {**bsds_images("train", True), **bsds_images("val", True), **bsds_images("test", True)}
    cols = ["photo", "given: core + r3", "linear fill (P0)", "patch memory (P3)"]
    fig, axs = plt.subplots(len(names), 4, figsize=(12, 2.9 * len(names)))
    for i, n in enumerate(names):
        X = np.transpose(col[n], (2, 0, 1))               # 3 channels as a batch of 3 grey images
        Bs = bands_of(X)
        given = sum(Bs[j] for j in range(NB - 2, NB))
        p0 = pm.complete(Bs, 2, "P0")[0]
        p3 = pm.complete(Bs, 2, "P3")[0]
        for j, im in enumerate([X, given, p0, p3]):
            ax = axs[i, j]
            ax.imshow(np.clip(np.transpose(im, (1, 2, 0)), 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=10)
            if j > 0:
                ps = 10 * np.log10(1 / ((np.clip(im, 0, 1) - X) ** 2).mean())
                ax.set_xlabel(f"{ps:.1f} dB", fontsize=8)
        axs[i, 0].set_ylabel(n + (" (train)" if n == "135069" else " (held out)"), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_patch_birds.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["mnist", "fashion", "bsds"]
    for w in what:
        if w == "bsds":
            bird_figure()
        else:
            shift_figure(w)
            if w == "fashion":
                defect_figure()
