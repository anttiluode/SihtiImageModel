"""Dataset loaders. All return float32 images (N, C, 32, 32) in [0, 1] and int labels.

mnist / fashion: idx .gz files in data/<name>/ (MNIST 28x28 is zero-padded to 32x32).
cifar10: python batches in data/cifar-10-batches-py/ ; downloaded from the
Toronto site on first use if missing (needs normal internet).
"""
import gzip
import os
import pickle
import tarfile
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
}

_IDX_URLS = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}


def _idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _load_idx(name, split):
    d = os.path.join(ROOT, name)
    pre = "train" if split == "train" else "t10k"
    files = [f"{pre}-images-idx3-ubyte.gz", f"{pre}-labels-idx1-ubyte.gz"]
    os.makedirs(d, exist_ok=True)
    for fn in files:
        p = os.path.join(d, fn)
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            urllib.request.urlretrieve(_IDX_URLS[name] + fn, p)
    X = _idx(os.path.join(d, files[0])).astype(np.float32) / 255.0
    y = _idx(os.path.join(d, files[1])).astype(np.int64)
    X = np.pad(X, ((0, 0), (2, 2), (2, 2)))[:, None]          # (N,1,32,32)
    return X, y


def _load_cifar(split):
    d = os.path.join(ROOT, "cifar-10-batches-py")
    if not os.path.isdir(d):
        tgz = os.path.join(ROOT, "cifar-10-python.tar.gz")
        os.makedirs(ROOT, exist_ok=True)
        if not os.path.exists(tgz):
            print("downloading CIFAR-10 (163 MB) ...")
            urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(ROOT)
    names = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
    Xs, ys = [], []
    for n in names:
        with open(os.path.join(d, n), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        Xs.append(b[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.array(b[b"labels"]))
    return np.concatenate(Xs).astype(np.float32) / 255.0, np.concatenate(ys).astype(np.int64)


def load(name, split="train"):
    if name == "cifar10":
        return _load_cifar(split)
    return _load_idx(name, split)
