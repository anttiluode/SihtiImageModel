# SihtiImageModel

A fixed sieve plus a residue memory, used as an image model. No backpropagation anywhere in the model.

```
image = r0 + r1 + r2 + r3 + core        (exact; the inverse is addition)
```

The sieve is [Sihti](https://github.com/anttiluode/Sihti)'s lattice arm: run heat diffusion on the pixel
lattice to depths 0, 1, 4, 16, 64 steps and keep every departure. `r0 = x - purify(x)` is a
centre-surround band. Nothing in the sieve is learned.

On top of it, everything is closed form, one pass over the data:

| part | what it is | how it is learned |
|---|---|---|
| vocabulary | per-band PCA basis: which shapes live in each band | unsupervised, one eigendecomposition per band |
| readout | band codes -> class | ridge regression (Hinton's "train only the last layer") |
| generator, linear | core ~ 8 Gaussian modes per class; each finer band = linear prediction from all coarser bands + noise | least squares + covariance |
| generator, memory | core imagined as above; each finer band **read** from the stored residues of the training images whose coarser state is closest (top-8, soft-weighted blend) | nothing: store the residues |

The memory mode is the "write the bird, keep its residue" idea taken literally. Writing an image
stores its residue codes. Reading is addressing by the coarse state.

## Run it

```
pip install numpy scipy pillow matplotlib scikit-learn
python gates.py mnist        # ~10 min on 2 CPU cores
python gates.py fashion
python gates.py cifar10      # birds; downloads CIFAR-10 once. NOT run yet (see below)
python export_web.py         # writes web/model_*.json for the browser page
python -m pytest tests

# patch residue memory
git clone --depth 1 --filter=blob:none --sparse https://github.com/BIDS/BSDS500.git data/bsds
(cd data/bsds && git sparse-checkout set BSDS500/data/images)      # 500 photos, ~25 MB
python patch_gates.py mnist && python patch_gates.py fashion && python patch_gates.py bsds
python patch_figures.py
python export_patch_web.py   # adds the patch memory and the photos to the page
```

`web/index.html` is the live page: pick a class, watch the sum build core -> +r3 -> +r2 -> +r1 -> +r0
with each added band shown in signed colour underneath, and the fill-in bench where a real test
image is cut down to its core (16 numbers) and the residues are put back.

## Results (pre-registered in PREREG.md, committed before any gate ran)

**The kill condition fired.** G2d failed on both datasets: with 10 examples per class, reading the
class from Sihti codes is slightly *worse* than reading it from raw pixels (MNIST 68.8% vs 70.1%,
Fashion 67.9% vs 69.2%). The sieve does not help a readout find classes. A lossless linear change
of coordinates cannot add linear information, which is the same lesson Sihti and
SighImageFactorization already filed: the coordinate change alone does not make objects.

What the sieve *is* good for here is generation and completion: a coarse-to-fine order to imagine
in, and an address (the coarse state) to read stored detail by.

| gate | MNIST | Fashion | |
|---|---|---|---|
| G0 sieve lossless (max error) | 2.2e-16 | 1.1e-16 | PASS |
| G1 core alone (16 numbers) reads the class | 74.6% | 68.9% | predicted >= 80%: **missed** |
| G1 best single band | r2 86.4% | r1 79.1% | |
| G2 ridge on raw pixels | 86.2% | 81.2% | |
| G2 ridge on Sihti codes | 86.4% | 81.1% | G2a PASS (same, as predicted) |
| G2 2000 random ReLU units + ridge, raw | 95.9% | 86.1% | |
| G2 2000 random ReLU units + ridge, Sihti | 96.0% | 86.6% | G2b PASS (barely) |
| G2 backprop MLP (judge, reference) | 98.1% | 89.4% | G2c PASS (backprop still wins) |
| G2d 10-shot raw -> Sihti | 70.1 -> 68.8% | 69.2 -> 67.9% | **FAIL** |
| G3 samples recognised by the judge, linear | 98.8% | 89.8% | G3a PASS |
| G3 samples recognised by the judge, memory | 100% | 97.7% | G3a PASS |
| G3 novelty ratio linear / memory | 1.06 / 0.69 | 0.97 / 0.67 | G3b PASS (>= 0.5) |
| G3 near-copies of a training image | 0% | 0% | |
| G4 fill from core: none -> linear -> memory (PSNR) | 12.6 -> 15.5 -> 16.8 dB | 13.1 -> 17.6 -> 18.7 dB | G4a PASS |
| G4 judge accuracy on the filled image (none -> memory) | 8.8% -> 77.7% | 15.2% -> 74.2% | |

Reading the table:

- The random ReLU layer is what lifts the readout from 86% to 96% on digits, with or without the
  sieve. That is Hinton's random-bird-net, last layer only, and it works; the sieve adds nothing to it.
- Generation works with no backprop. Memory-mode samples are clean and recognisable and are not
  copies (median distance to the nearest training image is 0.69 of the distance a real unseen test
  image has). They are closer to the training set than real test images are, so it is a blending
  memory, not a free imagination. `fig_samples_*.png` shows each memory sample beside its nearest
  training image so you can judge that yourself.
- Completion: from a 16-number blob the judge cannot read (9% on digits), the memory read puts back
  residues the judge reads at 78%. My prediction that the memory read would lose PSNR to the linear
  fill was wrong; it wins on PSNR too, because the linear fill is only linear.
- Where it fails visibly: an ambiguous core gets confidently filled with the wrong class (a 5 becomes
  a 1, a dress becomes trousers). See `fig_completion_*.png`.

## How much memory does memory mode need?  (`python bank_size.py`, run after the gates)

Each stored image is 392 numbers (the image is 1024 pixels; the finest band keeps 69% of its
detail). The fixed part of the model, which stores no images, is 1.43M numbers. Shrinking the bank:

| stored per class | MNIST recognised / novelty / fill | Fashion recognised / novelty / fill |
|---|---|---|
| 2 | 98.5% / 0.62 / 13.4 dB | 84.7% / 0.82 / 15.1 dB |
| 5 | 100% / 0.98 / 13.6 dB | 86.8% / 0.86 / 15.8 dB |
| 20 | 100% / 0.93 / 14.4 dB | 90.4% / 0.82 / 16.7 dB |
| 100 | 100% / 0.79 / 15.3 dB | 96.7% / 0.74 / 17.5 dB |
| 500 | 100% / 0.73 / 16.1 dB | 98.2% / 0.70 / 18.2 dB |
| 2000 | 100% / 0.69 / 16.7 dB | 97.7% / 0.67 / 18.7 dB |

Recognisable generation needs almost no stored images, because the class knowledge sits in the
fixed part (core modes, linear maps, vocabulary). Diversity was not measured: with 5 stored images
the fine detail can only be blended from 5 sources. Completion keeps improving with every stored
image, which is the signature of an example-based method: it can recombine what it has seen and
nothing else.

## Patch residue memory (PREREG_PATCH.md, committed before any patch run)

The whole-image memory addresses with the whole coarse image. The patch memory addresses with a
5x5 window of the coarser reconstruction around one place and returns the 4x4 residue patch for
that place. 512 local rules per band serve every location of every image. The main arm (P3) is a
k-means codebook of windows where each code holds a small linear map (window -> patch); windows are
contrast-normalised. Total size 0.90M numbers, fixed, and no class is used anywhere.

`python patch_gates.py mnist|fashion|bsds`, then `python patch_figures.py`.

| | MNIST | Fashion | |
|---|---|---|---|
| PG1 fill from core+r3: whole-image linear / whole-image memory / **patch P3** | 17.8 / 18.7 / **20.6 dB** | 18.8 / 19.8 / **20.4 dB** | PASS |
| fill from core only (16 numbers): whole-image memory / patch P3 | 16.8 / 16.0 dB | 18.7 / 17.6 dB | patch loses, as predicted |
| PG2a cost of shifting the image up to 4 px, patch P3 | 0.27 dB | **0.53 dB** | PASS / **FAIL** (threshold 0.5) |
| PG2b shifted, whole-image memory vs patch P3 | 14.1 vs 20.4 dB | 15.7 vs 19.9 dB | PASS (+6.3 / +4.2 dB) |
| PG4 stored numbers, patch P3 vs whole-image (fixed + bank) | 0.90M vs 9.3M | same | PASS |
| PG5 scratch localisation, pixel AUROC: high-pass / whole linear / patch P3 | | 0.867 / 0.917 / 0.937 | PASS |
| PG5 which images have a scratch, image AUROC, patch P3 | | 0.58 | barely above chance |

Real photos, BSDS500 (200 train, 300 held out, greyscale, half size). The whole-image model cannot
run here at all.

| fill from core+r3 | no fill | linear filter (P0) | stored-example lookup (P1, 30 imgs) | **patch P3** |
|---|---|---|---|---|
| mean PSNR | 22.76 | 24.87 | 23.72 (P3 on the same 30: 24.41) | **25.12** |

PG3 PASS: P3 beats the linear filter on all 300 held-out photos, by 0.25 dB on average. From the
core alone it does not (22.26 vs 22.27 dB, better on only 36% of photos).

Reading it:

- Fixing the address fixed the memory. Shifted digits wreck the whole-image memory (a shifted 5
  comes back as a 6, a 3 as a 5: `fig_patch_shift_mnist.png`); the patch memory follows the shape
  that is there. On Fashion the cost of shifting is 0.53 dB, just over the pre-registered 0.5;
  the likely cause is that clothing fills the frame and shifting pushes it into the border, where
  the windows run on edge padding. Logged as a fail, threshold not moved.
- Judge accuracies on shifted images are not usable: the backprop judge itself reads only 50% of
  the *real* shifted digits (43% of shifted clothing). Use PSNR there.
- The patch memory needs some local structure to address with. From a 16-number core there is
  almost none, and the global whole-image memory wins. A real system would use both: something
  global for layout, the patch memory for everything below it.
- On photos the gain over a linear filter is small but consistent. The visible part is cleaner
  edges without the light halo the linear filter rings around branches
  (`fig_patch_birds.png`, the held-out hawk on the branch). It does not invent texture that the
  coarse image did not contain: the ground bird's stripes do not come back.
- The codebook with local linear maps (P3) beats pure stored-example lookup (P1) on photos while
  storing 9x fewer numbers.
- Defects: the pre-registered pixel gate passes, but look at `fig_patch_defects_fashion.png` before
  believing it: the surprise map does not make scratches stand out, and deciding *whether* an image
  has a scratch is 0.58 AUROC. As a defect detector this is not usable.

Prior art for this part: example-based super-resolution (Freeman, Jones & Pasztor 2002), sparse-coding
super-resolution with coupled dictionaries (Yang, Wright, Huang & Ma 2010), anchored neighbourhood
regression, which is a codebook of local linear maps very close to P3 (Timofte, De Smet & Van Gool
2013, A+ 2014), and RAISR (Romano, Isidoro & Milanfar 2016, hashed local filters). Patch-memory
anomaly detection: PatchCore (Roth et al. 2022), which uses pretrained deep features.

## Correction made during the run

The first G4 run used "codes = 0" as the core-only baseline, which decodes to the core **plus the
mean of every residue band** (a ghost average digit), not the blurred image PREREG.md names. The
baseline was fixed to the core alone and G4 re-run from the cached models (`python gates.py <ds> --g4`).
G4a passed both before and after. Both runs are in `results/log_*.txt`.

## Not done

- **No CIFAR-10.** It could not be downloaded in the sandbox where this was built.
  `python gates.py cifar10` will download and run the whole-image model on it. Real bird photos
  were tested only through the patch memory (BSDS500), and only for filling in detail, not for
  generating a bird from nothing.
- Generating a new photo from scratch would need a global model for the layout; the patch memory
  only fills in below it.
- The sieve operator is the plain lattice. Sihti's own-colour-graph operator (objects stay in the
  core) was not tried as the generator's sieve.

## Prior art

Laplacian pyramid (Burt & Adelson 1983); coarse-to-fine generation on a Laplacian pyramid
(LAPGAN, Denton et al. 2015, with GANs where this uses least squares and lookup); non-parametric
multi-scale texture synthesis (Heeger & Bergen 1995; De Bonet 1997, which also reads finer bands
by matching coarser parents; Efros & Leung 1999); example-based super-resolution, which stores
high-frequency patches and looks them up by their low-frequency patch (Freeman, Jones & Pasztor 2002),
and face hallucination on a Laplacian pyramid (Baker & Kanade 2000), which is the fill-in bench here
with patches instead of whole images; retrieval-augmented diffusion (Blattmann et al. 2022), the
modern neural version of reading stored images; the perceptron with fixed random association units
and one trained layer (Rosenblatt 1958) and least-squares output training (Widrow & Hoff 1960); random features with a trained last layer (Rosenblatt; extreme
learning machines; Rahimi & Recht 2007); mixture of factor analysers / Gaussian models of images.
What is this repo's: putting the Sihti sieve under a residue memory and measuring, with
pre-registered gates, what that buys (generation, completion) and what it does not (classification).
