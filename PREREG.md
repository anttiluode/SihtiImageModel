# Pre-registration — Sihti image model

Written and committed **before** any gate was run.  The only thing run before this file
was a smoke test (fit on MNIST, look at one grid of samples), which is where `tau = 0.7`
for generation was chosen.

Datasets: MNIST and Fashion-MNIST, standard train/test splits, padded to 32x32.
Sieve: lattice diffusion, depths (0, 1, 4, 16, 64) -> r0, r1, r2, r3, core.
Vocabulary dims (fine -> coarse): 160, 128, 64, 24, 16.  K = 8 core modes per class.
No backpropagation anywhere in the model.  Seeds fixed (0).

## G0 — the sieve is lossless
float64 sieve on the whole test set: `max |sum(bands) - x| < 1e-12`.  PASS/FAIL.

## G1 — where does the class live?  (measurement, no pass/fail)
Ridge readout trained on each band alone, and on cumulative coarse -> fine sets.
Prediction: core alone (16 numbers) >= 80% on MNIST; the finest band r0 alone is the
weakest single band.

## G2 — reading the class without backprop
| arm | features | trained by |
|---|---|---|
| A | raw pixels | ridge (closed form) |
| B | 2000 fixed random ReLU units on raw pixels ("random bird net, last layer only") | ridge |
| C | Sihti band codes (each band scaled to unit variance) | ridge |
| D | 2000 fixed random ReLU units on Sihti codes | ridge |
| E | raw pixels, 1 hidden layer of 512, **backprop** (sklearn MLP, Adam, 20 epochs) | backprop |

Predictions:
- **G2a** |C - A| <= 1.0 point.  A (nearly) lossless linear change of coordinates should add
  nothing to a linear readout.  (If C beats A by more, it is the band scaling doing it, and
  that gets said.)
- **G2b** D >= B.  Weak prediction.
- **G2c** E > D: backprop still wins on full data.  Expected; this is the honest reference.
- **G2d (the one that would matter)** 10 examples per class, ridge readout, mean of 5 draws:
  Sihti codes beat raw pixels by >= 2 points.  PASS/FAIL on both datasets.

## G3 — generation
100 samples per class, modes `gauss` and `memory`, tau = 0.7.
Judge = the backprop MLP (E), trained only on real images, never on samples.
- **G3a recognisable**: fraction of samples the judge labels as the intended class >= 0.80.  PASS/FAIL per mode.
- **G3b not copies**: novelty ratio = median(NN pixel distance sample -> same-class train) /
  median(NN distance real test image -> same-class train).  PASS if >= 0.5.
  Copy rate = fraction of samples whose NN distance < 0.1 x the test median; reported.
- Predictions: memory mode more recognisable than gauss; memory mode lower novelty ratio.

## G4 — completion (the residues filled in from the coarse state)
2000 test images.  Given only the core (keep=1) or core+r3 (keep=2), fill the finer bands with:
zero (= the blurred image itself), gauss conditional mean (class-agnostic),
gauss conditional mean with the class read from the given coarse bands, memory read.
Score: PSNR of the full image vs truth, and judge accuracy on the completed image.
- **G4a**: class-agnostic gauss mean beats zero-fill by >= 1 dB at keep=1.  PASS/FAIL.
- Prediction: memory read has lower PSNR than the gauss mean (the mean is MSE-optimal)
  but equal or higher judge accuracy.

## Kill conditions stated in advance
- If G2d fails on both datasets, the sieve does not help reading classes; the README says
  so in the first paragraph of results.
- If G3b fails for memory mode, memory mode is called a lookup table, not a generator.
