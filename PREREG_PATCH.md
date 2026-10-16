# Pre-registration — patch residue memory

Written and committed before the patch code was run on any data.

## What changes
The whole-image memory addresses with the whole coarse image and stores whole-image residue codes.
The patch memory addresses with a **5x5 window of the coarser reconstruction** around a location
(sampled at stride 4 for r3, 2 for r2, 1 for r1 and r0) and predicts the **4x4 residue patch** there.
Patches sit on a stride-2 grid and overlap; overlaps are averaged. Each window is contrast-normalised
(mean removed, divided by its std + 0.02); the prediction is scaled back by the same std.
Completion runs coarse to fine: each predicted band joins the coarser reconstruction for the next.
No class information is used anywhere.

| arm | what it is | size |
|---|---|---|
| P0 | one linear map from the raw window to the patch (a learned convolution / deconvolution) | 26x16 per band |
| P1 | k-NN over 50 000 stored (window, patch) pairs, top-8 blend | grows with stored pairs |
| P2 | K = 512 k-means codebook on windows, each code returns its mean patch | fixed |
| **P3** | same codebook, each code a local linear map (window -> patch) | fixed, main arm |

Fit on training images only (MNIST / Fashion train; BSDS500 train split).

## Gates
- **PG1 completion, standard test** (2000 test images, given core+r3): report PSNR and judge accuracy
  for P0-P3 against the whole-image memory and whole-image linear fill.  PASS if P3 >= whole-image
  linear fill.  Prediction: from core only (keep=1) the patch memory will LOSE to whole-image memory,
  because a 16-number core has no local structure to address with; reported, not gated.
- **PG2 shift (the bird moves along the branch)**: same 2000 test images shifted by a random
  integer (dx, dy) in [-4, 4], not both 0, zero fill.  PASS if (a) P3 loses < 0.5 dB PSNR vs
  unshifted AND (b) P3 beats whole-image memory on the shifted set by >= 1 dB.  Given core+r3.
- **PG3 natural photos**: BSDS500, greyscale, half size. Codebooks fit on the 200 train images, tested
  on the 300 val+test images (the hawk on the branch, 42049, and the ground bird, 8023, are in val).
  Given core+r3, fill r2, r1, r0.  PASS if P3 > P0 in mean PSNR.  The whole-image model cannot run
  here at all (different image sizes); that is part of the point.
- **PG4 memory size**: P3 total stored numbers, all bands, <= 1M and independent of training-set size.
- **PG5 defects** (Fashion): 1000 test images, a synthetic scratch (line, length 6-14 px, +/-0.5) on a
  random half.  Anomaly map = |actual (r1+r0) - P3 prediction of (r1+r0) from core+r3+r2|, 3x3 mean.
  Baseline map = |actual (r1+r0)| (plain high-pass energy), and the whole-image linear fill as a
  second baseline.  PASS if P3 pixel AUROC beats the plain high-pass baseline by >= 0.05.
  Image-level AUROC (max of map) reported.

## Kill conditions
- If PG2 fails, the patch memory has not fixed the thing it exists to fix; the README says so first.
- If PG3 fails (P3 no better than a linear filter on real photos), the "bird fix" is not shown.
