# Pre-registration — does the patch memory know where it is guessing?

Committed before any uncertainty code was run on data.

## The maps
Fill = patch memory P3 (the cached models from PREREG_PATCH.md, unchanged).

- **U1 code spread.** For every code k of every band, the variance of the training residuals that
  code leaves behind (measured on a fresh draw of training windows). At test time each patch gets
  the variance of its code, scaled by its window contrast squared, overlap-averaged like the fill,
  summed over the predicted bands.  Main map.
- **U2 neighbour disagreement.** Spread of the 8 stored nearest examples' patches (P1 bank),
  same scaling and overlap.  Digits and clothing only (too slow for the photos).
- **U3 edge energy (baseline).** 5x5 average of the squared gradient of the given coarse image.
  If U1 cannot beat this, it only knows "errors happen at edges".

Truth signal: squared error of the P3 fill against the real image, per pixel.
Test sets: 1000 MNIST and 1000 Fashion test images (seed 4), 60 held-out BSDS photos.
Calibration (one global scale for U1, U2) fit on 1000 *training* images, never on test.
Given = core + r3 (keep 2) for the pixel gates; keep 1 and keep 2 for the flip gate.

## Gates
- **UG1 ranking.** Sparsification: drop pixels in order of predicted uncertainty and track the
  MSE of what remains; AUSE = mean gap to the oracle curve (dropping by true error), divided by
  full MSE. PASS per dataset if U1's AUSE is at least 10% lower than U3's.
- **UG2 calibration.** After the single training-set scale, the share of test pixels whose error
  lies within 2 predicted standard deviations is between 0.90 and 0.98. PASS per dataset.
- **UG3 confidently wrong.** Among test images the judge reads correctly when real, flag those
  whose P3 fill the judge reads as a different class. Score each image by its mean U1.
  PASS if AUROC >= 0.75 for flips at keep 1, on digits and on clothing. U3 reported alongside.

## Kill condition
If U1 fails UG1 on all three datasets, the memory has no usable self-knowledge beyond edge energy,
and the README says so.
