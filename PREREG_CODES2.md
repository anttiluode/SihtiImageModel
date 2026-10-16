# Pre-registration — codes of codes (a nonlinear coarse scale)

Committed before the code was written or run.

## Why
The patch memory P3 is nonlinear only inside 5x5 windows; information crosses the image only through
the blur, which is linear. UG3 failed (local strokes confident, whole digit wrong) and the CIFAR
whole-image model failed. Both point at the same gap: no nonlinearity at the scale where the object
is decided.

## The model (arm Q)
For each predicted band b (coarse -> fine), on the coarser reconstruction C_b:
1. **Level 1:** every 5x5 window (same sampling as P3) is contrast-normalised and assigned to one of
   K1 = 64 codes (k-means). This turns the coarse image into a *code map*.
2. **Context:** at each place, the one-hot codes of the 3x3 neighbouring places 2 grid steps apart
   (4 px) are concatenated (9 x 64 = 576), projected by PCA to 32 numbers (z), scaled so z carries
   the same total variance as the normalised window n.
3. **Level 2:** route by k-means (K = 512) on [n ; z]; each region holds a linear map
   [n ; z ; 1] -> 4x4 residue patch, scaled back by the window contrast.

The level-2 routing sees the codes of a neighbourhood about 3x wider than P3's window, so the choice of
expert depends nonlinearly on a wider region. All fitting is closed form; no class is used.
Same training data, sampling, test sets and seeds as the P3 gates.

## Gates
- **CG1 fill.** PSNR of Q vs P3 on 2000 test images (seed 4). PASS if Q >= P3 + 0.3 dB at keep 1 (core only)
  on BOTH MNIST and Fashion. Prediction: the gain is larger at keep 1 than at keep 2.
- **CG2 fewer wrong readings.** On the UG3 set (1000 test images, judge reads the real image correctly),
  judge flips of the keep-1 fill. PASS if Q's flips <= 0.8 x P3's flips on BOTH datasets.
- **CG3 knowing when the reading is ambiguous.** Per image, the disagreement between the Q fill and the
  P3 fill (mean squared difference). PASS if it predicts Q's keep-1 flips with AUROC >= 0.75 on BOTH.
  Q's own code-spread uncertainty is reported alongside.
- **CG4 still moves with the image.** Cost of a random 1-4 px shift for Q, keep 2, MNIST: < 0.5 dB.
- **Reported, not gated here:** BSDS500 photos (60 held out, keep 2), Q vs P3.
- **CG5 CIFAR-10 (run on Antti's machine, data not reachable here):** per-channel Q and P3 fills on the
  same 2000 test images as G4. PASS if Q beats the whole-image linear fill from his run
  (17.56 dB keep 1, 19.29 dB keep 2).

## Kill condition
If CG1 and CG2 both fail, codes of codes did not put useful nonlinearity at the coarse scale, and the
"in-between AI" stops at early vision. The README says so first.
