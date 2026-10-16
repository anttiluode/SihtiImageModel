# Pre-registration — does global context help the local experts?

Committed before the code was written or run. Design follows the external review of the codes-of-codes
run: isolate access to global relationships before adding any class-specific memory.

## The global summary
g = the 16 PCA coefficients of the given core band (the whole-image model's core vocabulary, fit on
training images). Unnormalised per window: the same 16 numbers for every place in an image. For keep 2
the given observation also includes r3; g still uses only the core, a subset of what is permitted.
No labels are used anywhere.

## Arms (all reuse P3's sampling, ridge strength and 4x4 overlap-add; no class information)
- **P3**: the existing patch memory (routing by the normalised window, local linear expert).
- **Q**: codes of codes (existing).
- **G**: P3's routing and codebook unchanged; each expert re-fit as a linear map of [n ; g ; 1].
  The only change from P3 is that the expert can read the global summary.
- **GP**: as G plus the window's position in the frame and position x g interactions
  [n ; g ; pos ; g*pos_x ; g*pos_y ; 1]. Expected to lose shift tolerance; that cost is measured.
- **GR**: routing by k-means on [n ; g] (512 regions), expert linear in [n ; g ; 1].

## Measurements (MNIST and Fashion; same test subsets and seeds as before)
- **Fidelity:** PSNR on the 2000-image set (seed 4), keep 1 (core only) and keep 2 (core + r3).
- **Identity:** judge flips at keep 1 on the 1000-image UG3 set (images the judge reads correctly when real).
- **Individuality:** for each of those 1000 images, the fill is compared with its own true image and 99
  other test images of the same class (fixed draw). Top-1 = the true image is the nearest. Measures
  whether a fill becomes a stereotyped member of its class instead of the particular one.
- **Where the error comes from:** per-band error energy of the predicted bands at keep 1 (r3, r2, r1, r0),
  separating first-stage error from propagation.
- **Shift cost:** keep 2 PSNR loss under a random 1-4 px shift (seed 11).
- **Information ceiling on the SAME 1000 images:** kNN (k = 15) and a backprop MLP reading the class from g.

## Gates
- **GG1 fidelity.** G >= P3 + 0.3 dB at keep 1 on BOTH datasets.
- **GG2 identity.** G's keep-1 flips <= 0.8 x P3's on BOTH datasets.
- **GG3 not stereotyped.** G's individuality top-1 >= P3's minus 2 points on BOTH datasets.
- GP and GR are reported against the same three gates but are not the gated arm.

## Kill condition
If G fails both GG1 and GG2 on both datasets, letting the local experts read the global summary
linearly does not help, and the diagnosis "the information is in the core but unused" stays untested
by this route. The README says so first.
