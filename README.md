# 2D Digital Image Correlation (DIC) Engine

A custom implementation of 2D Digital Image Correlation for full-field displacement and strain measurement, built entirely in **Python + NumPy**.

Based on the framework described in:
> Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). *Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review.* Measurement Science and Technology, 20(6), 062001.

## Features

- **IC-GN Sub-pixel Solver** — Forward Additive Gauss-Newton with pre-computed reference Hessian (Vendroux & Knauss 1998). First-order shape functions (6 DOF: translation, rotation, shear, normal strain).
- **ZNSSD Correlation Criterion** — Zero-mean Normalized Sum of Squared Differences. Invariant to affine lighting transforms (offset + linear scale).
- **Bicubic Interpolation** — Keys' convolution kernel (a = -0.5, Catmull-Rom) with analytical gradient computation. Eliminates pixel-locking artifacts.
- **Strain Field Estimation** — Pointwise local least-squares fitting over a configurable window. Outputs exx, eyy, exy, principal strains, and von Mises strain.
- **Binary Marker Tracker** — Adaptive thresholding + connected component labeling for tracking gauge markers in tensile test imagery. Computes engineering strain from Euclidean centroid distances.
- **Synthetic Validation Suite** — Speckle pattern generator with configurable displacement fields (translation, uniform strain, sinusoidal, simulated tensile test). Reports RMSE against ground truth.

## Dependencies

- Python 3.8+
- NumPy
- Pillow (image I/O only)
- Matplotlib (visualization only)

```bash
pip install numpy pillow matplotlib
```

## Project Structure

```
├── src/
│   ├── interpolation.py    # Bicubic interpolation (Keys kernel)
│   ├── solver.py           # DIC solver (IC-GN + ZNSSD)
│   ├── strain.py           # Strain estimation (local least-squares)
│   ├── tracker.py          # Binary marker tracker
│   ├── synthetic.py        # Synthetic speckle image generator
│   ├── visualize.py        # Plotting utilities
│   └── utils.py            # Image loading, gradients, timestamp parsing
├── main.py                 # Entry point
├── run_validation.py       # Synthetic validation tests
└── run_tracker.py          # Real data marker tracking
```

## Usage

### Quick Test
```bash
python3 main.py --quick
```
Generates a 128x128 synthetic speckle pair with known sub-pixel translation, runs the full DIC pipeline, and reports displacement error.

### Synthetic Validation
```bash
python3 main.py --validate
```
Runs four validation tests with increasing complexity:

| Test | Description | Expected Accuracy |
|------|-------------|-------------------|
| Translation | Rigid body shift (u=3.45, v=1.78 px) | RMSE < 0.02 px |
| Uniform Strain | exx=0.005, eyy=-0.0015 | < 3% relative error |
| Sinusoidal | Spatially varying u(x) = 2·sin(2πx/128) | RMSE < 0.1 px |
| Tensile | Simulated uniaxial test with Poisson contraction | Strain within 3% |

### Marker Tracking (Real Data)
```bash
python3 main.py --track
```
Tracks black gauge markers across tensile test image sequences. Expects image directories under `Images/`. Outputs gauge length evolution and engineering strain curves.

### Run Everything
```bash
python3 main.py --all
```

## Algorithm Pipeline

```
Reference Image ──┐
                   ├─→ ROI Grid ──→ Integer Search (ZNCC) ──→ IC-GN Sub-pixel ──→ Displacement Field
Deformed Image  ──┘                                            (ZNSSD + Bicubic)        │
                                                                                         ▼
                                                                              Local Least-Squares
                                                                                         │
                                                                                         ▼
                                                                                   Strain Field
```

1. **ROI & Grid** — User defines region of interest; grid points are placed at regular intervals.
2. **Integer Search** — Exhaustive ZNCC search within a configurable pixel radius for initial guess.
3. **IC-GN Refinement** — Iterative optimization of 6 shape function parameters to sub-pixel accuracy. Hessian pre-computed from reference image gradients.
4. **Strain Estimation** — Pointwise local least-squares fits a linear displacement plane over a (2m+1)×(2m+1) window. Strains are the fitted gradients.

## Configuration

Key parameters in `DICSolver`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subset_size` | 21 | Subset side length (pixels, must be odd) |
| `step` | 5 | Grid spacing between calculation points |
| `search_radius` | 20 | Integer search range (pixels) |
| `max_iter` | 50 | Maximum IC-GN iterations |
| `conv_threshold` | 1e-3 | Convergence criterion for parameter update norm |

## References

- Pan, B. et al. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. *Meas. Sci. Technol.* 20, 062001.
- Vendroux, G. & Knauss, W.G. (1998). Submicron deformation field measurements: Part 2. Improved digital image correlation. *Exp. Mech.* 38, 86–92.
- Bruck, H.A. et al. (1989). Digital image correlation using Newton–Raphson method of partial differential correction. *Exp. Mech.* 29, 261–7.
