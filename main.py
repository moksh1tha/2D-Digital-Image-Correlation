"""
Main entry point for the 2D Digital Image Correlation project.
Runs both synthetic validation and real data analysis.

Usage:
    python main.py --validate       Run synthetic DIC validation
    python main.py --track          Run marker tracking on real data
    python main.py --all            Run everything
    python main.py --quick          Quick test (small dataset, few frames)
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def run_quick_test():
    """Quick smoke test of the full pipeline."""
    print("=" * 60)
    print("QUICK TEST: Verifying DIC pipeline")
    print("=" * 60)

    from src.synthetic import generate_speckle_image, apply_displacement, uniform_translation
    from src.solver import DICSolver
    from src.strain import compute_strain_field

    # Generate small test images
    print("\n1. Generating 128x128 synthetic speckle images...")
    ref = generate_speckle_image(128, 128, n_speckles=500, seed=42)

    u_true, v_true = 2.35, -1.67
    u_func, v_func = uniform_translation(u_true, v_true)
    deformed = apply_displacement(ref, u_func, v_func)

    print(f"   Applied translation: u={u_true}, v={v_true}")

    # Run DIC
    print("\n2. Running DIC analysis...")
    solver = DICSolver(ref, subset_size=15, step=10, search_radius=8)
    results = solver.analyze(deformed, roi=(20, 20, 108, 108), verbose=True)

    # Check results
    valid = ~np.isnan(results['u'])
    if np.sum(valid) > 0:
        u_mean = np.nanmean(results['u'])
        v_mean = np.nanmean(results['v'])
        u_err = abs(u_mean - u_true)
        v_err = abs(v_mean - v_true)

        print(f"\n3. Results:")
        print(f"   Measured: u={u_mean:.4f}, v={v_mean:.4f}")
        print(f"   Error:    u_err={u_err:.4f} px, v_err={v_err:.4f} px")
        print(f"   Mean ZNCC: {np.nanmean(results['zncc']):.4f}")

        # Strain (should be ~0 for translation)
        exx, eyy, exy = compute_strain_field(
            results['grid_x'], results['grid_y'],
            results['u'], results['v'], window_size=3
        )
        print(f"   Strain (should be ~0): exx={np.nanmean(exx):.6f}, "
              f"eyy={np.nanmean(eyy):.6f}")

        if u_err < 0.1 and v_err < 0.1:
            print("\n   PASS: Sub-pixel accuracy achieved.")
        else:
            print("\n   WARNING: Error exceeds 0.1 px threshold.")
    else:
        print("\n   FAIL: No valid points computed.")

    return results


def run_validation():
    """Run full synthetic validation suite."""
    from run_validation import (
        validate_translation, validate_strain,
        validate_sinusoidal, validate_tensile
    )
    validate_translation()
    validate_strain()
    validate_sinusoidal()
    validate_tensile()


def run_tracking():
    """Run marker tracking on real data."""
    from run_tracker import analyze_dataset, IMAGE_DIRS, FRAME_STEP

    for name, img_dir in IMAGE_DIRS.items():
        if os.path.exists(img_dir):
            analyze_dataset(name, img_dir, frame_step=FRAME_STEP, max_frames=2000)
        else:
            print(f"Directory not found: {img_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='2D Digital Image Correlation Engine'
    )
    parser.add_argument('--validate', action='store_true',
                        help='Run synthetic DIC validation')
    parser.add_argument('--track', action='store_true',
                        help='Run marker tracking on real data')
    parser.add_argument('--all', action='store_true',
                        help='Run everything')
    parser.add_argument('--quick', action='store_true',
                        help='Quick pipeline test')

    args = parser.parse_args()

    if not any([args.validate, args.track, args.all, args.quick]):
        args.quick = True  # Default to quick test

    if args.quick or args.all:
        run_quick_test()

    if args.validate or args.all:
        run_validation()

    if args.track or args.all:
        run_tracking()


if __name__ == '__main__':
    main()
