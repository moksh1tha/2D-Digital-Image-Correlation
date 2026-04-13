"""
DIC Validation on Synthetic Speckle Images.
Generates reference + deformed images with known displacements,
runs the full DIC pipeline, and reports accuracy.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic import (
    generate_speckle_image, apply_displacement, add_noise,
    uniform_translation, uniform_strain, sinusoidal_displacement,
    tensile_test_field,
)
from src.solver import DICSolver
from src.strain import compute_strain_field, compute_principal_strains
from src.visualize import (
    plot_displacement_field, plot_strain_field, plot_zncc_map,
    plot_validation_error, plot_speckle_with_roi,
)


def validate_translation(u_true=3.45, v_true=1.78, noise_sigma=3.0):
    """Test 1: Sub-pixel rigid body translation."""
    print("=" * 60)
    print(f"TEST 1: Rigid Body Translation (u={u_true}, v={v_true})")
    print("=" * 60)

    ref = generate_speckle_image(256, 256, n_speckles=1500, seed=42)
    u_func, v_func = uniform_translation(u_true, v_true)
    deformed = apply_displacement(ref, u_func, v_func)
    deformed = add_noise(deformed, sigma=noise_sigma, seed=99)

    solver = DICSolver(ref, subset_size=21, step=10, search_radius=10)
    roi = (30, 30, 226, 226)
    results = solver.analyze(deformed, roi=roi)

    u_meas = results['u']
    v_meas = results['v']
    valid = ~np.isnan(u_meas)

    u_err = u_meas[valid] - u_true
    v_err = v_meas[valid] - v_true

    print(f"\nResults ({np.sum(valid)} valid points):")
    print(f"  U: mean={np.mean(u_meas[valid]):.4f}, "
          f"bias={np.mean(u_err):.4f}, std={np.std(u_err):.4f}")
    print(f"  V: mean={np.mean(v_meas[valid]):.4f}, "
          f"bias={np.mean(v_err):.4f}, std={np.std(v_err):.4f}")
    print(f"  RMSE_u={np.sqrt(np.mean(u_err**2)):.4f} px")
    print(f"  RMSE_v={np.sqrt(np.mean(v_err**2)):.4f} px")

    plot_displacement_field(results, title=f'Translation Test (u={u_true}, v={v_true})')
    plot_zncc_map(results)

    return results


def validate_strain(exx=0.005, eyy=-0.0015, noise_sigma=3.0):
    """Test 2: Uniform strain field."""
    print("=" * 60)
    print(f"TEST 2: Uniform Strain (exx={exx}, eyy={eyy})")
    print("=" * 60)

    ref = generate_speckle_image(256, 256, n_speckles=1500, seed=42)
    u_func, v_func = uniform_strain(exx=exx, eyy=eyy, cx=128, cy=128)
    deformed = apply_displacement(ref, u_func, v_func)
    deformed = add_noise(deformed, sigma=noise_sigma, seed=99)

    solver = DICSolver(ref, subset_size=21, step=7, search_radius=10)
    roi = (30, 30, 226, 226)
    results = solver.analyze(deformed, roi=roi)

    # Compute strains
    exx_m, eyy_m, exy_m = compute_strain_field(
        results['grid_x'], results['grid_y'],
        results['u'], results['v'],
        window_size=5
    )

    valid = ~np.isnan(exx_m)
    print(f"\nStrain results ({np.sum(valid)} valid points):")
    print(f"  exx: mean={np.nanmean(exx_m):.6f} (true={exx}), "
          f"std={np.nanstd(exx_m):.6f}")
    print(f"  eyy: mean={np.nanmean(eyy_m):.6f} (true={eyy}), "
          f"std={np.nanstd(eyy_m):.6f}")
    print(f"  exy: mean={np.nanmean(exy_m):.6f} (true=0.0), "
          f"std={np.nanstd(exy_m):.6f}")

    # Compute ground truth displacement at grid points
    gx = results['grid_x']
    gy = results['grid_y']
    u_true = exx * (gx - 128)
    v_true = eyy * (gy - 128)

    plot_validation_error(u_true, results['u'], v_true, results['v'], gx, gy)
    plot_strain_field(gx, gy, exx_m, eyy_m, exy_m,
                      title=f'Strain Field (true: exx={exx}, eyy={eyy})')

    return results


def validate_sinusoidal(amplitude=2.0, wavelength=128.0, noise_sigma=3.0):
    """Test 3: Sinusoidal displacement field."""
    print("=" * 60)
    print(f"TEST 3: Sinusoidal (amp={amplitude} px, lambda={wavelength} px)")
    print("=" * 60)

    ref = generate_speckle_image(256, 256, n_speckles=1500, seed=42)
    u_func, v_func = sinusoidal_displacement(amplitude, wavelength)
    deformed = apply_displacement(ref, u_func, v_func)
    deformed = add_noise(deformed, sigma=noise_sigma, seed=99)

    solver = DICSolver(ref, subset_size=21, step=5, search_radius=10)
    roi = (30, 30, 226, 226)
    results = solver.analyze(deformed, roi=roi)

    # Ground truth
    gx = results['grid_x']
    gy = results['grid_y']
    u_true = amplitude * np.sin(2 * np.pi * gx / wavelength)
    v_true = np.zeros_like(gx)

    valid = ~np.isnan(results['u'])
    err_u = results['u'][valid] - u_true[valid]
    print(f"\nResults ({np.sum(valid)} valid points):")
    print(f"  RMSE_u = {np.sqrt(np.mean(err_u**2)):.4f} px")

    plot_validation_error(u_true, results['u'], v_true, results['v'], gx, gy)

    # Strain from sinusoidal: du/dx = amplitude * 2*pi/wavelength * cos(...)
    exx_m, eyy_m, exy_m = compute_strain_field(
        gx, gy, results['u'], results['v'], window_size=5
    )
    exx_true = amplitude * (2 * np.pi / wavelength) * np.cos(2 * np.pi * gx / wavelength)
    plot_strain_field(gx, gy, exx_m, eyy_m, exy_m,
                      title='Strain Field (sinusoidal displacement)')

    return results


def validate_tensile(strain_rate=0.005, noise_sigma=3.0):
    """Test 4: Simulated tensile test."""
    print("=" * 60)
    print(f"TEST 4: Simulated Tensile Test (e={strain_rate})")
    print("=" * 60)

    ref = generate_speckle_image(256, 256, n_speckles=1500, seed=42)
    u_func, v_func = tensile_test_field(strain_rate, cx=128, cy=128)
    deformed = apply_displacement(ref, u_func, v_func)
    deformed = add_noise(deformed, sigma=noise_sigma, seed=99)

    solver = DICSolver(ref, subset_size=21, step=7, search_radius=10)
    roi = (30, 30, 226, 226)
    results = solver.analyze(deformed, roi=roi)

    exx_m, eyy_m, exy_m = compute_strain_field(
        results['grid_x'], results['grid_y'],
        results['u'], results['v'],
        window_size=5
    )

    print(f"\nStrain results:")
    print(f"  exx: mean={np.nanmean(exx_m):.6f} (true={strain_rate})")
    print(f"  eyy: mean={np.nanmean(eyy_m):.6f} (true={-0.3*strain_rate:.6f})")
    print(f"  exy: mean={np.nanmean(exy_m):.6f} (true=0.0)")

    e1, e2, gmax = compute_principal_strains(exx_m, eyy_m, exy_m)

    plot_displacement_field(results, title='Simulated Tensile Test')
    plot_strain_field(results['grid_x'], results['grid_y'],
                      exx_m, eyy_m, exy_m,
                      title='Strain Field (simulated tensile)')

    return results


if __name__ == '__main__':
    print("DIC VALIDATION SUITE")
    print("=" * 60)
    print("Running synthetic validation tests...\n")

    validate_translation()
    validate_strain()
    validate_sinusoidal()
    validate_tensile()

    print("\n" + "=" * 60)
    print("All validation tests complete.")
