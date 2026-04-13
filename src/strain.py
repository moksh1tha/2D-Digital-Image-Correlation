"""
Strain field estimation from DIC displacement fields.
Uses pointwise local least-squares fitting (Pan et al. 2009, Section 4).
Pure NumPy implementation.
"""

import numpy as np


def compute_strain_field(grid_x, grid_y, u, v, window_size=5):
    """
    Compute full-field strain from displacement fields using
    pointwise local least-squares fitting.

    For each point, a (2m+1)x(2m+1) window of displacement data is
    fit to a linear plane:
        u(x,y) = a0 + a1*x + a2*y
        v(x,y) = b0 + b1*x + b2*y

    Strains are then:
        exx = a1 = du/dx
        eyy = b2 = dv/dy
        exy = 0.5*(a2 + b1) = 0.5*(du/dy + dv/dx)

    Parameters
    ----------
    grid_x, grid_y : 2D arrays
        Coordinates of grid points.
    u, v : 2D arrays
        Displacement fields (may contain NaN for invalid points).
    window_size : int
        Side length of strain calculation window (must be odd).

    Returns
    -------
    exx, eyy, exy : 2D arrays
        Strain components. NaN where computation is not possible.
    """
    ny, nx = u.shape
    m = window_size // 2

    exx = np.full((ny, nx), np.nan)
    eyy = np.full((ny, nx), np.nan)
    exy = np.full((ny, nx), np.nan)

    # Grid spacing (assume uniform)
    if nx > 1:
        dx_step = grid_x[0, 1] - grid_x[0, 0]
    else:
        dx_step = 1.0
    if ny > 1:
        dy_step = grid_y[1, 0] - grid_y[0, 0]
    else:
        dy_step = 1.0

    for iy in range(ny):
        for ix in range(nx):
            if np.isnan(u[iy, ix]):
                continue

            # Window bounds (clamp to array edges)
            iy_min = max(0, iy - m)
            iy_max = min(ny - 1, iy + m)
            ix_min = max(0, ix - m)
            ix_max = min(nx - 1, ix + m)

            # Gather valid points in window
            local_x = []
            local_y = []
            local_u = []
            local_v = []

            for jy in range(iy_min, iy_max + 1):
                for jx in range(ix_min, ix_max + 1):
                    if not np.isnan(u[jy, jx]):
                        # Local coordinates relative to center point
                        local_x.append((jx - ix) * dx_step)
                        local_y.append((jy - iy) * dy_step)
                        local_u.append(u[jy, jx])
                        local_v.append(v[jy, jx])

            n_valid = len(local_u)
            if n_valid < 3:
                continue  # Need at least 3 points for 3 unknowns

            # Build system: [1, x, y] @ [a0, a1, a2]^T = u
            A = np.column_stack([
                np.ones(n_valid),
                np.array(local_x),
                np.array(local_y)
            ])

            u_vec = np.array(local_u)
            v_vec = np.array(local_v)

            # Solve via least-squares: A^T A @ coeffs = A^T b
            ATA = A.T @ A
            try:
                ATA_inv = np.linalg.inv(ATA)
            except np.linalg.LinAlgError:
                continue

            u_coeffs = ATA_inv @ (A.T @ u_vec)  # [a0, a1, a2]
            v_coeffs = ATA_inv @ (A.T @ v_vec)  # [b0, b1, b2]

            exx[iy, ix] = u_coeffs[1]       # du/dx
            eyy[iy, ix] = v_coeffs[2]       # dv/dy
            exy[iy, ix] = 0.5 * (u_coeffs[2] + v_coeffs[1])  # 0.5*(du/dy + dv/dx)

    return exx, eyy, exy


def compute_principal_strains(exx, eyy, exy):
    """
    Compute principal strains and maximum shear strain.

    Returns
    -------
    e1, e2 : 2D arrays (major, minor principal strains)
    gamma_max : 2D array (maximum shear strain)
    """
    avg = (exx + eyy) / 2.0
    diff = (exx - eyy) / 2.0
    R = np.sqrt(diff**2 + exy**2)

    e1 = avg + R
    e2 = avg - R
    gamma_max = 2.0 * R

    return e1, e2, gamma_max


def compute_von_mises_strain(exx, eyy, exy):
    """Compute equivalent von Mises strain."""
    return np.sqrt(exx**2 + eyy**2 - exx*eyy + 3.0*exy**2) * (2.0/3.0)**0.5
