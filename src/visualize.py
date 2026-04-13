"""
Visualization module for DIC results.
Displacement vector fields, strain heatmaps, marker tracking plots.
"""

import numpy as np


def plot_displacement_field(results, scale=1.0, title='Displacement Field',
                            save_path=None):
    """
    Plot displacement vector field (quiver plot).
    """
    import matplotlib.pyplot as plt

    gx = results['grid_x']
    gy = results['grid_y']
    u = results['u']
    v = results['v']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # U displacement heatmap
    im0 = axes[0].pcolormesh(gx, gy, u, cmap='RdBu_r', shading='auto')
    axes[0].set_title('U displacement (pixels)')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    plt.colorbar(im0, ax=axes[0])

    # V displacement heatmap
    im1 = axes[1].pcolormesh(gx, gy, v, cmap='RdBu_r', shading='auto')
    axes[1].set_title('V displacement (pixels)')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    plt.colorbar(im1, ax=axes[1])

    # Vector field
    # Subsample for readability
    step = max(1, min(gx.shape[0], gx.shape[1]) // 20)
    axes[2].quiver(gx[::step, ::step], gy[::step, ::step],
                   u[::step, ::step], v[::step, ::step],
                   scale=scale, angles='xy')
    axes[2].set_title('Displacement vectors')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_strain_field(grid_x, grid_y, exx, eyy, exy, title='Strain Field',
                      save_path=None):
    """
    Plot strain component heatmaps.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fields = [exx, eyy, exy]
    titles = [r'$\varepsilon_{xx}$', r'$\varepsilon_{yy}$', r'$\varepsilon_{xy}$']

    for ax, field, t in zip(axes, fields, titles):
        im = ax.pcolormesh(grid_x, grid_y, field, cmap='jet', shading='auto')
        ax.set_title(t, fontsize=14)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, format='%.4f')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_zncc_map(results, save_path=None):
    """Plot ZNCC correlation coefficient map."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(results['grid_x'], results['grid_y'],
                       results['zncc'], cmap='viridis', shading='auto',
                       vmin=0.8, vmax=1.0)
    ax.set_title('ZNCC Correlation Coefficient')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_marker_tracking(results, pairs=None, save_path=None):
    """
    Plot marker tracking results: gauge length and strain over time.

    Parameters
    ----------
    results : dict from MarkerTracker.run()
    pairs : list of pair keys like ['0-2', '1-3'], or None for all
    """
    import matplotlib.pyplot as plt

    frames = results['frame_indices']
    gauge = results['gauge_lengths']

    if pairs is None:
        pairs = list(gauge.keys())

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for key in pairs:
        d = gauge[key]
        d0 = d[0]
        if np.isnan(d0):
            continue
        strain = (d - d0) / d0

        axes[0].plot(frames, d, label=f'Pair {key}')
        axes[1].plot(frames, strain * 100, label=f'Pair {key}')

    axes[0].set_ylabel('Gauge Length (pixels)')
    axes[0].set_title('Gauge Length vs Frame')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Engineering Strain (%)')
    axes[1].set_title('Engineering Strain vs Frame')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_speckle_with_roi(image, roi=None, save_path=None):
    """Display image with optional ROI rectangle overlay."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    if roi is not None:
        x0, y0, x1, y1 = roi
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_title('Image with ROI')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_validation_error(true_u, measured_u, true_v, measured_v,
                          grid_x, grid_y, save_path=None):
    """Plot error maps for synthetic validation."""
    import matplotlib.pyplot as plt

    err_u = measured_u - true_u
    err_v = measured_v - true_v

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im00 = axes[0, 0].pcolormesh(grid_x, grid_y, true_u, cmap='RdBu_r', shading='auto')
    axes[0, 0].set_title('True U'); plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].pcolormesh(grid_x, grid_y, measured_u, cmap='RdBu_r', shading='auto')
    axes[0, 1].set_title('Measured U'); plt.colorbar(im01, ax=axes[0, 1])

    im10 = axes[1, 0].pcolormesh(grid_x, grid_y, err_u, cmap='RdBu_r', shading='auto')
    axes[1, 0].set_title(f'Error U (RMSE={np.nanstd(err_u):.4f} px)')
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].pcolormesh(grid_x, grid_y, err_v, cmap='RdBu_r', shading='auto')
    axes[1, 1].set_title(f'Error V (RMSE={np.nanstd(err_v):.4f} px)')
    plt.colorbar(im11, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.invert_yaxis()

    fig.suptitle('DIC Validation: Ground Truth vs Measured', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
