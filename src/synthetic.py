"""
Synthetic speckle image generator for DIC validation.
Generates reference + deformed image pairs with known ground truth.
Pure Python + NumPy implementation.
"""

import numpy as np


def generate_speckle_image(height=512, width=512, n_speckles=3000,
                           speckle_radius_range=(2, 5), seed=42):
    """
    Generate a synthetic speckle pattern image.

    Parameters
    ----------
    height, width : int
    n_speckles : int
        Number of speckle dots.
    speckle_radius_range : tuple (min, max)
        Range of speckle radii in pixels.
    seed : int

    Returns
    -------
    image : 2D ndarray (float64), values in [0, 255]
    """
    rng = np.random.RandomState(seed)
    image = np.ones((height, width), dtype=np.float64) * 220.0  # Light background

    # Add subtle background noise
    image += rng.normal(0, 5, (height, width))

    yy, xx = np.mgrid[0:height, 0:width]

    for _ in range(n_speckles):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        r = rng.uniform(*speckle_radius_range)
        intensity = rng.uniform(10, 80)  # Dark speckles

        dist2 = (xx - cx)**2 + (yy - cy)**2
        mask = dist2 < r**2
        # Gaussian-ish profile for smooth speckles
        profile = intensity * np.exp(-dist2 / (2 * (r / 2)**2))
        image = np.where(mask, np.minimum(image, 220 - profile), image)

    image = np.clip(image, 0, 255)
    return image


def apply_displacement(image, u_func, v_func):
    """
    Apply a known displacement field to a reference image.
    Uses backward mapping with bicubic interpolation.

    Parameters
    ----------
    image : 2D ndarray (reference image)
    u_func : callable(x, y) -> u displacement
    v_func : callable(x, y) -> v displacement

    Returns
    -------
    deformed : 2D ndarray
    """
    from .interpolation import BicubicInterpolator
    h, w = image.shape
    interp = BicubicInterpolator(image)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Backward mapping: for each deformed pixel, find reference location
    u_vals = u_func(xx, yy)
    v_vals = v_func(xx, yy)

    # Source coordinates in reference image
    x_src = xx - u_vals
    y_src = yy - v_vals

    # Clip to valid range
    x_src = np.clip(x_src, 0, w - 1.01)
    y_src = np.clip(y_src, 0, h - 1.01)

    deformed = interp.evaluate(y_src, x_src)
    deformed = np.clip(deformed, 0, 255)

    return deformed


def add_noise(image, sigma=3.0, seed=None):
    """Add Gaussian noise to an image."""
    rng = np.random.RandomState(seed)
    noisy = image + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255)


# --- Pre-defined displacement fields for validation ---

def uniform_translation(u_val, v_val):
    """Rigid body translation."""
    def u_func(x, y):
        return np.full_like(x, u_val)
    def v_func(x, y):
        return np.full_like(x, v_val)
    return u_func, v_func


def uniform_strain(exx=0.0, eyy=0.0, exy=0.0, cx=256, cy=256):
    """
    Uniform strain field centered at (cx, cy).
    u = exx*(x-cx) + exy*(y-cy)
    v = exy*(x-cx) + eyy*(y-cy)
    """
    def u_func(x, y):
        return exx * (x - cx) + exy * (y - cy)
    def v_func(x, y):
        return exy * (x - cx) + eyy * (y - cy)
    return u_func, v_func


def sinusoidal_displacement(amplitude=2.0, wavelength=128.0):
    """Sinusoidal displacement in x-direction."""
    def u_func(x, y):
        return amplitude * np.sin(2.0 * np.pi * x / wavelength)
    def v_func(x, y):
        return np.zeros_like(x)
    return u_func, v_func


def tensile_test_field(strain_rate=0.005, cx=256, cy=256):
    """
    Simulated uniaxial tensile test (x-direction).
    u = strain_rate * (x - cx)
    v = -0.3 * strain_rate * (y - cy)   (Poisson's ratio ~ 0.3)
    """
    def u_func(x, y):
        return strain_rate * (x - cx)
    def v_func(x, y):
        return -0.3 * strain_rate * (y - cy)
    return u_func, v_func
