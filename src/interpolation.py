"""
Bicubic interpolation in pure NumPy.
Uses Keys' convolution kernel (a=-0.5, Catmull-Rom) for C1 continuity.
Provides intensity values and analytical gradients at sub-pixel locations.
"""

import numpy as np


def _cubic_weight(t, a=-0.5):
    """
    Keys' cubic convolution kernel.
    a = -0.5 gives Catmull-Rom (C1 continuous).
    """
    t = np.abs(t)
    out = np.zeros_like(t)

    mask1 = t <= 1.0
    mask2 = (t > 1.0) & (t <= 2.0)

    t1 = t[mask1]
    out[mask1] = (a + 2.0) * t1**3 - (a + 3.0) * t1**2 + 1.0

    t2 = t[mask2]
    out[mask2] = a * t2**3 - 5.0 * a * t2**2 + 8.0 * a * t2 - 4.0 * a

    return out


def _cubic_weight_deriv(t, a=-0.5):
    """
    Derivative of Keys' cubic kernel with respect to t.
    """
    sign = np.sign(t)
    t_abs = np.abs(t)
    out = np.zeros_like(t)

    mask1 = t_abs <= 1.0
    mask2 = (t_abs > 1.0) & (t_abs <= 2.0)

    t1 = t_abs[mask1]
    out[mask1] = sign[mask1] * (3.0 * (a + 2.0) * t1**2 - 2.0 * (a + 3.0) * t1)

    t2 = t_abs[mask2]
    out[mask2] = sign[mask2] * (3.0 * a * t2**2 - 10.0 * a * t2 + 8.0 * a)

    return out


class BicubicInterpolator:
    """
    Precomputes padded image for fast bicubic interpolation.
    Evaluates intensity and gradients at arbitrary sub-pixel coordinates.
    """

    def __init__(self, image):
        """
        Parameters
        ----------
        image : 2D ndarray, shape (H, W)
            Grayscale image (float64).
        """
        self.h, self.w = image.shape
        # Pad by 2 pixels on each side for the 4x4 kernel support
        self.img = np.pad(image, 2, mode='reflect')

    def evaluate(self, y_coords, x_coords):
        """
        Evaluate interpolated intensity at sub-pixel coordinates.

        Parameters
        ----------
        y_coords, x_coords : ndarray (same shape)
            Sub-pixel coordinates in the original (unpadded) image frame.

        Returns
        -------
        values : ndarray, same shape as inputs
        """
        shape = y_coords.shape
        y = y_coords.ravel()
        x = x_coords.ravel()

        # Shift for padding offset
        y_p = y + 2.0
        x_p = x + 2.0

        iy = np.floor(y_p).astype(np.int64)
        ix = np.floor(x_p).astype(np.int64)

        dy = y_p - iy
        dx = x_p - ix

        n = len(y)
        values = np.zeros(n, dtype=np.float64)

        for j in range(-1, 3):
            wy = _cubic_weight(dy - j)
            for i in range(-1, 3):
                wx = _cubic_weight(dx - i)
                yy = np.clip(iy + j, 0, self.img.shape[0] - 1)
                xx = np.clip(ix + i, 0, self.img.shape[1] - 1)
                values += wy * wx * self.img[yy, xx]

        return values.reshape(shape)

    def evaluate_with_gradients(self, y_coords, x_coords):
        """
        Evaluate interpolated intensity and gradients at sub-pixel coordinates.

        Returns
        -------
        values, grad_x, grad_y : ndarrays, same shape as inputs
        """
        shape = y_coords.shape
        y = y_coords.ravel()
        x = x_coords.ravel()

        y_p = y + 2.0
        x_p = x + 2.0

        iy = np.floor(y_p).astype(np.int64)
        ix = np.floor(x_p).astype(np.int64)

        dy = y_p - iy
        dx = x_p - ix

        n = len(y)
        values = np.zeros(n, dtype=np.float64)
        grad_x = np.zeros(n, dtype=np.float64)
        grad_y = np.zeros(n, dtype=np.float64)

        for j in range(-1, 3):
            wy = _cubic_weight(dy - j)
            dwy = _cubic_weight_deriv(dy - j)
            for i in range(-1, 3):
                wx = _cubic_weight(dx - i)
                dwx = _cubic_weight_deriv(dx - i)

                yy = np.clip(iy + j, 0, self.img.shape[0] - 1)
                xx = np.clip(ix + i, 0, self.img.shape[1] - 1)
                pixel_vals = self.img[yy, xx]

                values += wy * wx * pixel_vals
                grad_x += wy * dwx * pixel_vals
                grad_y += dwy * wx * pixel_vals

        return (values.reshape(shape),
                grad_x.reshape(shape),
                grad_y.reshape(shape))
