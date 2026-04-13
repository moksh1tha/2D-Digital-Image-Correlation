"""
IC-GN (Inverse Compositional Gauss-Newton) DIC solver.
Uses ZNSSD correlation criterion and first-order shape functions.
Pure Python + NumPy implementation.
"""

import numpy as np
from .interpolation import BicubicInterpolator
from .utils import image_gradients


class DICSolver:
    """
    Full 2D Digital Image Correlation solver.

    Pipeline:
        1. Set reference image -> precompute gradients & interpolator
        2. Define ROI grid points
        3. For each grid point: integer search -> IC-GN sub-pixel refinement
        4. Output: full-field displacement (u, v) and displacement gradients
    """

    def __init__(self, ref_image, subset_size=21, step=5, search_radius=20,
                 max_iter=50, conv_threshold=1e-3):
        """
        Parameters
        ----------
        ref_image : 2D ndarray
            Reference (undeformed) grayscale image.
        subset_size : int
            Side length of square subset (must be odd).
        step : int
            Spacing between grid points in ROI.
        search_radius : int
            Pixel search radius for initial integer guess.
        max_iter : int
            Maximum IC-GN iterations per subset.
        conv_threshold : float
            Convergence threshold for parameter update norm.
        """
        self.ref = ref_image.astype(np.float64)
        self.h, self.w = self.ref.shape
        self.half = subset_size // 2
        self.subset_size = 2 * self.half + 1
        self.step = step
        self.search_radius = search_radius
        self.max_iter = max_iter
        self.conv_threshold = conv_threshold

        # Precompute reference image gradients and interpolator
        self.ref_gx, self.ref_gy = image_gradients(self.ref)
        self.ref_interp = BicubicInterpolator(self.ref)

    def _build_subset_coords(self, cx, cy):
        """Build local coordinate arrays for a subset centered at (cx, cy)."""
        M = self.half
        dy_range = np.arange(-M, M + 1, dtype=np.float64)
        dx_range = np.arange(-M, M + 1, dtype=np.float64)
        dy_grid, dx_grid = np.meshgrid(dy_range, dx_range, indexing='ij')
        return dy_grid, dx_grid  # (subset_size, subset_size)

    def _precompute_ref_subset(self, cx, cy):
        """
        Precompute reference subset data for IC-GN.
        Returns: ref_values, steepest_descent_images, hessian_inv, f_mean, delta_f
        """
        M = self.half
        dy_grid, dx_grid = self._build_subset_coords(cx, cy)

        # Pixel coordinates in reference image
        yy = cy + dy_grid
        xx = cx + dx_grid

        # Reference subset intensities
        f = self.ref_interp.evaluate(yy, xx)
        f_mean = np.mean(f)
        delta_f = np.sqrt(np.sum((f - f_mean) ** 2))
        if delta_f < 1e-10:
            return None  # Flat subset, skip

        # Reference gradients at subset pixels
        gx = self.ref_gx[int(cy) - M:int(cy) + M + 1,
                          int(cx) - M:int(cx) + M + 1]
        gy = self.ref_gy[int(cy) - M:int(cy) + M + 1,
                          int(cx) - M:int(cx) + M + 1]

        n_pixels = self.subset_size ** 2
        dx_flat = dx_grid.ravel()
        dy_flat = dy_grid.ravel()
        gx_flat = gx.ravel()
        gy_flat = gy.ravel()

        # Steepest descent images: SD[i] = [grad_f] @ [dW/dp]
        # For first-order shape function p = (u, ux, uy, v, vx, vy):
        # dW/dp at pixel with offset (dx, dy):
        #   [[1, dx, dy, 0,  0,  0],
        #    [0,  0,  0, 1, dx, dy]]
        # SD = [gx, gy] @ dW/dp = [gx, gx*dx, gx*dy, gy, gy*dx, gy*dy]
        SD = np.zeros((n_pixels, 6), dtype=np.float64)
        SD[:, 0] = gx_flat
        SD[:, 1] = gx_flat * dx_flat
        SD[:, 2] = gx_flat * dy_flat
        SD[:, 3] = gy_flat
        SD[:, 4] = gy_flat * dx_flat
        SD[:, 5] = gy_flat * dy_flat

        # Normalize SD by delta_f for ZNSSD
        SD_norm = SD / delta_f

        # Hessian: H = SD_norm^T @ SD_norm
        H = SD_norm.T @ SD_norm
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None  # Singular Hessian, skip

        return f, SD_norm, H_inv, f_mean, delta_f

    def _integer_search(self, def_image, cx, cy):
        """
        Integer-pixel displacement search using ZNCC.
        Returns (u_int, v_int) integer displacement.
        """
        M = self.half
        R = self.search_radius

        # Extract reference subset
        y0 = int(cy) - M
        y1 = int(cy) + M + 1
        x0 = int(cx) - M
        x1 = int(cx) + M + 1

        if y0 < 0 or x0 < 0 or y1 > self.h or x1 > self.w:
            return 0, 0

        ref_sub = self.ref[y0:y1, x0:x1].copy()
        ref_sub = ref_sub - np.mean(ref_sub)
        ref_norm = np.sqrt(np.sum(ref_sub ** 2))
        if ref_norm < 1e-10:
            return 0, 0

        best_zncc = -2.0
        best_u, best_v = 0, 0

        # Search bounds
        v_min = max(-R, M - cy)
        v_max = min(R, self.h - 1 - M - cy)
        u_min = max(-R, M - cx)
        u_max = min(R, self.w - 1 - M - cx)

        for dv in range(int(v_min), int(v_max) + 1):
            for du in range(int(u_min), int(u_max) + 1):
                sy = y0 + dv
                sx = x0 + du
                ey = y1 + dv
                ex = x1 + du

                if sy < 0 or sx < 0 or ey > self.h or ex > self.w:
                    continue

                def_sub = def_image[sy:ey, sx:ex].copy()
                def_sub = def_sub - np.mean(def_sub)
                def_norm = np.sqrt(np.sum(def_sub ** 2))
                if def_norm < 1e-10:
                    continue

                zncc = np.sum(ref_sub * def_sub) / (ref_norm * def_norm)
                if zncc > best_zncc:
                    best_zncc = zncc
                    best_u = du
                    best_v = dv

        return best_u, best_v

    def _icgn_subpixel(self, def_interp, cx, cy, u0, v0, ref_data):
        """
        Sub-pixel refinement using Forward Additive Gauss-Newton
        with pre-computed reference image Hessian (Vendroux & Knauss 1998).

        Uses ZNSSD correlation criterion and first-order shape functions.

        Parameters
        ----------
        def_interp : BicubicInterpolator for deformed image
        cx, cy : subset center in reference image
        u0, v0 : initial integer displacement guess
        ref_data : precomputed reference subset data

        Returns
        -------
        p : array [u, ux, uy, v, vx, vy] or None if failed
        zncc : correlation coefficient
        """
        f, SD_norm, H_inv, f_mean, delta_f = ref_data

        # Initialize parameter vector: [u, ux, uy, v, vx, vy]
        p = np.array([float(u0), 0.0, 0.0, float(v0), 0.0, 0.0])

        dy_grid, dx_grid = self._build_subset_coords(cx, cy)
        n_pixels = self.subset_size ** 2
        dx_flat = dx_grid.ravel()
        dy_flat = dy_grid.ravel()

        # Normalized reference subset
        f_flat = f.ravel()
        f_norm = (f_flat - f_mean) / delta_f

        zncc = -1.0

        for iteration in range(self.max_iter):
            u, ux, uy, v, vx, vy = p

            # Warp deformed image coordinates using current parameters
            x_def = cx + dx_flat + u + ux * dx_flat + uy * dy_flat
            y_def = cy + dy_flat + v + vx * dx_flat + vy * dy_flat

            # Check bounds
            if (np.any(x_def < 1) or np.any(x_def >= self.w - 2) or
                    np.any(y_def < 1) or np.any(y_def >= self.h - 2)):
                return None, -1.0

            # Interpolate deformed image at warped positions
            g_flat = def_interp.evaluate(
                y_def.reshape(dy_grid.shape),
                x_def.reshape(dx_grid.shape)
            ).ravel()

            g_mean = np.mean(g_flat)
            delta_g = np.sqrt(np.sum((g_flat - g_mean) ** 2))
            if delta_g < 1e-10:
                return None, -1.0

            # Normalized deformed subset
            g_norm = (g_flat - g_mean) / delta_g

            # ZNSSD and ZNCC
            diff = f_norm - g_norm
            znssd = np.sum(diff ** 2)
            zncc = 1.0 - znssd / 2.0

            # Forward additive: dp = H_inv @ SD^T @ (f_norm - g_norm)
            # SD uses reference gradients (pre-computed), so Hessian is constant
            dp = H_inv @ (SD_norm.T @ diff)

            # Convergence check
            dp_norm = np.sqrt(dp[0]**2 + dp[3]**2 +
                              (dp[1]**2 + dp[2]**2 + dp[4]**2 + dp[5]**2)
                              * (self.half ** 2))
            if dp_norm < self.conv_threshold:
                return p, zncc

            # Forward additive update
            p += dp

        return p, zncc

    def define_roi(self, x_start, y_start, x_end, y_end):
        """
        Define region of interest and generate grid points.

        Returns
        -------
        grid_y, grid_x : 1D arrays of grid point coordinates
        """
        M = self.half
        # Ensure grid points have enough margin for subsets
        x_s = max(x_start, M + 2)
        y_s = max(y_start, M + 2)
        x_e = min(x_end, self.w - M - 3)
        y_e = min(y_end, self.h - M - 3)

        grid_x = np.arange(x_s, x_e + 1, self.step)
        grid_y = np.arange(y_s, y_e + 1, self.step)
        return grid_y, grid_x

    def analyze(self, def_image, roi=None, verbose=True):
        """
        Run full DIC analysis: reference vs deformed image.

        Parameters
        ----------
        def_image : 2D ndarray
            Deformed grayscale image.
        roi : tuple (x_start, y_start, x_end, y_end) or None for full image.
        verbose : bool

        Returns
        -------
        results : dict with keys:
            'grid_x', 'grid_y' : 2D coordinate arrays
            'u', 'v' : 2D displacement arrays
            'ux', 'uy', 'vx', 'vy' : 2D displacement gradient arrays
            'zncc' : 2D correlation coefficient array
        """
        def_image = def_image.astype(np.float64)
        def_interp = BicubicInterpolator(def_image)

        if roi is None:
            roi = (0, 0, self.w, self.h)

        grid_y_1d, grid_x_1d = self.define_roi(*roi)
        ny = len(grid_y_1d)
        nx = len(grid_x_1d)

        if verbose:
            print(f"DIC grid: {ny} x {nx} = {ny * nx} points")

        # Create 2D grids
        gx_2d, gy_2d = np.meshgrid(grid_x_1d, grid_y_1d)

        # Output arrays
        u_field = np.full((ny, nx), np.nan)
        v_field = np.full((ny, nx), np.nan)
        ux_field = np.full((ny, nx), np.nan)
        uy_field = np.full((ny, nx), np.nan)
        vx_field = np.full((ny, nx), np.nan)
        vy_field = np.full((ny, nx), np.nan)
        zncc_field = np.full((ny, nx), np.nan)

        total = ny * nx
        computed = 0

        for iy in range(ny):
            for ix in range(nx):
                cx = grid_x_1d[ix]
                cy = grid_y_1d[iy]

                # Precompute reference subset data
                ref_data = self._precompute_ref_subset(cx, cy)
                if ref_data is None:
                    continue

                # Initial integer search
                u_int, v_int = self._integer_search(def_image, cx, cy)

                # IC-GN sub-pixel refinement
                result, zncc = self._icgn_subpixel(
                    def_interp, cx, cy, u_int, v_int, ref_data
                )

                if result is not None and zncc > 0.5:
                    u_field[iy, ix] = result[0]
                    ux_field[iy, ix] = result[1]
                    uy_field[iy, ix] = result[2]
                    v_field[iy, ix] = result[3]
                    vx_field[iy, ix] = result[4]
                    vy_field[iy, ix] = result[5]
                    zncc_field[iy, ix] = zncc

                computed += 1
                if verbose and computed % 100 == 0:
                    print(f"  {computed}/{total} points computed")

        if verbose:
            valid = np.sum(~np.isnan(u_field))
            print(f"  Done. {valid}/{total} valid points.")

        return {
            'grid_x': gx_2d,
            'grid_y': gy_2d,
            'u': u_field,
            'v': v_field,
            'ux': ux_field,
            'uy': uy_field,
            'vx': vx_field,
            'vy': vy_field,
            'zncc': zncc_field,
        }
