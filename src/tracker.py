"""
Binary marker tracker for real tensile test data.
Tracks black circular markers on specimen gauge section across frames.
Pure Python + NumPy implementation.
"""

import numpy as np
from .utils import load_image, parse_timestamp, list_image_files


def threshold_image(img, method='adaptive', block_size=31, offset=20):
    """
    Binarize grayscale image. Markers (dark) -> True.

    Parameters
    ----------
    img : 2D ndarray (float64)
    method : 'adaptive', 'otsu', or float threshold value
    block_size : int, window size for adaptive thresholding
    offset : float, pixel must be this much darker than local mean

    Returns
    -------
    binary : 2D bool array (True = marker pixel)
    """
    if method == 'adaptive':
        # Adaptive thresholding: pixel is foreground if significantly
        # darker than its local neighborhood mean
        h, w = img.shape
        pad = block_size // 2

        # Compute local mean using integral image (efficient)
        padded = np.pad(img, pad, mode='reflect')
        integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)

        # Local mean at each pixel
        y1 = np.arange(h)
        y2 = y1 + 2 * pad
        x1 = np.arange(w)
        x2 = x1 + 2 * pad

        y1g, x1g = np.meshgrid(y1, x1, indexing='ij')
        y2g, x2g = np.meshgrid(y2, x2, indexing='ij')

        area = block_size * block_size
        local_sum = (integral[y2g, x2g]
                     - integral[y1g, x2g]
                     - integral[y2g, x1g]
                     + integral[y1g, x1g])
        local_mean = local_sum / area

        return img < (local_mean - offset)

    elif method == 'otsu':
        hist, bin_edges = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = img.size
        sum_total = np.sum(bin_centers * hist)

        best_thresh = 0
        best_var = 0
        w0 = 0
        sum0 = 0

        for t in range(256):
            w0 += hist[t]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break
            sum0 += t * hist[t]
            mean0 = sum0 / w0
            mean1 = (sum_total - sum0) / w1
            var_between = w0 * w1 * (mean0 - mean1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = t

        return img < best_thresh
    else:
        return img < float(method)


def connected_components(binary):
    """
    Label connected components using two-pass algorithm.
    4-connectivity.

    Returns
    -------
    labels : 2D int array (0 = background)
    num_labels : int
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    current_label = 0

    # First pass
    for y in range(h):
        for x in range(w):
            if not binary[y, x]:
                continue

            neighbors = []
            if y > 0 and labels[y-1, x] > 0:
                neighbors.append(labels[y-1, x])
            if x > 0 and labels[y, x-1] > 0:
                neighbors.append(labels[y, x-1])

            if len(neighbors) == 0:
                current_label += 1
                labels[y, x] = current_label
                parent[current_label] = current_label
            else:
                min_label = min(neighbors)
                labels[y, x] = min_label
                for n in neighbors:
                    union(min_label, n)

    # Second pass: resolve labels
    label_map = {}
    new_label = 0
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                root = find(labels[y, x])
                if root not in label_map:
                    new_label += 1
                    label_map[root] = new_label
                labels[y, x] = label_map[root]

    return labels, new_label


def find_markers(binary, min_area=50, max_area=5000):
    """
    Find marker regions and compute their centroids.

    Returns
    -------
    markers : list of dict with keys 'centroid_x', 'centroid_y', 'area'
    """
    labels, num = connected_components(binary)

    markers = []
    for lbl in range(1, num + 1):
        ys, xs = np.where(labels == lbl)
        area = len(ys)
        if area < min_area or area > max_area:
            continue
        cx = np.mean(xs)
        cy = np.mean(ys)
        markers.append({
            'centroid_x': cx,
            'centroid_y': cy,
            'area': area,
            'label': lbl
        })

    return markers


def match_markers(prev_markers, curr_markers):
    """
    Match markers between consecutive frames using nearest centroid.
    Returns matched list in same order as prev_markers.
    """
    if len(curr_markers) == 0:
        return [None] * len(prev_markers)

    curr_xy = np.array([[m['centroid_x'], m['centroid_y']] for m in curr_markers])
    matched = []

    used = set()
    for pm in prev_markers:
        px, py = pm['centroid_x'], pm['centroid_y']
        dists = np.sqrt((curr_xy[:, 0] - px)**2 + (curr_xy[:, 1] - py)**2)

        # Sort by distance, pick closest unused
        order = np.argsort(dists)
        found = None
        for idx in order:
            if idx not in used and dists[idx] < 50:  # max 50px jump
                found = idx
                used.add(idx)
                break
        matched.append(curr_markers[found] if found is not None else None)

    return matched


class MarkerTracker:
    """
    Tracks gauge markers across a sequence of tensile test images.

    Expected setup: 4 black dots on specimen (2 pairs defining gauge lengths).
    """

    def __init__(self, image_dir, marker_count=4, min_area=50, max_area=5000,
                 roi=None, frame_step=1):
        """
        Parameters
        ----------
        image_dir : str
            Directory containing sequential image files.
        marker_count : int
            Expected number of markers.
        min_area, max_area : int
            Area filters for connected components.
        roi : tuple (x0, y0, x1, y1) or None
            Crop region to speed up processing.
        frame_step : int
            Process every Nth frame.
        """
        self.image_dir = image_dir
        self.files = list_image_files(image_dir)
        self.marker_count = marker_count
        self.min_area = min_area
        self.max_area = max_area
        self.roi = roi
        self.frame_step = frame_step

    def _crop(self, img):
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            return img[y0:y1, x0:x1]
        return img

    def run(self, max_frames=None, verbose=True):
        """
        Track markers across all frames.

        Returns
        -------
        results : dict with keys:
            'frame_indices' : list of int
            'timestamps' : list of datetime
            'centroids' : array shape (n_frames, n_markers, 2) -> (x, y)
            'gauge_lengths' : dict of pairwise distances
        """
        files = self.files[::self.frame_step]
        if max_frames is not None:
            files = files[:max_frames]

        n_frames = len(files)
        if verbose:
            print(f"Tracking {self.marker_count} markers across {n_frames} frames")

        centroids = np.full((n_frames, self.marker_count, 2), np.nan)
        frame_indices = []
        timestamps = []

        prev_markers = None

        for i, filepath in enumerate(files):
            _, fidx = parse_timestamp(filepath)
            frame_indices.append(fidx)

            img = load_image(filepath)
            img = self._crop(img)

            binary = threshold_image(img)
            markers = find_markers(binary, self.min_area, self.max_area)

            if prev_markers is None:
                # First frame: sort markers by position (top-to-bottom, left-to-right)
                markers.sort(key=lambda m: (m['centroid_y'], m['centroid_x']))
                if len(markers) >= self.marker_count:
                    markers = markers[:self.marker_count]
                    prev_markers = markers
                    for j, m in enumerate(markers):
                        centroids[i, j, 0] = m['centroid_x']
                        centroids[i, j, 1] = m['centroid_y']
            else:
                matched = match_markers(prev_markers, markers)
                for j, m in enumerate(matched):
                    if m is not None:
                        centroids[i, j, 0] = m['centroid_x']
                        centroids[i, j, 1] = m['centroid_y']
                # Update prev_markers with valid matches
                prev_markers = [m if m is not None else prev_markers[j]
                                for j, m in enumerate(matched)]

            if verbose and (i + 1) % 500 == 0:
                print(f"  Frame {i+1}/{n_frames}")

        # Compute pairwise gauge lengths
        gauge_lengths = {}
        for a in range(self.marker_count):
            for b in range(a + 1, self.marker_count):
                dx = centroids[:, b, 0] - centroids[:, a, 0]
                dy = centroids[:, b, 1] - centroids[:, a, 1]
                dist = np.sqrt(dx**2 + dy**2)
                gauge_lengths[f'{a}-{b}'] = dist

        if verbose:
            print("  Tracking complete.")

        return {
            'frame_indices': frame_indices,
            'centroids': centroids,
            'gauge_lengths': gauge_lengths,
        }


def compute_engineering_strain(gauge_lengths):
    """
    Compute engineering strain from gauge length arrays.
    e = (d - d0) / d0
    """
    strains = {}
    for key, dist in gauge_lengths.items():
        d0 = dist[0]
        if np.isnan(d0) or d0 < 1e-10:
            continue
        strains[key] = (dist - d0) / d0
    return strains
