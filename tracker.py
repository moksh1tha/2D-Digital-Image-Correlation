"""
Optical Tracking of Gauge Length Using Binary Image Analysis.

Tracks 4 black circular markers on a tensile specimen across video frames.
Computes gauge length evolution and engineering strain.

Uses OpenCV for image processing, NumPy for numerics.
"""

import cv2
import numpy as np
import os
import glob


def load_config(directory):
    """
    Read config1.dat calibration file.
    Returns dict with parsed fields.
    """
    config_path = os.path.join(directory, "config1.dat")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        vals = [float(x) for x in f.read().strip().split(",")]

    return {
        "n_frames": int(vals[0]),
        "start_frame": int(vals[1]),
        "x_center": vals[2],
        "y_top": vals[3],
        "y_bottom": vals[4],
        "y_gauge_end": vals[5],
        "mm_per_pixel_x": vals[6],
        "mm_per_pixel_y": vals[7],
    }


def get_image_files(directory):
    """Get sorted list of .jpg files in directory."""
    pattern = os.path.join(directory, "*.jpg")
    files = sorted(glob.glob(pattern))
    return files


def frame_index_from_filename(filepath):
    """Extract frame number from filename like prefix-NNNN.jpg"""
    import re
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Handle duplicates like "prefix-1234(1).jpg"
    match = re.search(r'-(\d+)(?:\(\d+\))?$', base)
    return int(match.group(1)) if match else 0


def detect_markers(gray, n_markers=4):
    """
    Detect dark circular markers on a bright specimen surface.

    Steps:
        1. Adaptive threshold to find dark blobs on local background
        2. Morphological cleanup
        3. Connected components with stats
        4. Filter by area and circularity
        5. Return sorted centroids

    Returns: list of (cx, cy) tuples sorted top-to-bottom then left-to-right,
             or None if fewer than n_markers found.
    """
    # Adaptive threshold — markers are darker than their neighborhood
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=25
    )

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filter: skip background (label 0), keep blobs with reasonable area
    candidates = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 20 or area > 2000:
            continue

        # Circularity check: width and height should be similar
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect < 0.4:
            continue

        cx, cy = centroids[i]
        candidates.append((cx, cy, area))

    if len(candidates) < n_markers:
        return None

    # Sort by area (largest first) and take top n_markers
    candidates.sort(key=lambda c: c[2], reverse=True)
    markers = candidates[:n_markers]

    # Sort: top-to-bottom, then left-to-right
    markers.sort(key=lambda c: (round(c[1] / 50), c[0]))

    return [(m[0], m[1]) for m in markers]


def match_markers(prev, curr, max_dist=50):
    """
    Match current markers to previous frame by nearest neighbor.
    prev, curr: lists of (cx, cy).
    Returns list aligned to prev ordering, with None for lost markers.
    """
    if curr is None:
        return [None] * len(prev)

    prev_arr = np.array(prev)
    curr_arr = np.array(curr)
    matched = [None] * len(prev)
    used = set()

    for i, p in enumerate(prev_arr):
        dists = np.sqrt(np.sum((curr_arr - p) ** 2, axis=1))
        for j in np.argsort(dists):
            if j not in used and dists[j] < max_dist:
                matched[i] = (curr_arr[j][0], curr_arr[j][1])
                used.add(j)
                break

    return matched


def track_markers(image_dir, frame_step=1, max_frames=None, roi=None):
    """
    Track 4 markers across all frames in a directory.

    Parameters
    ----------
    image_dir : str
    frame_step : int — process every Nth frame
    max_frames : int or None — limit number of frames
    roi : tuple (x0, y0, x1, y1) or None — crop region

    Returns
    -------
    dict with:
        frames    : list of frame indices
        centroids : array (n_frames, 4, 2) — x,y per marker
        config    : parsed config dict or None
    """
    files = get_image_files(image_dir)
    config = load_config(image_dir)

    files = files[::frame_step]
    if max_frames:
        files = files[:max_frames]

    n = len(files)
    centroids = np.full((n, 4, 2), np.nan)
    frames = []
    prev_markers = None

    print(f"Tracking 4 markers across {n} frames...")

    for i, path in enumerate(files):
        frames.append(frame_index_from_filename(path))

        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        if roi:
            x0, y0, x1, y1 = roi
            crop = gray[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0
        else:
            crop = gray
            offset_x, offset_y = 0, 0

        detected = detect_markers(crop)

        if prev_markers is None:
            # First frame — initialize
            if detected is not None:
                prev_markers = list(detected)  # local coords
                for j, (cx, cy) in enumerate(detected):
                    centroids[i, j] = [cx + offset_x, cy + offset_y]
        else:
            # Match detected markers to previous (both in local/crop coords)
            matched = match_markers(prev_markers, detected)

            for j, m in enumerate(matched):
                if m is not None:
                    centroids[i, j] = [m[0] + offset_x, m[1] + offset_y]
                    prev_markers[j] = m  # update with local coords
                else:
                    # Keep previous position for next match attempt
                    centroids[i, j] = centroids[i - 1, j]

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} frames processed")

    print(f"  Done. {n} frames processed.")
    return {"frames": frames, "centroids": centroids, "config": config}


def compute_gauge_and_strain(centroids, pair_a, pair_b, mm_per_pixel=1.0):
    """
    Compute gauge length and engineering strain for one marker pair.

    Parameters
    ----------
    centroids : (n_frames, 4, 2)
    pair_a, pair_b : marker indices (0-3)
    mm_per_pixel : calibration factor

    Returns
    -------
    gauge_px : gauge length in pixels per frame
    gauge_mm : gauge length in mm per frame
    strain   : engineering strain per frame
    """
    dx = centroids[:, pair_b, 0] - centroids[:, pair_a, 0]
    dy = centroids[:, pair_b, 1] - centroids[:, pair_a, 1]
    gauge_px = np.sqrt(dx ** 2 + dy ** 2)
    gauge_mm = gauge_px * mm_per_pixel

    d0 = gauge_px[0]
    strain = (gauge_px - d0) / d0

    return gauge_px, gauge_mm, strain
