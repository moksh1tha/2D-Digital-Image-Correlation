"""
Run binary marker tracking on real tensile test data.
Tracks 4 black markers across thousands of frames to measure gauge length
and engineering strain.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.tracker import MarkerTracker, compute_engineering_strain
from src.visualize import plot_marker_tracking
from src.utils import load_image


# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
IMAGE_DIRS = {
    'dbe0b': os.path.join(BASE_DIR, 'Images', 'dbe0b'),
    'dbe0c': os.path.join(BASE_DIR, 'Images', 'dbe0c'),
}

# ROI to crop images (reduces processing time and avoids background noise)
# Adjust these based on your images: (x0, y0, x1, y1)
# Markers are on the specimen surface: x~180-260, y~100-500
ROI = (170, 80, 260, 480)

# Frame step: process every Nth frame (1 = all frames, 10 = every 10th)
FRAME_STEP = 10


def analyze_dataset(name, image_dir, frame_step=10, max_frames=None):
    """Run marker tracking on one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")

    # Preview first image
    from src.utils import list_image_files
    files = list_image_files(image_dir)
    print(f"Found {len(files)} images")

    if len(files) == 0:
        print("No images found. Skipping.")
        return None

    # Show first frame info
    img = load_image(files[0])
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")

    # Run tracker
    tracker = MarkerTracker(
        image_dir,
        marker_count=4,
        min_area=30,
        max_area=3000,
        roi=ROI,
        frame_step=frame_step,
    )

    results = tracker.run(max_frames=max_frames, verbose=True)

    # Compute strain
    strains = compute_engineering_strain(results['gauge_lengths'])

    # Report
    print(f"\nGauge length pairs:")
    for key, dist in results['gauge_lengths'].items():
        d0 = dist[0]
        d_final = dist[-1]
        if not np.isnan(d0) and not np.isnan(d_final):
            s = (d_final - d0) / d0 * 100
            print(f"  Pair {key}: d0={d0:.1f} px -> d_final={d_final:.1f} px "
                  f"(strain={s:.2f}%)")

    # Identify primary gauge pairs (vertical separation = gauge length)
    # Markers sorted top-to-bottom, left-to-right: 0,1 (top), 2,3 (bottom)
    # Primary gauge pairs: 0-2 (left column), 1-3 (right column)
    print(f"\nPrimary gauge pairs (vertical):")
    for key in ['0-2', '1-3']:
        if key in strains:
            s = strains[key]
            valid = ~np.isnan(s)
            if np.any(valid):
                print(f"  Pair {key}: max strain = {np.nanmax(s)*100:.2f}%")

    # Plot
    plot_marker_tracking(results, pairs=['0-2', '1-3'],
                         save_path=os.path.join(BASE_DIR, f'{name}_tracking.png'))

    return results, strains


if __name__ == '__main__':
    print("MARKER TRACKING ANALYSIS")
    print("Tracking gauge markers in tensile test images\n")

    for name, img_dir in IMAGE_DIRS.items():
        if os.path.exists(img_dir):
            analyze_dataset(name, img_dir,
                            frame_step=FRAME_STEP,
                            max_frames=2000)  # Limit for quick test
        else:
            print(f"Directory not found: {img_dir}")

    print("\nAnalysis complete.")
