"""
Optical Tracking of Gauge Length — Main Script.

Tracks marker points on tensile specimens, computes gauge length and strain.

Usage:
    python3 main.py                     # Run on all datasets
    python3 main.py --dataset dbe0b     # Run on one dataset
    python3 main.py --step 50           # Process every 50th frame
    python3 main.py --max-frames 500    # Limit frames
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tracker import (
    track_markers, compute_gauge_and_strain, get_image_files
)


BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "dbe0b": os.path.join(BASE, "Images", "dbe0b"),
    "dbe0c": os.path.join(BASE, "Images", "dbe0c"),
}

# ROI covering the specimen gauge section with markers
# Determined from visual inspection of first frames
ROI = {
    "dbe0b": (150, 60, 300, 500),
    "dbe0c": (150, 60, 300, 500),
}

# Marker pairs that define vertical gauge length (tensile direction)
# Markers sorted: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
GAUGE_PAIRS = [(0, 2), (1, 3)]


def analyze(name, image_dir, frame_step=10, max_frames=None):
    """Run tracking on one dataset and plot results."""
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")

    n_files = len(get_image_files(image_dir))
    print(f"Total images: {n_files}")

    roi = ROI.get(name)
    results = track_markers(image_dir, frame_step=frame_step,
                            max_frames=max_frames, roi=roi)

    frames = np.array(results["frames"])
    centroids = results["centroids"]
    config = results["config"]

    # Calibration factor
    mm_px = config["mm_per_pixel_x"] if config else 1.0
    unit = "mm" if config else "px"

    # Print marker positions in first frame
    print(f"\nMarker positions (frame 0):")
    for j in range(4):
        cx, cy = centroids[0, j]
        print(f"  Marker {j}: ({cx:.1f}, {cy:.1f}) px")

    # Compute gauge lengths and strains for each pair
    print(f"\nGauge length results ({unit}):")
    pair_data = {}
    for a, b in GAUGE_PAIRS:
        g_px, g_mm, strain = compute_gauge_and_strain(centroids, a, b, mm_px)
        pair_data[(a, b)] = (g_px, g_mm, strain)

        valid = ~np.isnan(strain)
        if np.any(valid):
            s_max = np.nanmax(strain) * 100
            s_final = strain[valid][-1] * 100
            d0 = g_mm[0]
            print(f"  Pair {a}-{b}: L0={d0:.2f} {unit}, "
                  f"max strain={s_max:.2f}%, final strain={s_final:.2f}%")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Gauge length
    for (a, b), (g_px, g_mm, strain) in pair_data.items():
        axes[0].plot(frames, g_mm, label=f"Pair {a}-{b}")
    axes[0].set_ylabel(f"Gauge Length ({unit})")
    axes[0].set_title(f"{name} — Gauge Length vs Frame")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Engineering strain
    for (a, b), (g_px, g_mm, strain) in pair_data.items():
        axes[1].plot(frames, strain * 100, label=f"Pair {a}-{b}")
    axes[1].set_ylabel("Engineering Strain (%)")
    axes[1].set_title("Engineering Strain vs Frame")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Marker y-positions over time (shows separation)
    for j in range(4):
        axes[2].plot(frames, centroids[:, j, 1], label=f"Marker {j}")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylabel("Y position (px)")
    axes[2].set_title("Marker Y-Coordinates vs Frame")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(BASE, f"{name}_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.show()

    return results


def main():
    parser = argparse.ArgumentParser(description="Gauge Length Optical Tracker")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (dbe0b or dbe0c). Default: all.")
    parser.add_argument("--step", type=int, default=10,
                        help="Process every Nth frame. Default: 10.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process. Default: all.")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for name, path in targets.items():
        if not os.path.isdir(path):
            print(f"Not found: {path}")
            continue
        analyze(name, path, frame_step=args.step, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
