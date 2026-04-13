"""
Utility functions: image loading, timestamp parsing, helpers.
"""

import numpy as np
import os
import re
from datetime import datetime


def load_image(filepath):
    """Load an image file as a 2D float64 grayscale NumPy array."""
    from PIL import Image
    img = Image.open(filepath).convert('L')
    return np.array(img, dtype=np.float64)


def parse_timestamp(filename):
    """
    Extract timestamp and frame index from filename.
    Format: prefix_YYYY-MM-DD-HHMMSS-NNNN.jpg
    Returns (datetime, frame_index).
    """
    base = os.path.basename(filename)
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{6})-(\d+)', base)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from {base}")
    dt_str = match.group(1)
    frame_idx = int(match.group(2))
    dt = datetime.strptime(dt_str, "%Y-%m-%d-%H%M%S")
    return dt, frame_idx


def list_image_files(directory, extension='.jpg'):
    """List all image files in a directory, sorted by frame index."""
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(extension) and not f.startswith('.'):
            files.append(os.path.join(directory, f))
    files.sort()
    return files


def image_gradients(img):
    """
    Compute x and y gradients of an image using central differences.
    Returns (grad_x, grad_y) with same shape as input.
    """
    h, w = img.shape
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    # Central differences for interior
    grad_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    grad_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0

    # Forward/backward differences at boundaries
    grad_x[:, 0] = img[:, 1] - img[:, 0]
    grad_x[:, -1] = img[:, -1] - img[:, -2]
    grad_y[0, :] = img[1, :] - img[0, :]
    grad_y[-1, :] = img[-1, :] - img[-2, :]

    return grad_x, grad_y


def pad_image(img, pad):
    """Pad image with reflected boundary conditions."""
    return np.pad(img, pad, mode='reflect')
