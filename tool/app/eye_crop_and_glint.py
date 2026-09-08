"""
Robust eye region cropping + glint detection for near-eye IR camera images.

v5 - Contour-based pupil detection:
  - Primary: adaptive threshold + contour circularity filtering (rejects shadows)
  - Fallback A: OTSU threshold + contour filtering
  - Fallback B: legacy dark-blob centroid (tightened threshold)
  - Glint centroid = eye area anchor for search region
  - Radius from contour fitting (actual blob boundary), not radial profile
  - Circularity >= 0.55 rejects elongated eyelid shadows
  - Two-pass: raw detect -> Kalman smooth
  - Eye open/closed detection
  - 4-column output: original, debug, cropped, glint_debug
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import bisect
import re
import concurrent.futures

# All algorithms are pure Python/OpenCV — no external C++ dependencies

ALGORITHMS = ['contour', 'pupillabs', 'else_lite', 'starburst', 'combined']

# --- Hardware mask config per camera (240x240 near-eye cam) ---
# Each camera has the glasses hardware in a different corner.
# Per-batch hardware mask configs. V1/V2 cameras have different hardware positions.
CAMERA_MASK_CONFIGS = {
    'default': {
        'ri': {'block': 'upper_right', 'y_frac': 0.55, 'x_frac': 0.55},
        'li': {'block': 'upper_left',  'y_frac': 0.55, 'x_frac': 0.45},
        'ro': {'block': 'lower_right', 'y_frac': 0.50, 'x_frac': 0.50},
        'lo': {'block': 'lower_right', 'y_frac': 0.50, 'x_frac': 0.55},
    },
    'v1': {
        'ri': {'block': 'upper_right', 'y_frac': 0.25, 'x_frac': 0.58},
        'li': {'block': 'upper_left',  'y_frac': 0.20, 'x_frac': 0.33},
        'ro': {'block': 'lower_right', 'y_frac': 0.75, 'x_frac': 0.75},
        'lo': {'block': 'lower_left',  'y_frac': 0.55, 'x_frac': 0.40},
    },
}
CAMERA_MASK_CONFIGS['v2'] = CAMERA_MASK_CONFIGS['v1']
CAMERA_MASK_CONFIG = CAMERA_MASK_CONFIGS['default']
BORDER = 12


# ============================================================
# Kalman Filter for 2D position smoothing
# ============================================================
class BoxKalman:
    """Kalman filter for smooth (x, y) crop center tracking."""

    def __init__(self, process_noise=0.5, measurement_noise=10.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50
        self.initialized = False

    def init(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0].item()), float(pred[1].item())

    def correct(self, x, y):
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        return float(corrected[0].item()), float(corrected[1].item())


# ============================================================
# Detection
# ============================================================

def build_search_mask(h, w, camera='ri', camera_batch='default'):
    """Mask excluding hardware zone and borders. Camera and batch specific."""
    mask = np.ones((h, w), dtype=np.uint8) * 255
    batch_cfg = CAMERA_MASK_CONFIGS.get(camera_batch, CAMERA_MASK_CONFIGS['default'])
    cfg = batch_cfg.get(camera, batch_cfg.get('ri', CAMERA_MASK_CONFIGS['default']['ri']))
    block = cfg['block']
    y_cut = int(h * cfg['y_frac'])
    x_cut = int(w * cfg['x_frac'])

    if block == 'upper_right':
        mask[0:y_cut, x_cut:w] = 0
    elif block == 'upper_left':
        mask[0:y_cut, 0:x_cut] = 0
    elif block == 'lower_right':
        mask[y_cut:h, x_cut:w] = 0
    elif block == 'lower_left':
        mask[y_cut:h, 0:x_cut] = 0

    mask[0:BORDER, :] = 0
    mask[h - BORDER:h, :] = 0
    mask[:, 0:BORDER] = 0
    mask[:, w - BORDER:w] = 0
    return mask


def find_corneal_glints(gray, mask, target_glints=4, min_thresh_floors=None):
    """
    Find bright micro-spots (candidate glints) using multi-threshold cascade.

    Strategy: start with strict threshold, progressively relax until we find
    up to target_glints. New candidates at each relaxed level must be local
    maxima and spatially consistent with already-found glints.

    min_thresh_floors: optional list of 4 minimum thresholds (strict→permissive).
                       Default: [215, 200, 185, 170].
    """
    valid_pixels = gray[mask > 0]
    if len(valid_pixels) == 0:
        return []

    if min_thresh_floors is None:
        min_thresh_floors = [215, 200, 185, 170]

    # Multi-threshold cascade: strict → relaxed
    p998 = np.percentile(valid_pixels, 99.8)
    p996 = np.percentile(valid_pixels, 99.6)
    p992 = np.percentile(valid_pixels, 99.2)
    p988 = np.percentile(valid_pixels, 98.8)

    thresholds = [
        max(p998, min_thresh_floors[0]),
        max(p996, min_thresh_floors[1]),
        max(p992, min_thresh_floors[2]),
        max(p988, min_thresh_floors[3]),
    ]
    # Ensure descending and capped
    thresholds = [min(t, 254) for t in thresholds]

    all_spots = []
    seen_positions = set()  # (rounded x, rounded y) to avoid duplicates
    MIN_SPOT_DIST = 5  # minimum distance between distinct glints

    for thresh_val in thresholds:
        _, bright = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, mask)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1 or area > 30:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            # Skip if too close to an already-found glint
            too_close = False
            for sx, sy, _ in all_spots:
                if np.sqrt((cx - sx)**2 + (cy - sy)**2) < MIN_SPOT_DIST:
                    too_close = True
                    break
            if too_close:
                continue

            # Local maximum check: this spot should be the brightest
            # in its 7x7 neighborhood
            ix, iy = int(round(cx)), int(round(cy))
            h, w = gray.shape
            y1 = max(0, iy - 3)
            y2 = min(h, iy + 4)
            x1 = max(0, ix - 3)
            x2 = min(w, ix + 4)
            local_patch = gray[y1:y2, x1:x2]
            if local_patch.size > 0:
                local_max = np.max(local_patch)
                spot_val = gray[iy, ix] if 0 <= iy < h and 0 <= ix < w else 0
                if spot_val < local_max - 5:
                    continue  # not a local maximum

            all_spots.append((cx, cy, int(area)))

        if len(all_spots) >= target_glints:
            break

    # If we have more than target, keep the brightest ones
    if len(all_spots) > target_glints:
        # Score by pixel intensity at spot location
        h, w = gray.shape
        scored = []
        for sx, sy, sa in all_spots:
            ix, iy = int(round(sx)), int(round(sy))
            val = float(gray[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0
            scored.append((sx, sy, sa, val))
        scored.sort(key=lambda s: s[3], reverse=True)
        all_spots = [(s[0], s[1], s[2]) for s in scored[:target_glints]]

    return all_spots


def find_glints_robust(enhanced_gray, raw_gray, mask, target_glints=4):
    """
    Robust multi-pass glint detection. Tries multiple strategies to find
    up to target_glints:
      Pass 1: Enhanced image, normal thresholds
      Pass 2: Raw grayscale, normal thresholds (catches glints the enhancement missed)
      Pass 3: Enhanced image, lower thresholds (more permissive)
    Merges results, deduplicating by proximity (5px).
    """
    MIN_MERGE_DIST = 5

    def _merge(existing, new_spots):
        """Add new_spots to existing if not too close to any existing spot."""
        for nx, ny, na in new_spots:
            dup = False
            for ex, ey, _ in existing:
                if np.sqrt((nx - ex)**2 + (ny - ey)**2) < MIN_MERGE_DIST:
                    dup = True
                    break
            if not dup:
                existing.append((nx, ny, na))
        return existing

    # Pass 1: enhanced image, normal thresholds
    spots = find_corneal_glints(enhanced_gray, mask, target_glints=target_glints)

    if len(spots) >= target_glints:
        return spots

    # Pass 2: raw grayscale, normal thresholds
    raw_spots = find_corneal_glints(raw_gray, mask, target_glints=target_glints)
    spots = _merge(spots, raw_spots)

    if len(spots) >= target_glints:
        # Keep brightest up to target
        h, w = raw_gray.shape
        scored = []
        for sx, sy, sa in spots:
            ix, iy = int(round(sx)), int(round(sy))
            val = float(raw_gray[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0
            scored.append((sx, sy, sa, val))
        scored.sort(key=lambda s: s[3], reverse=True)
        return [(s[0], s[1], s[2]) for s in scored[:target_glints]]

    # Pass 3: enhanced image, lower thresholds
    low_spots = find_corneal_glints(enhanced_gray, mask, target_glints=target_glints,
                                     min_thresh_floors=[170, 150, 130, 110])
    spots = _merge(spots, low_spots)

    if len(spots) > target_glints:
        h, w = enhanced_gray.shape
        scored = []
        for sx, sy, sa in spots:
            ix, iy = int(round(sx)), int(round(sy))
            val = float(enhanced_gray[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0
            scored.append((sx, sy, sa, val))
        scored.sort(key=lambda s: s[3], reverse=True)
        return [(s[0], s[1], s[2]) for s in scored[:target_glints]]

    return spots


def radial_intensity_profile(gray_blurred, cx, cy, max_r=40, n_angles=36):
    """
    Compute average intensity at each radius from (cx, cy).
    Returns array of shape (max_r,) with mean intensity per radius ring.
    """
    h, w = gray_blurred.shape
    profile = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for r in range(1, max_r):
        for a in angles:
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            if 0 <= x < w and 0 <= y < h:
                profile[r] += gray_blurred[y, x]
                counts[r] += 1

    valid = counts > 0
    profile[valid] /= counts[valid]
    profile[~valid] = 255
    return profile


def estimate_pupil_radius(gray, cx, cy, min_r=6, max_r=35):
    """
    Find pupil radius using radial intensity profile.
    The pupil-iris boundary is where intensity jumps most (steepest gradient).

    Returns radius in pixels (clamped to [min_r, max_r]).
    """
    blurred = cv2.GaussianBlur(gray, (9, 9), 3)
    profile = radial_intensity_profile(blurred, cx, cy, max_r=max_r + 5)

    # Smooth the profile to avoid noise spikes
    kernel_size = 3
    smoothed = np.convolve(profile, np.ones(kernel_size) / kernel_size, mode='same')

    # Compute gradient (intensity change per pixel of radius)
    gradient = np.diff(smoothed)

    # Find the steepest positive gradient (dark->bright = pupil->iris boundary)
    # Only search in valid range
    search_start = max(min_r - 2, 3)
    search_end = min(max_r + 2, len(gradient))

    if search_start >= search_end:
        return min_r

    search_region = gradient[search_start:search_end]
    best_idx = np.argmax(search_region) + search_start

    radius = int(np.clip(best_idx, min_r, max_r))
    return radius


def _contour_circularity(contour):
    """Compute circularity: 4*pi*area / perimeter^2.  Perfect circle = 1.0."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def _find_pupil_contour(gray, search_mask, glint_pos=None, min_r=5, max_r=18):
    """
    Find the pupil contour using adaptive thresholding + circularity filtering.

    Returns (cx, cy, radius, circularity) or None.
    """
    # Work on the search region only
    roi = gray.copy()
    roi[search_mask == 0] = 255  # white-out masked areas

    blurred = cv2.GaussianBlur(roi, (7, 7), 2)

    # Adaptive threshold: adapts to local brightness, separates pupil from shadows
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=8
    )
    binary[search_mask == 0] = 0  # enforce mask

    # Morphological cleanup: remove eyelash noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.pi * min_r * min_r * 0.5
    max_area = np.pi * max_r * max_r * 2.0

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        circ = _contour_circularity(cnt)
        if circ < 0.55:
            continue
        # Compute center via enclosing circle
        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        # Score: circularity + proximity to glint (if available)
        score = circ * 0.6
        if glint_pos is not None:
            dist = np.sqrt((cx - glint_pos[0])**2 + (cy - glint_pos[1])**2)
            # Closer to glint = higher score (normalize by search radius)
            proximity = max(0.0, 1.0 - dist / 60.0)
            score += proximity * 0.4
        else:
            score += 0.2  # neutral proximity bonus

        candidates.append((cnt, cx, cy, r, circ, score))

    if not candidates:
        return None

    # Pick best candidate by score
    candidates.sort(key=lambda c: c[5], reverse=True)
    best_cnt, cx, cy, r, circ, score = candidates[0]

    # Refine with fitEllipse if enough points
    if len(best_cnt) >= 5:
        ellipse = cv2.fitEllipse(best_cnt)
        ecx, ecy = ellipse[0]
        axes = ellipse[1]
        radius = int(round((axes[0] + axes[1]) / 4.0))  # average semi-axis
    else:
        ecx, ecy = cx, cy
        radius = int(round(r))

    radius = int(np.clip(radius, min_r, max_r))
    return (int(round(ecx)), int(round(ecy)), radius, circ)


def _find_pupil_contour_otsu(gray, search_mask, glint_pos=None, min_r=5, max_r=18):
    """
    Fallback contour detection using OTSU thresholding.
    Handles cases where adaptive threshold fails.

    Returns (cx, cy, radius, circularity) or None.
    """
    roi = gray.copy()
    roi[search_mask == 0] = 255

    blurred = cv2.GaussianBlur(roi, (7, 7), 2)

    # Only threshold non-masked pixels — extract valid region stats
    valid_pixels = blurred[search_mask > 0]
    if len(valid_pixels) < 20:
        return None

    # OTSU on the search region
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[search_mask == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.pi * min_r * min_r * 0.5
    max_area = np.pi * max_r * max_r * 2.0

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        circ = _contour_circularity(cnt)
        if circ < 0.45:  # slightly more permissive than adaptive
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        score = circ * 0.6
        if glint_pos is not None:
            dist = np.sqrt((cx - glint_pos[0])**2 + (cy - glint_pos[1])**2)
            proximity = max(0.0, 1.0 - dist / 60.0)
            score += proximity * 0.4
        else:
            score += 0.2

        candidates.append((cnt, cx, cy, r, circ, score))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[5], reverse=True)
    best_cnt, cx, cy, r, circ, score = candidates[0]

    if len(best_cnt) >= 5:
        ellipse = cv2.fitEllipse(best_cnt)
        ecx, ecy = ellipse[0]
        axes = ellipse[1]
        radius = int(round((axes[0] + axes[1]) / 4.0))
    else:
        ecx, ecy = cx, cy
        radius = int(round(r))

    radius = int(np.clip(radius, min_r, max_r))
    return (int(round(ecx)), int(round(ecy)), radius, circ)


def _dark_blob_centroid_legacy(blurred, search_mask):
    """
    Find pupil center as the weighted centroid of the dark blob,
    not just the single darkest pixel.

    1. Find min intensity in the search region
    2. Threshold: all pixels within (min + margin) are "pupil candidates"
    3. Compute center-of-mass weighted by inverse brightness (darker = heavier)

    Returns (cx, cy) or None if no dark region found.
    """
    masked = blurred.copy()
    masked[search_mask == 0] = 255
    min_val, _, min_loc, _ = cv2.minMaxLoc(masked)

    if min_val >= 250:
        return None

    # Threshold: pixels within 15 intensity of darkest are pupil candidates
    # Tighter threshold to avoid pulling centroid toward shadows/eyelids
    thresh = min_val + 15
    dark_mask = (masked <= thresh) & (search_mask > 0)

    ys, xs = np.where(dark_mask)
    if len(xs) < 3:
        # Too few dark pixels, fall back to darkest point
        return min_loc

    # Weight by inverse brightness: darker pixels pull centroid more
    weights = (thresh - masked[ys, xs]).astype(np.float64) + 1.0
    total_w = weights.sum()
    cx = float(np.sum(xs * weights) / total_w)
    cy = float(np.sum(ys * weights) / total_w)
    return (int(round(cx)), int(round(cy)))


# ============================================================
# Alternative pupil detection algorithms
# ============================================================

def _get_search_region(gray, mask, glint_spots):
    """Compute glint-based search region (shared by all algorithms)."""
    h, w = gray.shape
    if len(glint_spots) >= 1:
        gx = np.mean([s[0] for s in glint_spots])
        gy = np.mean([s[1] for s in glint_spots])
        search_r = 35
        search_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(search_mask, (int(round(gx)), int(round(gy))), search_r, 255, -1)
        search_mask = cv2.bitwise_and(search_mask, mask)
        return search_mask, (gx, gy)
    else:
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(h, w) * 0.35), 255, -1)
        center_mask = cv2.bitwise_and(center_mask, mask)
        return center_mask, None


def detect_pupil_pupillabs(gray, mask, glint_spots):
    """
    Pure-Python reimplementation inspired by Pupil Labs Detector2D.

    Coarse-to-fine approach:
    1. Coarse: histogram-based threshold to find dark region candidates
    2. Filter candidates by size + position
    3. Fine: edge detection around best candidate boundary
    4. RANSAC-style ellipse fit on edge points
    5. Confidence = fraction of ellipse perimeter supported by edges
    """
    h, w = gray.shape
    search_mask, glint_pos = _get_search_region(gray, mask, glint_spots)

    roi = gray.copy()
    roi[search_mask == 0] = 255

    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)

    # --- Coarse: histogram-based adaptive threshold ---
    valid_pixels = blurred[search_mask > 0]
    if len(valid_pixels) < 30:
        return None

    # Use histogram to find the dark peak (pupil intensity cluster)
    hist = cv2.calcHist([blurred], [0], search_mask, [256], [0, 256]).flatten()
    # Find the strongest dark mode (below median)
    med = np.median(valid_pixels)
    dark_hist = hist[:int(med)]
    if len(dark_hist) < 5:
        return None
    dark_peak = np.argmax(dark_hist)
    # Threshold between dark peak and median
    coarse_thresh = int((dark_peak + med) / 2)

    _, binary = cv2.threshold(blurred, coarse_thresh, 255, cv2.THRESH_BINARY_INV)
    binary[search_mask == 0] = 0

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.pi * 5 * 5 * 0.4
    max_area = np.pi * 20 * 20 * 2.0

    # --- Filter and rank coarse candidates ---
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        circ = _contour_circularity(cnt)
        if circ < 0.35:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if not (0 <= cy_i < h and 0 <= cx_i < w and search_mask[cy_i, cx_i] > 0):
            continue

        score = circ * 0.5
        if glint_pos is not None:
            dist = np.sqrt((cx - glint_pos[0])**2 + (cy - glint_pos[1])**2)
            score += max(0.0, 1.0 - dist / 50.0) * 0.5
        candidates.append((cnt, cx_i, cy_i, int(round(r)), score))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[4], reverse=True)
    best_cnt, coarse_cx, coarse_cy, coarse_r, _ = candidates[0]

    # --- Fine: edge-based refinement around coarse candidate ---
    # Extract a local region around the coarse detection
    margin = coarse_r + 8
    x1 = max(0, coarse_cx - margin)
    y1 = max(0, coarse_cy - margin)
    x2 = min(w, coarse_cx + margin)
    y2 = min(h, coarse_cy + margin)
    local = blurred[y1:y2, x1:x2]

    if local.size < 20:
        radius = max(5, min(18, coarse_r))
        return (coarse_cx, coarse_cy, radius)

    # Edge detection on local region
    local_med = np.median(local)
    lo = int(max(0, 0.5 * local_med))
    hi = int(min(255, 1.2 * local_med))
    edges = cv2.Canny(local, lo, hi)

    edge_ys, edge_xs = np.where(edges > 0)
    if len(edge_xs) < 8:
        radius = max(5, min(18, coarse_r))
        return (coarse_cx, coarse_cy, radius)

    # Filter edge points: keep those near the expected pupil boundary
    local_cx = coarse_cx - x1
    local_cy = coarse_cy - y1
    dists = np.sqrt((edge_xs - local_cx)**2 + (edge_ys - local_cy)**2)
    near_boundary = (dists > coarse_r * 0.4) & (dists < coarse_r * 2.0)
    bx = edge_xs[near_boundary]
    by = edge_ys[near_boundary]

    if len(bx) < 5:
        radius = max(5, min(18, coarse_r))
        return (coarse_cx, coarse_cy, radius)

    # Fit ellipse to boundary edge points
    pts = np.stack([bx, by], axis=1).reshape(-1, 1, 2).astype(np.int32)
    try:
        ellipse = cv2.fitEllipse(pts)
        ecx, ecy = ellipse[0]
        axes = ellipse[1]
        radius = int(round((axes[0] + axes[1]) / 4.0))

        # Convert back to full-image coordinates
        final_cx = int(round(ecx)) + x1
        final_cy = int(round(ecy)) + y1

        # Sanity check
        if (0 <= final_cy < h and 0 <= final_cx < w
                and search_mask[final_cy, final_cx] > 0):
            radius = max(5, min(18, radius))
            return (final_cx, final_cy, radius)
    except cv2.error:
        pass

    radius = max(5, min(18, coarse_r))
    return (coarse_cx, coarse_cy, radius)


def detect_pupil_else_lite(gray, mask, glint_spots):
    """
    ElSe-inspired (Edge-based Pupil Detection) pure Python/OpenCV implementation.

    1. GaussianBlur → Canny edge detection (auto thresholds from median)
    2. Filter edge pixels: keep where interior side is dark
    3. Morphological close to connect edge fragments
    4. findContours → fit ellipse to each contour
    5. Score: mean intensity inside (dark) vs outside ring (bright) + shape regularity
    6. Return best scoring ellipse center + radius
    """
    h, w = gray.shape
    search_mask, glint_pos = _get_search_region(gray, mask, glint_spots)

    # Work in the search region
    roi = gray.copy()
    roi[search_mask == 0] = 128  # neutral gray outside

    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)

    # Auto Canny thresholds from median
    valid_pixels = blurred[search_mask > 0]
    if len(valid_pixels) < 20:
        return None
    med = np.median(valid_pixels)
    low_thresh = int(max(0, 0.5 * med))
    high_thresh = int(min(255, 1.3 * med))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    edges[search_mask == 0] = 0

    # Filter edges: keep only edges where the darker side faces inward
    # Compute gradient direction
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # For each edge pixel, sample a point 3px inward (toward darker side)
    # The gradient points from dark to bright, so "inward" = against gradient
    edge_ys, edge_xs = np.where(edges > 0)
    if len(edge_xs) < 10:
        return None

    # Quick filter: check that the inner side (opposite to gradient) is dark
    keep = np.ones(len(edge_xs), dtype=bool)
    for idx in range(len(edge_xs)):
        ex, ey = edge_xs[idx], edge_ys[idx]
        gx_val = grad_x[ey, ex]
        gy_val = grad_y[ey, ex]
        mag = np.sqrt(gx_val**2 + gy_val**2)
        if mag < 1:
            continue
        # Sample 3px against gradient (into darker region)
        dx, dy = -gx_val / mag, -gy_val / mag
        sx, sy = int(round(ex + dx * 3)), int(round(ey + dy * 3))
        if 0 <= sx < w and 0 <= sy < h:
            if blurred[sy, sx] > med:  # inner side should be dark, not bright
                keep[idx] = False

    # Create filtered edge mask
    filtered_edges = np.zeros_like(edges)
    filtered_edges[edge_ys[keep], edge_xs[keep]] = 255

    # Morphological close to connect fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.pi * 6 * 6 * 0.3
    max_area = np.pi * 30 * 30 * 2.0

    candidates = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        ellipse = cv2.fitEllipse(cnt)
        ecx, ecy = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]

        # Shape regularity: ratio of axes (reject very elongated)
        a_min, a_max = min(axes), max(axes)
        if a_max < 1:
            continue
        aspect = a_min / a_max
        if aspect < 0.4:
            continue

        radius = (axes[0] + axes[1]) / 4.0
        if radius < 4 or radius > 30:
            continue

        ecx_i, ecy_i = int(round(ecx)), int(round(ecy))
        if not (0 <= ecy_i < h and 0 <= ecx_i < w):
            continue
        if search_mask[ecy_i, ecx_i] == 0:
            continue

        # Score: dark inside, bright outside
        ell_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ell_mask, (ecx_i, ecy_i),
                     (int(radius * 0.8), int(radius * 0.8)), 0, 0, 360, 255, -1)
        inner_pixels = blurred[ell_mask > 0]

        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ring_mask, (ecx_i, ecy_i),
                     (int(radius * 1.5), int(radius * 1.5)), 0, 0, 360, 255, -1)
        ring_mask[ell_mask > 0] = 0
        ring_mask[search_mask == 0] = 0
        outer_pixels = blurred[ring_mask > 0]

        if len(inner_pixels) < 5 or len(outer_pixels) < 5:
            continue

        inner_mean = np.mean(inner_pixels)
        outer_mean = np.mean(outer_pixels)
        contrast_score = (outer_mean - inner_mean) / 255.0  # higher = better

        # Proximity to glints
        prox_score = 0.0
        if glint_pos is not None:
            dist = np.sqrt((ecx - glint_pos[0])**2 + (ecy - glint_pos[1])**2)
            prox_score = max(0.0, 1.0 - dist / 60.0)

        score = contrast_score * 0.5 + aspect * 0.2 + prox_score * 0.3
        candidates.append((ecx_i, ecy_i, int(round(radius)), score))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[3], reverse=True)
    best = candidates[0]
    return (best[0], best[1], max(6, min(25, best[2])))


def detect_pupil_starburst(gray, mask, glint_spots):
    """
    Starburst algorithm for pupil detection.

    1. Seed point: glint centroid if available, else image center
    2. Cast 24 rays outward from seed
    3. Along each ray, find the first significant intensity gradient (dark→bright)
    4. Collect edge points, filter outliers (distance from median)
    5. Fit ellipse to inlier edge points (cv2.fitEllipse)
    6. Use fitted center as new seed, repeat 2-3 iterations
    7. Return final ellipse center + average semi-axis as radius
    """
    h, w = gray.shape
    search_mask, glint_pos = _get_search_region(gray, mask, glint_spots)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Seed point
    if glint_pos is not None:
        seed_x, seed_y = float(glint_pos[0]), float(glint_pos[1])
    else:
        seed_x, seed_y = w / 2.0, h / 2.0

    n_rays = 24
    max_ray_len = 40
    gradient_thresh = 15  # minimum intensity jump to count as edge
    n_iterations = 3

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for iteration in range(n_iterations):
        edge_points = []

        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)
            prev_val = None

            for step in range(3, max_ray_len):
                rx = int(round(seed_x + dx * step))
                ry = int(round(seed_y + dy * step))

                if not (0 <= rx < w and 0 <= ry < h):
                    break
                if search_mask[ry, rx] == 0:
                    break

                val = float(blurred[ry, rx])
                if prev_val is not None:
                    gradient = val - prev_val
                    if gradient > gradient_thresh:
                        # Found dark→bright transition = pupil edge
                        edge_points.append((rx, ry))
                        break
                prev_val = val

        if len(edge_points) < 5:
            if iteration == 0:
                return None  # not enough edges even on first try
            break

        # Filter outliers by distance from median
        pts = np.array(edge_points, dtype=np.float32)
        med_x, med_y = np.median(pts[:, 0]), np.median(pts[:, 1])
        dists = np.sqrt((pts[:, 0] - med_x)**2 + (pts[:, 1] - med_y)**2)
        med_dist = np.median(dists)
        inliers = pts[dists < med_dist * 2.0 + 5]

        if len(inliers) < 5:
            break

        # Fit ellipse
        inlier_cnt = inliers.reshape(-1, 1, 2).astype(np.int32)
        try:
            ellipse = cv2.fitEllipse(inlier_cnt)
        except cv2.error:
            break

        ecx, ecy = ellipse[0]
        axes = ellipse[1]
        radius = (axes[0] + axes[1]) / 4.0

        # Sanity check
        if radius < 3 or radius > 35:
            break
        if not (0 <= ecx < w and 0 <= ecy < h):
            break

        # Update seed for next iteration
        seed_x, seed_y = ecx, ecy

    # Return last valid result
    ecx_i, ecy_i = int(round(seed_x)), int(round(seed_y))
    if not (0 <= ecy_i < h and 0 <= ecx_i < w):
        return None

    # Estimate radius from the edge points if we have them
    if len(edge_points) >= 5:
        pts = np.array(edge_points, dtype=np.float32)
        dists = np.sqrt((pts[:, 0] - seed_x)**2 + (pts[:, 1] - seed_y)**2)
        med_dist = np.median(dists)
        dists_filt = dists[dists < med_dist * 2.0 + 5]
        final_radius = int(round(np.median(dists_filt))) if len(dists_filt) > 0 else 15
    else:
        final_radius = 15

    final_radius = max(6, min(25, final_radius))
    return (ecx_i, ecy_i, final_radius)


def detect_pupil_combined(gray, mask, glint_spots):
    """
    Ensemble: run all 4 algorithms, combine with consensus voting.

    Strategy:
    1. Run contour, pupillabs, else_lite, starburst
    2. Collect all non-None detections
    3. Compute median center (robust to 1 outlier)
    4. Reject detections far from median (>12px = outlier)
    5. Final center = mean of inliers, final radius = median of inlier radii
    6. If only 1 detection: use it as-is (better than nothing)
    """
    SUB_ALGOS = [
        ('contour', lambda g, m, gl: _detect_pupil_contour_cascade(g, m, gl)),
        ('pupillabs', detect_pupil_pupillabs),
        ('else_lite', detect_pupil_else_lite),
        ('starburst', detect_pupil_starburst),
    ]

    detections = []
    for name, func in SUB_ALGOS:
        try:
            result = func(gray, mask, glint_spots)
            if result is not None:
                detections.append(result)
        except Exception:
            pass

    if len(detections) == 0:
        return None

    if len(detections) == 1:
        return detections[0]

    # Median center for outlier rejection
    xs = np.array([d[0] for d in detections], dtype=np.float64)
    ys = np.array([d[1] for d in detections], dtype=np.float64)
    rs = np.array([d[2] for d in detections], dtype=np.float64)

    med_x, med_y = np.median(xs), np.median(ys)

    # Keep detections within 12px of median center
    OUTLIER_DIST = 12
    dists = np.sqrt((xs - med_x)**2 + (ys - med_y)**2)
    inlier_mask = dists <= OUTLIER_DIST

    if not np.any(inlier_mask):
        # All far from each other — use the one closest to glints or median
        inlier_mask = dists <= np.min(dists) + 1  # keep closest to median

    inlier_xs = xs[inlier_mask]
    inlier_ys = ys[inlier_mask]
    inlier_rs = rs[inlier_mask]

    # Final: mean position of inliers, median radius
    final_cx = int(round(np.mean(inlier_xs)))
    final_cy = int(round(np.mean(inlier_ys)))
    final_r = int(round(np.median(inlier_rs)))
    final_r = max(5, min(18, final_r))

    return (final_cx, final_cy, final_r)


def _detect_pupil_contour_cascade(gray, mask, glint_spots):
    """Contour-based cascade (extracted for use by combined algorithm)."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 5)

    if len(glint_spots) >= 1:
        gx = np.mean([s[0] for s in glint_spots])
        gy = np.mean([s[1] for s in glint_spots])
        glint_pos = (gx, gy)

        if len(glint_spots) >= 2:
            spreads = [np.sqrt((s[0] - gx)**2 + (s[1] - gy)**2) for s in glint_spots]
            glint_spread = np.max(spreads)
            max_r = int(np.clip(glint_spread * 1.8, 10, 18))
            min_r = max(5, int(glint_spread * 0.4))
        else:
            min_r, max_r = 5, 18

        search_r = 35
        search_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(search_mask, (int(round(gx)), int(round(gy))), search_r, 255, -1)
        search_mask = cv2.bitwise_and(search_mask, mask)

        result = _find_pupil_contour(gray, search_mask, glint_pos=glint_pos,
                                     min_r=min_r, max_r=max_r)
        if result is not None:
            return result
        result = _find_pupil_contour_otsu(gray, search_mask, glint_pos=glint_pos,
                                          min_r=min_r, max_r=max_r)
        if result is not None:
            return result
        blob = _dark_blob_centroid_legacy(blurred, search_mask)
        if blob is None:
            cx, cy = int(round(gx)), int(round(gy))
        else:
            cx, cy = blob
        radius = estimate_pupil_radius(gray, cx, cy, min_r=min_r, max_r=max_r)
        return (cx, cy, radius)
    else:
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(h, w) * 0.35), 255, -1)
        center_mask = cv2.bitwise_and(center_mask, mask)
        result = _find_pupil_contour(gray, center_mask, min_r=5, max_r=18)
        if result is not None:
            return result
        result = _find_pupil_contour_otsu(gray, center_mask, min_r=5, max_r=18)
        if result is not None:
            return result
        blob = _dark_blob_centroid_legacy(blurred, center_mask)
        if blob is None:
            blob = _dark_blob_centroid_legacy(blurred, mask)
        if blob is None:
            return None
        cx, cy = blob
        radius = estimate_pupil_radius(gray, cx, cy, min_r=5, max_r=18)
        return (cx, cy, radius)


def detect_pupil_raw(gray, mask, glint_spots, algorithm='contour'):
    """
    Detect pupil using the specified algorithm.

    Returns (cx, cy, radius, circularity, method) or None.
    circularity: float 0-1 (contour methods) or None (other algorithms).
    method: 'adaptive', 'otsu', 'blob', or the algorithm name.

    Algorithms:
      - 'contour': cascading contour methods (adaptive + OTSU + blob) — default
      - 'pupillabs': Pupil Labs Detector2D inspired (pure Python)
      - 'else_lite': ElSe-inspired edge + ellipse fitting
      - 'starburst': Starburst ray-casting + RANSAC ellipse
      - 'combined': ensemble of all 4, median consensus

    Uses glints as eye-area anchor when available.
    """
    if algorithm == 'combined':
        r = detect_pupil_combined(gray, mask, glint_spots)
        return (*r[:3], None, 'combined') if r else None
    elif algorithm == 'pupillabs':
        r = detect_pupil_pupillabs(gray, mask, glint_spots)
        return (*r[:3], None, 'pupillabs') if r else None
    elif algorithm == 'else_lite':
        r = detect_pupil_else_lite(gray, mask, glint_spots)
        return (*r[:3], None, 'else_lite') if r else None
    elif algorithm == 'starburst':
        r = detect_pupil_starburst(gray, mask, glint_spots)
        return (*r[:3], None, 'starburst') if r else None

    # Default: contour-based cascade (original v5)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 5)

    if len(glint_spots) >= 1:
        # Glint centroid = eye area anchor (NOT pupil center)
        gx = np.mean([s[0] for s in glint_spots])
        gy = np.mean([s[1] for s in glint_spots])
        glint_pos = (gx, gy)

        # Radius constraints from glint spread
        if len(glint_spots) >= 2:
            spreads = [np.sqrt((s[0] - gx)**2 + (s[1] - gy)**2) for s in glint_spots]
            glint_spread = np.max(spreads)
            max_r = int(np.clip(glint_spread * 1.8, 10, 18))
            min_r = max(5, int(glint_spread * 0.4))
        else:
            min_r, max_r = 5, 18

        # Build search mask near glints (35px radius for contour enclosure)
        search_r = 35
        search_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(search_mask, (int(round(gx)), int(round(gy))), search_r, 255, -1)
        search_mask = cv2.bitwise_and(search_mask, mask)

        # Primary: adaptive threshold contour detection
        result = _find_pupil_contour(gray, search_mask, glint_pos=glint_pos,
                                     min_r=min_r, max_r=max_r)
        if result is not None:
            return (*result, 'adaptive')  # (cx, cy, r, circ, method)

        # Fallback A: OTSU threshold contour detection
        result = _find_pupil_contour_otsu(gray, search_mask, glint_pos=glint_pos,
                                          min_r=min_r, max_r=max_r)
        if result is not None:
            return (*result, 'otsu')

        # Fallback B: legacy dark blob centroid + radial profile radius
        blob = _dark_blob_centroid_legacy(blurred, search_mask)
        if blob is None:
            cx, cy = int(round(gx)), int(round(gy))
        else:
            cx, cy = blob
        radius = estimate_pupil_radius(gray, cx, cy, min_r=min_r, max_r=max_r)
        return (cx, cy, radius, 0.0, 'blob')

    else:
        # No glints: search center region
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(h, w) * 0.35), 255, -1)
        center_mask = cv2.bitwise_and(center_mask, mask)

        # Primary: contour-based
        result = _find_pupil_contour(gray, center_mask, min_r=5, max_r=18)
        if result is not None:
            return (*result, 'adaptive')

        # Fallback A: OTSU contour
        result = _find_pupil_contour_otsu(gray, center_mask, min_r=5, max_r=18)
        if result is not None:
            return (*result, 'otsu')

        # Fallback B: legacy blob centroid
        blob = _dark_blob_centroid_legacy(blurred, center_mask)
        if blob is None:
            blob = _dark_blob_centroid_legacy(blurred, mask)
        if blob is None:
            return None
        cx, cy = blob
        radius = estimate_pupil_radius(gray, cx, cy, min_r=5, max_r=18)
        return (cx, cy, radius, 0.0, 'blob')


def detect_eye_closed(gray, center, radius=30):
    """
    Open eye: high contrast (dark pupil + bright iris)
    Closed eye: low contrast (uniform eyelid)
    """
    h, w = gray.shape
    cx, cy = center
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return True, 1.0

    std = float(np.std(region))
    p5 = np.percentile(region, 5)
    p95 = np.percentile(region, 95)
    contrast = float(p95 - p5)
    dark_frac = float(np.sum(region < 80)) / region.size

    is_closed = (std < 25 and contrast < 60) or (dark_frac < 0.02 and contrast < 50)
    confidence = 1.0 - min(1.0, std / 40.0)
    return is_closed, confidence


def crop_region(gray, center, crop_size):
    """Crop square ROI centered on target. Zero-pads when near edges."""
    h, w = gray.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = crop_size // 2

    # Ideal crop box (may extend outside image)
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Clamp source region to image bounds
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(w, x2)
    sy2 = min(h, y2)

    # Padding amounts
    pad_left = sx1 - x1
    pad_top = sy1 - y1

    # Extract valid region
    roi = gray[sy1:sy2, sx1:sx2]

    # If full crop fits in image, fast path (no padding needed)
    if roi.shape[0] == crop_size and roi.shape[1] == crop_size:
        local = (cx - x1, cy - y1)
        return roi, (sx1, sy1, sx2, sy2), local

    # Zero-pad to exact crop_size
    if len(gray.shape) == 3:
        padded = np.zeros((crop_size, crop_size, gray.shape[2]), dtype=gray.dtype)
    else:
        padded = np.zeros((crop_size, crop_size), dtype=gray.dtype)
    padded[pad_top:pad_top + roi.shape[0], pad_left:pad_left + roi.shape[1]] = roi

    # Local coords: always at center of crop
    local = (half, half)
    # Bbox uses clamped source coords (for mapping back to original image)
    return padded, (sx1, sy1, sx2, sy2), local


def detect_glints_in_crop(cropped, pupil_pos, pupil_radius=None, max_glints=4):
    """
    Detect glints in cropped image using multi-threshold cascade.
    Aims for 4 glints without increasing false positives.
    """
    h, w = cropped.shape
    px, py = pupil_pos
    sr = pupil_radius * 4 if pupil_radius else min(w, h) * 0.45

    # Multi-threshold cascade
    p995 = np.percentile(cropped, 99.5)
    p990 = np.percentile(cropped, 99.0)
    p985 = np.percentile(cropped, 98.5)
    p975 = np.percentile(cropped, 97.5)

    thresholds = [
        max(p995, 210),
        max(p990, 195),
        max(p985, 180),
        max(p975, 165),
    ]
    thresholds = [min(t, 254) for t in thresholds]

    glints = []
    MIN_GLINT_DIST = 4

    for thresh_val in thresholds:
        _, bright_mask = cv2.threshold(cropped, int(thresh_val), 255, cv2.THRESH_BINARY)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, 8)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            if area < 1 or area > 40:
                continue
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist > sr:
                continue

            # Skip if too close to already-found glint
            too_close = False
            for g in glints:
                if np.sqrt((cx - g["x"])**2 + (cy - g["y"])**2) < MIN_GLINT_DIST:
                    too_close = True
                    break
            if too_close:
                continue

            # Local maximum check (5x5 neighborhood)
            ix, iy = int(round(cx)), int(round(cy))
            y1, y2 = max(0, iy - 2), min(h, iy + 3)
            x1, x2 = max(0, ix - 2), min(w, ix + 3)
            local_patch = cropped[y1:y2, x1:x2]
            if local_patch.size > 0:
                local_max = int(np.max(local_patch))
                spot_val = int(cropped[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0
                if spot_val < local_max - 5:
                    continue

            comp = (labels == i).astype(np.uint8)
            intensity = cv2.mean(cropped, mask=comp)[0]
            glints.append({"x": cx, "y": cy, "area": int(area),
                            "intensity": float(intensity), "dist_to_pupil": float(dist)})

        if len(glints) >= max_glints:
            break

    # Keep brightest if over target
    glints.sort(key=lambda g: g["intensity"], reverse=True)
    return glints[:max_glints]


# ============================================================
# Eye region segmentation (standalone, no detection info needed)
# ============================================================

def segment_eye_region(cropped_gray):
    """
    Segment inner eye region using intensity-based analysis.
    Fully standalone — does NOT use pupil/glint detection results.

    Approach: multi-level Otsu thresholding exploits the natural intensity
    structure of IR eye images (pupil=dark, iris=medium, sclera=bright).
    Spatial coherence and morphology assign final classes.

    Input:  cropped grayscale image (e.g. 150x150)
    Output: class map (H, W) uint8: 0=background, 1=sclera, 2=iris, 3=pupil
    """
    h, w = cropped_gray.shape
    class_map = np.zeros((h, w), dtype=np.uint8)

    # --- Preprocessing ---
    blurred = cv2.GaussianBlur(cropped_gray, (7, 7), 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # --- Multi-level Otsu: find 2 thresholds to separate 3 intensity bands ---
    # Band 0 (dark): pupil candidate
    # Band 1 (medium): iris candidate
    # Band 2 (bright): sclera + skin/background
    # Use 2-level Otsu via histogram analysis
    hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256]).flatten()
    total_pixels = h * w

    # Exhaustive search for optimal 2-threshold Otsu
    best_variance = 0
    t1_best, t2_best = 50, 150
    cumsum = np.cumsum(hist)
    cumsum_val = np.cumsum(hist * np.arange(256))

    for t1 in range(20, 120, 2):
        for t2 in range(t1 + 20, 220, 2):
            w0 = cumsum[t1]
            w1 = cumsum[t2] - cumsum[t1]
            w2 = total_pixels - cumsum[t2]
            if w0 == 0 or w1 == 0 or w2 == 0:
                continue
            m0 = cumsum_val[t1] / w0
            m1 = (cumsum_val[t2] - cumsum_val[t1]) / w1
            m2 = (cumsum_val[255] - cumsum_val[t2]) / w2
            mt = cumsum_val[255] / total_pixels
            var = w0 * (m0 - mt)**2 + w1 * (m1 - mt)**2 + w2 * (m2 - mt)**2
            if var > best_variance:
                best_variance = var
                t1_best, t2_best = t1, t2

    # --- Create initial band masks ---
    dark_mask = (enhanced <= t1_best).astype(np.uint8)    # pupil candidates
    mid_mask = ((enhanced > t1_best) & (enhanced <= t2_best)).astype(np.uint8)  # iris candidates
    bright_mask = (enhanced > t2_best).astype(np.uint8)   # sclera + skin

    # --- Morphological cleanup ---
    kern_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Clean pupil: open to remove noise, close to fill gaps
    pupil_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kern_s, iterations=1)
    pupil_clean = cv2.morphologyEx(pupil_clean, cv2.MORPH_CLOSE, kern_m, iterations=2)

    # --- Find pupil as the most circular dark blob near image center ---
    contours, _ = cv2.findContours(pupil_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cx, img_cy = w // 2, h // 2
    min_pupil_area = 80   # minimum ~5px radius
    max_pupil_area = 2500  # maximum ~28px radius

    best_pupil = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_pupil_area or area > max_pupil_area:
            continue
        circ = _contour_circularity(cnt)
        if circ < 0.35:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # Score: circularity + proximity to center
        dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        proximity = max(0, 1.0 - dist / (min(h, w) * 0.4))
        score = circ * 0.5 + proximity * 0.5
        if score > best_score:
            best_score = score
            best_pupil = (cnt, int(round(cx)), int(round(cy)), int(round(r)))

    if best_pupil is None:
        # Fallback: use darkest region near center
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (img_cx, img_cy), min(h, w) // 3, 255, -1)
        masked = enhanced.copy()
        masked[center_mask == 0] = 255
        min_val, _, min_loc, _ = cv2.minMaxLoc(masked)
        pcx, pcy = min_loc
        pr = 8
    else:
        _, pcx, pcy, pr = best_pupil
        pr = max(5, pr)

    # --- Estimate iris boundary via radial intensity profile ---
    profile = radial_intensity_profile(blurred, pcx, pcy,
                                       max_r=min(60, min(h, w) // 2), n_angles=36)
    gradient = np.diff(profile)
    search_start = pr + 2
    search_end = min(len(gradient), pr * 5, min(h, w) // 2)

    iris_radius = int(pr * 2.5)  # default
    if search_start < search_end:
        search_region = gradient[search_start:search_end]
        if len(search_region) >= 3:
            kernel_1d = np.ones(3) / 3
            search_region = np.convolve(search_region, kernel_1d, mode='same')
        if len(search_region) > 0 and np.max(search_region) > 1.5:
            iris_radius = int(np.argmax(search_region) + search_start)

    iris_radius = max(pr + 3, min(iris_radius, min(h, w) // 2 - 2))

    # --- Detect eye opening (sclera boundary) ---
    # Eye opening = bright + medium region that's spatially connected to the iris
    eye_region = ((enhanced > t1_best)).astype(np.uint8) * 255  # everything above pupil threshold
    eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_CLOSE, kern_m, iterations=2)
    eye_region = cv2.morphologyEx(eye_region, cv2.MORPH_OPEN, kern_s, iterations=1)

    # Find connected component touching the iris area
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(eye_region, connectivity=8)
    # Check which component contains the iris boundary points
    eye_label = 0
    for dy, dx in [(iris_radius, 0), (-iris_radius, 0), (0, iris_radius), (0, -iris_radius),
                   (pr, 0), (-pr, 0), (0, pr), (0, -pr)]:
        ty, tx = pcy + dy, pcx + dx
        if 0 <= ty < h and 0 <= tx < w and labels[ty, tx] > 0:
            eye_label = labels[ty, tx]
            break

    if eye_label > 0:
        eye_opening = (labels == eye_label).astype(np.uint8) * 255
    else:
        eye_opening = eye_region

    # Limit to reasonable distance from detected pupil center
    dist_limit = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(dist_limit, (pcx, pcy), iris_radius + 25, 255, -1)
    eye_opening = cv2.bitwise_and(eye_opening, dist_limit)

    # --- Assign classes ---
    # 3: Pupil
    pupil_mask = np.zeros((h, w), dtype=np.uint8)
    if best_pupil is not None:
        cv2.drawContours(pupil_mask, [best_pupil[0]], -1, 255, -1)
    else:
        cv2.circle(pupil_mask, (pcx, pcy), pr, 255, -1)
    class_map[pupil_mask > 0] = 3

    # 2: Iris (ring between pupil and iris boundary, within eye opening)
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, (pcx, pcy), iris_radius, 255, -1)
    iris_mask[pupil_mask > 0] = 0
    iris_mask = cv2.bitwise_and(iris_mask, eye_opening)
    class_map[iris_mask > 0] = 2

    # 1: Sclera (eye opening outside iris)
    sclera_mask = eye_opening.copy()
    cv2.circle(sclera_mask, (pcx, pcy), iris_radius, 0, -1)
    class_map[sclera_mask > 0] = 1

    return class_map


_worldcoin_sess = None

def _get_worldcoin_session():
    """Lazy-load Worldcoin ONNX session (singleton)."""
    global _worldcoin_sess
    if _worldcoin_sess is None:
        import onnxruntime as ort
        model_path = str(Path(__file__).parent / "models" / "iris_semseg_upp_scse_mobilenetv2.onnx")
        _worldcoin_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return _worldcoin_sess


def segment_eye_worldcoin(cropped_gray):
    """
    Segment eye region using Worldcoin iris-semantic-segmentation model.
    UNet++ with MobileNetV2 encoder, trained on IR eye images.

    Input:  cropped grayscale image (any size, resized to 640x480 internally)
    Output: class map (H, W) uint8: 0=background, 1=sclera, 2=iris, 3=pupil
    """
    sess = _get_worldcoin_session()
    h_orig, w_orig = cropped_gray.shape

    # Resize to model input: 640w x 480h
    resized = cv2.resize(cropped_gray, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Preprocessing: 3-channel (duplicate grayscale), ImageNet normalization
    img3 = np.stack([resized, resized, resized], axis=0).astype(np.float32)
    img3 = img3[np.newaxis] / 255.0  # (1, 3, 480, 640)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    img3 = (img3 - mean) / std

    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: img3})[0]  # (1, 4, 480, 640)

    # Worldcoin classes: 0=eyeball, 1=iris, 2=pupil, 3=eyelashes
    # Map to our classes: 0=background, 1=sclera/eyeball, 2=iris, 3=pupil
    probs = 1.0 / (1.0 + np.exp(-out[0]))  # sigmoid
    masks = (probs > 0.5).astype(np.uint8)

    # Build class map with priority: pupil > iris > eyeball > background
    class_map_full = np.zeros((480, 640), dtype=np.uint8)
    class_map_full[masks[0] > 0] = 1   # eyeball → sclera
    class_map_full[masks[1] > 0] = 2   # iris
    class_map_full[masks[2] > 0] = 3   # pupil
    # eyelashes (masks[3]) → leave as background

    # Morphological cleanup
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cls_val in [1, 2, 3]:
        m = (class_map_full == cls_val).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern, iterations=1)
        class_map_full[m > 0] = cls_val

    # Resize back to original crop size
    class_map = cv2.resize(class_map_full, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return class_map


SEG_ALGORITHMS = ['worldcoin', 'ritnet', 'classical']

# ── RITnet segmentation model (lightweight, 249K params, >300fps) ──
_ritnet_sess = None

def _get_ritnet_session():
    """Lazy-load RITnet ONNX session (singleton)."""
    global _ritnet_sess
    if _ritnet_sess is None:
        import onnxruntime as ort
        model_path = str(Path(__file__).parent / "models" / "ritnet.onnx")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"RITnet model not found: {model_path}")
        _ritnet_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return _ritnet_sess


def segment_eye_ritnet(cropped_gray):
    """Segment eye using RITnet (DenseNet2D, 4-class).

    RITnet outputs 4 classes: 0=background, 1=sclera, 2=iris, 3=pupil
    (Same mapping as our pipeline standard.)

    Input: grayscale cropped image (any size)
    Output: class map same size as input (0=bg, 1=sclera, 2=iris, 3=pupil)
    """
    sess = _get_ritnet_session()
    h_orig, w_orig = cropped_gray.shape[:2]

    # RITnet expects input divisible by 16
    # Resize to 640x400 (width x height, matching exported ONNX dims)
    target_h, target_w = 400, 640
    resized = cv2.resize(cropped_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1] (RITnet trained with transforms.Normalize(mean=0.5, std=0.5))
    inp = (resized.astype(np.float32) / 127.5) - 1.0
    inp = inp[np.newaxis, np.newaxis]  # (1, 1, 400, 640)

    # Run inference
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: inp})[0]  # (1, 4, H', W')

    # Output may be slightly different size due to 16-pixel quantization
    n_classes = out.shape[1]
    out_h, out_w = out.shape[2], out.shape[3]

    # Argmax to get class map
    class_map = np.argmax(out[0], axis=0).astype(np.uint8)  # (H', W')

    # Resize back to original dimensions
    if out_h != h_orig or out_w != w_orig:
        class_map = cv2.resize(class_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Morphological cleanup per class (same as worldcoin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cls in [1, 2, 3]:
        mask = (class_map == cls).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        class_map[mask > 0] = cls

    # Ensure priority: pupil > iris > sclera
    final = np.zeros_like(class_map)
    final[class_map == 1] = 1
    final[class_map == 2] = 2
    final[class_map == 3] = 3

    return final




def draw_segmentation_overlay(gray, seg_mask, alpha=0.4):
    """
    Draw colored segmentation overlay on grayscale image.

    Colors (BGR):
      Background: transparent
      Sclera:     cyan   (255, 255, 0)
      Iris:       blue   (255, 100, 0)
      Pupil:      red    (0, 0, 255)

    Also draws yellow contour outline of inner eye boundary
    (sclera+iris+pupil combined vs background).

    Returns: BGR overlay image.
    """
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    # Color map: class -> BGR
    colors = {
        1: (255, 255, 0),    # sclera: cyan
        2: (255, 100, 0),    # iris: blue
        3: (0, 0, 255),      # pupil: red
    }

    for cls, color in colors.items():
        mask = (seg_mask == cls)
        overlay[mask] = color

    # Alpha blend (only where non-background)
    fg_mask = seg_mask > 0
    result = base.copy()
    result[fg_mask] = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)[fg_mask]

    # Draw yellow contour outline of inner eye boundary
    eye_mask = (seg_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)

    # Add class labels
    cv2.putText(result, "SEG", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return result


def enhance_for_glint_detection(gray):
    """
    Enhance image for glint detection (not visualization).
    Uses white top-hat to isolate small bright spots + CLAHE for contrast.
    Returns a grayscale image optimized for find_corneal_glints().
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(tophat)
    if enhanced.max() > 0:
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced


def enhance_for_pupil_detection(gray):
    """
    Enhance image for pupil detection (not visualization).
    Uses bilateral filter (edge-preserving) + CLAHE + DoG at pupil scale.
    Returns a grayscale image where the pupil blob is maximally enhanced.
    """
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    # DoG: large-scale mean minus local = bright where dark blobs exist
    g_small = cv2.GaussianBlur(enhanced, (0, 0), 2)
    g_large = cv2.GaussianBlur(enhanced, (0, 0), 12)
    dog = cv2.subtract(g_large, g_small)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return dog


def detect_pupil_in_seg_mask(gray, seg_mask, eye_region=None, glint_positions=None):
    """
    Detect pupil center within the segmented eye region.
    PRIMARY: uses seg mask class 3 (pupil) contour from the ML model.
    FALLBACK: adaptive thresholding on grayscale to find darkest blob.
    Returns (cx, cy, radius) in cropped coords, or None.
    """
    # --- Primary: seg mask class 3 contour (most reliable) ---
    pupil_mask = (seg_mask == 3).astype(np.uint8) * 255
    if cv2.countNonZero(pupil_mask) >= 10:
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest class 3 contour
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area >= 10:
                (cx, cy), radius = cv2.minEnclosingCircle(biggest)
                return (float(cx), float(cy), max(int(radius), 3))

    # --- Fallback: class 3 centroid (if contour too fragmented) ---
    pupil_ys, pupil_xs = np.where(seg_mask == 3)
    if len(pupil_xs) >= 5:
        px = float(np.mean(pupil_xs))
        py = float(np.mean(pupil_ys))
        radius = max(int(np.sqrt(len(pupil_xs) / np.pi)), 3)
        return (px, py, radius)

    # --- Last resort: adaptive threshold darkest blob near glints ---
    if eye_region is None:
        eye_region = (seg_mask > 0).astype(np.uint8) * 255
        eye_region = cv2.dilate(eye_region, np.ones((5, 5), np.uint8))

    if cv2.countNonZero(eye_region) < 20:
        return None

    glint_cx, glint_cy = None, None
    MAX_PUPIL_GLINT_DIST = 50
    if glint_positions and len(glint_positions) >= 2:
        glint_cx = np.mean([g["x"] for g in glint_positions])
        glint_cy = np.mean([g["y"] for g in glint_positions])
        h, w = gray.shape[:2]
        glint_focus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(glint_focus, (int(glint_cx), int(glint_cy)),
                   MAX_PUPIL_GLINT_DIST, 255, -1)
        eye_region = cv2.bitwise_and(eye_region, glint_focus)

    if cv2.countNonZero(eye_region) < 20:
        return None

    masked = gray.copy()
    masked[eye_region == 0] = 200

    blurred = cv2.GaussianBlur(masked, (9, 9), 2)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 4)
    th = cv2.bitwise_and(th, eye_region)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 8000:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            continue
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < 0.25:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        score = area * circ
        if score > best_score:
            best_score = score
            best = cnt

    if best is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(best)
        return (float(cx), float(cy), max(int(radius), 3))

    return None


def preprocess_for_glints(gray):
    """
    Preprocess cropped eye image to enhance glint (corneal reflection) visibility.
    Uses white top-hat transform to isolate small bright specular reflections,
    then CLAHE for local contrast enhancement. Displayed as a heatmap overlay.
    """
    # White top-hat: morphological opening subtracted from original
    # Extracts bright features smaller than the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # CLAHE on top-hat to boost local bright spots
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(tophat)

    # Normalize to full range
    if enhanced.max() > 0:
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Colormap for striking visualization
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)

    # Blend with dim original for spatial context
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(gray_bgr, 0.3, heatmap, 0.7, 0)

    cv2.putText(result, "GLINT ENHANCED", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
    return result


def preprocess_for_pupil(gray):
    """
    Preprocess cropped eye image to enhance pupil visibility.
    Uses bilateral filtering (edge-preserving smooth), CLAHE for contrast,
    and Difference-of-Gaussians (DoG) for dark blob enhancement at pupil scale.
    Overlays Canny edges for boundary clarity.
    """
    # Bilateral filter: smooths noise while preserving the pupil boundary
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # Difference of Gaussians at pupil scale (~10-20px radius)
    # Large sigma captures the dark pupil blob, small sigma the texture
    g_small = cv2.GaussianBlur(enhanced, (0, 0), 2)
    g_large = cv2.GaussianBlur(enhanced, (0, 0), 12)
    # g_large - g_small: positive where large-scale mean is brighter than local = dark blobs
    dog = cv2.subtract(g_large, g_small)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Canny edges on the enhanced image for boundary detection
    med = np.median(enhanced)
    edges = cv2.Canny(enhanced, int(max(0, 0.5 * med)), int(min(255, 1.5 * med)))
    # Dilate edges slightly for visibility
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    # Compose: enhanced as base, DoG as blue tint for pupil region, edges as green overlay
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    # Blue tint where DoG is strong (dark blob = pupil candidate)
    result[:, :, 0] = cv2.add(result[:, :, 0], (dog * 0.4).astype(np.uint8))
    # Green edges for pupil boundary
    result[:, :, 1] = cv2.add(result[:, :, 1], (edges * 0.7).astype(np.uint8))

    cv2.putText(result, "PUPIL ENHANCED", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return result


# ============================================================
# Two-pass offline processing
# ============================================================

def process_all(input_dir, output_dir, crop_size=150, camera='ri', cam_calib=None, algorithm='contour',
                seg_enabled=False, seg_algo='worldcoin', camera_batch='default', crop_method='blob'):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    subs = ["cropped", "debug", "glint_debug"]
    if seg_enabled:
        subs.append("seg_debug")
        subs.append("seg_gaze_debug")
        subs.append("seg_masks")
        subs.append("glint_enhanced")
        subs.append("pupil_enhanced")
        subs.append("seg_glint_debug")
    for sub in subs:
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    frames = sorted(input_path.glob("*.png"), key=lambda p: _nk(p.name))
    print(f"[{camera.upper()}] Found {len(frames)} frames\n")

    # ---- Pass 1: Raw detection with temporal fallback ----
    print(f"Pass 1: Raw detection (algorithm={algorithm})...")
    mask = None
    raw = []
    prev_pupil = None  # temporal tracking for no-glint fallback
    MAX_JUMP = 40      # reject detections that jump more than this from previous

    for i, fp in enumerate(frames):
        gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raw.append(None)
            continue
        if mask is None:
            mask = build_search_mask(*gray.shape, camera=camera, camera_batch=camera_batch)
            # Pre-build fallback mask (no hardware block, borders only) for adaptive retry
            _h, _w = gray.shape
            fallback_mask = np.ones((_h, _w), dtype=np.uint8) * 255
            fallback_mask[0:BORDER, :] = 0
            fallback_mask[_h - BORDER:_h, :] = 0
            fallback_mask[:, 0:BORDER] = 0
            fallback_mask[:, _w - BORDER:_w] = 0

        glints = find_corneal_glints(gray, mask)
        pupil = detect_pupil_raw(gray, mask, glints, algorithm=algorithm)

        # Extract detection metadata from 5-tuple, trim pupil to 3-tuple
        det_circularity = None
        det_method = None
        if pupil is not None and len(pupil) >= 5:
            det_circularity = pupil[3]
            det_method = pupil[4]
            pupil = (pupil[0], pupil[1], pupil[2])

        # Adaptive mask fallback: if primary mask yields no pupil AND no glints, try open mask
        if pupil is None and len(glints) == 0 and fallback_mask is not None:
            glints = find_corneal_glints(gray, fallback_mask)
            pupil = detect_pupil_raw(gray, fallback_mask, glints, algorithm=algorithm)
            if pupil is not None and len(pupil) >= 5:
                det_circularity = pupil[3]
                det_method = 'fallback_mask_' + str(pupil[4])
                pupil = (pupil[0], pupil[1], pupil[2])
            elif pupil is not None:
                det_method = 'fallback_mask'

        # If no glints and we have history, re-detect using previous position as anchor
        # (temporal fallback only for contour algorithm, others handle this internally)
        if algorithm == 'contour' and len(glints) == 0 and prev_pupil is not None:
            h, w = gray.shape
            # Search near previous pupil position using contour detection
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(local_mask, (prev_pupil[0], prev_pupil[1]), 50, 255, -1)
            local_mask = cv2.bitwise_and(local_mask, mask)

            # Try contour-based detection near previous position
            result = _find_pupil_contour(gray, local_mask, min_r=5, max_r=18)
            if result is not None:
                det_circularity = result[3]
                det_method = 'adaptive_temporal'
                pupil = (result[0], result[1], result[2])
            else:
                result = _find_pupil_contour_otsu(gray, local_mask, min_r=5, max_r=18)
                if result is not None:
                    det_circularity = result[3]
                    det_method = 'otsu_temporal'
                    pupil = (result[0], result[1], result[2])
                else:
                    # Last resort: legacy blob centroid
                    blurred = cv2.GaussianBlur(gray, (15, 15), 5)
                    blob = _dark_blob_centroid_legacy(blurred, local_mask)
                    if blob is not None:
                        cx, cy = blob
                        radius = estimate_pupil_radius(gray, cx, cy, min_r=5, max_r=18)
                        pupil = (cx, cy, radius)
                        det_circularity = 0.0
                        det_method = 'blob_temporal'

        # Jump rejection: if detection jumps too far from previous, keep previous
        if pupil is not None and prev_pupil is not None:
            dx = pupil[0] - prev_pupil[0]
            dy = pupil[1] - prev_pupil[1]
            jump = np.sqrt(dx**2 + dy**2)
            if jump > MAX_JUMP:
                # Likely false detection, use previous position with re-estimated radius
                pupil = (prev_pupil[0], prev_pupil[1],
                         estimate_pupil_radius(gray, prev_pupil[0], prev_pupil[1], 6, 25))
                det_method = 'jump_rejected'
                det_circularity = None

        if pupil:
            prev_pupil = pupil
            ec, ec_conf = detect_eye_closed(gray, (pupil[0], pupil[1]))
        else:
            ec, ec_conf = True, 1.0

        raw.append({"pupil": pupil, "glints": glints, "eye_closed": ec, "ec_conf": ec_conf,
                     "circularity": det_circularity, "det_method": det_method})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(frames)}")

    # ---- Pass 2: Kalman smooth on GLINT centroid + generate outputs ----
    # Kalman tracks glint positions (very stable) not pupil (moves with gaze).
    # Very high measurement noise = sluggish, won't jump around.
    # Crop center: prefer glint centroid. Only use pupil if close to glints.
    print("\nPass 2: Kalman smoothing (glint-based) + output...")
    kalman = BoxKalman(process_noise=0.05, measurement_noise=80.0)
    all_results = []
    PUPIL_GLINT_CLOSE = 25  # px: if pupil < this from glints, they agree

    for sub in ["gaze_debug"]:
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(frames):
        gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        r = raw[i]
        out = {
            "frame": fp.name, "pupil_detected": False, "pupil_center": None,
            "pupil_radius": None, "smoothed_center": None, "crop_bbox": None,
            "eye_closed": True, "closed_confidence": 1.0, "glints": [],
            "gaze_angle_deg": None, "gaze_vector": None,
            "detection_confidence": 0.0, "detection_method": None,
            "pupil_circularity": None,
        }

        if r is None or gray is None:
            all_results.append(out)
            continue

        pupil = r["pupil"]
        glint_spots = r["glints"]
        out["eye_closed"] = r["eye_closed"]
        out["closed_confidence"] = r["ec_conf"]
        out["detection_method"] = r.get("det_method")
        out["pupil_circularity"] = round(r["circularity"], 3) if r.get("circularity") is not None else None

        if pupil:
            out["pupil_detected"] = True
            out["pupil_center"] = [pupil[0], pupil[1]]
            out["pupil_radius"] = pupil[2]

            # Composite detection confidence (0.0 - 1.0)
            conf = 0.2  # base: pupil was detected

            # Circularity bonus (0 - 0.25): higher = more circular = more reliable
            circ = r.get("circularity")
            if circ is not None and circ > 0:
                conf += min(0.25, circ * 0.25)

            # Glint count bonus (0 - 0.25): 4 glints = ideal
            n_glints = len(glint_spots)
            conf += min(0.25, n_glints / 4.0 * 0.25)

            # Detection method bonus (0 - 0.15): primary > OTSU > blob
            method = r.get("det_method", "")
            if method in ('adaptive', 'adaptive_temporal'):
                conf += 0.15
            elif method in ('otsu', 'otsu_temporal'):
                conf += 0.08
            elif method == 'jump_rejected':
                conf += 0.0  # unreliable
            elif method in ('blob', 'blob_temporal'):
                conf += 0.02

            # Eye-open bonus (0 - 0.1): open eye = more reliable detection
            if not r["eye_closed"]:
                conf += 0.1

            # Pupil-glint separation reasonableness (0 - 0.05)
            if pupil and n_glints >= 1:
                gcx = float(np.mean([s[0] for s in glint_spots]))
                gcy = float(np.mean([s[1] for s in glint_spots]))
                pg_dist = np.sqrt((pupil[0] - gcx)**2 + (pupil[1] - gcy)**2)
                # Reasonable range: 3-25px. Outside = less confident
                if 3.0 <= pg_dist <= 25.0:
                    conf += 0.05
                elif pg_dist <= 35.0:
                    conf += 0.02

            out["detection_confidence"] = round(min(1.0, conf), 3)

        # --- Crop center: pupil + glint agreement check ---
        # Find glints and pupil. If distance is unreasonable, use the
        # densest glint cluster center (glints are always on the cornea,
        # so the densest cluster reliably marks the eye region).
        PUPIL_GLINT_UNREASONABLE = 40  # px: beyond this, pupil detection is suspect

        def _densest_glint_center(spots):
            """Find center of the densest glint cluster."""
            if len(spots) <= 2:
                # With 1-2 glints just use centroid
                return float(np.mean([s[0] for s in spots])), float(np.mean([s[1] for s in spots]))
            # For 3+ glints: find the pair with minimum mutual distance,
            # then include all glints within 30px of that pair's midpoint
            pts = np.array([[s[0], s[1]] for s in spots], dtype=float)
            best_d = 1e9
            best_i, best_j = 0, 1
            for ii in range(len(pts)):
                for jj in range(ii + 1, len(pts)):
                    d = np.linalg.norm(pts[ii] - pts[jj])
                    if d < best_d:
                        best_d = d
                        best_i, best_j = ii, jj
            core = (pts[best_i] + pts[best_j]) / 2.0
            # Gather all glints within 30px of core
            cluster = [pts[k] for k in range(len(pts))
                       if np.linalg.norm(pts[k] - core) < 30.0]
            if not cluster:
                cluster = [pts[best_i], pts[best_j]]
            c = np.mean(cluster, axis=0)
            return float(c[0]), float(c[1])

        gcx, gcy = None, None
        if len(glint_spots) >= 1:
            gcx, gcy = _densest_glint_center(glint_spots)

        if gcx is not None and pupil:
            pg_dist = np.sqrt((pupil[0] - gcx)**2 + (pupil[1] - gcy)**2)
            if pg_dist < PUPIL_GLINT_UNREASONABLE:
                # Reasonable distance — pupil and glints agree, trust pupil
                feed_x, feed_y = float(pupil[0]), float(pupil[1])
            else:
                # Unreasonable — pupil detection is wrong, use glint cluster
                feed_x, feed_y = gcx, gcy
                out["detection_method"] = (out.get("detection_method") or "") + "_glint_override"
        elif gcx is not None:
            # No pupil detected, use glint cluster
            feed_x, feed_y = gcx, gcy
        elif pupil:
            feed_x, feed_y = float(pupil[0]), float(pupil[1])
        else:
            # Fallback: dark-blob centroid (pupil is darkest region in IR image)
            blurred_fb = cv2.GaussianBlur(gray, (21, 21), 7)
            dark_thresh = np.percentile(blurred_fb[mask > 0], 15) if np.any(mask > 0) else 50
            dark_region = ((blurred_fb < dark_thresh) & (mask > 0)).astype(np.uint8)
            # Morphological cleanup
            kernel_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dark_region = cv2.morphologyEx(dark_region, cv2.MORPH_CLOSE, kernel_fb)
            dark_region = cv2.morphologyEx(dark_region, cv2.MORPH_OPEN, kernel_fb)
            # Find largest dark blob
            num_cc, labels_cc, stats_cc, centroids_cc = cv2.connectedComponentsWithStats(
                dark_region, connectivity=8)
            if num_cc > 1:
                # Skip background (label 0)
                areas = stats_cc[1:, cv2.CC_STAT_AREA]
                largest = 1 + int(np.argmax(areas))
                if areas[largest - 1] > 20:  # minimum 20px area
                    feed_x = float(centroids_cc[largest][0])
                    feed_y = float(centroids_cc[largest][1])
                    out["detection_method"] = "dark_blob_fallback"
                else:
                    feed_x, feed_y = None, None
            else:
                feed_x, feed_y = None, None

        if feed_x is not None:
            if not kalman.initialized:
                kalman.init(feed_x, feed_y)
                sx, sy = feed_x, feed_y
            else:
                kx, ky = kalman.predict()
                dist_to_kalman = np.sqrt((feed_x - kx)**2 + (feed_y - ky)**2)

                SNAP_THRESHOLD = 15
                if dist_to_kalman > SNAP_THRESHOLD:
                    kalman.init(feed_x, feed_y)
                    sx, sy = feed_x, feed_y
                else:
                    sx, sy = kalman.correct(feed_x, feed_y)
        else:
            if kalman.initialized:
                sx, sy = kalman.predict()
            else:
                # Last resort: image center (ensures we always produce a crop)
                h_img, w_img = gray.shape[:2]
                sx, sy = float(w_img // 2), float(h_img // 2)
                out["detection_method"] = "image_center_fallback"

        sc = (int(round(sx)), int(round(sy)))
        out["smoothed_center"] = list(sc)

        # Crop using smoothed center
        cropped, bbox, p_local = crop_region(gray, sc, crop_size)
        out["crop_bbox"] = list(bbox)

        # Glint detection on cropped
        pr = pupil[2] if pupil else 15
        fine_glints = detect_glints_in_crop(cropped, p_local, pupil_radius=pr)
        for g in fine_glints:
            g["x_orig"] = g["x"] + bbox[0]
            g["y_orig"] = g["y"] + bbox[1]
        # Filter out glints that fall in the hardware-masked area
        h_img, w_img = gray.shape
        fine_glints = [g for g in fine_glints
                       if 0 <= int(g["y_orig"]) < h_img
                       and 0 <= int(g["x_orig"]) < w_img
                       and mask[int(g["y_orig"]), int(g["x_orig"])] > 0]
        out["glints"] = [
            {"x": g["x"], "y": g["y"], "x_orig": g.get("x_orig"), "y_orig": g.get("y_orig"),
             "area": g["area"], "intensity": g["intensity"]}
            for g in fine_glints
        ]

        # --- Gaze angle estimation ---
        gaze_vx, gaze_vy, gaze_angle = 0.0, 0.0, None
        if pupil and len(fine_glints) >= 1:
            # Pixel-space gaze (for visualization arrows)
            glint_cx = np.mean([g["x"] for g in fine_glints])
            glint_cy = np.mean([g["y"] for g in fine_glints])
            pcx = pupil[0] - bbox[0]
            pcy = pupil[1] - bbox[1]
            gaze_vx = float(pcx - glint_cx)
            gaze_vy = float(pcy - glint_cy)

            if cam_calib is not None:
                # Undistorted normalized gaze using camera intrinsics
                glint_cx_orig = np.mean([g["x_orig"] for g in fine_glints])
                glint_cy_orig = np.mean([g["y_orig"] for g in fine_glints])
                gnx, gny = undistort_gaze(
                    (float(pupil[0]), float(pupil[1])),
                    (float(glint_cx_orig), float(glint_cy_orig)),
                    cam_calib['K'], cam_calib['dist'])
                out["gaze_vector_norm"] = [round(gnx, 6), round(gny, 6)]
                gaze_mag_n = np.sqrt(gnx**2 + gny**2)
                if gaze_mag_n > 0.001:
                    gaze_angle = float(np.degrees(np.arctan2(-gny, gnx)))
            else:
                # Fallback: raw pixel gaze
                gaze_mag = np.sqrt(gaze_vx**2 + gaze_vy**2)
                if gaze_mag > 1.0:
                    gaze_angle = float(np.degrees(np.arctan2(-gaze_vy, gaze_vx)))

            if gaze_angle is not None:
                out["gaze_angle_deg"] = round(gaze_angle, 1)
                out["gaze_vector"] = [round(gaze_vx, 1), round(gaze_vy, 1)]

        # ---- Save cropped ----
        cv2.imwrite(str(out_path / "cropped" / (fp.stem + "_cropped.png")), cropped)

        # ---- Save segmentation overlay on cropped (if enabled) ----
        if seg_enabled:
            try:
                # Try loading cached seg mask first
                seg_cache_path = out_path / "seg_masks" / (fp.stem + "_seg.npz")
                if seg_cache_path.exists():
                    cached = np.load(str(seg_cache_path))
                    seg_mask = cached["mask"]
                else:
                    if seg_algo == 'worldcoin':
                        seg_mask = segment_eye_worldcoin(cropped)
                    elif seg_algo == 'ritnet':
                        seg_mask = segment_eye_ritnet(cropped)
                    else:
                        seg_mask = segment_eye_region(cropped)
                    # Cache mask + crop bbox (for mapping back to original coords)
                    np.savez_compressed(str(seg_cache_path), mask=seg_mask,
                                        bbox=np.array(bbox, dtype=np.int32))
                seg_img = draw_segmentation_overlay(cropped, seg_mask)

                # Clean seg mask: keep only the largest connected component.
                # Removes scattered fragments from glasses hardware, eyelids, edges.
                seg_binary = (seg_mask > 0).astype(np.uint8)
                num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(
                    seg_binary, connectivity=8)
                if num_cc > 2:  # background + 2+ foreground = needs cleaning
                    largest_label = 1 + np.argmax(stats_cc[1:, cv2.CC_STAT_AREA])
                    clean_mask = (labels_cc == largest_label).astype(np.uint8)
                    seg_mask = seg_mask * clean_mask

                # ========== TWO-PASS RECENTER: check if seg pupil is far from crop center ==========
                # Quick seg pupil check on first-pass crop
                _rc_pupil_ys, _rc_pupil_xs = np.where(seg_mask == 3)
                _rc_iris_ys, _rc_iris_xs = np.where(seg_mask == 2)
                _rc_seg_px = float(np.mean(_rc_pupil_xs)) if len(_rc_pupil_xs) > 20 else None
                _rc_seg_py = float(np.mean(_rc_pupil_ys)) if len(_rc_pupil_xs) > 20 else None
                _rc_iris_cx = float(np.mean(_rc_iris_xs)) if len(_rc_iris_xs) > 20 else None
                _rc_iris_cy = float(np.mean(_rc_iris_ys)) if len(_rc_iris_xs) > 20 else None

                RECENTER_THRESHOLD = 10  # px: only recenter if seg center is this far from crop center
                _rc_half = crop_size // 2
                _rc_crop_cx, _rc_crop_cy = float(_rc_half), float(_rc_half)  # crop center in local coords

                if _rc_seg_px is not None:
                    _rc_dist = np.sqrt((_rc_seg_px - _rc_crop_cx)**2 + (_rc_seg_py - _rc_crop_cy)**2)
                    if _rc_dist > RECENTER_THRESHOLD:
                        # Compute weighted center from available sources (in ORIGINAL image coords)
                        _rc_weights = []
                        _rc_points = []
                        # seg pupil (weight 3) - most reliable
                        _rc_points.append((_rc_seg_px + bbox[0], _rc_seg_py + bbox[1]))
                        _rc_weights.append(3.0)
                        # seg iris (weight 2) - good backup
                        if _rc_iris_cx is not None:
                            _rc_points.append((_rc_iris_cx + bbox[0], _rc_iris_cy + bbox[1]))
                            _rc_weights.append(2.0)
                        # contour pupil (weight 1) - least reliable
                        if pupil is not None:
                            _rc_points.append((float(pupil[0]), float(pupil[1])))
                            _rc_weights.append(1.0)

                        _rc_w_total = sum(_rc_weights)
                        _rc_new_cx = sum(p[0] * w for p, w in zip(_rc_points, _rc_weights)) / _rc_w_total
                        _rc_new_cy = sum(p[1] * w for p, w in zip(_rc_points, _rc_weights)) / _rc_w_total
                        _rc_new_center = (int(round(_rc_new_cx)), int(round(_rc_new_cy)))

                        # Re-crop with padded crop_region (handles edge cases with zero-padding)
                        cropped, bbox, p_local = crop_region(gray, _rc_new_center, crop_size)
                        out["crop_bbox"] = list(bbox)
                        out["smoothed_center"] = list(_rc_new_center)
                        sc = _rc_new_center

                        # Re-detect glints on recentered crop
                        pr = pupil[2] if pupil else 15
                        fine_glints = detect_glints_in_crop(cropped, p_local, pupil_radius=pr)
                        for g in fine_glints:
                            g["x_orig"] = g["x"] + bbox[0]
                            g["y_orig"] = g["y"] + bbox[1]
                        h_img, w_img = gray.shape
                        fine_glints = [g for g in fine_glints
                                       if 0 <= int(g["y_orig"]) < h_img
                                       and 0 <= int(g["x_orig"]) < w_img
                                       and mask[int(g["y_orig"]), int(g["x_orig"])] > 0]
                        out["glints"] = [
                            {"x": g["x"], "y": g["y"], "x_orig": g.get("x_orig"), "y_orig": g.get("y_orig"),
                             "area": g["area"], "intensity": g["intensity"]}
                            for g in fine_glints
                        ]

                        # Re-run segmentation on recentered crop
                        if seg_algo == 'worldcoin':
                            seg_mask = segment_eye_worldcoin(cropped)
                        elif seg_algo == 'ritnet':
                            seg_mask = segment_eye_ritnet(cropped)
                        else:
                            seg_mask = segment_eye_region(cropped)

                        # Update seg cache with recentered crop
                        np.savez_compressed(str(seg_cache_path), mask=seg_mask,
                                            bbox=np.array(bbox, dtype=np.int32))

                        # Re-clean seg mask (largest connected component)
                        seg_binary = (seg_mask > 0).astype(np.uint8)
                        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(
                            seg_binary, connectivity=8)
                        if num_cc > 2:
                            largest_label = 1 + np.argmax(stats_cc[1:, cv2.CC_STAT_AREA])
                            clean_mask = (labels_cc == largest_label).astype(np.uint8)
                            seg_mask = seg_mask * clean_mask

                        seg_img = draw_segmentation_overlay(cropped, seg_mask)

                        # Update cropped image save
                        cv2.imwrite(str(out_path / "cropped" / (fp.stem + "_cropped.png")), cropped)

                        # Mark that recentering happened
                        out["recentered"] = True
                        out["recenter_shift_px"] = round(_rc_dist, 1)
                # ========== END TWO-PASS RECENTER ==========

                # Eye region for basic seg_glints
                eye_region = (seg_mask > 0).astype(np.uint8) * 255
                eye_region = cv2.dilate(eye_region, np.ones((7, 7), np.uint8))
                seg_glints_raw = find_corneal_glints(cropped, eye_region)
                seg_fine_glints = []
                for gx, gy, ga in seg_glints_raw:
                    seg_fine_glints.append({
                        "x": round(float(gx), 2),
                        "y": round(float(gy), 2),
                        "area": int(ga),
                        "x_orig": round(float(gx) + bbox[0], 2),
                        "y_orig": round(float(gy) + bbox[1], 2),
                    })
                out["seg_glints"] = seg_fine_glints

                # Compute grayscale for enhanced detection
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

                # --- Enhanced seg detection: glints on iris+pupil only (no sclera) ---
                enh_eye_region = (seg_mask >= 2).astype(np.uint8) * 255  # iris(2) + pupil(3) only
                enh_eye_region = cv2.dilate(enh_eye_region, np.ones((15, 15), np.uint8))
                glint_enh_gray = enhance_for_glint_detection(cropped_gray)
                enh_glints_raw = find_glints_robust(glint_enh_gray, cropped_gray, enh_eye_region)
                # Post-filter: reject glints outside dilated iris+pupil boundary
                enh_glints = []
                for gx, gy, ga in enh_glints_raw:
                    ix, iy = int(round(gx)), int(round(gy))
                    if 0 <= iy < enh_eye_region.shape[0] and 0 <= ix < enh_eye_region.shape[1] and enh_eye_region[iy, ix] > 0:
                        enh_glints.append({
                            "x": round(float(gx), 2),
                            "y": round(float(gy), 2),
                            "area": int(ga),
                            "x_orig": round(float(gx) + bbox[0], 2),
                            "y_orig": round(float(gy) + bbox[1], 2),
                        })
                out["seg_enh_glints"] = enh_glints

                # Pupil detection from seg mask (class 3 contour primary)
                seg_pupil_result = detect_pupil_in_seg_mask(
                    cropped_gray, seg_mask, glint_positions=enh_glints)
                if seg_pupil_result:
                    enh_px, enh_py, enh_pr = seg_pupil_result
                    out["seg_enh_pupil"] = [round(float(enh_px) + bbox[0], 1),
                                            round(float(enh_py) + bbox[1], 1)]
                    out["seg_enh_pupil_radius"] = int(enh_pr)

                # Extract pupil center from segmentation mask and draw gaze
                pupil_ys, pupil_xs = np.where(seg_mask == 3)
                if len(pupil_xs) > 0:
                    seg_px = float(np.mean(pupil_xs))
                    seg_py = float(np.mean(pupil_ys))
                    # Pupil center marker (red filled + white outline for visibility)
                    cv2.circle(seg_img, (int(seg_px), int(seg_py)), 5, (255, 255, 255), 2)
                    cv2.circle(seg_img, (int(seg_px), int(seg_py)), 4, (0, 0, 255), -1)

                    # Draw glint positions on seg overlay
                    for g in fine_glints:
                        cv2.circle(seg_img, (int(g["x"]), int(g["y"])), 4, (0, 0, 255), 1)
                        cv2.circle(seg_img, (int(g["x"]), int(g["y"])), 1, (0, 0, 255), -1)

                    # Extract iris center from segmentation mask
                    iris_ys, iris_xs = np.where(seg_mask == 2)
                    seg_iris_cx = float(np.mean(iris_xs)) if len(iris_xs) > 0 else None
                    seg_iris_cy = float(np.mean(iris_ys)) if len(iris_ys) > 0 else None

                    # Save seg pupil center to results (always, regardless of gaze method)
                    out["seg_pupil_center"] = [round(seg_px + bbox[0], 1),
                                                round(seg_py + bbox[1], 1)]
                    if seg_iris_cx is not None:
                        out["seg_iris_center"] = [round(seg_iris_cx + bbox[0], 1),
                                                   round(seg_iris_cy + bbox[1], 1)]

                    # --- Seg gaze: use enhanced pupil + enhanced glints ---
                    ref_cx, ref_cy = None, None
                    ref_cx_orig, ref_cy_orig = None, None
                    seg_gaze_method = None

                    # Use enhanced pupil center (class 3 contour), fallback to centroid
                    seg_gaze_px = seg_pupil_result[0] if seg_pupil_result else seg_px
                    seg_gaze_py = seg_pupil_result[1] if seg_pupil_result else seg_py

                    if len(enh_glints) >= 2:
                        # Primary: pupil - enhanced glint centroid
                        ref_cx = np.mean([g["x"] for g in enh_glints])
                        ref_cy = np.mean([g["y"] for g in enh_glints])
                        ref_cx_orig = np.mean([g["x_orig"] for g in enh_glints])
                        ref_cy_orig = np.mean([g["y_orig"] for g in enh_glints])
                        seg_gaze_method = "pupil_glint"
                    elif seg_iris_cx is not None:
                        # Fallback: pupil - iris center (pure segmentation)
                        ref_cx = seg_iris_cx
                        ref_cy = seg_iris_cy
                        ref_cx_orig = seg_iris_cx + bbox[0]
                        ref_cy_orig = seg_iris_cy + bbox[1]
                        seg_gaze_method = "pupil_iris"

                    if ref_cx is not None:
                        gv_x = seg_gaze_px - ref_cx
                        gv_y = seg_gaze_py - ref_cy
                        gv_mag = np.sqrt(gv_x**2 + gv_y**2)

                        out["seg_gaze_vector"] = [round(float(gv_x), 1), round(float(gv_y), 1)]
                        out["seg_gaze_method"] = seg_gaze_method

                        # Undistorted normalized seg gaze (PCCR in 3D)
                        seg_gaze_h_deg = 0.0
                        seg_gaze_v_deg = 0.0
                        if cam_calib is not None:
                            seg_gaze_px_orig = seg_gaze_px + bbox[0]
                            seg_gaze_py_orig = seg_gaze_py + bbox[1]
                            sg_nx, sg_ny = undistort_gaze(
                                (float(seg_gaze_px_orig), float(seg_gaze_py_orig)),
                                (float(ref_cx_orig), float(ref_cy_orig)),
                                cam_calib['K'], cam_calib['dist'])
                            out["seg_gaze_vector_norm"] = [round(sg_nx, 6), round(sg_ny, 6)]
                            # 3D gaze angles from normalized PCCR
                            seg_gaze_h_deg = float(np.degrees(np.arctan(sg_nx)))
                            seg_gaze_v_deg = float(np.degrees(np.arctan(-sg_ny)))
                        else:
                            # Fallback: pixel-space angle (no calibration)
                            seg_gaze_h_deg = float(np.degrees(np.arctan2(gv_x, 1.0))) if gv_mag > 0.01 else 0.0
                            seg_gaze_v_deg = float(np.degrees(np.arctan2(-gv_y, 1.0))) if gv_mag > 0.01 else 0.0

                        out["seg_gaze_angle_h_deg"] = round(seg_gaze_h_deg, 2)
                        out["seg_gaze_angle_v_deg"] = round(seg_gaze_v_deg, 2)
                        out["seg_gaze_vector"] = [round(float(gv_x), 1), round(float(gv_y), 1)]
                        out["seg_gaze_method"] = seg_gaze_method

                        # Gaze arrow on seg overlay (use normalized vector for direction)
                        if cam_calib is not None:
                            arrow_scale = 80.0  # scale normalized coords to pixel arrow
                            ax = int(seg_gaze_px + sg_nx * arrow_scale)
                            ay = int(seg_gaze_py + sg_ny * arrow_scale)
                        else:
                            scale = 4.0
                            ax = int(seg_gaze_px + gv_x * scale)
                            ay = int(seg_gaze_py + gv_y * scale)
                        cv2.arrowedLine(seg_img, (int(seg_gaze_px), int(seg_gaze_py)),
                                        (ax, ay), (0, 255, 255), 2, tipLength=0.3)
                        label = f"H:{seg_gaze_h_deg:+.0f} V:{seg_gaze_v_deg:+.0f}"
                        if seg_gaze_method == "pupil_iris":
                            label += " [iris]"
                        cv2.putText(seg_img, label,
                                    (4, seg_img.shape[0] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                        # --- Seg gaze debug image (separate from seg overlay) ---
                        sg = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
                        sg_h, sg_w = sg.shape[:2]
                        # Draw reference point (blue cross = glint, cyan cross = iris)
                        ref_color = (255, 100, 0) if seg_gaze_method == "pupil_glint" else (255, 255, 0)
                        cv2.drawMarker(sg, (int(ref_cx), int(ref_cy)),
                                       ref_color, cv2.MARKER_CROSS, 8, 1)
                        # Draw seg pupil center (magenta dot)
                        cv2.circle(sg, (int(seg_gaze_px), int(seg_gaze_py)), 3, (255, 0, 255), -1)
                        # Draw iris center (cyan dot) if available
                        if seg_iris_cx is not None:
                            cv2.circle(sg, (int(seg_iris_cx), int(seg_iris_cy)), 3, (255, 255, 0), 1)
                        # Arrow from seg pupil center in gaze direction (use normalized if available)
                        if cam_calib is not None:
                            sax = int(seg_gaze_px + sg_nx * 80.0)
                            say = int(seg_gaze_py + sg_ny * 80.0)
                        else:
                            sax = int(seg_gaze_px + gv_x * 4.0)
                            say = int(seg_gaze_py + gv_y * 4.0)
                        cv2.arrowedLine(sg, (int(seg_gaze_px), int(seg_gaze_py)),
                                        (sax, say), (255, 0, 255), 2, tipLength=0.3)
                        angle_str = f"H:{seg_gaze_h_deg:+.0f} V:{seg_gaze_v_deg:+.0f}"
                        if seg_gaze_method == "pupil_iris":
                            angle_str += " [iris]"
                        cv2.putText(sg, angle_str, (4, sg_h - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                        method_label = "SEG GAZE" if seg_gaze_method == "pupil_glint" else "SEG GAZE [iris]"
                        cv2.putText(sg, method_label, (4, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                        cv2.imwrite(str(out_path / "seg_gaze_debug" / (fp.stem + "_seg_gaze.png")), sg)

                cv2.imwrite(str(out_path / "seg_debug" / (fp.stem + "_seg.png")), seg_img)

                # Preprocessing visualizations (analysis only, not used in pipeline)
                glint_img = preprocess_for_glints(cropped_gray)
                pupil_img = preprocess_for_pupil(cropped_gray)
                cv2.imwrite(str(out_path / "glint_enhanced" / (fp.stem + "_glint_enh.png")), glint_img)
                cv2.imwrite(str(out_path / "pupil_enhanced" / (fp.stem + "_pupil_enh.png")), pupil_img)

                # --- Generate seg_glint_debug image ---
                sgd = cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)
                sgd_h, sgd_w = sgd.shape[:2]

                # Draw iris+pupil boundary (cyan outline, no sclera)
                iris_pupil_boundary = (seg_mask >= 2).astype(np.uint8) * 255
                boundary_contours, _ = cv2.findContours(iris_pupil_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(sgd, boundary_contours, -1, (200, 200, 0), 1)

                # Draw detected glints (green circles with labels)
                for j, g in enumerate(enh_glints):
                    gx_i, gy_i = int(g["x"]), int(g["y"])
                    cv2.circle(sgd, (gx_i, gy_i), 7, (0, 255, 0), 1)
                    cv2.circle(sgd, (gx_i, gy_i), 2, (0, 255, 0), -1)
                    cv2.putText(sgd, f"G{j}", (gx_i + 9, gy_i - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                # Draw pupil (magenta circle)
                if seg_pupil_result:
                    epx, epy, epr = int(enh_px), int(enh_py), int(enh_pr)
                    cv2.circle(sgd, (epx, epy), epr, (255, 0, 255), 1)
                    cv2.circle(sgd, (epx, epy), 2, (255, 0, 255), -1)
                    cv2.putText(sgd, "pupil", (epx + epr + 3, epy + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 0, 255), 1)

                # Draw glint centroid (yellow cross)
                if len(enh_glints) >= 2:
                    gcx = np.mean([g["x"] for g in enh_glints])
                    gcy = np.mean([g["y"] for g in enh_glints])
                    cv2.drawMarker(sgd, (int(gcx), int(gcy)),
                                   (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
                    cv2.putText(sgd, "ref", (int(gcx) + 8, int(gcy) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

                # Label and metadata
                cv2.putText(sgd, "SEG ENH GLINTS", (4, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
                meta_str = f"seg_enh: {len(enh_glints)}"
                if seg_pupil_result:
                    meta_str += " | pupil_seg"
                cv2.putText(sgd, meta_str, (4, sgd_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 200), 1)

                cv2.imwrite(str(out_path / "seg_glint_debug" / (fp.stem + "_seg_glint.png")), sgd)

            except Exception as e:
                if i == 0:
                    print(f"  [SEG] Warning: segmentation failed: {e}")

        # ---- Save debug ----
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if pupil:
            cv2.circle(dbg, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 1)
            cv2.circle(dbg, (pupil[0], pupil[1]), 2, (0, 255, 0), -1)
        cv2.drawMarker(dbg, sc, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
        for gx, gy, _ in r["glints"]:
            cv2.circle(dbg, (int(gx), int(gy)), 5, (0, 255, 255), 1)
        for g in fine_glints:
            cv2.circle(dbg, (int(g["x_orig"]), int(g["y_orig"])), 4, (0, 0, 255), 1)
            cv2.circle(dbg, (int(g["x_orig"]), int(g["y_orig"])), 1, (0, 0, 255), -1)
        if out["eye_closed"]:
            cv2.putText(dbg, "EYE CLOSED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(out_path / "debug" / (fp.stem + "_debug.png")), dbg)

        # ---- Save glint debug (cropped + glint overlay) ----
        gv = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        if pupil:
            pupil_crop_x = pupil[0] - bbox[0]
            pupil_crop_y = pupil[1] - bbox[1]
            ch, cw = cropped.shape[:2]
            if 0 <= pupil_crop_x < cw and 0 <= pupil_crop_y < ch:
                cv2.circle(gv, (pupil_crop_x, pupil_crop_y), pr, (0, 255, 0), 1)
                cv2.circle(gv, (pupil_crop_x, pupil_crop_y), 2, (0, 255, 0), -1)
        for j, g in enumerate(fine_glints):
            gx, gy = int(g["x"]), int(g["y"])
            cv2.circle(gv, (gx, gy), 6, (0, 0, 255), 1)
            cv2.circle(gv, (gx, gy), 2, (0, 0, 255), -1)
            cv2.putText(gv, f"G{j}", (gx + 8, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        cv2.putText(gv, f"Glints: {len(fine_glints)}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        if out["eye_closed"]:
            cv2.putText(gv, "CLOSED", (4, gv.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imwrite(str(out_path / "glint_debug" / (fp.stem + "_glint.png")), gv)

        # ---- Save gaze debug (arrow showing gaze direction) ----
        gz = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        ch, cw = cropped.shape[:2]
        if pupil and len(fine_glints) >= 1:
            glint_cx = np.mean([g["x"] for g in fine_glints])
            glint_cy = np.mean([g["y"] for g in fine_glints])
            pcx = pupil[0] - bbox[0]
            pcy = pupil[1] - bbox[1]

            # Draw glint centroid (blue cross)
            cv2.drawMarker(gz, (int(glint_cx), int(glint_cy)),
                           (255, 100, 0), cv2.MARKER_CROSS, 8, 1)
            # Draw pupil center (green dot)
            if 0 <= pcx < cw and 0 <= pcy < ch:
                cv2.circle(gz, (int(pcx), int(pcy)), 3, (0, 255, 0), -1)

            # Gaze arrow: from glint centroid toward pupil, scaled up
            gaze_mag = np.sqrt(gaze_vx**2 + gaze_vy**2)
            scale = 4.0
            arrow_end_x = int(glint_cx + gaze_vx * scale)
            arrow_end_y = int(glint_cy + gaze_vy * scale)
            cv2.arrowedLine(gz, (int(glint_cx), int(glint_cy)),
                            (arrow_end_x, arrow_end_y),
                            (0, 255, 255), 2, tipLength=0.3)
            # Angle text
            angle_str = f"{gaze_angle:+.0f} deg" if gaze_angle is not None else ""
            cv2.putText(gz, angle_str, (4, ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        else:
            cv2.putText(gz, "No gaze", (4, ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(gz, "GAZE", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.imwrite(str(out_path / "gaze_debug" / (fp.stem + "_gaze.png")), gz)

        all_results.append(out)
        if (i + 1) % 200 == 0:
            det = sum(1 for o in all_results if o.get("pupil_detected"))
            cl = sum(1 for o in all_results if o.get("eye_closed"))
            print(f"  {i + 1}/{len(frames)} (det={det} closed={cl})")

    # Summary
    total = len(all_results)
    det = sum(1 for o in all_results if o.get("pupil_detected"))
    cl = sum(1 for o in all_results if o.get("eye_closed"))
    ag = sum(len(o.get("glints", [])) for o in all_results) / max(1, total)
    print(f"\n{'=' * 50}")
    print(f"Frames:       {total}")
    print(f"Detected:     {det} ({100 * det / max(1, total):.1f}%)")
    print(f"Eye closed:   {cl} ({100 * cl / max(1, total):.1f}%)")
    print(f"Avg glints:   {ag:.1f}/frame")
    print(f"Crop size:    {crop_size}x{crop_size}")
    print(f"Output:       {out_path}")
    print(f"{'=' * 50}")

    with open(str(out_path / "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def load_calibration(calib_dir):
    """Load per-camera intrinsics and stereo geometry from calibration files.

    Supports two formats:
    1. New JSON: directory with {pair}/out/stereo_calib_summary.json
    2. Old NPZ: directory with stereo_calib_results_{pair}.npz (fallback)
    """
    calib_path = Path(calib_dir)
    calib = {}

    # --- Try new JSON format first ---
    pairs_json = [
        ("ro_ri", "ro", "ri", "right"),
        ("lo_li", "lo", "li", "left"),
        ("ro_lo", "ro", "lo", "cross"),
    ]
    json_found = False
    for pair_name, cam_l, cam_r, pair_key in pairs_json:
        json_path = calib_path / pair_name / "out" / "stereo_calib_summary.json"
        if not json_path.exists():
            continue
        json_found = True
        with open(json_path) as f:
            data = json.load(f)

        # Extract per-camera intrinsics (intra-eye pairs loaded first take priority)
        pair_intrinsics = {}
        for side, cam_id in [("camera_params_l", cam_l), ("camera_params_r", cam_r)]:
            intr = data[side]["intrinsic"]
            K = np.array(intr["K"], dtype=np.float64)
            dist_raw = intr["dist"]
            dist = np.array(dist_raw[0] if isinstance(dist_raw[0], list) else dist_raw,
                            dtype=np.float64).reshape(1, -1)
            pair_intrinsics[cam_id] = {'K': K, 'dist': dist}
            if cam_id not in calib:
                calib[cam_id] = {'K': K, 'dist': dist}

        # Extract stereo R, T (store pair's own intrinsics for consistent use)
        R = np.array(data["R"], dtype=np.float64)
        T_raw = data["T"]
        T = np.array([t[0] if isinstance(t, list) else t for t in T_raw],
                      dtype=np.float64).reshape(3, 1)
        calib[pair_key] = {'R': R, 'T': T, 'cam1': cam_l, 'cam2': cam_r,
                           'intrinsics': pair_intrinsics}

        # Extract reprojection errors per camera (for weighting)
        for side, cam_id, err_key in [
            ("camera_params_l", cam_l, "errs_mono_reproj_l"),
            ("camera_params_r", cam_r, "errs_mono_reproj_r"),
        ]:
            if err_key in data:
                err = data[err_key]
                reproj = {'mean_px': err['mean_px'], 'std_px': err['std_px']}
                # Store best (lowest) reproj error across pairs for each camera
                if 'reproj_err' not in calib.get(cam_id, {}):
                    calib[cam_id]['reproj_err'] = reproj
                elif err['mean_px'] < calib[cam_id]['reproj_err']['mean_px']:
                    calib[cam_id]['reproj_err'] = reproj

        # Extract physical sensor params per camera
        for side, cam_id in [("camera_params_l", cam_l), ("camera_params_r", cam_r)]:
            intr = data[side]["intrinsic"]
            if 'f_mm' not in calib.get(cam_id, {}):
                phys = {}
                for key in ['f_mm', 'px_mm', 'py_mm', 'W_mm', 'H_mm']:
                    if key in intr:
                        phys[key] = intr[key]
                if phys:
                    calib[cam_id].update(phys)

    if json_found and len(calib) >= 4:
        src = "JSON"
        for cam_id in ['ro', 'ri', 'lo', 'li']:
            if cam_id in calib:
                K = calib[cam_id]['K']
                d = calib[cam_id]['dist'].flatten()
                reproj = calib[cam_id].get('reproj_err', {})
                f_mm = calib[cam_id].get('f_mm')
                px_mm = calib[cam_id].get('px_mm')
                extras = ""
                if reproj:
                    extras += f" reproj={reproj['mean_px']:.3f}px"
                if f_mm and px_mm:
                    ang_res = px_mm / f_mm * 180 / np.pi
                    extras += f" f={f_mm:.3f}mm ({ang_res:.3f}°/px)"
                print(f"  {cam_id.upper()}: fx={K[0,0]:.2f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} "
                      f"dist=[{d[0]:.3f}, {d[1]:.3f}]{extras}")
        for pk in ['right', 'left', 'cross']:
            if pk in calib:
                bl = float(np.linalg.norm(calib[pk]['T']))
                print(f"  {pk}: baseline={bl:.2f}mm")
        print(f"  [CALIB] Loaded from {src} ({calib_path})")
        return calib

    # --- Fallback: old NPZ format ---
    calib = {}
    ro_ri_f = calib_path / "stereo_calib_results_ro_ri.npz"
    if ro_ri_f.exists():
        d = np.load(str(ro_ri_f))
        calib['ro'] = {'K': d['K1'], 'dist': d['dist1']}
        calib['ri'] = {'K': d['K2'], 'dist': d['dist2']}
        calib['right'] = {'R': d['R'], 'T': d['T'],
                          'cam1': 'ro', 'cam2': 'ri'}

    lo_li_f = calib_path / "stereo_calib_results_lo_li.npz"
    if lo_li_f.exists():
        d = np.load(str(lo_li_f))
        calib['lo'] = {'K': d['K1'], 'dist': d['dist1']}
        calib['li'] = {'K': d['K2'], 'dist': d['dist2']}
        calib['left'] = {'R': d['R'], 'T': d['T'],
                         'cam1': 'lo', 'cam2': 'li'}

    if len(calib) >= 4:
        print(f"  [CALIB] Loaded from NPZ ({calib_path})")
        return calib
    return None


def load_led_positions(calib_pkl_path):
    """Load 3D IR LED positions from full_calibration_data.pkl.

    LED positions are in the pickle's camera 0 (RO) coordinate frame.
    Also returns the stereo R,T from the pickle so we can transform
    LED positions to RI frame.

    Returns: dict with 'positions' (list of 4 numpy (3,) arrays in RO frame, mm)
             and 'R', 'T' (stereo rotation/translation RO->RI from pickle).
             Returns None if file doesn't exist or fails.
    """
    import pickle as _pickle
    pkl_path = Path(calib_pkl_path)
    if not pkl_path.exists():
        print(f"[LED] Calibration pickle not found: {pkl_path}")
        return None

    class _FlexibleUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                class _Dummy:
                    def __init__(self, *a, **kw):
                        pass
                _Dummy.__name__ = name
                _Dummy.__module__ = module
                return _Dummy

    try:
        with open(str(pkl_path), 'rb') as f:
            data = _FlexibleUnpickler(f).load()
        glints = data.get('glints')
        if glints is None or len(glints) < 4:
            print(f"[LED] Pickle missing 'glints' key or < 4 LEDs")
            return None
        led_positions = [np.asarray(g, dtype=np.float64) for g in glints[:4]]

        # Get stereo R,T from pickle (transforms RO frame -> RI frame)
        stereo_calib = data.get('stereo_calibration', {})
        R_pkl = stereo_calib.get('R')
        T_pkl = stereo_calib.get('T')
        if R_pkl is None or T_pkl is None:
            print(f"[LED] Pickle missing stereo R,T — cannot transform to RI frame")
            return None

        R_pkl = np.asarray(R_pkl, dtype=np.float64)
        T_pkl = np.asarray(T_pkl, dtype=np.float64).flatten()

        print(f"[LED] Loaded {len(led_positions)} LED 3D positions from {pkl_path.name} (RO frame)")
        for i, p in enumerate(led_positions):
            print(f"  LED[{i}] RO: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] mm")

        # Transform to RI frame: P_ri = R @ P_ro + T
        led_ri = [R_pkl @ p + T_pkl for p in led_positions]
        print(f"[LED] Transformed to RI frame:")
        for i, p in enumerate(led_ri):
            print(f"  LED[{i}] RI: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] mm")

        return {
            'positions_ro': led_positions,
            'positions_ri': led_ri,
            'R': R_pkl,
            'T': T_pkl,
        }
    except Exception as e:
        print(f"[LED] Failed to load pickle: {e}")
        return None


def match_glints_to_leds(detected_glints_2d, led_3d_positions, K, dist):
    """Match detected 2D glints to known 3D LED positions using angular ordering.

    The physical LEDs are NOT in the camera's field of view — only their
    corneal reflections (glints) are visible. We can't project LED positions
    into the image. Instead, we match by angular ordering: the convex cornea
    preserves the spatial topology of the LED arrangement in its reflections.

    1. Compute angular direction of each LED from their centroid (as seen from camera)
    2. Compute angle of each detected glint from their centroid
    3. Sort both by angle and match by order

    detected_glints_2d: list of (x, y) pixel positions of detected glints
    led_3d_positions: list of 4 numpy arrays (3,) in RI camera frame (mm)
    K: 3x3 camera intrinsic matrix
    dist: distortion coefficients

    Returns: (matched_glints_2d, matched_leds_3d) as Nx2, Nx3 arrays, or (None, None)
    """
    n_det = len(detected_glints_2d)
    n_led = len(led_3d_positions)
    if n_det < 3 or n_led < 3:
        return None, None

    led_3d_arr = np.array(led_3d_positions, dtype=np.float64)

    # Compute LED angular directions as seen from camera origin.
    # Direction = LED_pos / |LED_pos|, then use atan2 of x,y components.
    led_dirs = []
    for p in led_3d_arr:
        d = p / np.linalg.norm(p)
        led_dirs.append(d)
    led_dirs = np.array(led_dirs)
    led_centroid = np.mean(led_dirs[:, :2], axis=0)
    led_angles = np.array([np.arctan2(d[1] - led_centroid[1],
                                       d[0] - led_centroid[0])
                           for d in led_dirs])

    # Compute glint angles from their centroid in pixel space
    det_arr = np.array(detected_glints_2d, dtype=np.float64)
    glint_centroid = np.mean(det_arr, axis=0)
    glint_angles = np.array([np.arctan2(g[1] - glint_centroid[1],
                                         g[0] - glint_centroid[0])
                             for g in det_arr])

    # Sort LEDs and glints by angle
    led_order = np.argsort(led_angles)
    glint_order = np.argsort(glint_angles)

    # Match min(n_det, n_led) pairs
    # Corneal reflection (convex mirror) preserves angular order but may
    # introduce a rotational offset. Try all cyclic shifts and pick the
    # best one (smallest total angular difference).
    n_match = min(n_det, n_led)

    # Sorted LEDs and glints
    sorted_led_idx = led_order[:n_led]
    sorted_glint_idx = glint_order[:n_det]

    # If we have more of one than the other, we need to pick a subset.
    # Try matching n_match from the larger set to the smaller set with
    # all possible cyclic offsets.
    best_cost = float('inf')
    best_match = None

    if n_det >= n_led:
        # More glints than LEDs: try each cyclic offset of glints
        for offset in range(n_det):
            cost = 0
            pairs = []
            for k in range(n_led):
                gi = sorted_glint_idx[(offset + k) % n_det]
                li = sorted_led_idx[k]
                # Angular difference (normalized)
                diff = abs(glint_angles[gi] - led_angles[li])
                diff = min(diff, 2 * np.pi - diff)
                cost += diff
                pairs.append((gi, li))
            if cost < best_cost:
                best_cost = cost
                best_match = pairs
    else:
        # More LEDs than glints: try each cyclic offset of LEDs
        for offset in range(n_led):
            cost = 0
            pairs = []
            for k in range(n_det):
                gi = sorted_glint_idx[k]
                li = sorted_led_idx[(offset + k) % n_led]
                diff = abs(glint_angles[gi] - led_angles[li])
                diff = min(diff, 2 * np.pi - diff)
                cost += diff
                pairs.append((gi, li))
            if cost < best_cost:
                best_cost = cost
                best_match = pairs

    # Also try reversed order (in case of reflection flip)
    sorted_glint_idx_rev = sorted_glint_idx[::-1]
    glint_angles_rev = -glint_angles  # reverse angles
    if n_det >= n_led:
        for offset in range(n_det):
            cost = 0
            pairs = []
            for k in range(n_led):
                gi = sorted_glint_idx_rev[(offset + k) % n_det]
                li = sorted_led_idx[k]
                diff = abs(-glint_angles[gi] - led_angles[li])
                diff = min(diff, 2 * np.pi - diff)
                cost += diff
                pairs.append((gi, li))
            if cost < best_cost:
                best_cost = cost
                best_match = pairs
    else:
        for offset in range(n_led):
            cost = 0
            pairs = []
            for k in range(n_det):
                gi = sorted_glint_idx_rev[k]
                li = sorted_led_idx[(offset + k) % n_led]
                diff = abs(-glint_angles[gi] - led_angles[li])
                diff = min(diff, 2 * np.pi - diff)
                cost += diff
                pairs.append((gi, li))
            if cost < best_cost:
                best_cost = cost
                best_match = pairs

    if best_match is None or len(best_match) < 3:
        return None, None

    matched_g = np.array([det_arr[gi] for gi, li in best_match], dtype=np.float64)
    matched_l = np.array([led_3d_arr[li] for gi, li in best_match], dtype=np.float64)
    return matched_g, matched_l


def estimate_corneal_center(glint_2d_pts, led_3d_pts, K, dist,
                            corneal_radius=7.8):
    """Estimate 3D corneal sphere center from matched glint-LED pairs.

    Uses the corneal reflection geometry: each glint is a specular reflection
    of an LED on the corneal sphere surface. The corneal center lies along
    the bisector of the angle between the camera ray to the glint and the
    LED-to-reflection-point line.

    Method: For each (LED, glint) pair, the reflection point on the cornea
    lies on the ray from the camera through the glint. The corneal center
    is at distance `corneal_radius` from the reflection point, along the
    surface normal. We estimate the corneal center by least-squares
    intersection of multiple constraint rays.

    glint_2d_pts: Nx2 array of detected glint pixel positions
    led_3d_pts: Nx3 array of corresponding 3D LED positions (camera frame, mm)
    K: 3x3 camera intrinsic matrix
    dist: distortion coefficients
    corneal_radius: corneal sphere radius in mm (default 7.8)

    Returns: corneal_center (3,) in camera frame (mm), or None
    """
    n = len(glint_2d_pts)
    if n < 2:
        return None

    # Undistort glint points to get normalized camera rays
    glint_rays = []
    for i in range(n):
        nx, ny = _undistort_single(glint_2d_pts[i][0], glint_2d_pts[i][1],
                                   K, dist)
        ray = np.array([nx, ny, 1.0], dtype=np.float64)
        ray = ray / np.linalg.norm(ray)
        glint_rays.append(ray)

    # For each (LED, glint_ray) pair, the corneal center must satisfy:
    # The reflection point R is on the glint ray: R = t * ray_dir
    # The surface normal at R points from R toward corneal center C:
    #   C = R + corneal_radius * normal
    # The reflection law: incident ray (LED->R), reflected ray (R->camera),
    # and normal are coplanar, with equal angles.
    #
    # Simplified approach: assume the reflection point is at the closest
    # point between the glint ray and the LED position. Then the corneal
    # center is offset from that point along the bisector direction.

    # Estimate corneal center using midpoint method:
    # For each pair, find where the glint ray is closest to the midpoint
    # between camera and LED. The corneal center is approximately at
    # the LED-glint-ray midpoint at the right depth.

    # Practical approach: least-squares fit of corneal center
    # such that |C - reflection_point_i| = corneal_radius for all i,
    # where reflection_point_i = t_i * ray_i (on the camera-to-glint ray).
    #
    # This is equivalent to: for each ray_i and LED_i, find C such that
    # the reflection law holds. We use iterative optimization.

    mean_ray = np.mean(glint_rays, axis=0)
    mean_ray = mean_ray / np.linalg.norm(mean_ray)

    def _corneal_residuals(C):
        """Residuals: for each (LED, glint_ray), the reflection law should hold."""
        residuals = []
        for i in range(n):
            ray = glint_rays[i]
            led = led_3d_pts[i]

            a_coeff = np.dot(ray, ray)  # = 1 since ray is normalized
            b_coeff = -2 * np.dot(ray, C)
            c_coeff = np.dot(C, C) - corneal_radius**2

            disc = b_coeff**2 - 4 * a_coeff * c_coeff
            if disc < 0:
                residuals.append(np.sqrt(abs(disc)))
                continue

            t1 = (-b_coeff - np.sqrt(disc)) / (2 * a_coeff)
            t2 = (-b_coeff + np.sqrt(disc)) / (2 * a_coeff)
            t = t1 if t1 > 0 else t2
            if t <= 0:
                residuals.append(10.0)
                continue

            R = t * ray
            normal = (C - R) / np.linalg.norm(C - R)

            inc = R - led
            inc_norm = np.linalg.norm(inc)
            if inc_norm < 1e-6:
                residuals.append(10.0)
                continue
            inc = inc / inc_norm

            refl = -ray
            residuals.append(np.dot(inc, normal) - np.dot(refl, normal))

        return residuals

    from scipy.optimize import least_squares

    # Multi-start optimization: try several initial depths to avoid local minima.
    # The cornea is ~15-25mm from the near-eye camera.
    best_C = None
    best_cost = float('inf')
    min_valid_depth = corneal_radius + 2.0  # CC must be at least radius+2mm from camera

    for init_depth in [12.0, 18.0, 25.0, 35.0]:
        C0 = mean_ray * init_depth
        try:
            result = least_squares(_corneal_residuals, C0, method='lm',
                                   max_nfev=500)
            C = result.x
            # Validate: corneal center must be at reasonable depth
            if C[2] <= min_valid_depth or C[2] > 80:
                continue
            if result.cost < best_cost:
                best_cost = result.cost
                best_C = C
        except Exception:
            continue

    if best_C is not None and best_cost < 5.0:
        return best_C
    return None


def _fit_sphere(points_3d, radius_hint=7.8):
    """Fit a sphere to 3D points using algebraic least-squares.

    Given N points that should lie on a sphere, find center C and radius r.
    Uses linear solve: |P_i - C|^2 = r^2  =>  -2*P_i @ C + (|C|^2 - r^2) = -|P_i|^2

    Returns: (center (3,), radius) or (None, None)
    """
    pts = np.array(points_3d, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None, None

    # Linear system: A @ [Cx, Cy, Cz, d]^T = b
    # where d = |C|^2 - r^2
    A = np.zeros((n, 4))
    b = np.zeros(n)
    for i in range(n):
        A[i, :3] = -2 * pts[i]
        A[i, 3] = 1.0
        b[i] = -(pts[i] @ pts[i])

    x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    C = x[:3]
    d = x[3]
    r = np.sqrt(np.abs(C @ C - d))

    # Validate
    if r < 1.0 or r > 30.0:  # unreasonable sphere radius
        return None, None
    if C[2] <= 0:  # behind camera
        return None, None

    return C, r


def _match_glints_stereo(glints1, glints2, K1, dist1, K2, dist2, F):
    """Match glints between two cameras using epipolar constraint.

    For each glint in cam1, find best matching glint in cam2 by
    distance to epipolar line.

    Returns: list of (idx1, idx2) matched pairs
    """
    if len(glints1) < 2 or len(glints2) < 2:
        return []

    g1 = np.array(glints1, dtype=np.float64)
    g2 = np.array(glints2, dtype=np.float64)

    # Compute epipolar distances: for each g1[i], epipolar line in cam2
    # l2 = F @ [x1, y1, 1]^T
    # Distance of g2[j] to l2: |l2^T @ [x2, y2, 1]| / sqrt(l2[0]^2 + l2[1]^2)
    dist_mat = np.zeros((len(g1), len(g2)))
    for i in range(len(g1)):
        l2 = F @ np.array([g1[i, 0], g1[i, 1], 1.0])
        for j in range(len(g2)):
            d = abs(l2[0] * g2[j, 0] + l2[1] * g2[j, 1] + l2[2])
            d /= np.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
            dist_mat[i, j] = d

    # Greedy matching by smallest epipolar distance
    pairs = []
    used1, used2 = set(), set()
    flat = [(dist_mat[i, j], i, j)
            for i in range(len(g1)) for j in range(len(g2))]
    flat.sort()
    for d, i, j in flat:
        if i in used1 or j in used2:
            continue
        if d > 20.0:  # max 20px epipolar distance
            continue
        pairs.append((i, j))
        used1.add(i)
        used2.add(j)

    return pairs


def compute_corneal_3d_gaze(output_dir, calib, led_data, crop_size=150):
    """Compute gaze using stereo 3D corneal model (two-pass).

    Both RO and RI cameras see reflections of the same 4 IR LEDs on the cornea.

    Pass 1: For frames with 3+ stereo glint matches, triangulate glint
    reflection points, fit sphere -> corneal center. Take the MEDIAN corneal
    center across all successful frames as the stable eye center.

    Pass 2: For ALL frames with stereo pupil detection, compute:
    gaze = normalize(pupil_3d - median_corneal_center)

    This gives maximum coverage because the corneal center (physical eye center)
    is stable across frames, while pupil movement drives gaze variation.

    Outputs to {eye}_corneal_3d/ directory.
    """
    out_base = Path(output_dir)

    for eye, pair_key in [("right", "right"), ("left", "left")]:
        stereo = calib.get(pair_key)
        if stereo is None:
            print(f"  [CORNEAL 3D] No stereo calibration for {eye} eye, skipping")
            continue

        cam1, cam2 = stereo['cam1'], stereo['cam2']  # ro (outer), ri (inner)
        R = stereo['R']
        T = stereo['T']
        K1, dist1 = calib[cam1]['K'], calib[cam1]['dist']
        K2, dist2 = calib[cam2]['K'], calib[cam2]['dist']

        # Compute fundamental matrix from stereo calibration
        T_x = np.array([[0, -T[2, 0], T[1, 0]],
                         [T[2, 0], 0, -T[0, 0]],
                         [-T[1, 0], T[0, 0], 0]])
        E = T_x @ R
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        # Load per-camera results from BOTH cameras
        r1_path = out_base / cam1 / "results.json"
        r2_path = out_base / cam2 / "results.json"
        if not r1_path.exists() or not r2_path.exists():
            print(f"  [CORNEAL 3D] Missing results for {cam1} or {cam2}, skipping")
            continue
        with open(r1_path) as f:
            res1 = json.load(f)
        with open(r2_path) as f:
            res2 = json.load(f)

        n_frames = min(len(res1), len(res2))

        # ===== PASS 1: Estimate corneal center from best frames =====
        print(f"  [{eye.upper()} CORNEAL 3D] Pass 1: Estimating corneal center "
              f"from stereo glint triangulation...")
        corneal_centers = []  # [(frame_name, cc_array), ...]

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            pass1_frame_name = r1.get("frame", f"frame_{i}")
            # Prefer seg_enh_glints (enhanced top-hat + CLAHE detection) over
            # seg_glints/raw glints for better detection consistency
            glints1 = r1.get("seg_enh_glints") or r1.get("seg_glints") or r1.get("glints", [])
            glints2 = r2.get("seg_enh_glints") or r2.get("seg_glints") or r2.get("glints", [])

            if (len(glints1) < 3 or len(glints2) < 3
                    or r1.get("eye_closed") or r2.get("eye_closed")):
                continue

            g1_2d = [(g["x_orig"], g["y_orig"]) for g in glints1]
            g2_2d = [(g["x_orig"], g["y_orig"]) for g in glints2]

            matches = _match_glints_stereo(g1_2d, g2_2d, K1, dist1,
                                           K2, dist2, F)
            if len(matches) < 3:
                continue

            reflection_pts = []
            for idx1, idx2 in matches:
                pt1_n = _undistort_single(
                    g1_2d[idx1][0], g1_2d[idx1][1], K1, dist1)
                pt2_n = _undistort_single(
                    g2_2d[idx2][0], g2_2d[idx2][1], K2, dist2)
                pt3d = _triangulate_point(pt1_n, pt2_n, R, T)
                if pt3d is not None and 0 < pt3d[2] < 80:
                    reflection_pts.append(pt3d)

            if len(reflection_pts) >= 3:
                cc, radius = _fit_sphere(reflection_pts)
                if cc is not None and 3.0 < radius < 20.0:
                    corneal_centers.append((pass1_frame_name, cc))

        if len(corneal_centers) < 3:
            print(f"  [{eye.upper()} CORNEAL 3D] Only {len(corneal_centers)} "
                  f"corneal centers found — not enough for robust estimate")
            # Write empty results
            comb_dir = out_base / f"{eye}_corneal_3d" / "combined_gaze"
            comb_dir.mkdir(parents=True, exist_ok=True)
            combined_results = [
                {"frame": res2[i].get("frame", f"frame_{i}"), "eye": eye,
                 "corneal_3d_gaze_deg": None, "corneal_3d_gaze_norm": None,
                 "corneal_center_3d": None, "method": None}
                for i in range(n_frames)]
            with open(str(out_base / f"{eye}_corneal_3d" / "combined_results.json"), "w") as f:
                json.dump(combined_results, f, indent=2)
            continue

        # Save per-frame CC observations for fair calibration
        cc_obs_key = f"cc_observations_{eye}"
        cc_obs_path = out_base / f"cc_observations_corneal3d_{eye}.json"
        cc_obs_list = [
            {"frame": fn, "cc": [round(float(cc[k]), 4) for k in range(3)]}
            for fn, cc in corneal_centers
        ]
        with open(str(cc_obs_path), "w") as f:
            json.dump(cc_obs_list, f, indent=2)
        print(f"  [{eye.upper()} CORNEAL 3D] Saved {len(cc_obs_list)} CC observations to {cc_obs_path}")

        # Median corneal center (robust to outliers)
        cc_arr = np.array([cc for _, cc in corneal_centers])
        median_cc = np.median(cc_arr, axis=0)
        cc_std = np.std(cc_arr, axis=0)
        print(f"  [{eye.upper()} CORNEAL 3D] Corneal center from "
              f"{len(corneal_centers)} frames:")
        print(f"    Median: [{median_cc[0]:.2f}, {median_cc[1]:.2f}, "
              f"{median_cc[2]:.2f}] mm")
        print(f"    Std:    [{cc_std[0]:.2f}, {cc_std[1]:.2f}, "
              f"{cc_std[2]:.2f}] mm")

        # ===== PASS 2: Compute gaze for all frames (two sub-passes) =====
        # The raw gaze (pupil_proj - cc_proj) has a large constant bias from
        # the camera-eye geometry (the CC is deep inside the eye, not on the
        # surface). Sub-pass A collects raw gaze vectors, sub-pass B subtracts
        # the median to isolate actual eye rotation, then draws debug images.
        print(f"  [{eye.upper()} CORNEAL 3D] Pass 2: Computing gaze "
              f"for all frames using fixed corneal center...")

        comb_dir = out_base / f"{eye}_corneal_3d" / "combined_gaze"
        comb_dir.mkdir(parents=True, exist_ok=True)

        # --- Pass 2: Stereo pupil triangulation + fixed CC ---
        # For each frame, triangulate the pupil from both cameras to get
        # pupil_3d in cam1 (RO) frame, then compute gaze as:
        #   gaze_3d = normalize(pupil_3d - median_cc)
        # Project gaze_3d onto RO normalized plane for 2D gaze vector.
        print(f"  [{eye.upper()} CORNEAL 3D] Pass 2: Stereo pupil "
              f"triangulation with fixed corneal center...")

        comb_dir = out_base / f"{eye}_corneal_3d" / "combined_gaze"
        comb_dir.mkdir(parents=True, exist_ok=True)

        combined_results = []

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r2.get("frame", f"frame_{i}")

            entry = {"frame": frame_name, "eye": eye,
                     "corneal_3d_gaze_deg": None,
                     "corneal_3d_gaze_norm": None,
                     "corneal_center_3d": [
                         round(float(median_cc[0]), 4),
                         round(float(median_cc[1]), 4),
                         round(float(median_cc[2]), 4)],
                     "pupil_3d": None,
                     "method": None}

            gaze_result = None

            # Need pupil detected in BOTH cameras for stereo
            pc1 = r1.get("pupil_center")
            pc2 = r2.get("pupil_center")
            closed1 = r1.get("eye_closed", False)
            closed2 = r2.get("eye_closed", False)

            if (pc1 is not None and pc2 is not None
                    and not closed1 and not closed2):
                # Undistort pupil from both cameras
                p1_n = _undistort_single(pc1[0], pc1[1], K1, dist1)
                p2_n = _undistort_single(pc2[0], pc2[1], K2, dist2)

                # Triangulate pupil in 3D (cam1/RO frame)
                pupil_3d = _triangulate_point(p1_n, p2_n, R, T)

                if pupil_3d is not None:
                    # Gaze direction = pupil - corneal center (in 3D)
                    gaze_3d = pupil_3d - median_cc
                    gaze_len = np.linalg.norm(gaze_3d)

                    if gaze_len > 0.001:
                        gaze_dir = gaze_3d / gaze_len

                        # Project onto RO normalized plane (X/Z, Y/Z)
                        # Use pupil and CC projections for consistency
                        # with combined glint gaze method
                        pupil_proj = np.array([pupil_3d[0] / pupil_3d[2],
                                               pupil_3d[1] / pupil_3d[2]])
                        cc_proj = np.array([median_cc[0] / median_cc[2],
                                            median_cc[1] / median_cc[2]])
                        gaze_norm_2d = pupil_proj - cc_proj

                        angle = float(np.degrees(
                            np.arctan2(-gaze_norm_2d[1], gaze_norm_2d[0])))
                        entry["corneal_3d_gaze_deg"] = round(angle, 1)
                        entry["corneal_3d_gaze_norm"] = [
                            round(float(gaze_norm_2d[0]), 6),
                            round(float(gaze_norm_2d[1]), 6)]
                        entry["pupil_3d"] = [round(float(pupil_3d[k]), 4)
                                             for k in range(3)]
                        entry["method"] = "stereo_cc"
                        gaze_result = gaze_norm_2d

                    # Refraction-corrected pupil_3d
                    if pc1 is not None and pc2 is not None:
                        pupil_refracted = _correct_pupil_refraction(
                            pc1, pc2, K1, dist1, K2, dist2, R, T,
                            median_cc, corneal_radius=7.8)
                        if pupil_refracted is not None:
                            gaze_ref = pupil_refracted - median_cc
                            if np.linalg.norm(gaze_ref) > 0.001:
                                p_proj = np.array([pupil_refracted[0] / pupil_refracted[2],
                                                   pupil_refracted[1] / pupil_refracted[2]])
                                cc_proj_r = np.array([median_cc[0] / median_cc[2],
                                                      median_cc[1] / median_cc[2]])
                                gaze_ref_2d = p_proj - cc_proj_r
                                entry["corneal_3d_gaze_norm_refracted"] = [
                                    round(float(gaze_ref_2d[0]), 6),
                                    round(float(gaze_ref_2d[1]), 6)]
                                entry["pupil_3d_refracted"] = [
                                    round(float(pupil_refracted[k]), 4)
                                    for k in range(3)]

            combined_results.append(entry)

            # --- Draw debug image on cam1 (RO) crop ---
            # Gaze is in cam1 frame, so draw on cam1's image
            stem = Path(frame_name).stem
            crop_path = out_base / cam1 / "cropped" / f"{stem}_cropped.png"
            if crop_path.exists():
                base_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                sz = crop_size
                canvas = np.zeros((sz, sz, 3), dtype=np.uint8) + 35

            ch, cw = canvas.shape[:2]
            pcx, pcy = cw // 2, ch // 2
            if r1.get("pupil_center") and r1.get("crop_bbox"):
                bb = r1["crop_bbox"]
                pcx = int(r1["pupil_center"][0] - bb[0])
                pcy = int(r1["pupil_center"][1] - bb[1])
                pcx = max(0, min(cw - 1, pcx))
                pcy = max(0, min(ch - 1, pcy))
            origin = (pcx, pcy)

            arrow_scale = min(ch, cw) * 3.0

            if gaze_result is not None and entry["corneal_3d_gaze_deg"] is not None:
                end_c = (int(origin[0] + gaze_result[0] * arrow_scale),
                         int(origin[1] + gaze_result[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end_c, (255, 229, 0), 2,
                                tipLength=0.2)
                cv2.putText(canvas, f"{entry['corneal_3d_gaze_deg']:+.0f} deg",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 229, 0), 1)
            else:
                cv2.putText(canvas, "No gaze", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

            if entry["corneal_center_3d"] is not None:
                cv2.drawMarker(canvas, origin, (255, 229, 0),
                               cv2.MARKER_DIAMOND, 8, 1)

            cv2.putText(canvas, "3D CORNEAL", (4, ch - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 229, 0), 1)
            cv2.putText(canvas, f"stereo CC",
                        (4, ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

            cv2.imwrite(str(comb_dir / f"{stem}_combined.png"), canvas)

            if (i + 1) % 200 == 0:
                valid = sum(1 for e in combined_results
                            if e.get("corneal_3d_gaze_deg") is not None)
                print(f"  [{eye.upper()} CORNEAL 3D] {i+1}/{n_frames} (valid={valid})")

        with open(str(out_base / f"{eye}_corneal_3d" / "combined_results.json"), "w") as f:
            json.dump(combined_results, f, indent=2)

        valid = sum(1 for e in combined_results
                    if e.get("corneal_3d_gaze_deg") is not None)
        print(f"  [{eye.upper()} CORNEAL 3D] {n_frames} frames, {valid} valid "
              f"({100*valid/max(1,n_frames):.0f}%)")


def undistort_gaze(pupil_xy, glint_centroid_xy, K, dist):
    """Compute gaze vector in undistorted normalized camera coordinates."""
    pts = np.array([[[pupil_xy[0], pupil_xy[1]]],
                    [[glint_centroid_xy[0], glint_centroid_xy[1]]]],
                   dtype=np.float32)  # shape (2, 1, 2)
    und = cv2.undistortPoints(pts, K, dist)  # returns normalized coords
    return float(und[0][0][0] - und[1][0][0]), float(und[0][0][1] - und[1][0][1])


def _undistort_single(px, py, K, dist):
    """Undistort a single pixel point to normalized camera coordinates."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    und = cv2.undistortPoints(pt, K, dist)
    return float(und[0][0][0]), float(und[0][0][1])


def _triangulate_point(pt1_norm, pt2_norm, R, T):
    """
    Triangulate a 3D point from undistorted normalized coordinates
    in two cameras.

    pt1_norm: (x, y) undistorted normalized coords in cam1
    pt2_norm: (x, y) undistorted normalized coords in cam2
    R: 3x3 rotation matrix (cam1 -> cam2)
    T: 3x1 translation vector (cam1 -> cam2)

    Returns: 3D point (X, Y, Z) in cam1's coordinate frame, or None if bad.
    """
    P1 = np.float64([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
    P2 = np.hstack([R, T.reshape(3, 1)]).astype(np.float64)

    pts1 = np.float64([[pt1_norm[0]], [pt1_norm[1]]])
    pts2 = np.float64([[pt2_norm[0]], [pt2_norm[1]]])

    X = cv2.triangulatePoints(P1, P2, pts1, pts2)
    w = X[3, 0]
    if abs(w) < 1e-10:
        return None
    pt3d = X[:3, 0] / w
    # Reject if behind camera (z <= 0) or unreasonably far
    if pt3d[2] <= 0:
        return None
    return pt3d


def _transform_gaze_to_frame(gaze_norm, R):
    """
    Transform gaze displacement from cam2's normalized frame to cam1's frame.
    Used as fallback when only one camera has valid detection.
    """
    ref_3d = np.array([0.0, 0.0, 1.0])
    gaze_3d = np.array([gaze_norm[0], gaze_norm[1], 1.0])
    ref_in_cam1 = R.T @ ref_3d
    gaze_in_cam1 = R.T @ gaze_3d
    ref_proj = (ref_in_cam1[0] / ref_in_cam1[2], ref_in_cam1[1] / ref_in_cam1[2])
    gaze_proj = (gaze_in_cam1[0] / gaze_in_cam1[2], gaze_in_cam1[1] / gaze_in_cam1[2])
    return (gaze_proj[0] - ref_proj[0], gaze_proj[1] - ref_proj[1])


# ============================================================
# Corneal refraction correction helpers
# ============================================================

def _ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    """Intersect a ray with a sphere. Returns closest intersection point or None.

    ray_origin, ray_dir: numpy arrays (3,)
    sphere_center: numpy array (3,)
    sphere_radius: float (mm)

    Solves |O + t*D - C|^2 = R^2 for t >= 0.
    Returns the nearest intersection point (smallest positive t).
    """
    oc = ray_origin - sphere_center
    a = float(np.dot(ray_dir, ray_dir))
    b = 2.0 * float(np.dot(oc, ray_dir))
    c = float(np.dot(oc, oc)) - sphere_radius ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # Take nearest positive intersection
    t = t1 if t1 > 1e-6 else t2
    if t < 1e-6:
        return None
    return ray_origin + t * ray_dir


def _sagittal_radius(R, Q, theta):
    """Sagittal radius of curvature for a conic cornea at angle theta.

    R: base corneal radius (mm)
    Q: asphericity (Q=0 → sphere, Q<0 → prolate ellipsoid)
    theta: angle between surface normal and corneal optical axis (radians)

    Returns rho_s = R / sqrt(1 - (1+Q) * sin^2(theta)).
    For Q=0: rho_s = R (sphere). For Q=-0.26: rho_s < R at periphery.
    """
    sin2 = np.sin(theta) ** 2
    arg = 1.0 - (1.0 + Q) * sin2
    # Clamp to avoid sqrt of negative at extreme angles
    if arg < 0.01:
        arg = 0.01
    return R / np.sqrt(arg)


def _ray_conic_intersect(ray_origin, ray_dir, conic_center, conic_axis, R, Q):
    """Intersect a ray with a conic corneal surface.

    The conic surface in local coords (origin=conic_center, z=conic_axis):
        x^2 + y^2 + (1+Q)*z^2 - 2*R*z = 0

    ray_origin, ray_dir: numpy arrays (3,)
    conic_center: numpy array (3,), center of corneal curvature
    conic_axis: numpy array (3,), unit vector along corneal optical axis
    R: corneal radius (mm)
    Q: asphericity parameter

    Returns intersection point in world coords, or None.
    """
    # Build local coordinate frame: z = conic_axis
    az = conic_axis / np.linalg.norm(conic_axis)
    # Choose an arbitrary perpendicular axis
    if abs(az[0]) < 0.9:
        ax = np.cross(az, np.array([1.0, 0.0, 0.0]))
    else:
        ax = np.cross(az, np.array([0.0, 1.0, 0.0]))
    ax = ax / np.linalg.norm(ax)
    ay = np.cross(az, ax)

    # Transform ray to local coords
    o_local = ray_origin - conic_center
    ox = float(np.dot(o_local, ax))
    oy = float(np.dot(o_local, ay))
    oz = float(np.dot(o_local, az))
    dx = float(np.dot(ray_dir, ax))
    dy = float(np.dot(ray_dir, ay))
    dz = float(np.dot(ray_dir, az))

    kk = 1.0 + Q
    # Quadratic: a*t^2 + b*t + c = 0
    a = dx * dx + dy * dy + kk * dz * dz
    b = 2.0 * (ox * dx + oy * dy + kk * oz * dz - R * dz)
    c = ox * ox + oy * oy + kk * oz * oz - 2.0 * R * oz

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t = t1 if t1 > 1e-6 else t2
    if t < 1e-6:
        return None
    return ray_origin + t * ray_dir


def _conic_normal(hit_point, conic_center, conic_axis, R, Q):
    """Compute outward unit normal of conic corneal surface at hit_point.

    Gradient of F(x,y,z) = x^2 + y^2 + (1+Q)*z^2 - 2*R*z in local coords:
        grad_F = [2x, 2y, 2*(1+Q)*z - 2*R]
    Transformed back to world coords and normalized.
    """
    az = conic_axis / np.linalg.norm(conic_axis)
    if abs(az[0]) < 0.9:
        ax = np.cross(az, np.array([1.0, 0.0, 0.0]))
    else:
        ax = np.cross(az, np.array([0.0, 1.0, 0.0]))
    ax = ax / np.linalg.norm(ax)
    ay = np.cross(az, ax)

    local = hit_point - conic_center
    x = float(np.dot(local, ax))
    y = float(np.dot(local, ay))
    z = float(np.dot(local, az))

    kk = 1.0 + Q
    grad_x = 2.0 * x
    grad_y = 2.0 * y
    grad_z = 2.0 * kk * z - 2.0 * R

    # Transform gradient back to world
    normal = grad_x * ax + grad_y * ay + grad_z * az
    n_len = np.linalg.norm(normal)
    if n_len < 1e-12:
        return None
    return normal / n_len


def _refract_ray(incident_dir, surface_normal, n1, n2):
    """Snell's law vector form: compute refracted ray direction.

    incident_dir: unit vector, incoming ray direction (pointing INTO surface)
    surface_normal: unit vector, outward surface normal
    n1: refractive index of medium the ray comes from
    n2: refractive index of medium the ray enters

    Returns refracted direction (unit vector) or None for total internal reflection.

    Vector Snell's law:
      t = (n1/n2) * i + (n1/n2 * cos_i - cos_t) * n
    where cos_i = -dot(i, n), cos_t = sqrt(1 - (n1/n2)^2 * (1 - cos_i^2))
    """
    ratio = n1 / n2
    cos_i = -float(np.dot(incident_dir, surface_normal))
    if cos_i < 0:
        # Ray hitting from inside — flip normal
        surface_normal = -surface_normal
        cos_i = -cos_i
    sin2_t = ratio * ratio * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None  # total internal reflection
    cos_t = np.sqrt(1.0 - sin2_t)
    refracted = ratio * incident_dir + (ratio * cos_i - cos_t) * surface_normal
    refracted = refracted / np.linalg.norm(refracted)
    return refracted


def _correct_pupil_refraction(pc1_xy, pc2_xy, K1, dist1, K2, dist2,
                               R_stereo, T_stereo, corneal_center,
                               corneal_radius=7.8, n_cornea=1.376):
    """Correct pupil 3D position for corneal refraction using Snell's law.

    For each camera, traces the ray from camera through corneal surface,
    applies refraction, and re-triangulates to find the true pupil position
    behind the refracting corneal sphere.

    pc1_xy: (x, y) pixel coords of pupil in cam1 (outer camera)
    pc2_xy: (x, y) pixel coords of pupil in cam2 (inner camera)
    K1, dist1: cam1 intrinsics
    K2, dist2: cam2 intrinsics
    R_stereo, T_stereo: stereo extrinsics (cam1 -> cam2)
    corneal_center: 3D corneal center in cam1 frame (3,)
    corneal_radius: corneal radius in mm
    n_cornea: refractive index of cornea (1.376)

    Returns corrected pupil_3d in cam1 frame, or None if correction fails.
    """
    N_AIR = 1.0
    cc = np.array(corneal_center, dtype=np.float64)

    # Inner camera origin in cam1 (outer camera) frame
    cam2_origin = (-np.array(R_stereo).T @ np.array(T_stereo).reshape(3, 1)).flatten()

    refracted_rays = []  # list of (origin, direction) for each camera

    for cam_origin, px, py, K, dist in [
        (np.zeros(3), pc1_xy[0], pc1_xy[1], K1, dist1),       # cam1 at origin
        (cam2_origin, pc2_xy[0], pc2_xy[1], K2, dist2),       # cam2
    ]:
        # Undistort pixel to normalized coords
        pt = np.array([[[px, py]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, K, dist)
        nx, ny = float(und[0][0][0]), float(und[0][0][1])

        # Ray direction in the camera's own frame
        ray_dir_cam = np.array([nx, ny, 1.0])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

        # For cam2, transform ray to cam1 frame
        if np.linalg.norm(cam_origin) > 1e-6:  # cam2
            ray_dir = (np.array(R_stereo).T @ ray_dir_cam.reshape(3, 1)).flatten()
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
        else:
            ray_dir = ray_dir_cam

        # Intersect ray with corneal sphere
        hit = _ray_sphere_intersect(cam_origin, ray_dir, cc, corneal_radius)
        if hit is None:
            return None

        # Surface normal at intersection (outward from sphere center)
        normal = (hit - cc)
        normal = normal / np.linalg.norm(normal)

        # Refract ray at corneal surface (air -> cornea)
        refracted_dir = _refract_ray(ray_dir, normal, N_AIR, n_cornea)
        if refracted_dir is None:
            return None

        refracted_rays.append((hit, refracted_dir))

    if len(refracted_rays) < 2:
        return None

    # Re-triangulate: closest approach of the two refracted rays
    p1, d1 = refracted_rays[0]
    p2, d2 = refracted_rays[1]

    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None

    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom

    closest1 = p1 + sc * d1
    closest2 = p2 + tc * d2
    pupil_3d_corrected = (closest1 + closest2) / 2.0

    # Sanity check: corrected pupil should be behind the corneal surface
    # (deeper into the eye, further from camera than CC)
    if pupil_3d_corrected[2] <= 0:
        return None

    return pupil_3d_corrected


def compute_combined_gaze(output_dir, calib, crop_size=150):
    """
    Post-process: combine gaze from inner+outer camera of each eye
    using stereo calibration to bring vectors into a common frame.
    """
    out_base = Path(output_dir)

    for eye, pair_key in [("right", "right"), ("left", "left")]:
        stereo = calib.get(pair_key)
        if stereo is None:
            continue

        cam1, cam2 = stereo['cam1'], stereo['cam2']  # outer, inner
        R = stereo['R']   # rotation cam1->cam2
        T = stereo['T']   # translation cam1->cam2
        K1, dist1 = calib[cam1]['K'], calib[cam1]['dist']
        K2, dist2 = calib[cam2]['K'], calib[cam2]['dist']

        # Load per-camera results
        r1_path = out_base / cam1 / "results.json"
        r2_path = out_base / cam2 / "results.json"
        if not r1_path.exists() or not r2_path.exists():
            continue
        with open(r1_path) as f:
            res1 = json.load(f)
        with open(r2_path) as f:
            res2 = json.load(f)

        comb_dir = out_base / f"{eye}_combined" / "combined_gaze"
        comb_dir.mkdir(parents=True, exist_ok=True)

        combined_results = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")
            stem = Path(frame_name).stem

            entry = {"frame": frame_name, "eye": eye,
                     "combined_gaze_deg": None,
                     "combined_gaze_norm": None,
                     "cam1_gaze_norm": None, "cam2_gaze_norm": None}

            g1_norm, g2_norm = None, None
            pc1_px, gc1_px, pc2_px, gc2_px = None, None, None, None
            MAX_GAZE_MAG = 0.4  # reject gaze with magnitude > ~23 deg

            # Cam1 (outer) - extract pixel coords and normalized gaze
            if (r1.get("pupil_center") and r1.get("glints")
                    and len(r1["glints"]) >= 1 and not r1.get("eye_closed")):
                pc1_px = r1["pupil_center"]
                gc1_px = (np.mean([g["x_orig"] for g in r1["glints"]]),
                          np.mean([g["y_orig"] for g in r1["glints"]]))
                g1_norm = undistort_gaze(pc1_px, gc1_px, K1, dist1)
                if np.sqrt(g1_norm[0]**2 + g1_norm[1]**2) > MAX_GAZE_MAG:
                    g1_norm = None
                    pc1_px, gc1_px = None, None
                else:
                    entry["cam1_gaze_norm"] = [round(g1_norm[0], 6), round(g1_norm[1], 6)]

            # Cam2 (inner) - extract pixel coords and normalized gaze
            if (r2.get("pupil_center") and r2.get("glints")
                    and len(r2["glints"]) >= 1 and not r2.get("eye_closed")):
                pc2_px = r2["pupil_center"]
                gc2_px = (np.mean([g["x_orig"] for g in r2["glints"]]),
                          np.mean([g["y_orig"] for g in r2["glints"]]))
                g2_norm = undistort_gaze(pc2_px, gc2_px, K2, dist2)
                if np.sqrt(g2_norm[0]**2 + g2_norm[1]**2) > MAX_GAZE_MAG:
                    g2_norm = None
                    pc2_px, gc2_px = None, None
                else:
                    entry["cam2_gaze_norm"] = [round(g2_norm[0], 6), round(g2_norm[1], 6)]

            # --- Combine using stereo triangulation when both cameras valid ---
            # Both cameras see the same pupil and same glint reflections.
            # Triangulate their 3D positions, then compute the true 3D gaze
            # vector = pupil_3d - glint_3d. Project back to cam1's normalized
            # plane for 2D output. This is geometrically correct (intersection
            # of rays) unlike averaging 2D projections from different viewpoints.
            combined = None
            g2_in_cam1 = None

            if pc1_px is not None and pc2_px is not None:
                # Undistort each point individually
                p1_n = _undistort_single(pc1_px[0], pc1_px[1], K1, dist1)
                g1_n = _undistort_single(gc1_px[0], gc1_px[1], K1, dist1)
                p2_n = _undistort_single(pc2_px[0], pc2_px[1], K2, dist2)
                g2_n = _undistort_single(gc2_px[0], gc2_px[1], K2, dist2)

                # Triangulate 3D positions in cam1's frame
                pupil_3d = _triangulate_point(p1_n, p2_n, R, T)
                glint_3d = _triangulate_point(g1_n, g2_n, R, T)

                if pupil_3d is not None and glint_3d is not None:
                    # 3D gaze direction: from glint to pupil
                    gaze_3d = pupil_3d - glint_3d

                    # Project back to cam1's normalized plane for 2D output:
                    # perspective-project both 3D points, take difference
                    pupil_proj = (pupil_3d[0] / pupil_3d[2],
                                  pupil_3d[1] / pupil_3d[2])
                    glint_proj = (glint_3d[0] / glint_3d[2],
                                  glint_3d[1] / glint_3d[2])
                    combined = np.array([pupil_proj[0] - glint_proj[0],
                                         pupil_proj[1] - glint_proj[1]])

                    # Store 3D gaze info for debugging
                    entry["gaze_3d"] = [round(float(gaze_3d[0]), 6),
                                        round(float(gaze_3d[1]), 6),
                                        round(float(gaze_3d[2]), 6)]
                    entry["method"] = "triangulation"

            # Fallback: single camera only
            if combined is None:
                if g1_norm and g2_norm:
                    # Both have 2D gaze but triangulation failed - average
                    g2_in_cam1 = _transform_gaze_to_frame(g2_norm, R)
                    combined = np.array([(g1_norm[0] + g2_in_cam1[0]) / 2.0,
                                         (g1_norm[1] + g2_in_cam1[1]) / 2.0])
                    entry["method"] = "average_fallback"
                elif g1_norm:
                    combined = np.array([g1_norm[0], g1_norm[1]])
                    entry["method"] = "cam1_only"
                elif g2_norm:
                    g2_in_cam1 = _transform_gaze_to_frame(g2_norm, R)
                    combined = np.array([g2_in_cam1[0], g2_in_cam1[1]])
                    entry["method"] = "cam2_only"

            if combined is not None:
                cx, cy = float(combined[0]), float(combined[1])
                mag = np.sqrt(cx**2 + cy**2)
                if mag > 0.001:
                    angle = float(np.degrees(np.arctan2(-cy, cx)))
                    entry["combined_gaze_deg"] = round(angle, 1)
                    entry["combined_gaze_norm"] = [round(cx, 6), round(cy, 6)]

            combined_results.append(entry)

            # --- Draw combined gaze on outer camera (cam1/RO) crop ---
            # Gaze is in cam1 frame, so draw on cam1's image
            crop_path = out_base / cam1 / "cropped" / f"{stem}_cropped.png"
            if crop_path.exists():
                base_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                sz = crop_size
                canvas = np.zeros((sz, sz, 3), dtype=np.uint8) + 35

            ch, cw = canvas.shape[:2]
            pcx, pcy = cw // 2, ch // 2
            if r1.get("pupil_center") and r1.get("crop_bbox"):
                bb = r1["crop_bbox"]
                pcx = int(r1["pupil_center"][0] - bb[0])
                pcy = int(r1["pupil_center"][1] - bb[1])
                pcx = max(0, min(cw - 1, pcx))
                pcy = max(0, min(ch - 1, pcy))
            origin = (pcx, pcy)

            arrow_scale = min(ch, cw) * 3.0

            # Cam1 (outer) arrow - blue
            if g1_norm:
                end1 = (int(origin[0] + g1_norm[0] * arrow_scale),
                        int(origin[1] + g1_norm[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end1, (255, 130, 0), 1, tipLength=0.25)

            # Cam2 (inner) arrow - green (transformed to cam1 frame)
            if g2_norm:
                if g2_in_cam1 is not None:
                    end2 = (int(origin[0] + g2_in_cam1[0] * arrow_scale),
                            int(origin[1] + g2_in_cam1[1] * arrow_scale))
                else:
                    g2t = _transform_gaze_to_frame(g2_norm, R)
                    end2 = (int(origin[0] + g2t[0] * arrow_scale),
                            int(origin[1] + g2t[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end2, (0, 200, 0), 1, tipLength=0.25)

            # Combined arrow - bright yellow, thicker
            if combined is not None and entry["combined_gaze_deg"] is not None:
                end_c = (int(origin[0] + combined[0] * arrow_scale),
                         int(origin[1] + combined[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end_c, (0, 255, 255), 2, tipLength=0.2)
                cv2.putText(canvas, f"{entry['combined_gaze_deg']:+.0f} deg",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            else:
                cv2.putText(canvas, "No gaze", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

            # Legend
            cv2.putText(canvas, f"{cam1.upper()}", (4, ch - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 130, 0), 1)
            cv2.putText(canvas, f"{cam2.upper()}", (4, ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 0), 1)

            cv2.imwrite(str(comb_dir / f"{stem}_combined.png"), canvas)

            if (i + 1) % 200 == 0:
                valid = sum(1 for e in combined_results if e.get("combined_gaze_deg") is not None)
                print(f"  [{eye.upper()}] {i+1}/{n_frames} (valid={valid})")

        with open(str(out_base / f"{eye}_combined" / "combined_results.json"), "w") as f:
            json.dump(combined_results, f, indent=2)

        valid = sum(1 for e in combined_results if e.get("combined_gaze_deg") is not None)
        print(f"  [{eye.upper()} COMBINED] {n_frames} frames, {valid} valid ({100*valid/max(1,n_frames):.0f}%)")


def compute_combined_seg_gaze(output_dir, calib, crop_size=150):
    """
    Post-process: combine seg-based gaze from inner+outer camera of each eye
    using stereo calibration. Uses seg_enh_pupil (contour-fit) and seg_enh_glints
    (enhanced, filtered within seg mask). Requires >= 2 glints per camera.
    """
    out_base = Path(output_dir)

    for eye, pair_key in [("right", "right"), ("left", "left")]:
        stereo = calib.get(pair_key)
        if stereo is None:
            continue

        cam1, cam2 = stereo['cam1'], stereo['cam2']  # outer, inner
        R = stereo['R']
        T = stereo['T']
        K1, dist1 = calib[cam1]['K'], calib[cam1]['dist']
        K2, dist2 = calib[cam2]['K'], calib[cam2]['dist']

        # Load per-camera results
        r1_path = out_base / cam1 / "results.json"
        r2_path = out_base / cam2 / "results.json"
        if not r1_path.exists() or not r2_path.exists():
            continue
        with open(r1_path) as f:
            res1 = json.load(f)
        with open(r2_path) as f:
            res2 = json.load(f)

        comb_dir = out_base / f"{eye}_seg_combined" / "combined_gaze"
        comb_dir.mkdir(parents=True, exist_ok=True)

        combined_results = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")
            stem = Path(frame_name).stem

            entry = {"frame": frame_name, "eye": eye,
                     "seg_combined_gaze_deg": None,
                     "seg_combined_gaze_norm": None,
                     "cam1_seg_gaze_norm": None, "cam2_seg_gaze_norm": None}

            g1_norm, g2_norm = None, None
            pc1_px, gc1_px, pc2_px, gc2_px = None, None, None, None
            MAX_GAZE_MAG = 0.4

            # Cam1 (outer) - use seg_enh_pupil + seg_enh_glints (aligned with :5056 pipeline)
            if (r1.get("seg_enh_pupil") and r1.get("seg_enh_glints")
                    and len(r1["seg_enh_glints"]) >= 2 and not r1.get("eye_closed")):
                pc1_px = r1["seg_enh_pupil"]
                gc1_px = (np.mean([g["x_orig"] for g in r1["seg_enh_glints"]]),
                          np.mean([g["y_orig"] for g in r1["seg_enh_glints"]]))
                g1_norm = undistort_gaze(pc1_px, gc1_px, K1, dist1)
                if np.sqrt(g1_norm[0]**2 + g1_norm[1]**2) > MAX_GAZE_MAG:
                    g1_norm = None
                    pc1_px, gc1_px = None, None
                else:
                    entry["cam1_seg_gaze_norm"] = [round(g1_norm[0], 6), round(g1_norm[1], 6)]

            # Cam2 (inner) - use seg_enh_pupil + seg_enh_glints (aligned with :5056 pipeline)
            if (r2.get("seg_enh_pupil") and r2.get("seg_enh_glints")
                    and len(r2["seg_enh_glints"]) >= 2 and not r2.get("eye_closed")):
                pc2_px = r2["seg_enh_pupil"]
                gc2_px = (np.mean([g["x_orig"] for g in r2["seg_enh_glints"]]),
                          np.mean([g["y_orig"] for g in r2["seg_enh_glints"]]))
                g2_norm = undistort_gaze(pc2_px, gc2_px, K2, dist2)
                if np.sqrt(g2_norm[0]**2 + g2_norm[1]**2) > MAX_GAZE_MAG:
                    g2_norm = None
                    pc2_px, gc2_px = None, None
                else:
                    entry["cam2_seg_gaze_norm"] = [round(g2_norm[0], 6), round(g2_norm[1], 6)]

            # Combine using stereo triangulation
            combined = None
            g2_in_cam1 = None

            if pc1_px is not None and pc2_px is not None:
                p1_n = _undistort_single(pc1_px[0], pc1_px[1], K1, dist1)
                g1_n = _undistort_single(gc1_px[0], gc1_px[1], K1, dist1)
                p2_n = _undistort_single(pc2_px[0], pc2_px[1], K2, dist2)
                g2_n = _undistort_single(gc2_px[0], gc2_px[1], K2, dist2)

                pupil_3d = _triangulate_point(p1_n, p2_n, R, T)
                glint_3d = _triangulate_point(g1_n, g2_n, R, T)

                if pupil_3d is not None and glint_3d is not None:
                    gaze_3d = pupil_3d - glint_3d
                    pupil_proj = (pupil_3d[0] / pupil_3d[2],
                                  pupil_3d[1] / pupil_3d[2])
                    glint_proj = (glint_3d[0] / glint_3d[2],
                                  glint_3d[1] / glint_3d[2])
                    combined = np.array([pupil_proj[0] - glint_proj[0],
                                         pupil_proj[1] - glint_proj[1]])
                    entry["method"] = "triangulation"
                    entry["pupil_3d"] = [round(float(pupil_3d[k]), 4) for k in range(3)]

                    # --- Pupil diameter in mm (pinhole model) ---
                    depth = float(pupil_3d[2])
                    if depth > 0:
                        pr1 = r1.get("seg_enh_pupil_radius") or r1.get("pupil_radius")
                        pr2 = r2.get("seg_enh_pupil_radius") or r2.get("pupil_radius")
                        fx1 = float(K1[0, 0])
                        fx2 = float(K2[0, 0])
                        d1_mm = 2.0 * pr1 * depth / fx1 if pr1 and fx1 > 0 else None
                        d2_mm = 2.0 * pr2 * depth / fx2 if pr2 and fx2 > 0 else None
                        if d1_mm is not None and d2_mm is not None:
                            entry["pupil_diameter_mm"] = round((d1_mm + d2_mm) / 2.0, 2)
                        elif d1_mm is not None:
                            entry["pupil_diameter_mm"] = round(d1_mm, 2)
                        elif d2_mm is not None:
                            entry["pupil_diameter_mm"] = round(d2_mm, 2)

            # Fallback: single camera only
            if combined is None:
                if g1_norm and g2_norm:
                    g2_in_cam1 = _transform_gaze_to_frame(g2_norm, R)
                    combined = np.array([(g1_norm[0] + g2_in_cam1[0]) / 2.0,
                                         (g1_norm[1] + g2_in_cam1[1]) / 2.0])
                    entry["method"] = "average_fallback"
                elif g1_norm:
                    combined = np.array([g1_norm[0], g1_norm[1]])
                    entry["method"] = "cam1_only"
                elif g2_norm:
                    g2_in_cam1 = _transform_gaze_to_frame(g2_norm, R)
                    combined = np.array([g2_in_cam1[0], g2_in_cam1[1]])
                    entry["method"] = "cam2_only"

            if combined is not None:
                cx, cy = float(combined[0]), float(combined[1])
                mag = np.sqrt(cx**2 + cy**2)
                if mag > 0.001:
                    angle = float(np.degrees(np.arctan2(-cy, cx)))
                    entry["seg_combined_gaze_deg"] = round(angle, 1)
                    entry["seg_combined_gaze_norm"] = [round(cx, 6), round(cy, 6)]

            combined_results.append(entry)

            # --- Draw seg combined gaze on inner camera's cropped image ---
            crop_path = out_base / cam2 / "cropped" / f"{stem}_cropped.png"
            if crop_path.exists():
                base_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                sz = crop_size
                canvas = np.zeros((sz, sz, 3), dtype=np.uint8) + 35

            ch, cw = canvas.shape[:2]
            # Use seg pupil position from inner camera as arrow origin
            pcx, pcy = cw // 2, ch // 2
            if r2.get("seg_pupil_center") and r2.get("crop_bbox"):
                bb = r2["crop_bbox"]
                pcx = int(r2["seg_pupil_center"][0] - bb[0])
                pcy = int(r2["seg_pupil_center"][1] - bb[1])
                pcx = max(0, min(cw - 1, pcx))
                pcy = max(0, min(ch - 1, pcy))
            origin = (pcx, pcy)

            arrow_scale = min(ch, cw) * 3.0

            # Cam1 (outer) arrow - dark magenta
            if g1_norm:
                end1 = (int(origin[0] + g1_norm[0] * arrow_scale),
                        int(origin[1] + g1_norm[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end1, (180, 0, 130), 1, tipLength=0.25)

            # Cam2 (inner) arrow - pink
            if g2_norm:
                if g2_in_cam1 is not None:
                    end2 = (int(origin[0] + g2_in_cam1[0] * arrow_scale),
                            int(origin[1] + g2_in_cam1[1] * arrow_scale))
                else:
                    g2t = _transform_gaze_to_frame(g2_norm, R)
                    end2 = (int(origin[0] + g2t[0] * arrow_scale),
                            int(origin[1] + g2t[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end2, (180, 105, 255), 1, tipLength=0.25)

            # Combined arrow - bright magenta, thicker
            if combined is not None and entry["seg_combined_gaze_deg"] is not None:
                end_c = (int(origin[0] + combined[0] * arrow_scale),
                         int(origin[1] + combined[1] * arrow_scale))
                cv2.arrowedLine(canvas, origin, end_c, (255, 0, 255), 2, tipLength=0.2)
                cv2.putText(canvas, f"{entry['seg_combined_gaze_deg']:+.0f} deg",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            else:
                cv2.putText(canvas, "No gaze", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

            # Legend
            cv2.putText(canvas, "SEG COMBINED", (4, ch - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 0, 255), 1)
            cv2.putText(canvas, f"{cam1.upper()} {cam2.upper()}", (4, ch - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 105, 255), 1)

            cv2.imwrite(str(comb_dir / f"{stem}_combined.png"), canvas)

            if (i + 1) % 200 == 0:
                valid = sum(1 for e in combined_results if e.get("seg_combined_gaze_deg") is not None)
                print(f"  [{eye.upper()} SEG COMBINED] {i+1}/{n_frames} (valid={valid})")

        with open(str(out_base / f"{eye}_seg_combined" / "combined_results.json"), "w") as f:
            json.dump(combined_results, f, indent=2)

        valid = sum(1 for e in combined_results if e.get("seg_combined_gaze_deg") is not None)
        print(f"  [{eye.upper()} SEG COMBINED] {n_frames} frames, {valid} valid ({100*valid/max(1,n_frames):.0f}%)")

    # --- Compute convergence using seg_combined_gaze_norm ---
    # Same gaze as the binocular visualization in the frontend.
    # Ray origins = camera positions (RO at origin, LO from cross-pair R,T).
    # Gaze directions = seg_combined_gaze_norm [gx, gy] -> normalize([gx, gy, 1.0]).
    # IPD from triangulated pupil_3d positions (per-eye stereo).
    cross = calib.get("cross")
    if cross is None:
        print("  [CONVERGENCE] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    # LO camera origin in RO frame: P_LO=0 -> P_RO = R^T * (0 - T) = -R^T * T
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load seg combined results for both eyes
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [CONVERGENCE] Need seg combined results for both eyes, skipping")
        return

    with open(r_seg_path) as f:
        r_seg = json.load(f)
    with open(l_seg_path) as f:
        l_seg = json.load(f)

    n_conv = min(len(r_seg), len(l_seg))
    convergence_results = []

    for i in range(n_conv):
        r_entry = r_seg[i]
        l_entry = l_seg[i]
        frame_name = r_entry.get("frame", f"frame_{i}")

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        r_gaze_norm = r_entry.get("seg_combined_gaze_norm")
        l_gaze_norm = l_entry.get("seg_combined_gaze_norm")

        if r_gaze_norm and l_gaze_norm:
            # Right gaze direction in RO frame: [gx, gy, 1.0]
            r_dir = np.array([r_gaze_norm[0], r_gaze_norm[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)

            # Left gaze direction in LO frame: [gx, gy, 1.0]
            l_dir_lo = np.array([l_gaze_norm[0], l_gaze_norm[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)

            # Transform left direction to RO frame: d_RO = R^T * d_LO
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            # Ray-ray closest approach: RO origin [0,0,0], LO origin in RO frame
            w0 = -lo_origin_ro  # [0,0,0] - lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d = float(np.dot(r_dir, w0))
            e = float(np.dot(l_dir_ro, w0))
            denom = a * c - b * b
            if abs(denom) > 1e-10:
                sc = (b * e - c * d) / denom
                tc = (a * e - b * d) / denom
                if sc > 0 and tc > 0:
                    closest_r = sc * r_dir
                    closest_l = lo_origin_ro + tc * l_dir_ro
                    convergence_point = (closest_r + closest_l) / 2.0
                    ray_miss = float(np.linalg.norm(closest_r - closest_l))
                    cam_mid = lo_origin_ro / 2.0  # midpoint between cameras
                    fixation_dist = float(np.linalg.norm(convergence_point - cam_mid))

                    entry["fixation_distance_mm"] = round(fixation_dist, 2)
                    entry["convergence_mm"] = round(fixation_dist, 2)
                    entry["convergence_point"] = [round(float(convergence_point[k]), 2) for k in range(3)]
                    entry["ray_miss_mm"] = round(ray_miss, 2)

            # IPD from triangulated pupil_3d (if available from seg combined)
            r_pupil = r_entry.get("pupil_3d")
            l_pupil = l_entry.get("pupil_3d")
            if r_pupil and l_pupil:
                rp = np.array(r_pupil)     # right pupil in RO frame
                lp_lo = np.array(l_pupil)  # left pupil in LO frame
                lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                ipd = float(np.linalg.norm(rp - lp_ro))
                entry["ipd_mm"] = round(ipd, 2)
                entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
                entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # Compute stats
    fix_vals = [e["fixation_distance_mm"] for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    if fix_vals:
        median_fix = float(np.median(fix_vals))
        mean_fix = float(np.mean(fix_vals))
        std_fix = float(np.std(fix_vals))
        print(f"  [CONVERGENCE] {len(fix_vals)} frames | "
              f"fixation: median={median_fix/10:.1f}cm mean={mean_fix/10:.1f}cm std={std_fix/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"] for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [CONVERGENCE] ray miss: median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [CONVERGENCE] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm std={np.std(ipd_vals):.1f}mm")

    # Save convergence metadata
    conv_meta = {
        "method": "seg_combined_gaze",
        "description": "Camera-origin rays using seg_combined_gaze_norm (same as binocular view)",
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }
    conv_path = out_base / "convergence_meta.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)


def _load_led_positions_from_calib_dir(calib_dir):
    """Load LED 3D positions from calibration directory pickles for all 4 cameras.

    Looks for:
      {calib_dir}/ro_ri/out/leds/full_calibration_data.pkl  (right eye pair)
      {calib_dir}/lo_li/out/leds/full_calibration_data.pkl  (left eye pair)

    Returns dict: { 'ro': [...], 'ri': [...], 'lo': [...], 'li': [...] }
    where each value is a list of 4 numpy (3,) arrays (LED positions in that camera's frame, mm).
    Returns None if neither pickle exists.
    """
    import pickle as _pickle

    class _FlexUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                class _D:
                    def __init__(self, *a, **kw): pass
                _D.__name__ = name
                _D.__module__ = module
                return _D

    calib_path = Path(calib_dir)
    result = {}

    for pair_name, outer_cam, inner_cam in [("ro_ri", "ro", "ri"), ("lo_li", "lo", "li")]:
        pkl_path = calib_path / pair_name / "out" / "leds" / "full_calibration_data.pkl"
        if not pkl_path.exists():
            print(f"  [LED] No LED pickle for {pair_name}: {pkl_path}")
            continue

        try:
            with open(str(pkl_path), 'rb') as f:
                data = _FlexUnpickler(f).load()

            glints = data.get('glints')
            if not glints or 'CAM_BOTH' not in glints:
                print(f"  [LED] Pickle {pair_name} missing glints/CAM_BOTH")
                continue

            cam_both = glints['CAM_BOTH']
            led_outer = []
            for idx in ['0', '1', '2', '3']:
                if idx not in cam_both:
                    break
                pw = cam_both[idx].get('point_world')
                if pw is None:
                    break
                led_outer.append(np.asarray(pw, dtype=np.float64))

            if len(led_outer) < 4:
                print(f"  [LED] {pair_name}: only {len(led_outer)} LEDs in CAM_BOTH, need 4")
                continue

            result[outer_cam] = led_outer
            print(f"  [LED] {outer_cam.upper()} frame: {len(led_outer)} LEDs loaded")
            for i, p in enumerate(led_outer):
                print(f"    LED[{i}]: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm")

            # Transform to inner camera frame: P_inner = R * P_outer + T
            sc = data.get('stereo_calibration', {})
            R_pkl = sc.get('R')
            T_pkl = sc.get('T')
            if R_pkl is not None and T_pkl is not None:
                R_pkl = np.asarray(R_pkl, dtype=np.float64)
                T_pkl = np.asarray(T_pkl, dtype=np.float64).flatten()
                led_inner = [R_pkl @ p + T_pkl for p in led_outer]
                result[inner_cam] = led_inner
                print(f"  [LED] {inner_cam.upper()} frame: {len(led_inner)} LEDs (transformed)")
            else:
                print(f"  [LED] {pair_name}: no stereo R,T in pickle, cannot transform to {inner_cam.upper()}")

        except Exception as e:
            print(f"  [LED] Failed to load {pkl_path}: {e}")

    return result if result else None



def _load_led_positions_from_calib_dir_field(calib_dir, field='point_unmirrored'):
    """Load LED 3D positions using a specified field name (e.g. point_unmirrored).

    Same logic as _load_led_positions_from_calib_dir but reads a different field
    from the calibration pickle. Used for physical (unmirrored) LED positions.

    Returns dict: { 'ro': [...], 'ri': [...], 'lo': [...], 'li': [...] }
    or None if data not available.
    """
    import pickle as _pickle

    class _FlexUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                class _D:
                    def __init__(self, *a, **kw): pass
                _D.__name__ = name
                _D.__module__ = module
                return _D

    calib_path = Path(calib_dir)
    result = {}

    for pair_name, outer_cam, inner_cam in [("ro_ri", "ro", "ri"), ("lo_li", "lo", "li")]:
        pkl_path = calib_path / pair_name / "out" / "leds" / "full_calibration_data.pkl"
        if not pkl_path.exists():
            continue

        try:
            with open(str(pkl_path), 'rb') as f:
                data = _FlexUnpickler(f).load()

            glints = data.get('glints')
            if not glints or 'CAM_BOTH' not in glints:
                continue

            cam_both = glints['CAM_BOTH']
            led_outer = []
            for idx in ['0', '1', '2', '3']:
                if idx not in cam_both:
                    break
                pw = cam_both[idx].get(field)
                if pw is None:
                    break
                led_outer.append(np.asarray(pw, dtype=np.float64))

            if len(led_outer) < 4:
                print(f"  [LED-{field}] {pair_name}: only {len(led_outer)} LEDs with '{field}', need 4")
                continue

            result[outer_cam] = led_outer
            print(f"  [LED-{field}] {outer_cam.upper()} frame: {len(led_outer)} LEDs loaded ({field})")
            for i, p in enumerate(led_outer):
                print(f"    LED[{i}]: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm")

            # Transform to inner camera frame
            sc = data.get('stereo_calibration', {})
            R_pkl = sc.get('R')
            T_pkl = sc.get('T')
            if R_pkl is not None and T_pkl is not None:
                R_pkl = np.asarray(R_pkl, dtype=np.float64)
                T_pkl = np.asarray(T_pkl, dtype=np.float64).flatten()
                led_inner = [R_pkl @ p + T_pkl for p in led_outer]
                result[inner_cam] = led_inner
                print(f"  [LED-{field}] {inner_cam.upper()} frame: {len(led_inner)} LEDs (transformed)")
            else:
                print(f"  [LED-{field}] {pair_name}: no stereo R,T, cannot transform to {inner_cam.upper()}")

        except Exception as e:
            print(f"  [LED-{field}] Failed to load {pkl_path}: {e}")

    return result if result else None



def _load_cc_observations(out_base, method, eye):
    """Load CC observations from saved JSON file."""
    obs_path = Path(out_base) / f"cc_observations_{method}_{eye}.json"
    if obs_path.exists():
        with open(str(obs_path)) as f:
            return json.load(f)
    return None


def _recompute_median_cc_from_observations(cc_observations, cal_cutoff_time):
    """Recompute median CC using only calibration-phase frames.

    cc_observations: list of {"frame": "...", "cc": [x, y, z]}
    cal_cutoff_time: frame timestamp cutoff (cal frames have ts <= cutoff)
    Returns: numpy array of median CC, or None if not enough observations.
    """
    if not cc_observations or cal_cutoff_time is None:
        return None

    cal_ccs = []
    for obs in cc_observations:
        ft = _frame_timestamp(obs["frame"])
        if ft is not None and ft <= cal_cutoff_time:
            cal_ccs.append(obs["cc"])

    if len(cal_ccs) < 3:
        return None

    return np.median(np.array(cal_ccs), axis=0)


def compute_corneal_3d_convergence(output_dir, calib, crop_size=150):
    """Compute convergence using stereo corneal 3D model gaze.

    Uses the corneal center from stereo glint triangulation + sphere fitting
    (computed by compute_corneal_3d_gaze) and pupil_3d from stereo pupil
    triangulation. Gaze = normalize(pupil_3d - corneal_center).

    This is an alternative to the PCCR-based seg_combined_gaze convergence.
    Saves to convergence_meta_corneal3d.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [CORNEAL 3D CONV] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load corneal 3D results for both eyes
    r_c3d_path = out_base / "right_corneal_3d" / "combined_results.json"
    l_c3d_path = out_base / "left_corneal_3d" / "combined_results.json"
    if not r_c3d_path.exists() or not l_c3d_path.exists():
        print("  [CORNEAL 3D CONV] Need corneal 3D results for both eyes, skipping")
        return

    with open(r_c3d_path) as f:
        r_c3d = json.load(f)
    with open(l_c3d_path) as f:
        l_c3d = json.load(f)

    # Extract corneal centers (fixed median per eye)
    right_cc = None
    left_cc = None
    for entry in r_c3d:
        cc = entry.get("corneal_center_3d")
        if cc:
            right_cc = np.array(cc)
            break
    for entry in l_c3d:
        cc = entry.get("corneal_center_3d")
        if cc:
            left_cc = np.array(cc)
            break

    if right_cc is None or left_cc is None:
        print("  [CORNEAL 3D CONV] Missing corneal centers, skipping")
        return

    print(f"  [CORNEAL 3D CONV] Right CC (RO frame): [{right_cc[0]:.2f}, {right_cc[1]:.2f}, {right_cc[2]:.2f}]")
    print(f"  [CORNEAL 3D CONV] Left CC (LO frame):  [{left_cc[0]:.2f}, {left_cc[1]:.2f}, {left_cc[2]:.2f}]")

    # Build frame lookups from corneal 3D results (need corneal_3d_gaze_norm)
    r_by_frame = {e["frame"]: e for e in r_c3d if e.get("corneal_3d_gaze_norm")}
    l_by_frame = {e["frame"]: e for e in l_c3d if e.get("corneal_3d_gaze_norm")}

    # Also load seg combined for IPD (pupil_3d)
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame = {}
    l_seg_by_frame = {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    # Process common frames
    common_frames = sorted(set(r_by_frame.keys()) & set(l_by_frame.keys()))
    convergence_results = []

    for frame_name in common_frames:
        r_entry = r_by_frame[frame_name]
        l_entry = l_by_frame[frame_name]

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        # Helper: compute convergence from a pair of gaze norms
        def _conv_from_gaze(r_gn, l_gn):
            r_d = np.array([r_gn[0], r_gn[1], 1.0])
            r_d = r_d / np.linalg.norm(r_d)
            l_d_lo = np.array([l_gn[0], l_gn[1], 1.0])
            l_d_lo = l_d_lo / np.linalg.norm(l_d_lo)
            l_d_ro = (R_cross.T @ l_d_lo.reshape(3, 1)).flatten()
            l_d_ro = l_d_ro / np.linalg.norm(l_d_ro)
            w0 = -lo_origin_ro
            aa = float(np.dot(r_d, r_d))
            bb = float(np.dot(r_d, l_d_ro))
            cc = float(np.dot(l_d_ro, l_d_ro))
            dd = float(np.dot(r_d, w0))
            ee = float(np.dot(l_d_ro, w0))
            den = aa * cc - bb * bb
            if abs(den) < 1e-10:
                return None
            sc_v = (bb * ee - cc * dd) / den
            tc_v = (aa * ee - bb * dd) / den
            if sc_v <= 0 or tc_v <= 0:
                return None
            cl_r = sc_v * r_d
            cl_l = lo_origin_ro + tc_v * l_d_ro
            cp = (cl_r + cl_l) / 2.0
            rm = float(np.linalg.norm(cl_r - cl_l))
            cm = lo_origin_ro / 2.0
            fd = float(np.linalg.norm(cp - cm))
            return {"fixation_distance_mm": round(fd, 2),
                    "convergence_mm": round(fd, 2),
                    "convergence_point": [round(float(cp[k]), 2) for k in range(3)],
                    "ray_miss_mm": round(rm, 2)}

        # Standard convergence
        r_gaze_norm = r_entry["corneal_3d_gaze_norm"]
        l_gaze_norm = l_entry["corneal_3d_gaze_norm"]
        result = _conv_from_gaze(r_gaze_norm, l_gaze_norm)
        if result:
            entry.update(result)

        # Refraction-corrected convergence
        r_gaze_ref = r_entry.get("corneal_3d_gaze_norm_refracted")
        l_gaze_ref = l_entry.get("corneal_3d_gaze_norm_refracted")
        if r_gaze_ref and l_gaze_ref:
            ref_result = _conv_from_gaze(r_gaze_ref, l_gaze_ref)
            if ref_result:
                entry["fixation_distance_mm_refracted"] = ref_result["fixation_distance_mm"]
                entry["convergence_mm_refracted"] = ref_result["convergence_mm"]
                entry["convergence_point_refracted"] = ref_result["convergence_point"]
                entry["ray_miss_mm_refracted"] = ref_result["ray_miss_mm"]

        # IPD from seg combined pupil_3d (same stereo triangulation)
        r_seg = r_seg_by_frame.get(frame_name)
        l_seg = l_seg_by_frame.get(frame_name)
        if r_seg and l_seg:
            r_pupil = r_seg.get("pupil_3d")
            l_pupil = l_seg.get("pupil_3d")
            if r_pupil and l_pupil:
                rp_s = np.array(r_pupil)
                lp_s_lo = np.array(l_pupil)
                lp_s_ro = (R_cross.T @ (lp_s_lo.reshape(3, 1) - T_cross)).flatten()
                ipd = float(np.linalg.norm(rp_s - lp_s_ro))
                entry["ipd_mm"] = round(ipd, 2)
                entry["right_pupil_3d"] = [round(float(rp_s[k]), 4) for k in range(3)]
                entry["left_pupil_3d_ro"] = [round(float(lp_s_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"] for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    if fix_vals:
        median_fix = float(np.median(fix_vals))
        print(f"  [CORNEAL 3D CONV] {len(fix_vals)} frames | "
              f"fixation: median={median_fix/10:.1f}cm mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"] for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [CORNEAL 3D CONV] ray miss: median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [CORNEAL 3D CONV] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    # Refraction stats
    fix_vals_ref = [e["fixation_distance_mm_refracted"] for e in convergence_results
                    if e.get("fixation_distance_mm_refracted")]
    if fix_vals_ref:
        print(f"  [CORNEAL 3D CONV] Refracted: {len(fix_vals_ref)} frames | "
              f"fixation: median={np.median(fix_vals_ref)/10:.1f}cm "
              f"std={np.std(fix_vals_ref)/10:.1f}cm")

    conv_meta = {
        "method": "corneal_3d",
        "description": "Stereo corneal center (sphere fit) + pupil_3d gaze",
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "median_fixation_mm_refracted": round(float(np.median(fix_vals_ref)), 2) if fix_vals_ref else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": [round(float(right_cc[k]), 4) for k in range(3)],
        "left_corneal_center_lo": [round(float(left_cc[k]), 4) for k in range(3)],
        "cc_observations_right": _load_cc_observations(out_base, "corneal3d", "right"),
        "cc_observations_left": _load_cc_observations(out_base, "corneal3d", "left"),
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "fixation_distance_mm_refracted": e.get("fixation_distance_mm_refracted"),
            "convergence_mm_refracted": e.get("convergence_mm_refracted"),
            "convergence_point_refracted": e.get("convergence_point_refracted"),
            "ray_miss_mm_refracted": e.get("ray_miss_mm_refracted"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_corneal3d.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CORNEAL 3D CONV] Saved to {conv_path}")


def _solve_nray_fixation(rays, huber_delta=None, n_huber_iter=5):
    """Solve for the 3D fixation point from N camera gaze rays.

    Args:
        rays: list of (origin, direction, cam_name, static_weight) tuples
        huber_delta: if not None, apply Huber-weighted IRLS with this threshold (mm).
                     Rays with residual > delta are downweighted: w = delta / |r|.
        n_huber_iter: number of IRLS iterations (default 5).

    Returns:
        dict with keys: P, residuals, rms_residual, huber_weights (or None if failed)
    """
    if len(rays) < 2:
        return None

    I3 = np.eye(3)
    dyn_weights = np.ones(len(rays))
    n_iter = n_huber_iter if huber_delta is not None else 1
    P = None

    for iteration in range(n_iter):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for j, (origin, direction, _, static_w) in enumerate(rays):
            w = static_w * dyn_weights[j]
            d = direction.reshape(3, 1)
            M = I3 - d @ d.T
            A += w * M
            b += w * (M @ origin)

        try:
            P = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        if P[2] <= 0:
            return None

        # Update Huber weights if requested
        if huber_delta is not None and iteration < n_iter - 1:
            for j, (origin, direction, _, _) in enumerate(rays):
                diff = P - origin
                proj_len = np.dot(diff, direction)
                if proj_len < 0:
                    dyn_weights[j] = 0.1
                    continue
                perp = diff - proj_len * direction
                r_mm = np.linalg.norm(perp)
                dyn_weights[j] = 1.0 if r_mm <= huber_delta else huber_delta / r_mm

    # Final residuals
    residuals = {}
    for j, (origin, direction, cam_name, _) in enumerate(rays):
        diff = P - origin
        proj_len = np.dot(diff, direction)
        if proj_len < 0:
            continue
        perp = diff - proj_len * direction
        residuals[cam_name] = float(np.linalg.norm(perp))

    if not residuals:
        return None

    rms_residual = float(np.sqrt(np.mean([r**2 for r in residuals.values()])))
    cam_mid = np.mean([o for o, _, _, _ in rays], axis=0)
    fixation_dist = float(np.linalg.norm(P - cam_mid))

    result = {
        'P': P,
        'fixation_dist': fixation_dist,
        'residuals': residuals,
        'rms_residual': rms_residual,
    }
    if huber_delta is not None:
        result['huber_weights'] = {rays[j][2]: float(dyn_weights[j])
                                    for j in range(len(rays))}

    return result


def compute_4ray_convergence(output_dir, calib, crop_size=150):
    """Compute convergence from all 4 individual camera rays simultaneously.

    Instead of first combining cameras per-eye then intersecting 2 eye rays,
    this directly solves for the 3D fixation point P that minimizes the sum of
    squared distances to all camera gaze rays.

    For each camera i with origin O_i and unit gaze direction d_i:
      M_i = I - d_i * d_i^T   (perpendicular projection matrix)
      P = (Σ M_i)^{-1} * (Σ M_i * O_i)   (3x3 linear solve)

    Uses per-camera seg_gaze_vector_norm (individual PCCR gaze).
    Camera chain: RI <-(RO-RI)-> RO <-(RO-LO)-> LO <-(LO-LI)-> LI.
    Saves to convergence_meta_4ray.json.
    """
    out_base = Path(output_dir)

    # Need all three stereo pairs for full camera chain
    right_pair = calib.get("right")   # RO-RI
    left_pair = calib.get("left")     # LO-LI
    cross_pair = calib.get("cross")   # RO-LO
    if not all([right_pair, left_pair, cross_pair]):
        print("  [4-RAY] Need all 3 stereo pairs (right, left, cross), skipping")
        return

    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)
    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)

    # Camera origins in RO frame (mm)
    # Convention: P_cam2 = R * P_cam1 + T, so cam2 origin in cam1 frame = -R^T * T
    ro_origin = np.zeros(3)
    ri_origin = (-R_right.T @ T_right).flatten()
    lo_origin = (-R_cross.T @ T_cross).flatten()
    # LI: first in LO frame, then transform LO→RO
    li_in_lo = (-R_left.T @ T_left).flatten()
    li_origin = (R_cross.T @ (li_in_lo.reshape(3, 1) - T_cross)).flatten()

    cam_origins = {'ro': ro_origin, 'ri': ri_origin,
                   'lo': lo_origin, 'li': li_origin}

    # Camera weights from calibration reprojection error (inverse error weighting)
    cam_weights = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        reproj = calib.get(cam, {}).get('reproj_err')
        if reproj and reproj['mean_px'] > 0:
            cam_weights[cam] = 1.0 / reproj['mean_px']
        else:
            cam_weights[cam] = 1.0  # default weight if no reproj data
    # Normalize weights so mean = 1.0
    w_mean = np.mean(list(cam_weights.values()))
    if w_mean > 0:
        cam_weights = {cam: w / w_mean for cam, w in cam_weights.items()}

    # Physical sensor params for angular gaze
    cam_physical = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        f_mm = calib.get(cam, {}).get('f_mm')
        px_mm = calib.get(cam, {}).get('px_mm')
        if f_mm and px_mm:
            cam_physical[cam] = {'f_mm': f_mm, 'px_mm': px_mm,
                                 'deg_per_px': px_mm / f_mm * 180 / np.pi}

    print(f"  [4-RAY] Camera origins in RO frame (mm):")
    for cam, orig in cam_origins.items():
        w = cam_weights.get(cam, 1.0)
        phys = cam_physical.get(cam, {})
        extras = f" w={w:.3f}"
        if phys:
            extras += f" ({phys['deg_per_px']:.3f}°/px)"
        print(f"    {cam.upper()}: [{orig[0]:.2f}, {orig[1]:.2f}, {orig[2]:.2f}]{extras}")

    # Rotation matrices: transform gaze direction from each camera frame → RO frame
    # d_RO = R_pair^T * d_cam  (inverse of the stereo R that maps RO→cam)
    cam_rotations = {
        'ro': np.eye(3),
        'ri': R_right.T,
        'lo': R_cross.T,
        'li': R_cross.T @ R_left.T,   # LI→LO via R_left^T, then LO→RO via R_cross^T
    }

    # Load per-camera results
    cam_results = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        rpath = out_base / cam / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                cam_results[cam] = json.load(f)

    if len(cam_results) < 2:
        print(f"  [4-RAY] Need at least 2 cameras, only found {len(cam_results)}")
        return

    # Load seg combined for IPD (pupil_3d from stereo triangulation)
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame, l_seg_by_frame = {}, {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    n_frames = min(len(v) for v in cam_results.values())
    I3 = np.eye(3)
    convergence_results = []

    for i in range(n_frames):
        frame_name = None
        rays = []   # (origin, direction_ro, cam_name)

        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")

            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue

            # Gaze direction in camera frame: [gx, gy, 1.0] normalised
            d_cam = np.array([gaze_norm[0], gaze_norm[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)

            # Rotate to RO frame
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)

            # Sanity: gaze should be mostly forward (positive Z in RO frame)
            if d_ro[2] < 0.3:
                continue

            rays.append((cam_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "n_cameras": len(rays), "per_camera_residual": None,
                 "gaze_angles_deg": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        # Compute per-camera gaze angles in true degrees using physical params
        gaze_angles = {}
        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None:
                continue
            phys = cam_physical.get(cam)
            if phys:
                # gaze_norm is in normalized camera coords (already divided by f)
                # True angle = atan(gaze_norm_component) in radians
                h_deg = float(np.degrees(np.arctan(gaze_norm[0])))
                v_deg = float(np.degrees(np.arctan(gaze_norm[1])))
                gaze_angles[cam] = {'h_deg': round(h_deg, 3),
                                    'v_deg': round(v_deg, 3)}
        if gaze_angles:
            entry["gaze_angles_deg"] = gaze_angles

        if len(rays) >= 2:
            # Build weighted linear system: (Σ w_i M_i) P = Σ w_i M_i O_i
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for origin, direction, _, weight in rays:
                d = direction.reshape(3, 1)
                M = I3 - d @ d.T
                A += weight * M
                b += weight * (M @ origin)

            try:
                P = np.linalg.solve(A, b)

                if P[2] > 0:   # fixation point in front of cameras
                    # Per-camera residuals (perpendicular distance from P to each ray)
                    residuals = {}
                    for origin, direction, cam_name, _ in rays:
                        diff = P - origin
                        proj_len = np.dot(diff, direction)
                        if proj_len < 0:
                            continue   # ray points away from fixation
                        perp = diff - proj_len * direction
                        residuals[cam_name] = float(np.linalg.norm(perp))

                    if residuals:
                        rms_residual = float(np.sqrt(
                            np.mean([r**2 for r in residuals.values()])))

                        # Fixation distance from midpoint of participating cameras
                        cam_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                        fixation_dist = float(np.linalg.norm(P - cam_mid))

                        entry["fixation_distance_mm"] = round(fixation_dist, 2)
                        entry["convergence_mm"] = round(fixation_dist, 2)
                        entry["convergence_point"] = [
                            round(float(P[k]), 2) for k in range(3)]
                        entry["ray_miss_mm"] = round(rms_residual, 2)
                        entry["per_camera_residual"] = {
                            k: round(v, 2) for k, v in residuals.items()}
            except np.linalg.LinAlgError:
                pass

        # IPD from seg combined pupil_3d
        if frame_name:
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if r_seg and l_seg:
                r_pupil = r_seg.get("pupil_3d")
                l_pupil = l_seg.get("pupil_3d")
                if r_pupil and l_pupil:
                    rp = np.array(r_pupil)
                    lp_lo = np.array(l_pupil)
                    lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                    ipd = float(np.linalg.norm(rp - lp_ro))
                    entry["ipd_mm"] = round(ipd, 2)
                    entry["right_pupil_3d"] = [
                        round(float(rp[k]), 4) for k in range(3)]
                    entry["left_pupil_3d_ro"] = [
                        round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # --- Statistics ---
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    n_cam_vals = [e["n_cameras"]
                  for e in convergence_results if e["fixation_distance_mm"]]

    if fix_vals:
        print(f"  [4-RAY] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [4-RAY] residual (RMS): median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
        if n_cam_vals:
            print(f"  [4-RAY] cameras/frame: mean={np.mean(n_cam_vals):.1f} "
                  f"min={min(n_cam_vals)} max={max(n_cam_vals)}")
        # Per-camera average residual
        cam_resids = {cam: [] for cam in ['ro', 'ri', 'lo', 'li']}
        for e in convergence_results:
            pcr = e.get("per_camera_residual")
            if pcr:
                for cam, val in pcr.items():
                    cam_resids[cam].append(val)
        for cam in ['ro', 'ri', 'lo', 'li']:
            vals = cam_resids[cam]
            if vals:
                print(f"  [4-RAY]   {cam.upper()} avg residual: "
                      f"{np.mean(vals):.2f}mm ({len(vals)} frames)")
    if ipd_vals:
        print(f"  [4-RAY] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "4ray_weighted",
        "description": "4-camera weighted least-squares fixation point (inverse reproj error)",
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "camera_origins_ro": {cam: [round(float(v), 2) for v in orig]
                              for cam, orig in cam_origins.items()},
        "camera_weights": {cam: round(w, 4) for cam, w in cam_weights.items()},
        "camera_physical": {cam: {k: round(v, 4) for k, v in phys.items()}
                            for cam, phys in cam_physical.items()},
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "n_cameras": e.get("n_cameras"),
            "per_camera_residual": e.get("per_camera_residual"),
            "gaze_angles_deg": e.get("gaze_angles_deg"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_4ray.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [4-RAY] Saved to {conv_path}")


def compute_huber_4ray_convergence(output_dir, calib, crop_size=150,
                                    huber_delta=5.0, n_iterations=5):
    """Compute convergence from all 4 camera rays with Huber-weighted IRLS.

    Same as compute_4ray_convergence(), but wraps the weighted least-squares
    solve in an Iteratively Reweighted Least Squares (IRLS) loop using the
    Huber loss function to downweight outlier rays.

    Per iteration:
      1. Solve weighted N-ray LS: P = (Σ w_i M_i)^{-1} * (Σ w_i M_i * O_i)
      2. Compute per-ray residual r_i = perpendicular distance from P to ray i
      3. Update weights: w_i = w_static_i * huber(r_i) where
         huber(r) = 1 if |r| <= delta, else delta / |r|

    huber_delta: threshold in mm. Rays with residual > delta are downweighted.
    n_iterations: number of IRLS iterations (default 5).

    Saves to convergence_meta_huber4ray.json.
    """
    out_base = Path(output_dir)

    # Need all three stereo pairs for full camera chain
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    cross_pair = calib.get("cross")
    if not all([right_pair, left_pair, cross_pair]):
        print("  [HUBER-4R] Need all 3 stereo pairs (right, left, cross), skipping")
        return

    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)
    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)

    # Camera origins in RO frame (mm)
    ro_origin = np.zeros(3)
    ri_origin = (-R_right.T @ T_right).flatten()
    lo_origin = (-R_cross.T @ T_cross).flatten()
    li_in_lo = (-R_left.T @ T_left).flatten()
    li_origin = (R_cross.T @ (li_in_lo.reshape(3, 1) - T_cross)).flatten()

    cam_origins = {'ro': ro_origin, 'ri': ri_origin,
                   'lo': lo_origin, 'li': li_origin}

    # Camera weights from calibration reprojection error (inverse error weighting)
    cam_weights = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        reproj = calib.get(cam, {}).get('reproj_err')
        if reproj and reproj['mean_px'] > 0:
            cam_weights[cam] = 1.0 / reproj['mean_px']
        else:
            cam_weights[cam] = 1.0
    w_mean = np.mean(list(cam_weights.values()))
    if w_mean > 0:
        cam_weights = {cam: w / w_mean for cam, w in cam_weights.items()}

    cam_rotations = {
        'ro': np.eye(3),
        'ri': R_right.T,
        'lo': R_cross.T,
        'li': R_cross.T @ R_left.T,
    }

    # Load per-camera results
    cam_results = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        rpath = out_base / cam / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                cam_results[cam] = json.load(f)

    if len(cam_results) < 2:
        print(f"  [HUBER-4R] Need at least 2 cameras, only found {len(cam_results)}")
        return

    # Load seg combined for IPD
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame, l_seg_by_frame = {}, {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    print(f"  [HUBER-4R] Huber delta={huber_delta}mm, iterations={n_iterations}")
    print(f"  [HUBER-4R] Camera weights: "
          + ", ".join(f"{c.upper()}={cam_weights[c]:.3f}" for c in ['ro','ri','lo','li']))

    n_frames = min(len(v) for v in cam_results.values())
    I3 = np.eye(3)
    convergence_results = []
    total_downweighted = 0
    total_rays = 0

    for i in range(n_frames):
        frame_name = None
        rays = []

        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")

            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue

            d_cam = np.array([gaze_norm[0], gaze_norm[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)

            if d_ro[2] < 0.3:
                continue

            rays.append((cam_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "n_cameras": len(rays), "per_camera_residual": None,
                 "huber_weights": None}

        if len(rays) >= 2:
            # --- IRLS loop with Huber weights ---
            # Initialize dynamic weights to 1.0
            dyn_weights = np.ones(len(rays))

            P = None
            for iteration in range(n_iterations):
                A = np.zeros((3, 3))
                b = np.zeros(3)
                for j, (origin, direction, _, static_w) in enumerate(rays):
                    w = static_w * dyn_weights[j]
                    d = direction.reshape(3, 1)
                    M = I3 - d @ d.T
                    A += w * M
                    b += w * (M @ origin)

                try:
                    P = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    P = None
                    break

                if P[2] <= 0:
                    P = None
                    break

                # Compute per-ray residuals and update Huber weights
                for j, (origin, direction, _, _) in enumerate(rays):
                    diff = P - origin
                    proj_len = np.dot(diff, direction)
                    if proj_len < 0:
                        dyn_weights[j] = 0.1  # strongly downweight backward rays
                        continue
                    perp = diff - proj_len * direction
                    r_mm = np.linalg.norm(perp)

                    # Huber weight: 1.0 if |r| <= delta, else delta/|r|
                    if r_mm <= huber_delta:
                        dyn_weights[j] = 1.0
                    else:
                        dyn_weights[j] = huber_delta / r_mm

            if P is not None and P[2] > 0:
                # Final per-camera residuals
                residuals = {}
                for j, (origin, direction, cam_name, _) in enumerate(rays):
                    diff = P - origin
                    proj_len = np.dot(diff, direction)
                    if proj_len < 0:
                        continue
                    perp = diff - proj_len * direction
                    residuals[cam_name] = float(np.linalg.norm(perp))

                if residuals:
                    rms_residual = float(np.sqrt(
                        np.mean([r**2 for r in residuals.values()])))
                    cam_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                    fixation_dist = float(np.linalg.norm(P - cam_mid))

                    entry["fixation_distance_mm"] = round(fixation_dist, 2)
                    entry["convergence_mm"] = round(fixation_dist, 2)
                    entry["convergence_point"] = [
                        round(float(P[k]), 2) for k in range(3)]
                    entry["ray_miss_mm"] = round(rms_residual, 2)
                    entry["per_camera_residual"] = {
                        k: round(v, 2) for k, v in residuals.items()}

                    # Record final Huber weights
                    hw = {}
                    for j, (_, _, cam_name, _) in enumerate(rays):
                        hw[cam_name] = round(float(dyn_weights[j]), 4)
                    entry["huber_weights"] = hw

                    # Track downweighted rays
                    total_rays += len(rays)
                    total_downweighted += sum(1 for w in dyn_weights if w < 0.99)

        # IPD from seg combined pupil_3d
        if frame_name:
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if r_seg and l_seg:
                r_pupil = r_seg.get("pupil_3d")
                l_pupil = l_seg.get("pupil_3d")
                if r_pupil and l_pupil:
                    rp = np.array(r_pupil)
                    lp_lo = np.array(l_pupil)
                    lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                    ipd = float(np.linalg.norm(rp - lp_ro))
                    entry["ipd_mm"] = round(ipd, 2)
                    entry["right_pupil_3d"] = [
                        round(float(rp[k]), 4) for k in range(3)]
                    entry["left_pupil_3d_ro"] = [
                        round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # --- Statistics ---
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]

    if fix_vals:
        print(f"  [HUBER-4R] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [HUBER-4R] residual (RMS): median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
        if total_rays > 0:
            pct = 100.0 * total_downweighted / total_rays
            print(f"  [HUBER-4R] Huber downweighted: {total_downweighted}/{total_rays} "
                  f"rays ({pct:.1f}%)")
        # Per-camera average residual
        cam_resids = {cam: [] for cam in ['ro', 'ri', 'lo', 'li']}
        for e in convergence_results:
            pcr = e.get("per_camera_residual")
            if pcr:
                for cam, val in pcr.items():
                    cam_resids[cam].append(val)
        for cam in ['ro', 'ri', 'lo', 'li']:
            vals = cam_resids[cam]
            if vals:
                print(f"  [HUBER-4R]   {cam.upper()} avg residual: "
                      f"{np.mean(vals):.2f}mm ({len(vals)} frames)")
    if ipd_vals:
        print(f"  [HUBER-4R] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "huber_4ray_weighted",
        "description": "4-camera Huber-weighted IRLS fixation point (robust to outlier rays)",
        "huber_delta_mm": huber_delta,
        "n_iterations": n_iterations,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "camera_origins_ro": {cam: [round(float(v), 2) for v in orig]
                              for cam, orig in cam_origins.items()},
        "camera_weights": {cam: round(w, 4) for cam, w in cam_weights.items()},
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "n_cameras": e.get("n_cameras"),
            "per_camera_residual": e.get("per_camera_residual"),
            "huber_weights": e.get("huber_weights"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_huber4ray.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [HUBER-4R] Saved to {conv_path}")


def _estimate_personal_corneal_radius(cc_observations_by_R, default_R=7.8):
    """Estimate the individual's corneal radius from glint reflection residuals.

    For each candidate R, compute CC = glint_3d - R * normal for all observations.
    The R that minimizes the scatter (std) of the resulting CC cloud is the best estimate.

    cc_observations_by_R: list of dicts, each with:
        'glint_3d': numpy array (3,)
        'normal': numpy array (3,), unit outward surface normal
    default_R: population average corneal radius (mm)

    Returns: dict with 'personal_R_mm', 'confidence', 'default_R_mm',
             'scatter_at_personal_R', 'scatter_at_default_R'
             or None if estimation fails.
    """
    if len(cc_observations_by_R) < 10:
        return None

    from scipy.optimize import minimize_scalar

    def cc_scatter(R):
        ccs = []
        for obs in cc_observations_by_R:
            cc = obs['glint_3d'] - R * obs['normal']
            if 5 < cc[2] < 80:
                ccs.append(cc)
        if len(ccs) < 5:
            return 1e6
        arr = np.array(ccs)
        # Use total std (sum of per-axis std) as scatter metric
        return float(np.sum(np.std(arr, axis=0)))

    # Optimize R in physiological range [6.5, 9.5] mm
    result = minimize_scalar(cc_scatter, bounds=(6.5, 9.5), method='bounded',
                              options={'xatol': 0.01, 'maxiter': 200})

    personal_R = result.x
    scatter_personal = cc_scatter(personal_R)
    scatter_default = cc_scatter(default_R)

    # Confidence based on how much improvement personal R gives
    if scatter_default > 1e-6:
        improvement = (scatter_default - scatter_personal) / scatter_default
    else:
        improvement = 0.0

    # If scatter is nearly flat (< 2% improvement), estimation is unreliable
    if improvement < 0.02:
        confidence = "low"
    elif improvement < 0.10:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        'personal_R_mm': round(float(personal_R), 2),
        'confidence': confidence,
        'improvement_pct': round(float(improvement * 100), 1),
        'default_R_mm': default_R,
        'scatter_at_personal_R': round(float(scatter_personal), 4),
        'scatter_at_default_R': round(float(scatter_default), 4),
    }





def _estimate_cc_specular(observations, initial_cc, corneal_radius=7.8):
    """Estimate corneal center by solving the specular reflection equation.

    For each observation (cam_origin, glint_ray_dir, physical_led_pos):
    1. Find reflection point P on sphere (CC, R) along camera ray
    2. Compute surface normal at P
    3. Compute incident ray from LED to P, then reflected ray
    4. Error = angular mismatch between reflected ray and camera direction

    Optimizes CC (3 params) to minimize total angular error.

    Parameters:
        observations: list of (cam_origin, ray_dir, led_pos) tuples
            cam_origin: numpy (3,) camera position in outer frame
            ray_dir: numpy (3,) unit direction from camera toward glint
            led_pos: numpy (3,) physical LED position in outer frame
        initial_cc: numpy (3,) initial guess for corneal center
        corneal_radius: float, corneal sphere radius in mm

    Returns: numpy array CC (3,), or None if optimization fails.
    """
    from scipy.optimize import minimize as _minimize

    if len(observations) < 4:
        return None

    R = corneal_radius

    def _specular_error(cc_flat):
        cc = np.array(cc_flat)
        total_err = 0.0
        n_valid = 0
        for cam_origin, ray_dir, led_pos in observations:
            # Ray-sphere intersection: O + t*d, sphere at cc with radius R
            u = cam_origin - cc
            a_coeff = np.dot(ray_dir, ray_dir)  # should be ~1 if unit
            b_coeff = 2.0 * np.dot(ray_dir, u)
            c_coeff = np.dot(u, u) - R * R
            disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
            if disc < 0:
                total_err += 1.0  # penalty for miss
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)
            t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
            # Pick front intersection (closest to camera, positive t)
            if t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                total_err += 1.0
                continue
            # Reflection point on sphere
            P = cam_origin + t * ray_dir
            # Surface normal (outward)
            n = (P - cc) / R
            # Incident ray: from LED to P
            inc = P - led_pos
            inc_len = np.linalg.norm(inc)
            if inc_len < 1e-6:
                continue
            i_hat = inc / inc_len
            # Reflected ray: r = i - 2(i.n)n
            r_hat = i_hat - 2.0 * np.dot(i_hat, n) * n
            r_norm = np.linalg.norm(r_hat)
            if r_norm < 1e-6:
                continue
            r_hat = r_hat / r_norm
            # Error: reflected ray should point toward camera (i.e. -ray_dir)
            # cos(angle) = dot(r_hat, -ray_dir), error = 1 - cos
            cos_angle = np.dot(r_hat, -ray_dir)
            err = 1.0 - cos_angle
            total_err += err * err
            n_valid += 1
        if n_valid == 0:
            return 1e6
        return total_err / n_valid

    # Bounds: x,y within +-20mm of initial, z in [10, 80]
    bounds = [
        (initial_cc[0] - 20, initial_cc[0] + 20),
        (initial_cc[1] - 20, initial_cc[1] + 20),
        (max(10.0, initial_cc[2] - 20), min(80.0, initial_cc[2] + 20)),
    ]

    result = _minimize(_specular_error, initial_cc, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})

    if not result.success and result.fun > 0.1:
        print(f"    [PHYS-REFLECT] Optimization warning: {result.message}")

    cc_opt = np.array(result.x)
    final_err = result.fun

    # Report angular error in degrees
    avg_angular_err_deg = np.degrees(np.arccos(max(0, 1.0 - np.sqrt(final_err))))
    print(f"    [PHYS-REFLECT] CC optimized: [{cc_opt[0]:.2f}, {cc_opt[1]:.2f}, {cc_opt[2]:.2f}]")
    print(f"    [PHYS-REFLECT] Mean angular error: {avg_angular_err_deg:.2f} deg")

    return cc_opt

def _estimate_corneal_asphericity(eye_observations_map, default_R=7.8,
                                   default_Q=-0.26):
    """Jointly estimate corneal radius R and asphericity Q from glint observations.

    Uses a bootstrap approach:
    1. First pass: Compute CC with sphere (Q=0) → estimate optical axis per eye
    2. Second pass: With known optical axis, compute theta for each observation
    3. Optimize (R, Q) jointly to minimize CC scatter

    eye_observations_map: dict of 'right' -> [obs], 'left' -> [obs]
        each obs has 'glint_3d' and 'normal' (unit, outward)
    default_R: population average corneal radius (mm)
    default_Q: population average asphericity

    Returns: dict with 'R_mm', 'Q', 'confidence', 'scatter', per-eye CC,
             or None if insufficient data.
    """
    from scipy.optimize import minimize

    # Flatten all observations for joint estimation (only 'right'/'left' keys)
    all_obs = []
    for eye, obs_list in eye_observations_map.items():
        if eye not in ("right", "left"):
            continue
        for obs in obs_list:
            all_obs.append({**obs, 'eye': eye})

    if len(all_obs) < 20:
        return None

    # --- Pass 1: Compute CC with sphere (Q=0) to get optical axis ---
    eye_optical_axes = {}
    for eye, obs_list in eye_observations_map.items():
        if eye not in ("right", "left"):
            continue
        if len(obs_list) < 5:
            continue
        # CC = glint_3d - R * normal (sphere model)
        ccs = []
        for obs in obs_list:
            cc = obs['glint_3d'] - default_R * obs['normal']
            if 5 < cc[2] < 80:
                ccs.append(cc)
        if len(ccs) < 5:
            continue
        median_cc = np.median(np.array(ccs), axis=0)

        # Optical axis ≈ direction from CC to mean glint_3d (approximation)
        # Better: direction from CC outward along the corneal apex
        # For a cornea looking roughly at the camera, this is roughly -Z
        # But we use the mean normal direction as a proxy for the optical axis
        normals = np.array([obs['normal'] for obs in obs_list])
        mean_normal = np.mean(normals, axis=0)
        mn_len = np.linalg.norm(mean_normal)
        if mn_len < 1e-6:
            continue
        optical_axis = mean_normal / mn_len
        eye_optical_axes[eye] = optical_axis

    if not eye_optical_axes:
        return None

    # --- Pass 2: Joint (R, Q) optimization ---
    def cc_scatter_asph(params):
        R, Q = params
        ccs = []
        for obs in all_obs:
            eye = obs['eye']
            opt_ax = eye_optical_axes.get(eye)
            if opt_ax is None:
                continue
            normal = obs['normal']
            # theta = angle between surface normal and optical axis
            cos_theta = float(np.dot(normal, opt_ax))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            rho_s = _sagittal_radius(R, Q, theta)
            cc = obs['glint_3d'] - rho_s * normal
            if 5 < cc[2] < 80:
                ccs.append(cc)
        if len(ccs) < 10:
            return 1e6
        arr = np.array(ccs)
        return float(np.sum(np.std(arr, axis=0)))

    # Optimize with bounds: R in [6.5, 9.5], Q in [-0.6, 0.0]
    result = minimize(cc_scatter_asph, [default_R, default_Q],
                      method='L-BFGS-B',
                      bounds=[(6.5, 9.5), (-0.6, 0.0)],
                      options={'maxiter': 500, 'ftol': 1e-8})

    R_opt, Q_opt = result.x
    scatter_asph = cc_scatter_asph([R_opt, Q_opt])
    scatter_sphere = cc_scatter_asph([default_R, 0.0])

    # Confidence based on improvement over sphere model
    if scatter_sphere > 1e-6:
        improvement = (scatter_sphere - scatter_asph) / scatter_sphere
    else:
        improvement = 0.0

    if improvement < 0.01:
        confidence = "low"
    elif improvement < 0.05:
        confidence = "medium"
    else:
        confidence = "high"

    # Recompute per-eye CC with optimal (R, Q)
    eye_cc_asph = {}
    for eye, obs_list in eye_observations_map.items():
        if eye not in ("right", "left"):
            continue
        opt_ax = eye_optical_axes.get(eye)
        if opt_ax is None:
            continue
        ccs = []
        for obs in obs_list:
            normal = obs['normal']
            cos_theta = float(np.dot(normal, opt_ax))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            rho_s = _sagittal_radius(R_opt, Q_opt, theta)
            cc = obs['glint_3d'] - rho_s * normal
            if 5 < cc[2] < 80:
                ccs.append(cc)
        if ccs:
            eye_cc_asph[eye] = np.median(np.array(ccs), axis=0)

    return {
        'R_mm': round(float(R_opt), 3),
        'Q': round(float(Q_opt), 4),
        'confidence': confidence,
        'improvement_pct': round(float(improvement * 100), 1),
        'scatter_aspherical': round(float(scatter_asph), 4),
        'scatter_sphere': round(float(scatter_sphere), 4),
        'eye_cc_asph': eye_cc_asph,  # numpy arrays (not JSON-serializable)
        'optical_axes': eye_optical_axes,  # numpy arrays (not JSON-serializable)
    }


def compute_reflection_4ray_convergence(output_dir, calib, led_positions_all,
                                         crop_size=150):
    """Compute convergence using reflection-law constrained corneal centers + 4 gaze rays.

    Instead of generic sphere fitting, uses the specular reflection law with known
    LED 3D positions to estimate corneal center from each glint observation. Each
    (3D_glint, LED, camera) triple constrains the surface normal via the bisector
    of the LED-to-glint and camera-to-glint directions, giving CC = glint - R*normal.

    Then uses the estimated CC as ray origins (not camera origins) for weighted
    4-ray least-squares convergence.

    Saves to convergence_meta_reflect4ray.json.
    """
    out_base = Path(output_dir)

    # Need all three stereo pairs for full camera chain
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    cross_pair = calib.get("cross")
    if not all([right_pair, left_pair, cross_pair]):
        print("  [REFLECT-4R] Need all 3 stereo pairs (right, left, cross), skipping")
        return

    if not led_positions_all:
        print("  [REFLECT-4R] Need LED positions for all cameras, skipping")
        return

    CORNEAL_RADIUS = 7.8  # mm

    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)
    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)

    # Camera origins in RO frame (mm)
    ro_origin = np.zeros(3)
    ri_origin = (-R_right.T @ T_right).flatten()
    lo_origin = (-R_cross.T @ T_cross).flatten()
    li_in_lo = (-R_left.T @ T_left).flatten()
    li_origin = (R_cross.T @ (li_in_lo.reshape(3, 1) - T_cross)).flatten()

    # Camera rotations: gaze direction from each camera frame → RO frame
    cam_rotations = {
        'ro': np.eye(3),
        'ri': R_right.T,
        'lo': R_cross.T,
        'li': R_cross.T @ R_left.T,
    }

    # Camera weights from calibration reprojection error
    cam_weights = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        reproj = calib.get(cam, {}).get('reproj_err')
        if reproj and reproj['mean_px'] > 0:
            cam_weights[cam] = 1.0 / reproj['mean_px']
        else:
            cam_weights[cam] = 1.0
    w_mean = np.mean(list(cam_weights.values()))
    if w_mean > 0:
        cam_weights = {cam: w / w_mean for cam, w in cam_weights.items()}

    # Physical sensor params for angular gaze
    cam_physical = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        f_mm = calib.get(cam, {}).get('f_mm')
        px_mm = calib.get(cam, {}).get('px_mm')
        if f_mm and px_mm:
            cam_physical[cam] = {'f_mm': f_mm, 'px_mm': px_mm,
                                 'deg_per_px': px_mm / f_mm * 180 / np.pi}

    # ===== STEP A: Reflection-law CC estimation per eye =====
    # Collect (glint_3d, normal) observations for personal R estimation,
    # then compute CC with default R and optionally with personal R.
    eye_cc = {}  # 'right' -> CC in RO frame, 'left' -> CC in LO frame
    eye_cc_personal = {}  # same but with personal R
    eye_cc_asph = {}  # same but with aspherical model
    all_cc_observations = []  # for personal R estimation
    eye_observations_map = {}  # 'right' -> [obs], 'left' -> [obs] for asphericity
    personal_radius_info = None
    asphericity_info = None

    for eye, pair_key in [("right", "right"), ("left", "left")]:
        stereo = calib.get(pair_key)
        cam1 = stereo['cam1']  # outer: ro or lo
        cam2 = stereo['cam2']  # inner: ri or li
        R_s = stereo['R']
        T_s = stereo['T']
        K1, dist1 = calib[cam1]['K'], calib[cam1]['dist']
        K2, dist2 = calib[cam2]['K'], calib[cam2]['dist']

        # LED positions in each camera's frame
        leds_outer = led_positions_all.get(cam1)
        leds_inner = led_positions_all.get(cam2)
        if not leds_outer or not leds_inner or len(leds_outer) < 4:
            print(f"  [REFLECT-4R] No LED positions for {eye} eye "
                  f"({cam1}/{cam2}), skipping")
            continue

        # Inner camera origin in outer camera frame
        inner_cam_origin = (-np.array(R_s).T @ np.array(T_s).reshape(3, 1)).flatten()

        # Load per-camera results
        r1_path = out_base / cam1 / "results.json"
        r2_path = out_base / cam2 / "results.json"
        if not r1_path.exists() or not r2_path.exists():
            print(f"  [REFLECT-4R] Missing results for {cam1} or {cam2}, "
                  f"skipping {eye}")
            continue
        with open(r1_path) as f:
            res1 = json.load(f)
        with open(r2_path) as f:
            res2 = json.load(f)

        n_frames = min(len(res1), len(res2))
        cc_candidates = []  # [(frame_name, cc_array), ...]
        eye_observations = []  # (glint_3d, normal) pairs for this eye

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")

            # Get glints from both cameras
            glints1 = (r1.get("seg_enh_glints") or r1.get("seg_glints")
                       or r1.get("glints", []))
            glints2 = (r2.get("seg_enh_glints") or r2.get("seg_glints")
                       or r2.get("glints", []))

            if (len(glints1) < 3 or len(glints2) < 3
                    or r1.get("eye_closed") or r2.get("eye_closed")):
                continue

            g1_2d = [(g["x_orig"], g["y_orig"]) for g in glints1]
            g2_2d = [(g["x_orig"], g["y_orig"]) for g in glints2]

            # Match glints to LEDs independently in each camera
            matched_g1, matched_l1 = match_glints_to_leds(
                g1_2d, leds_outer, K1, dist1)
            matched_g2, matched_l2 = match_glints_to_leds(
                g2_2d, leds_inner, K2, dist2)

            if matched_g1 is None or matched_g2 is None:
                continue

            # Find LED index for each matched pair by comparing to original LED list
            led_idx_cam1 = {}  # led_index -> glint_2d (pixel)
            for j in range(len(matched_l1)):
                for led_i in range(len(leds_outer)):
                    if np.allclose(matched_l1[j], leds_outer[led_i], atol=0.1):
                        led_idx_cam1[led_i] = matched_g1[j]
                        break

            led_idx_cam2 = {}
            for j in range(len(matched_l2)):
                for led_i in range(len(leds_inner)):
                    if np.allclose(matched_l2[j], leds_inner[led_i], atol=0.1):
                        led_idx_cam2[led_i] = matched_g2[j]
                        break

            # Stereo pairs: same LED index seen in both cameras
            common_leds = set(led_idx_cam1.keys()) & set(led_idx_cam2.keys())
            if len(common_leds) < 2:
                continue

            for led_i in common_leds:
                g1_px = led_idx_cam1[led_i]
                g2_px = led_idx_cam2[led_i]

                pt1_n = _undistort_single(g1_px[0], g1_px[1], K1, dist1)
                pt2_n = _undistort_single(g2_px[0], g2_px[1], K2, dist2)

                # Triangulate → 3D in outer camera frame
                glint_3d = _triangulate_point(pt1_n, pt2_n, R_s, T_s)
                if glint_3d is None or glint_3d[2] <= 0 or glint_3d[2] > 80:
                    continue

                led_pos = np.array(leds_outer[led_i], dtype=np.float64)

                # --- Reflection law from outer camera (origin = [0,0,0]) ---
                to_led = led_pos - glint_3d
                to_led_len = np.linalg.norm(to_led)
                if to_led_len < 1e-6:
                    continue
                to_led = to_led / to_led_len

                to_cam = -glint_3d  # camera at origin
                to_cam_len = np.linalg.norm(to_cam)
                if to_cam_len < 1e-6:
                    continue
                to_cam = to_cam / to_cam_len

                normal = to_led + to_cam
                normal_len = np.linalg.norm(normal)
                if normal_len < 1e-6:
                    continue
                normal = normal / normal_len

                cc = glint_3d - CORNEAL_RADIUS * normal
                if 5 < cc[2] < 80:
                    cc_candidates.append((frame_name, cc))
                # Store observation for personal R estimation
                eye_observations.append({
                    'glint_3d': glint_3d.copy(),
                    'normal': normal.copy()
                })

                # --- Reflection law from inner camera ---
                to_cam_inner = inner_cam_origin - glint_3d
                to_cam_inner_len = np.linalg.norm(to_cam_inner)
                if to_cam_inner_len < 1e-6:
                    continue
                to_cam_inner = to_cam_inner / to_cam_inner_len

                normal_inner = to_led + to_cam_inner
                normal_inner_len = np.linalg.norm(normal_inner)
                if normal_inner_len < 1e-6:
                    continue
                normal_inner = normal_inner / normal_inner_len

                cc_inner = glint_3d - CORNEAL_RADIUS * normal_inner
                if 5 < cc_inner[2] < 80:
                    cc_candidates.append((frame_name, cc_inner))
                eye_observations.append({
                    'glint_3d': glint_3d.copy(),
                    'normal': normal_inner.copy()
                })

        all_cc_observations.extend(eye_observations)
        eye_observations_map[eye] = eye_observations

        if len(cc_candidates) < 3:
            print(f"  [REFLECT-4R] {eye}: only {len(cc_candidates)} CC "
                  f"candidates, need >= 3")
            continue

        cc_arr = np.array([cc for _, cc in cc_candidates])
        median_cc = np.median(cc_arr, axis=0)
        cc_std = np.std(cc_arr, axis=0)
        eye_cc[eye] = median_cc

        # Save per-frame CC observations for fair calibration
        eye_observations_map[f"cc_observations_{eye}"] = [
            {"frame": fn, "cc": [round(float(cc[k]), 4) for k in range(3)]}
            for fn, cc in cc_candidates
        ]

        frame_label = "RO" if eye == "right" else "LO"
        print(f"  [REFLECT-4R] {eye} CC ({frame_label} frame) from "
              f"{len(cc_candidates)} observations:")
        print(f"    Median: [{median_cc[0]:.2f}, {median_cc[1]:.2f}, "
              f"{median_cc[2]:.2f}] mm")
        print(f"    Std:    [{cc_std[0]:.2f}, {cc_std[1]:.2f}, "
              f"{cc_std[2]:.2f}] mm")

    if not eye_cc:
        print("  [REFLECT-4R] No corneal centers estimated, skipping")
        return

    # ===== Personal corneal radius estimation =====
    personal_radius_info = _estimate_personal_corneal_radius(
        all_cc_observations, default_R=CORNEAL_RADIUS)
    if personal_radius_info:
        pr = personal_radius_info['personal_R_mm']
        print(f"  [PERSONAL-R] Estimated corneal radius: {pr:.2f} mm "
              f"(default: {CORNEAL_RADIUS} mm)")
        print(f"  [PERSONAL-R] Confidence: {personal_radius_info['confidence']} "
              f"(improvement: {personal_radius_info['improvement_pct']:.1f}%)")

        # Recompute CC with personal R for each eye
        for eye, pair_key in [("right", "right"), ("left", "left")]:
            # Re-derive from stored observations using personal R
            # We stored all_cc_observations but need per-eye. Use eye_cc existence
            # as check that this eye had enough data.
            if eye not in eye_cc:
                continue
            stereo = calib.get(pair_key)
            cam1 = stereo['cam1']
            cam2 = stereo['cam2']

            # Recompute CC candidates with personal R from ALL observations for this eye
            # We need to re-iterate but we can filter from all_cc_observations
            # Simpler: just recompute CC = glint_3d - personalR * normal for all obs
            # But we stored observations from both eyes mixed. Let's recompute per-eye.
            pass

        # Simpler approach: recompute CC from the stored observations
        # We need per-eye separation. Let's restructure to collect per-eye.
        # Since we already have all_cc_observations from both eyes combined,
        # recompute per-eye CC using personal R from the same per-eye loop data.
        # Actually, the simplest approach is to just recompute CC = glint - R*normal
        # for all observations and take the median, which IS per-eye since each eye's
        # loop collected its own observations. Let me fix this by collecting per-eye.
        pass
    else:
        print("  [PERSONAL-R] Not enough observations for estimation")

    # Recompute per-eye CC with personal R (using all observations)
    # We need to redo the CC computation with personal R per eye.
    # The cleanest way: use the observation normals we already collected.
    # Store per-eye observations:
    # Preserve cc_observations_* entries, reset only per-eye observation lists
    cc_obs_backup = {k: v for k, v in eye_observations_map.items()
                     if k.startswith("cc_observations_")}
    eye_observations_map = {}  # 'right' -> [obs], 'left' -> [obs]
    eye_observations_map.update(cc_obs_backup)
    # We'll track which observations belong to which eye by index
    # Actually, we already lost the per-eye tracking. Let me refactor slightly:
    # We need to re-derive personal CC. Since we have eye_cc (default R), and
    # know the observations, we can just recompute from the stored normals.
    # But we mixed them. The fix: recompute CC with personal R from the full
    # set — this gives an "average" CC, not per-eye. That's fine for the
    # overall estimation, but for per-eye CC we need per-eye observations.
    #
    # Solution: Re-run the per-eye CC computation with personal R.
    if personal_radius_info:
        pr = personal_radius_info['personal_R_mm']
        for eye, pair_key in [("right", "right"), ("left", "left")]:
            if eye not in eye_cc:
                continue
            stereo = calib.get(pair_key)
            cam1 = stereo['cam1']
            cam2 = stereo['cam2']
            R_s = stereo['R']
            T_s = stereo['T']
            K1, dist1 = calib[cam1]['K'], calib[cam1]['dist']
            K2, dist2 = calib[cam2]['K'], calib[cam2]['dist']
            leds_outer = led_positions_all.get(cam1)
            leds_inner = led_positions_all.get(cam2)
            if not leds_outer or not leds_inner:
                continue
            inner_cam_origin = (-np.array(R_s).T @ np.array(T_s).reshape(3, 1)).flatten()

            r1_path = out_base / cam1 / "results.json"
            r2_path = out_base / cam2 / "results.json"
            with open(r1_path) as f:
                res1 = json.load(f)
            with open(r2_path) as f:
                res2 = json.load(f)

            n_frames = min(len(res1), len(res2))
            cc_candidates_pr = []
            for i in range(n_frames):
                r1, r2 = res1[i], res2[i]
                glints1 = (r1.get("seg_enh_glints") or r1.get("seg_glints")
                           or r1.get("glints", []))
                glints2 = (r2.get("seg_enh_glints") or r2.get("seg_glints")
                           or r2.get("glints", []))
                if (len(glints1) < 3 or len(glints2) < 3
                        or r1.get("eye_closed") or r2.get("eye_closed")):
                    continue
                g1_2d = [(g["x_orig"], g["y_orig"]) for g in glints1]
                g2_2d = [(g["x_orig"], g["y_orig"]) for g in glints2]
                matched_g1, matched_l1 = match_glints_to_leds(g1_2d, leds_outer, K1, dist1)
                matched_g2, matched_l2 = match_glints_to_leds(g2_2d, leds_inner, K2, dist2)
                if matched_g1 is None or matched_g2 is None:
                    continue
                led_idx_cam1 = {}
                for j in range(len(matched_l1)):
                    for led_i in range(len(leds_outer)):
                        if np.allclose(matched_l1[j], leds_outer[led_i], atol=0.1):
                            led_idx_cam1[led_i] = matched_g1[j]
                            break
                led_idx_cam2 = {}
                for j in range(len(matched_l2)):
                    for led_i in range(len(leds_inner)):
                        if np.allclose(matched_l2[j], leds_inner[led_i], atol=0.1):
                            led_idx_cam2[led_i] = matched_g2[j]
                            break
                common_leds = set(led_idx_cam1.keys()) & set(led_idx_cam2.keys())
                if len(common_leds) < 2:
                    continue
                for led_i in common_leds:
                    g1_px = led_idx_cam1[led_i]
                    g2_px = led_idx_cam2[led_i]
                    pt1_n = _undistort_single(g1_px[0], g1_px[1], K1, dist1)
                    pt2_n = _undistort_single(g2_px[0], g2_px[1], K2, dist2)
                    glint_3d = _triangulate_point(pt1_n, pt2_n, R_s, T_s)
                    if glint_3d is None or glint_3d[2] <= 0 or glint_3d[2] > 80:
                        continue
                    led_pos = np.array(leds_outer[led_i], dtype=np.float64)
                    # Outer camera normal
                    to_led = led_pos - glint_3d
                    tl = np.linalg.norm(to_led)
                    if tl < 1e-6: continue
                    to_led = to_led / tl
                    to_cam = -glint_3d
                    tc_len = np.linalg.norm(to_cam)
                    if tc_len < 1e-6: continue
                    to_cam = to_cam / tc_len
                    normal = to_led + to_cam
                    nl = np.linalg.norm(normal)
                    if nl < 1e-6: continue
                    normal = normal / nl
                    cc_pr = glint_3d - pr * normal
                    if 5 < cc_pr[2] < 80:
                        cc_candidates_pr.append(cc_pr)
                    # Inner camera normal
                    to_ci = inner_cam_origin - glint_3d
                    tcl = np.linalg.norm(to_ci)
                    if tcl < 1e-6: continue
                    to_ci = to_ci / tcl
                    ni = to_led + to_ci
                    nil = np.linalg.norm(ni)
                    if nil < 1e-6: continue
                    ni = ni / nil
                    cc_pr_i = glint_3d - pr * ni
                    if 5 < cc_pr_i[2] < 80:
                        cc_candidates_pr.append(cc_pr_i)

            if len(cc_candidates_pr) >= 3:
                cc_arr_pr = np.array(cc_candidates_pr)
                median_cc_pr = np.median(cc_arr_pr, axis=0)
                eye_cc_personal[eye] = median_cc_pr
                frame_label = "RO" if eye == "right" else "LO"
                print(f"  [PERSONAL-R] {eye} CC (R={pr}mm, {frame_label} frame) "
                      f"from {len(cc_candidates_pr)} obs: "
                      f"[{median_cc_pr[0]:.2f}, {median_cc_pr[1]:.2f}, "
                      f"{median_cc_pr[2]:.2f}]")

    # ===== Asphericity (R, Q) estimation =====
    if eye_observations_map:
        asphericity_info = _estimate_corneal_asphericity(
            eye_observations_map, default_R=CORNEAL_RADIUS)
        if asphericity_info:
            R_a = asphericity_info['R_mm']
            Q_a = asphericity_info['Q']
            print(f"  [ASPH] Estimated R={R_a:.3f}mm, Q={Q_a:.4f} "
                  f"(confidence: {asphericity_info['confidence']}, "
                  f"improvement: {asphericity_info['improvement_pct']:.1f}%)")
            print(f"  [ASPH] Scatter: aspherical={asphericity_info['scatter_aspherical']:.4f} "
                  f"vs sphere={asphericity_info['scatter_sphere']:.4f}")
            if asphericity_info.get('eye_cc_asph'):
                for eye, cc in asphericity_info['eye_cc_asph'].items():
                    eye_cc_asph[eye] = cc
                    frame_label = "RO" if eye == "right" else "LO"
                    print(f"  [ASPH] {eye} CC ({frame_label} frame): "
                          f"[{cc[0]:.2f}, {cc[1]:.2f}, {cc[2]:.2f}]")
        else:
            print("  [ASPH] Not enough observations for asphericity estimation")

    # ===== STEP B: 4-Ray convergence with CC origins =====

    # Transform CCs to RO frame
    right_cc_ro = eye_cc.get("right")  # already in RO frame
    left_cc_lo = eye_cc.get("left")    # in LO frame
    left_cc_ro = None
    if left_cc_lo is not None:
        left_cc_ro = (R_cross.T @ (left_cc_lo.reshape(3, 1) - T_cross)).flatten()

    # Ray origin for each camera = CC of corresponding eye in RO frame
    ray_origins = {}
    if right_cc_ro is not None:
        ray_origins['ro'] = right_cc_ro
        ray_origins['ri'] = right_cc_ro
    if left_cc_ro is not None:
        ray_origins['lo'] = left_cc_ro
        ray_origins['li'] = left_cc_ro

    if len(ray_origins) < 2:
        print("  [REFLECT-4R] Need CC for at least 2 cameras, skipping")
        return

    print(f"  [REFLECT-4R] Ray origins (CC in RO frame):")
    for cam, orig in ray_origins.items():
        print(f"    {cam.upper()}: [{orig[0]:.2f}, {orig[1]:.2f}, "
              f"{orig[2]:.2f}]")

    # Load per-camera results
    cam_results = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        rpath = out_base / cam / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                cam_results[cam] = json.load(f)

    if len(cam_results) < 2:
        print(f"  [REFLECT-4R] Need at least 2 cameras with results")
        return

    # Load seg combined for IPD (pupil_3d from stereo triangulation)
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame, l_seg_by_frame = {}, {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    n_frames = min(len(v) for v in cam_results.values())
    I3 = np.eye(3)
    convergence_results = []

    for i in range(n_frames):
        frame_name = None
        rays = []

        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            if cam not in ray_origins:
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")

            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue

            d_cam = np.array([gaze_norm[0], gaze_norm[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)

            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)

            if d_ro[2] < 0.3:
                continue

            rays.append((ray_origins[cam], d_ro, cam,
                         cam_weights.get(cam, 1.0)))

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "n_cameras": len(rays), "per_camera_residual": None,
                 "gaze_angles_deg": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        # Per-camera gaze angles in true degrees
        gaze_angles = {}
        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None:
                continue
            phys = cam_physical.get(cam)
            if phys:
                h_deg = float(np.degrees(np.arctan(gaze_norm[0])))
                v_deg = float(np.degrees(np.arctan(gaze_norm[1])))
                gaze_angles[cam] = {'h_deg': round(h_deg, 3),
                                    'v_deg': round(v_deg, 3)}
        if gaze_angles:
            entry["gaze_angles_deg"] = gaze_angles

        if len(rays) >= 2:
            # Weighted N-ray least-squares: (Σ w_i M_i) P = Σ w_i M_i O_i
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for origin, direction, _, weight in rays:
                d = direction.reshape(3, 1)
                M = I3 - d @ d.T
                A += weight * M
                b += weight * (M @ origin)

            try:
                P = np.linalg.solve(A, b)

                if P[2] > 0:
                    residuals = {}
                    for origin, direction, cam_name, _ in rays:
                        diff = P - origin
                        proj_len = np.dot(diff, direction)
                        if proj_len < 0:
                            continue
                        perp = diff - proj_len * direction
                        residuals[cam_name] = float(np.linalg.norm(perp))

                    if residuals:
                        rms_residual = float(np.sqrt(
                            np.mean([r**2 for r in residuals.values()])))

                        ray_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                        fixation_dist = float(np.linalg.norm(P - ray_mid))

                        entry["fixation_distance_mm"] = round(fixation_dist, 2)
                        entry["convergence_mm"] = round(fixation_dist, 2)
                        entry["convergence_point"] = [
                            round(float(P[k]), 2) for k in range(3)]
                        entry["ray_miss_mm"] = round(rms_residual, 2)
                        entry["per_camera_residual"] = {
                            k: round(v, 2) for k, v in residuals.items()}
            except np.linalg.LinAlgError:
                pass

        # IPD from seg combined pupil_3d
        if frame_name:
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if r_seg and l_seg:
                r_pupil = r_seg.get("pupil_3d")
                l_pupil = l_seg.get("pupil_3d")
                if r_pupil and l_pupil:
                    rp = np.array(r_pupil)
                    lp_lo = np.array(l_pupil)
                    lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                    ipd = float(np.linalg.norm(rp - lp_ro))
                    entry["ipd_mm"] = round(ipd, 2)
                    entry["right_pupil_3d"] = [
                        round(float(rp[k]), 4) for k in range(3)]
                    entry["left_pupil_3d_ro"] = [
                        round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # --- Statistics ---
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    n_cam_vals = [e["n_cameras"]
                  for e in convergence_results if e["fixation_distance_mm"]]

    if fix_vals:
        print(f"  [REFLECT-4R] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [REFLECT-4R] residual (RMS): "
                  f"median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
        if n_cam_vals:
            print(f"  [REFLECT-4R] cameras/frame: "
                  f"mean={np.mean(n_cam_vals):.1f} "
                  f"min={min(n_cam_vals)} max={max(n_cam_vals)}")
        cam_resids = {cam: [] for cam in ['ro', 'ri', 'lo', 'li']}
        for e in convergence_results:
            pcr = e.get("per_camera_residual")
            if pcr:
                for cam, val in pcr.items():
                    cam_resids[cam].append(val)
        for cam in ['ro', 'ri', 'lo', 'li']:
            vals = cam_resids[cam]
            if vals:
                print(f"  [REFLECT-4R]   {cam.upper()} avg residual: "
                      f"{np.mean(vals):.2f}mm ({len(vals)} frames)")
    if ipd_vals:
        print(f"  [REFLECT-4R] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "reflection_4ray_weighted",
        "description": "Reflection-law constrained corneal center + 4-camera "
                       "weighted gaze rays",
        "cc_estimation_method": "reflection_law",
        "corneal_radius_mm": CORNEAL_RADIUS,
        "personal_radius": personal_radius_info,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": [round(float(v), 4) for v in eye_cc["right"]]
            if "right" in eye_cc else None,
        "left_corneal_center_lo": [round(float(v), 4) for v in eye_cc["left"]]
            if "left" in eye_cc else None,
        "right_corneal_center_ro_personalR": [round(float(v), 4) for v in eye_cc_personal["right"]]
            if "right" in eye_cc_personal else None,
        "left_corneal_center_lo_personalR": [round(float(v), 4) for v in eye_cc_personal["left"]]
            if "left" in eye_cc_personal else None,
        "asphericity": {k: v for k, v in asphericity_info.items()
                    if k not in ('eye_cc_asph', 'optical_axes')} if asphericity_info else None,
        "right_corneal_center_ro_asph": [round(float(v), 4) for v in eye_cc_asph["right"]]
            if "right" in eye_cc_asph else None,
        "left_corneal_center_lo_asph": [round(float(v), 4) for v in eye_cc_asph["left"]]
            if "left" in eye_cc_asph else None,
        "right_R_asph": asphericity_info['R_mm'] if asphericity_info else None,
        "right_Q_asph": asphericity_info['Q'] if asphericity_info else None,
        "left_R_asph": asphericity_info['R_mm'] if asphericity_info else None,
        "left_Q_asph": asphericity_info['Q'] if asphericity_info else None,
        "cc_observations_right": eye_observations_map.get("cc_observations_right"),
        "cc_observations_left": eye_observations_map.get("cc_observations_left"),
        "camera_weights": {cam: round(w, 4) for cam, w in cam_weights.items()},
        "camera_physical": {cam: {k: round(v, 4) for k, v in phys.items()}
                            for cam, phys in cam_physical.items()},
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "n_cameras": e.get("n_cameras"),
            "per_camera_residual": e.get("per_camera_residual"),
            "gaze_angles_deg": e.get("gaze_angles_deg"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_reflect4ray.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [REFLECT-4R] Saved to {conv_path}")

    # --- Huber variant for Reflect 4-Ray ---
    huber_r4r_results = []
    for i in range(n_frames):
        frame_name = None
        rays = []
        for cam in ['ro', 'ri', 'lo', 'li']:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            if cam not in ray_origins:
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")
            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue
            d_cam = np.array([gaze_norm[0], gaze_norm[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)
            if d_ro[2] < 0.3:
                continue
            rays.append((ray_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))

        h_entry = {"frame": frame_name, "convergence_mm": None,
                   "fixation_distance_mm": None, "convergence_point": None,
                   "ray_miss_mm": None, "ipd_mm": None,
                   "n_cameras": len(rays), "per_camera_residual": None,
                   "huber_weights": None}
        if len(rays) >= 2:
            h_sol = _solve_nray_fixation(rays, huber_delta=5.0)
            if h_sol:
                h_entry["fixation_distance_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_point"] = [round(float(h_sol['P'][k]), 2) for k in range(3)]
                h_entry["ray_miss_mm"] = round(h_sol['rms_residual'], 2)
                h_entry["per_camera_residual"] = {k: round(v, 2) for k, v in h_sol['residuals'].items()}
                h_entry["huber_weights"] = {k: round(v, 4) for k, v in h_sol.get('huber_weights', {}).items()}
        # Copy IPD from standard results
        std_entry = convergence_results[i]
        h_entry["ipd_mm"] = std_entry.get("ipd_mm")
        huber_r4r_results.append(h_entry)

    h_fix = [e["fixation_distance_mm"] for e in huber_r4r_results if e["fixation_distance_mm"]]
    if h_fix:
        print(f"  [REFLECT-4R HUBER] {len(h_fix)} frames | "
              f"fixation: median={np.median(h_fix)/10:.1f}cm "
              f"std={np.std(h_fix)/10:.1f}cm")
    h_meta = dict(conv_meta)
    h_meta["method"] = "huber_reflect4ray_weighted"
    h_meta["huber_delta_mm"] = 5.0
    h_meta["median_fixation_mm"] = round(float(np.median(h_fix)), 2) if h_fix else None
    h_meta["per_frame"] = [{
        "frame": e["frame"], "convergence_mm": e["convergence_mm"],
        "fixation_distance_mm": e["fixation_distance_mm"],
        "convergence_point": e.get("convergence_point"),
        "ray_miss_mm": e.get("ray_miss_mm"), "n_cameras": e.get("n_cameras"),
        "per_camera_residual": e.get("per_camera_residual"),
        "huber_weights": e.get("huber_weights"),
        "ipd_mm": e.get("ipd_mm"),
    } for e in huber_r4r_results if e["fixation_distance_mm"]]
    h_path = out_base / "convergence_meta_huber_reflect4ray.json"
    with open(str(h_path), "w") as f:
        json.dump(h_meta, f, indent=2)
    print(f"  [REFLECT-4R HUBER] Saved to {h_path}")

    # ===== Also compute 2-ray 3D model convergence using reflection-law CC =====
    # Gaze = normalize(pupil_3d - CC) per eye, then 2-ray intersection.
    # This uses the CC from the reflection law with stereo pupil triangulation,
    # which is more physically accurate than PCCR + CC-origin rays.
    _compute_reflect_c3d_convergence(out_base, calib, eye_cc,
                                     R_cross, T_cross,
                                     eye_cc_personal=eye_cc_personal,
                                     personal_radius_info=personal_radius_info,
                                     eye_cc_asph=eye_cc_asph,
                                     asphericity_info=asphericity_info,
                                     cc_observations_map=eye_observations_map)


def _compute_reflect_c3d_convergence(out_base, calib, eye_cc,
                                      R_cross, T_cross,
                                      eye_cc_personal=None,
                                      personal_radius_info=None,
                                      eye_cc_asph=None,
                                      asphericity_info=None,
                                      cc_observations_map=None):
    """2-ray convergence using reflection-law CC + 3D corneal model gaze.

    For each eye: gaze = normalize(pupil_3d - CC) projected onto the outer
    camera's normalized plane. Then intersect right and left gaze rays.
    Same approach as compute_corneal_3d_convergence() but with reflection-law CC
    instead of sphere-fit CC.

    Also computes:
    - Refraction-corrected variants (pupil_3d corrected for corneal refraction)
    - Personal R variants (using personal corneal radius CC)
    - Aspherical variants (using asphericity-corrected CC)

    Saves to convergence_meta_reflectc3d.json.
    """
    cross = calib.get("cross")
    if cross is None:
        print("  [REFLECT-C3D] No cross-pair calibration, skipping")
        return

    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    right_cc = eye_cc.get("right")  # in RO frame
    left_cc = eye_cc.get("left")    # in LO frame
    if right_cc is None or left_cc is None:
        print("  [REFLECT-C3D] Need CC for both eyes, skipping")
        return

    print(f"  [REFLECT-C3D] Right CC (RO): [{right_cc[0]:.2f}, "
          f"{right_cc[1]:.2f}, {right_cc[2]:.2f}]")
    print(f"  [REFLECT-C3D] Left CC (LO):  [{left_cc[0]:.2f}, "
          f"{left_cc[1]:.2f}, {left_cc[2]:.2f}]")

    # Personal R CC
    if eye_cc_personal is None:
        eye_cc_personal = {}
    right_cc_pr = eye_cc_personal.get("right")
    left_cc_pr = eye_cc_personal.get("left")
    has_personal = right_cc_pr is not None and left_cc_pr is not None

    # Aspherical CC
    if eye_cc_asph is None:
        eye_cc_asph = {}
    right_cc_asph = eye_cc_asph.get("right")
    left_cc_asph = eye_cc_asph.get("left")
    has_asph = right_cc_asph is not None and left_cc_asph is not None
    if has_asph:
        print(f"  [REFLECT-C3D] Aspherical CC available for both eyes")

    # Load seg combined results (need pupil_3d for both eyes)
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [REFLECT-C3D] Need seg combined results for both eyes, skipping")
        return

    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    # Load per-camera results for refraction correction (need 2D pupil pixel coords)
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    r_cam1_results, r_cam2_results = {}, {}
    l_cam1_results, l_cam2_results = {}, {}
    if right_pair:
        cam1, cam2 = right_pair['cam1'], right_pair['cam2']
        rp1 = out_base / cam1 / "results.json"
        rp2 = out_base / cam2 / "results.json"
        if rp1.exists():
            with open(rp1) as f:
                for e in json.load(f):
                    r_cam1_results[e.get("frame", "")] = e
        if rp2.exists():
            with open(rp2) as f:
                for e in json.load(f):
                    r_cam2_results[e.get("frame", "")] = e
    if left_pair:
        cam1, cam2 = left_pair['cam1'], left_pair['cam2']
        lp1 = out_base / cam1 / "results.json"
        lp2 = out_base / cam2 / "results.json"
        if lp1.exists():
            with open(lp1) as f:
                for e in json.load(f):
                    l_cam1_results[e.get("frame", "")] = e
        if lp2.exists():
            with open(lp2) as f:
                for e in json.load(f):
                    l_cam2_results[e.get("frame", "")] = e

    def _converge(r_pupil_3d, l_pupil_3d_lo, r_cc, l_cc):
        """Compute convergence point from pupil_3d and CC for both eyes."""
        r_gaze_3d = r_pupil_3d - r_cc
        if np.linalg.norm(r_gaze_3d) < 1e-6:
            return None
        r_pupil_proj = np.array([r_pupil_3d[0]/r_pupil_3d[2], r_pupil_3d[1]/r_pupil_3d[2]])
        r_cc_proj = np.array([r_cc[0]/r_cc[2], r_cc[1]/r_cc[2]])
        r_gaze_norm = r_pupil_proj - r_cc_proj

        l_gaze_3d = l_pupil_3d_lo - l_cc
        if np.linalg.norm(l_gaze_3d) < 1e-6:
            return None
        l_pupil_proj = np.array([l_pupil_3d_lo[0]/l_pupil_3d_lo[2], l_pupil_3d_lo[1]/l_pupil_3d_lo[2]])
        l_cc_proj = np.array([l_cc[0]/l_cc[2], l_cc[1]/l_cc[2]])
        l_gaze_norm = l_pupil_proj - l_cc_proj

        r_dir = np.array([r_gaze_norm[0], r_gaze_norm[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze_norm[0], l_gaze_norm[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d = float(np.dot(r_dir, w0))
        e = float(np.dot(l_dir_ro, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return None
        sc_val = (b * e - c * d) / denom
        tc_val = (a * e - b * d) / denom
        if sc_val <= 0 or tc_val <= 0:
            return None
        closest_r = sc_val * r_dir
        closest_l = lo_origin_ro + tc_val * l_dir_ro
        conv_pt = (closest_r + closest_l) / 2.0
        ray_miss = float(np.linalg.norm(closest_r - closest_l))
        cam_mid = lo_origin_ro / 2.0
        fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
        return {
            'fixation_distance_mm': round(fix_dist, 2),
            'convergence_mm': round(fix_dist, 2),
            'convergence_point': [round(float(conv_pt[k]), 2) for k in range(3)],
            'ray_miss_mm': round(ray_miss, 2),
        }

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    convergence_results = []
    corneal_radius_default = 7.8
    corneal_radius_personal = personal_radius_info['personal_R_mm'] if personal_radius_info else None

    for frame_name in common_frames:
        r_entry = r_seg[frame_name]
        l_entry = l_seg[frame_name]

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        r_pupil = r_entry.get("pupil_3d")
        l_pupil = l_entry.get("pupil_3d")
        if not r_pupil or not l_pupil:
            convergence_results.append(entry)
            continue

        rp = np.array(r_pupil)
        lp_lo = np.array(l_pupil)

        # Standard convergence (default CC)
        conv = _converge(rp, lp_lo, right_cc, left_cc)
        if conv:
            entry.update(conv)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        # --- Refraction correction ---
        # Try to correct right eye pupil_3d
        rp_refracted = None
        if right_pair:
            r1_res = r_cam1_results.get(frame_name)
            r2_res = r_cam2_results.get(frame_name)
            if r1_res and r2_res:
                pc1 = r1_res.get("seg_pupil_center") or r1_res.get("pupil_center")
                pc2 = r2_res.get("seg_pupil_center") or r2_res.get("pupil_center")
                if pc1 and pc2:
                    K1 = calib[right_pair['cam1']]['K']
                    d1 = calib[right_pair['cam1']]['dist']
                    K2 = calib[right_pair['cam2']]['K']
                    d2 = calib[right_pair['cam2']]['dist']
                    R_s = np.array(right_pair['R'])
                    T_s = np.array(right_pair['T']).reshape(3, 1)
                    rp_refracted = _correct_pupil_refraction(
                        pc1, pc2, K1, d1, K2, d2, R_s, T_s,
                        right_cc, corneal_radius=corneal_radius_default)

        # Try to correct left eye pupil_3d
        lp_refracted = None
        if left_pair:
            l1_res = l_cam1_results.get(frame_name)
            l2_res = l_cam2_results.get(frame_name)
            if l1_res and l2_res:
                pc1 = l1_res.get("seg_pupil_center") or l1_res.get("pupil_center")
                pc2 = l2_res.get("seg_pupil_center") or l2_res.get("pupil_center")
                if pc1 and pc2:
                    K1 = calib[left_pair['cam1']]['K']
                    d1 = calib[left_pair['cam1']]['dist']
                    K2 = calib[left_pair['cam2']]['K']
                    d2 = calib[left_pair['cam2']]['dist']
                    R_s = np.array(left_pair['R'])
                    T_s = np.array(left_pair['T']).reshape(3, 1)
                    lp_refracted = _correct_pupil_refraction(
                        pc1, pc2, K1, d1, K2, d2, R_s, T_s,
                        left_cc, corneal_radius=corneal_radius_default)

        # Refracted convergence
        if rp_refracted is not None and lp_refracted is not None:
            conv_ref = _converge(rp_refracted, lp_refracted, right_cc, left_cc)
            if conv_ref:
                entry["fixation_distance_mm_refracted"] = conv_ref['fixation_distance_mm']
                entry["convergence_mm_refracted"] = conv_ref['convergence_mm']
                entry["convergence_point_refracted"] = conv_ref['convergence_point']
                entry["ray_miss_mm_refracted"] = conv_ref['ray_miss_mm']
            # Store refracted pupil_3d per eye for use by calibration
            entry["right_pupil_3d_refracted"] = [round(float(rp_refracted[k]), 4) for k in range(3)]
            entry["left_pupil_3d_lo_refracted"] = [round(float(lp_refracted[k]), 4) for k in range(3)]

        # Personal R convergence
        if has_personal:
            conv_pr = _converge(rp, lp_lo, right_cc_pr, left_cc_pr)
            if conv_pr:
                entry["fixation_distance_mm_personalR"] = conv_pr['fixation_distance_mm']
                entry["convergence_mm_personalR"] = conv_pr['convergence_mm']
                entry["convergence_point_personalR"] = conv_pr['convergence_point']
                entry["ray_miss_mm_personalR"] = conv_pr['ray_miss_mm']

            # Refracted + personal R
            if rp_refracted is not None and lp_refracted is not None:
                conv_both = _converge(rp_refracted, lp_refracted,
                                      right_cc_pr, left_cc_pr)
                if conv_both:
                    entry["fixation_distance_mm_refracted_personalR"] = conv_both['fixation_distance_mm']
                    entry["convergence_mm_refracted_personalR"] = conv_both['convergence_mm']
                    entry["convergence_point_refracted_personalR"] = conv_both['convergence_point']
                    entry["ray_miss_mm_refracted_personalR"] = conv_both['ray_miss_mm']

        # Aspherical convergence
        if has_asph:
            conv_asph = _converge(rp, lp_lo, right_cc_asph, left_cc_asph)
            if conv_asph:
                entry["fixation_distance_mm_aspherical"] = conv_asph['fixation_distance_mm']
                entry["convergence_mm_aspherical"] = conv_asph['convergence_mm']
                entry["convergence_point_aspherical"] = conv_asph['convergence_point']
                entry["ray_miss_mm_aspherical"] = conv_asph['ray_miss_mm']

        convergence_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    if fix_vals:
        print(f"  [REFLECT-C3D] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [REFLECT-C3D] ray miss: median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")

    # Refracted stats
    fix_vals_ref = [e["fixation_distance_mm_refracted"]
                    for e in convergence_results if e.get("fixation_distance_mm_refracted")]
    if fix_vals_ref:
        miss_ref = [e["ray_miss_mm_refracted"]
                    for e in convergence_results if e.get("ray_miss_mm_refracted") is not None]
        print(f"  [REFLECT-C3D REFR] {len(fix_vals_ref)} frames | "
              f"fixation: median={np.median(fix_vals_ref)/10:.1f}cm "
              f"ray miss: {np.median(miss_ref):.2f}mm" if miss_ref else "")

    # Personal R stats
    fix_vals_pr = [e["fixation_distance_mm_personalR"]
                   for e in convergence_results if e.get("fixation_distance_mm_personalR")]
    if fix_vals_pr:
        print(f"  [REFLECT-C3D PR] {len(fix_vals_pr)} frames | "
              f"fixation: median={np.median(fix_vals_pr)/10:.1f}cm")

    # Aspherical stats
    fix_vals_asph = [e["fixation_distance_mm_aspherical"]
                     for e in convergence_results if e.get("fixation_distance_mm_aspherical")]
    if fix_vals_asph:
        miss_asph = [e["ray_miss_mm_aspherical"]
                     for e in convergence_results if e.get("ray_miss_mm_aspherical") is not None]
        print(f"  [REFLECT-C3D ASPH] {len(fix_vals_asph)} frames | "
              f"fixation: median={np.median(fix_vals_asph)/10:.1f}cm "
              f"ray miss: {np.median(miss_asph):.2f}mm" if miss_asph else "")

    if ipd_vals:
        print(f"  [REFLECT-C3D] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "reflection_corneal_3d",
        "description": "Reflection-law CC + 3D corneal model gaze (pupil_3d - CC)",
        "cc_estimation_method": "reflection_law",
        "personal_radius": personal_radius_info,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_fixation_mm_refracted": round(float(np.median(fix_vals_ref)), 2) if fix_vals_ref else None,
        "median_fixation_mm_personalR": round(float(np.median(fix_vals_pr)), 2) if fix_vals_pr else None,
        "median_fixation_mm_aspherical": round(float(np.median(fix_vals_asph)), 2) if fix_vals_asph else None,
        "asphericity": {k: v for k, v in asphericity_info.items()
                    if k not in ('eye_cc_asph', 'optical_axes')} if asphericity_info else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": [round(float(v), 4) for v in eye_cc["right"]],
        "left_corneal_center_lo": [round(float(v), 4) for v in eye_cc["left"]],
        "right_corneal_center_ro_personalR": [round(float(v), 4) for v in right_cc_pr]
            if right_cc_pr is not None else None,
        "left_corneal_center_lo_personalR": [round(float(v), 4) for v in left_cc_pr]
            if left_cc_pr is not None else None,
        "right_corneal_center_ro_asph": [round(float(v), 4) for v in right_cc_asph]
            if right_cc_asph is not None else None,
        "left_corneal_center_lo_asph": [round(float(v), 4) for v in left_cc_asph]
            if left_cc_asph is not None else None,
        "cc_observations_right": cc_observations_map.get("cc_observations_right") if cc_observations_map else None,
        "cc_observations_left": cc_observations_map.get("cc_observations_left") if cc_observations_map else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "fixation_distance_mm_refracted": e.get("fixation_distance_mm_refracted"),
            "convergence_mm_refracted": e.get("convergence_mm_refracted"),
            "convergence_point_refracted": e.get("convergence_point_refracted"),
            "ray_miss_mm_refracted": e.get("ray_miss_mm_refracted"),
            "right_pupil_3d_refracted": e.get("right_pupil_3d_refracted"),
            "left_pupil_3d_lo_refracted": e.get("left_pupil_3d_lo_refracted"),
            "fixation_distance_mm_personalR": e.get("fixation_distance_mm_personalR"),
            "convergence_mm_personalR": e.get("convergence_mm_personalR"),
            "convergence_point_personalR": e.get("convergence_point_personalR"),
            "ray_miss_mm_personalR": e.get("ray_miss_mm_personalR"),
            "fixation_distance_mm_aspherical": e.get("fixation_distance_mm_aspherical"),
            "convergence_mm_aspherical": e.get("convergence_mm_aspherical"),
            "convergence_point_aspherical": e.get("convergence_point_aspherical"),
            "ray_miss_mm_aspherical": e.get("ray_miss_mm_aspherical"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_reflectc3d.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [REFLECT-C3D] Saved to {conv_path}")

    # Save separate JSONs for variant methods (so they get scene calibration + compare_methods support)
    _variant_suffixes = [
        ("refracted", "Refraction-corrected (Snell's law) pupil_3d + reflection-law CC"),
        ("personalR", "Personalized corneal radius + reflection-law CC"),
        ("aspherical", "Aspherical cornea model (R,Q) + reflection-law CC"),
    ]
    for suffix, desc in _variant_suffixes:
        variant_frames = []
        for e in conv_meta["per_frame"]:
            cp = e.get(f"convergence_point_{suffix}")
            fd = e.get(f"fixation_distance_mm_{suffix}")
            if cp and fd:
                variant_frames.append({
                    "frame": e["frame"],
                    "convergence_point": cp,
                    "fixation_distance_mm": fd,
                    "convergence_mm": e.get(f"convergence_mm_{suffix}"),
                    "ray_miss_mm": e.get(f"ray_miss_mm_{suffix}"),
                    "ipd_mm": e.get("ipd_mm"),
                })
        if variant_frames:
            variant_meta = {
                "method": f"reflection_corneal_3d_{suffix}",
                "description": desc,
                "parent_method": "convergence_meta_reflectc3d.json",
                "median_fixation_mm": conv_meta.get(f"median_fixation_mm_{suffix}"),
                "n_frames": len(variant_frames),
                "per_frame": variant_frames,
            }
            vpath = out_base / f"convergence_meta_{suffix}.json"
            with open(str(vpath), "w") as f:
                json.dump(variant_meta, f, indent=2)
            print(f"  [REFLECT-C3D] Saved variant {suffix} -> {vpath} ({len(variant_frames)} frames)")






def compute_cor_c3d_convergence(output_dir, calib):
    """2-ray convergence using Center of Rotation (COR) as ray origin.

    COR is the eye's center of rotation, located ~5.7mm behind the corneal
    center (CC) along the optical axis. Using COR instead of CC as the
    gaze ray origin provides a more anatomically correct model because
    the eye rotates around COR, not CC.

    COR position: COR = CC + COR_OFFSET * normalize(CC)
    where COR_OFFSET = 5.7mm (distance from CC to COR along optical axis)

    Gaze direction: normalize(pupil_3d - COR) projected onto normalized plane
    Then intersect right and left gaze rays for convergence.

    Loads CC from convergence_meta_reflectc3d.json (reflection-law CC estimate).
    Saves to convergence_meta_corc3d.json.
    """
    COR_OFFSET_MM = 5.7  # CC to COR distance along optical axis

    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [COR-C3D] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load CC from reflection-law estimate
    meta_rc3d_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_rc3d_path.exists():
        print("  [COR-C3D] No reflect c3d meta (need CC positions), skipping")
        return
    with open(meta_rc3d_path) as f:
        rc3d_meta = json.load(f)

    if not rc3d_meta.get("right_corneal_center_ro") or not rc3d_meta.get("left_corneal_center_lo"):
        print("  [COR-C3D] Missing CC in reflect c3d meta, skipping")
        return

    right_cc = np.array(rc3d_meta["right_corneal_center_ro"])
    left_cc = np.array(rc3d_meta["left_corneal_center_lo"])

    # Compute COR = CC + offset along optical axis (camera-to-CC direction)
    right_cc_norm = np.linalg.norm(right_cc)
    left_cc_norm = np.linalg.norm(left_cc)
    if right_cc_norm < 1e-6 or left_cc_norm < 1e-6:
        print("  [COR-C3D] CC too close to camera origin, skipping")
        return

    right_cor = right_cc + COR_OFFSET_MM * (right_cc / right_cc_norm)
    left_cor = left_cc + COR_OFFSET_MM * (left_cc / left_cc_norm)

    print(f"  [COR-C3D] Right CC  (RO): [{right_cc[0]:.2f}, "
          f"{right_cc[1]:.2f}, {right_cc[2]:.2f}]")
    print(f"  [COR-C3D] Right COR (RO): [{right_cor[0]:.2f}, "
          f"{right_cor[1]:.2f}, {right_cor[2]:.2f}]")
    print(f"  [COR-C3D] Left CC   (LO): [{left_cc[0]:.2f}, "
          f"{left_cc[1]:.2f}, {left_cc[2]:.2f}]")
    print(f"  [COR-C3D] Left COR  (LO): [{left_cor[0]:.2f}, "
          f"{left_cor[1]:.2f}, {left_cor[2]:.2f}]")
    print(f"  [COR-C3D] COR offset: {COR_OFFSET_MM}mm along optical axis")

    # Load seg combined results (need pupil_3d for both eyes)
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [COR-C3D] Need seg combined results for both eyes, skipping")
        return

    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    # Transform left COR to RO frame for convergence computation
    l_cor_ro = (R_cross.T @ (left_cor.reshape(3, 1) - T_cross)).flatten()

    def _converge_cor(r_pupil_3d, l_pupil_3d_lo, r_cor, l_cor):
        """Compute convergence using COR as ray origin.

        The gaze direction is the same as RC3D (projection-based: pupil_proj - cc_proj),
        because COR lies on the same camera ray as CC (so cor_proj == cc_proj).
        The key difference is the RAY ORIGIN: COR instead of camera center (0,0,0).
        This models the eye rotating around COR, giving physically distinct ray geometry.
        """
        # Gaze direction: same projection-based approach as RC3D
        # (cor_proj == cc_proj, so this equals pupil_proj - cc_proj)
        r_pupil_proj = np.array([r_pupil_3d[0]/r_pupil_3d[2], r_pupil_3d[1]/r_pupil_3d[2]])
        r_cor_proj = np.array([r_cor[0]/r_cor[2], r_cor[1]/r_cor[2]])
        r_gaze_norm = r_pupil_proj - r_cor_proj

        l_pupil_proj = np.array([l_pupil_3d_lo[0]/l_pupil_3d_lo[2], l_pupil_3d_lo[1]/l_pupil_3d_lo[2]])
        l_cor_proj = np.array([l_cor[0]/l_cor[2], l_cor[1]/l_cor[2]])
        l_gaze_norm = l_pupil_proj - l_cor_proj

        r_dir = np.array([r_gaze_norm[0], r_gaze_norm[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze_norm[0], l_gaze_norm[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        # KEY DIFFERENCE: Ray origins are COR positions, not camera centers
        # Ray 1: P = r_cor + s * r_dir  (right COR in RO frame)
        # Ray 2: P = l_cor_ro + t * l_dir_ro  (left COR in RO frame)
        w0 = r_cor - l_cor_ro
        a = float(np.dot(r_dir, r_dir))
        b = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d = float(np.dot(r_dir, w0))
        e = float(np.dot(l_dir_ro, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return None
        sc_val = (b * e - c * d) / denom
        tc_val = (a * e - b * d) / denom
        if sc_val <= 0 or tc_val <= 0:
            return None
        closest_r = r_cor + sc_val * r_dir
        closest_l = l_cor_ro + tc_val * l_dir_ro
        conv_pt = (closest_r + closest_l) / 2.0
        ray_miss = float(np.linalg.norm(closest_r - closest_l))
        cam_mid = lo_origin_ro / 2.0
        fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
        return {
            'fixation_distance_mm': round(fix_dist, 2),
            'convergence_mm': round(fix_dist, 2),
            'convergence_point': [round(float(conv_pt[k]), 2) for k in range(3)],
            'ray_miss_mm': round(ray_miss, 2),
        }

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    convergence_results = []

    for frame_name in common_frames:
        r_entry = r_seg[frame_name]
        l_entry = l_seg[frame_name]

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        r_pupil = r_entry.get("pupil_3d")
        l_pupil = l_entry.get("pupil_3d")
        if not r_pupil or not l_pupil:
            convergence_results.append(entry)
            continue

        rp = np.array(r_pupil)
        lp_lo = np.array(l_pupil)

        # COR convergence
        conv = _converge_cor(rp, lp_lo, right_cor, left_cor)
        if conv:
            entry.update(conv)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    if fix_vals:
        print(f"  [COR-C3D] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [COR-C3D] ray miss: median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [COR-C3D] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "cor_corneal_3d",
        "description": "Center of Rotation (COR) + 3D corneal model gaze (pupil_3d - COR)",
        "cc_estimation_method": "reflection_law",
        "cor_offset_mm": COR_OFFSET_MM,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_cor_ro": [round(float(v), 4) for v in right_cor],
        "left_cor_lo": [round(float(v), 4) for v in left_cor],
        "right_corneal_center_ro": [round(float(v), 4) for v in right_cc],
        "left_corneal_center_lo": [round(float(v), 4) for v in left_cc],
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_corc3d.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [COR-C3D] Saved to {conv_path}")





def _sliding_cc_for_frame(cc_by_frame, frames_sorted, cc_global, frame_name,
                           half_win, min_obs=5, causal=True):
    """Compute sliding-window CC for a specific frame with MAD outlier rejection.

    Args:
        cc_by_frame: dict mapping frame_name -> np.array CC [x,y,z]
        frames_sorted: sorted list of frame names with CC observations
        cc_global: np.array global median CC (fallback)
        frame_name: the frame to compute CC for
        half_win: half the window size (e.g. 25 for window_size=50)
        min_obs: minimum valid observations; falls back to global if fewer
        causal: if True (default), window only looks backward (past-only)
                to prevent temporal leakage. Uses full window size backward.
                If False, uses symmetric window (original behavior).
    Returns:
        np.array CC [x,y,z]
    """
    idx = bisect.bisect_left(frames_sorted, frame_name)
    idx = min(idx, len(frames_sorted) - 1)

    if causal:
        window = half_win * 2  # use full window size backward
        start = max(0, idx - window)
        end = idx + 1  # include current frame only
    else:
        start = max(0, idx - half_win)
        end = min(len(frames_sorted), idx + half_win + 1)
    nearby_frames = frames_sorted[start:end]
    nearby_ccs = np.array([cc_by_frame[f] for f in nearby_frames])

    if len(nearby_ccs) < min_obs:
        return cc_global

    # Outlier rejection using MAD (median absolute deviation)
    local_median = np.median(nearby_ccs, axis=0)
    deviations = np.linalg.norm(nearby_ccs - local_median, axis=1)
    mad = np.median(deviations)
    if mad < 1e-10:
        return local_median

    # Reject observations > 2 * MAD from local median
    valid_mask = deviations <= 2.0 * mad
    valid_ccs = nearby_ccs[valid_mask]

    if len(valid_ccs) < min_obs:
        return cc_global

    return np.median(valid_ccs, axis=0)


def compute_sliding_cc_convergence(output_dir, calib, window_size=50, min_obs=5, causal=True):
    """2-ray convergence using per-frame sliding-window corneal center.

    Instead of a fixed session-wide median CC, this method adapts the CC
    estimate per frame using a sliding window of nearby CC observations.
    This makes the convergence robust to glasses slip during the session.

    Args:
        window_size: Number of nearest frames for CC estimate (default 50).
        min_obs: Minimum valid CC observations in window; falls back to global
                 median if fewer.

    Saves to convergence_meta_slidingcc.json.
    """
    out_base = Path(output_dir)
    TAG = "SLIDING-CC"

    # Load cross calibration
    cross = calib.get("cross")
    if cross is None:
        print(f"  [{TAG}] No cross-pair calibration, skipping")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load CC observations from reflection-law meta (NOT corneal3d — different CC method!)
    meta_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_path.exists():
        print(f"  [{TAG}] No reflect c3d results (need per-frame CC observations)")
        return
    with open(meta_path) as f:
        _reflect_meta = json.load(f)
    r_cc_obs = _reflect_meta.get("cc_observations_right")
    l_cc_obs = _reflect_meta.get("cc_observations_left")
    if not r_cc_obs or not l_cc_obs:
        print(f"  [{TAG}] No per-frame CC observations in reflect c3d meta")
        return

    # Build frame -> CC lookup and sorted frame lists
    r_cc_by_frame = {obs["frame"]: np.array(obs["cc"]) for obs in r_cc_obs}
    l_cc_by_frame = {obs["frame"]: np.array(obs["cc"]) for obs in l_cc_obs}
    r_frames_sorted = sorted(r_cc_by_frame.keys())
    l_frames_sorted = sorted(l_cc_by_frame.keys())

    # Global median CC (fallback)
    r_cc_all = np.array([obs["cc"] for obs in r_cc_obs])
    l_cc_all = np.array([obs["cc"] for obs in l_cc_obs])
    r_cc_global = np.median(r_cc_all, axis=0)
    l_cc_global = np.median(l_cc_all, axis=0)

    # Load pupil_3d for both eyes
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print(f"  [{TAG}] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    half_win = window_size // 2

    print(f"  [{TAG}] {len(common_frames)} common frames, window={window_size}, "
          f"R CC obs={len(r_cc_obs)}, L CC obs={len(l_cc_obs)}")

    convergence_results = []
    n_converged = 0
    for frame_name in common_frames:
        rp_3d = r_seg[frame_name].get("pupil_3d")
        lp_3d = l_seg[frame_name].get("pupil_3d")
        if not rp_3d or not lp_3d:
            continue

        rp = np.array(rp_3d)
        lp_lo = np.array(lp_3d)

        # Get sliding-window CC for this frame
        r_cc = _sliding_cc_for_frame(r_cc_by_frame, r_frames_sorted, r_cc_global,
                                      frame_name, half_win, min_obs, causal=causal)
        l_cc = _sliding_cc_for_frame(l_cc_by_frame, l_frames_sorted, l_cc_global,
                                      frame_name, half_win, min_obs, causal=causal)

        # Compute gaze direction (pupil_proj - cc_proj on normalized plane)
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_cc_proj = np.array([r_cc[0] / r_cc[2], r_cc[1] / r_cc[2]])
        r_gaze = r_pupil_proj - r_cc_proj

        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_cc_proj = np.array([l_cc[0] / l_cc[2], l_cc[1] / l_cc[2]])
        l_gaze = l_pupil_proj - l_cc_proj

        # Build 3D gaze rays
        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        # Closest point of approach (2-ray intersection)
        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val

        entry = {
            "frame": frame_name,
            "convergence_mm": None,
            "fixation_distance_mm": None,
            "convergence_point": None,
            "ray_miss_mm": None,
            "ipd_mm": None,
        }

        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))

                entry["convergence_mm"] = round(fix_dist, 2)
                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [round(float(conv_pt[0]), 2),
                                               round(float(conv_pt[1]), 2),
                                               round(float(conv_pt[2]), 2)]
                entry["ray_miss_mm"] = round(ray_miss, 2)
                n_converged += 1

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)

        convergence_results.append(entry)

    # Statistics
    fix_vals = [e["fixation_distance_mm"] for e in convergence_results
                if e["fixation_distance_mm"] is not None]
    miss_vals = [e["ray_miss_mm"] for e in convergence_results
                 if e["ray_miss_mm"] is not None]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]

    print(f"  [{TAG}] {n_converged}/{len(convergence_results)} frames converged")
    if fix_vals:
        print(f"  [{TAG}] fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [{TAG}] ray miss: median={np.median(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [{TAG}] IPD: median={np.median(ipd_vals):.2f}mm")

    # Save JSON
    conv_meta = {
        "method": "sliding_cc_convergence",
        "description": "2-ray convergence with per-frame sliding-window corneal center",
        "cc_estimation_method": "sliding_window",
        "window_size": window_size,
        "min_observations": min_obs,
        "right_corneal_center_ro_global": [round(float(x), 4) for x in r_cc_global],
        "left_corneal_center_lo_global": [round(float(x), 4) for x in l_cc_global],
        "right_corneal_center_ro": _reflect_meta.get("right_corneal_center_ro"),
        "left_corneal_center_lo": _reflect_meta.get("left_corneal_center_lo"),
        "n_right_cc_observations": len(r_cc_obs),
        "n_left_cc_observations": len(l_cc_obs),
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ray_miss_mm": round(float(np.median(miss_vals)), 2) if miss_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "per_frame": convergence_results,
    }

    conv_path = out_base / "convergence_meta_slidingcc.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [{TAG}] Saved to {conv_path}")

def compute_physical_reflection_convergence(output_dir, calib, led_positions_physical,
                                             led_positions_virtual, crop_size=150):
    """Compute convergence using physical LED positions + specular reflection solver.

    Uses point_unmirrored (physical LED positions) and solves the specular reflection
    equation on the corneal sphere. Initial CC estimate from virtual-image approach.

    Saves to convergence_meta_physreflectc3d.json.
    """
    out_base = Path(output_dir)

    cross_pair = calib.get("cross")
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    if not all([cross_pair, right_pair, left_pair]):
        print("  [PHYS-REFLECT] Need all 3 stereo pairs, skipping")
        return

    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)
    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)

    lo_origin_ro = (-R_cross.T @ T_cross).flatten()
    CORNEAL_RADIUS = 7.8

    # Camera origins in their own outer frame
    cam_origins = {
        'ro': np.zeros(3),
        'ri': (-R_right.T @ T_right).flatten(),
        'lo': np.zeros(3),
        'li': (-R_left.T @ T_left).flatten(),
    }

    # ===== STEP 1: Get initial CC from virtual-image approach =====
    # Load existing reflect c3d meta for initial CC guess
    meta_rc3d_path = out_base / "convergence_meta_reflectc3d.json"
    meta_r4r_path = out_base / "convergence_meta_reflect4ray.json"
    initial_cc = {}

    for meta_path in [meta_rc3d_path, meta_r4r_path]:
        if meta_path.exists() and not initial_cc:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("right_corneal_center_ro"):
                initial_cc["right"] = np.array(meta["right_corneal_center_ro"])
            if meta.get("left_corneal_center_lo"):
                initial_cc["left"] = np.array(meta["left_corneal_center_lo"])

    if "right" not in initial_cc or "left" not in initial_cc:
        print("  [PHYS-REFLECT] Need initial CC from virtual-image method (reflect C3D or R4R), skipping")
        return

    print(f"  [PHYS-REFLECT] Initial CC (virtual-image):")
    print(f"    Right (RO): [{initial_cc['right'][0]:.2f}, {initial_cc['right'][1]:.2f}, {initial_cc['right'][2]:.2f}]")
    print(f"    Left (LO):  [{initial_cc['left'][0]:.2f}, {initial_cc['left'][1]:.2f}, {initial_cc['left'][2]:.2f}]")

    # ===== STEP 2: Collect specular reflection observations =====
    eye_cc_phys = {}

    for eye, pair_key, outer_cam, inner_cam in [
        ("right", "right", "ro", "ri"),
        ("left", "left", "lo", "li"),
    ]:
        stereo = calib.get(pair_key)
        cam1 = stereo['cam1']  # outer
        cam2 = stereo['cam2']  # inner
        K1 = np.array(stereo.get('K1', calib[cam1]['K']))
        d1 = np.array(stereo.get('dist1', calib[cam1]['dist']))
        K2 = np.array(stereo.get('K2', calib[cam2]['K']))
        d2 = np.array(stereo.get('dist2', calib[cam2]['dist']))
        R_s = np.array(stereo['R'])
        T_s = np.array(stereo['T']).reshape(3, 1)

        # LED positions in outer camera frame (physical)
        leds_outer = led_positions_physical.get(outer_cam)
        leds_inner = led_positions_physical.get(inner_cam)
        if not leds_outer:
            print(f"  [PHYS-REFLECT] No physical LED positions for {outer_cam.upper()}, skipping {eye}")
            continue

        # Load per-camera results for glint detections
        cam1_results_path = out_base / cam1 / "results.json"
        cam2_results_path = out_base / cam2 / "results.json"
        if not cam1_results_path.exists() or not cam2_results_path.exists():
            print(f"  [PHYS-REFLECT] Missing results for {cam1}/{cam2}, skipping {eye}")
            continue

        with open(cam1_results_path) as f:
            cam1_results = json.load(f)
        with open(cam2_results_path) as f:
            cam2_results = json.load(f)

        observations = []

        for entry in cam1_results:
            glints_raw = (entry.get("seg_enh_glints") or entry.get("seg_glints")
                          or entry.get("glints", []))
            if not glints_raw or len(glints_raw) < 3:
                continue
            # Convert dict glints to (x, y) tuples using original image coords
            glints = [(g["x_orig"], g["y_orig"]) for g in glints_raw
                      if "x_orig" in g and "y_orig" in g]
            if len(glints) < 3:
                continue
            # Match glints to LEDs (virtual positions for matching topology)
            leds_virt_outer = led_positions_virtual.get(outer_cam)
            if not leds_virt_outer:
                continue
            matched_g, matched_l = match_glints_to_leds(glints, leds_virt_outer, K1, d1)
            if matched_g is None:
                continue
            # For each matched glint, create observation with physical LED pos
            for gi in range(len(matched_g)):
                gx, gy = matched_g[gi]
                # Find which LED index this matched to
                led_3d_virt = matched_l[gi]
                # Find the index in the original list
                led_idx = None
                for li, lv in enumerate(leds_virt_outer):
                    if np.allclose(lv, led_3d_virt, atol=0.01):
                        led_idx = li
                        break
                if led_idx is None or led_idx >= len(leds_outer):
                    continue
                # Camera ray direction (undistorted)
                nx, ny = _undistort_single(gx, gy, K1, d1)
                ray_dir = np.array([nx, ny, 1.0])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                # Physical LED position in outer frame
                led_phys = leds_outer[led_idx]
                observations.append((cam_origins[outer_cam], ray_dir, led_phys))

        # Also collect from inner camera
        for entry in cam2_results:
            glints_raw = (entry.get("seg_enh_glints") or entry.get("seg_glints")
                          or entry.get("glints", []))
            if not glints_raw or len(glints_raw) < 3:
                continue
            glints = [(g["x_orig"], g["y_orig"]) for g in glints_raw
                      if "x_orig" in g and "y_orig" in g]
            if len(glints) < 3:
                continue
            leds_virt_inner = led_positions_virtual.get(inner_cam)
            if not leds_virt_inner or not leds_inner:
                continue
            matched_g, matched_l = match_glints_to_leds(glints, leds_virt_inner, K2, d2)
            if matched_g is None:
                continue
            for gi in range(len(matched_g)):
                gx, gy = matched_g[gi]
                led_3d_virt = matched_l[gi]
                led_idx = None
                for li, lv in enumerate(leds_virt_inner):
                    if np.allclose(lv, led_3d_virt, atol=0.01):
                        led_idx = li
                        break
                if led_idx is None or led_idx >= len(leds_inner):
                    continue
                # Ray in inner camera frame
                nx, ny = _undistort_single(gx, gy, K2, d2)
                ray_dir_inner = np.array([nx, ny, 1.0])
                ray_dir_inner = ray_dir_inner / np.linalg.norm(ray_dir_inner)
                # Transform ray to outer frame
                ray_dir_outer = (R_s.T @ ray_dir_inner.reshape(3, 1)).flatten()
                ray_dir_outer = ray_dir_outer / np.linalg.norm(ray_dir_outer)
                # Camera origin in outer frame
                inner_origin_outer = (-R_s.T @ T_s).flatten()
                # Physical LED in inner frame -> transform to outer frame
                led_phys_inner = leds_inner[led_idx]
                led_phys_outer = (R_s.T @ (led_phys_inner.reshape(3, 1) - T_s)).flatten()
                observations.append((inner_origin_outer, ray_dir_outer, led_phys_outer))

        print(f"  [PHYS-REFLECT] {eye.upper()}: {len(observations)} specular observations")

        if len(observations) < 10:
            print(f"  [PHYS-REFLECT] Too few observations for {eye}, skipping")
            continue

        # Optimize CC
        cc_phys = _estimate_cc_specular(observations, initial_cc[eye], CORNEAL_RADIUS)
        if cc_phys is not None:
            eye_cc_phys[eye] = cc_phys
            diff = cc_phys - initial_cc[eye]
            print(f"  [PHYS-REFLECT] {eye.upper()} CC shift (phys - virtual): "
                  f"[{diff[0]:.2f}, {diff[1]:.2f}, {diff[2]:.2f}] mm "
                  f"(|shift| = {np.linalg.norm(diff):.2f} mm)")

    if "right" not in eye_cc_phys or "left" not in eye_cc_phys:
        print("  [PHYS-REFLECT] Could not compute physical CC for both eyes, skipping convergence")
        return

    # ===== STEP 3: Compute C3D convergence with physical CC =====
    # Same approach as _compute_reflect_c3d_convergence but with physical CC
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [PHYS-REFLECT] Need seg combined results, skipping")
        return

    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    right_cc = eye_cc_phys["right"]
    left_cc = eye_cc_phys["left"]

    def _converge_phys(r_pupil_3d, l_pupil_3d_lo, r_cc, l_cc):
        r_gaze_3d = r_pupil_3d - r_cc
        if np.linalg.norm(r_gaze_3d) < 1e-6:
            return None
        r_pupil_proj = np.array([r_pupil_3d[0]/r_pupil_3d[2], r_pupil_3d[1]/r_pupil_3d[2]])
        r_cc_proj = np.array([r_cc[0]/r_cc[2], r_cc[1]/r_cc[2]])
        r_gaze_norm = r_pupil_proj - r_cc_proj

        l_gaze_3d = l_pupil_3d_lo - l_cc
        if np.linalg.norm(l_gaze_3d) < 1e-6:
            return None
        l_pupil_proj = np.array([l_pupil_3d_lo[0]/l_pupil_3d_lo[2], l_pupil_3d_lo[1]/l_pupil_3d_lo[2]])
        l_cc_proj = np.array([l_cc[0]/l_cc[2], l_cc[1]/l_cc[2]])
        l_gaze_norm = l_pupil_proj - l_cc_proj

        r_dir = np.array([r_gaze_norm[0], r_gaze_norm[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze_norm[0], l_gaze_norm[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d = float(np.dot(r_dir, w0))
        e = float(np.dot(l_dir_ro, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return None
        sc_val = (b * e - c * d) / denom
        tc_val = (a * e - b * d) / denom
        if sc_val <= 0 or tc_val <= 0:
            return None
        closest_r = sc_val * r_dir
        closest_l = lo_origin_ro + tc_val * l_dir_ro
        conv_pt = (closest_r + closest_l) / 2.0
        ray_miss = float(np.linalg.norm(closest_r - closest_l))
        cam_mid = lo_origin_ro / 2.0
        fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
        return {
            'fixation_distance_mm': round(fix_dist, 2),
            'convergence_mm': round(fix_dist, 2),
            'convergence_point': [round(float(conv_pt[k]), 2) for k in range(3)],
            'ray_miss_mm': round(ray_miss, 2),
        }

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    convergence_results = []

    for frame_name in common_frames:
        r_entry = r_seg[frame_name]
        l_entry = l_seg[frame_name]

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None}

        r_pupil = r_entry.get("pupil_3d")
        l_pupil = l_entry.get("pupil_3d")
        if not r_pupil or not l_pupil:
            convergence_results.append(entry)
            continue

        rp = np.array(r_pupil)
        lp_lo = np.array(l_pupil)

        conv = _converge_phys(rp, lp_lo, right_cc, left_cc)
        if conv:
            entry.update(conv)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        convergence_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in convergence_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in convergence_results if e["ipd_mm"]]
    if fix_vals:
        print(f"  [PHYS-REFLECT] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
        miss_vals = [e["ray_miss_mm"]
                     for e in convergence_results if e["ray_miss_mm"] is not None]
        if miss_vals:
            print(f"  [PHYS-REFLECT] ray miss: median={np.median(miss_vals):.2f}mm "
                  f"mean={np.mean(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [PHYS-REFLECT] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    conv_meta = {
        "method": "physical_reflection_c3d",
        "description": "Physical LED positions + specular reflection CC solver + C3D gaze",
        "cc_estimation_method": "specular_reflection",
        "corneal_radius_mm": CORNEAL_RADIUS,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro_physical": [round(float(v), 4) for v in eye_cc_phys["right"]],
        "left_corneal_center_lo_physical": [round(float(v), 4) for v in eye_cc_phys["left"]],
        "right_corneal_center_ro_virtual": [round(float(v), 4) for v in initial_cc["right"]],
        "left_corneal_center_lo_virtual": [round(float(v), 4) for v in initial_cc["left"]],
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
        } for e in convergence_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_physreflectc3d.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [PHYS-REFLECT] Saved to {conv_path}")

def _compute_cal_cutoff(task_dir):
    """Compute the frame timestamp that separates calibration from test frames.

    Calibration = frames from start until the dot_off of the last NEW unique dot position.
    This covers the first complete cycle of all unique positions (e.g. 9 for 3x3 grid).

    Returns (cal_cutoff_frame_time, n_unique_positions, n_total_dots) or (None, 0, 0).
    """
    logs_path = Path(task_dir) / "logs.json"
    if not logs_path.exists():
        return None, 0, 0

    with open(logs_path) as f:
        logs = json.load(f)

    # Parse all events with timestamps
    events = []  # (ts_sec, event_type, position_key, marker_id)
    first_event_ts = None

    for msg in logs.get("websocket_messages", []):
        parsed = json.loads(msg["ws_message"])
        et = parsed.get("eventType", "")
        ts_raw = parsed.get("timestamp", "")

        # Handle both ISO strings (with "T") and Unix timestamps (float/int)
        if not ts_raw:
            continue

        try:
            if isinstance(ts_raw, (int, float)):
                # Unix timestamp — could be seconds or milliseconds
                ts_sec = float(ts_raw)
                if ts_sec > 1e12:  # milliseconds
                    ts_sec = ts_sec / 1000.0
            elif isinstance(ts_raw, str) and "T" in ts_raw:
                # ISO string: "2026-02-20T14:30:45.123"
                h, m, s = ts_raw.split("T")[1].split(":")
                ts_sec = int(h) * 3600 + int(m) * 60 + float(s)
            else:
                continue
        except (ValueError, IndexError):
            continue

        if first_event_ts is None:
            first_event_ts = ts_sec

        if et == "VisualDot":
            details = parsed.get("details", {})
            rel = details.get("relativePosition", {})
            pos_key = (rel.get("x"), rel.get("y"))
            events.append((ts_sec, "dot_on", pos_key, parsed.get("markerId")))
        elif et == "Event":
            action = parsed.get("details", {}).get("action", "")
            if action == "dot_off":
                events.append((ts_sec, "dot_off", None, None))

    if not events or first_event_ts is None:
        return None, 0, 0

    # Track unique positions to find the last NEW dot
    dot_on_events = [(ts, pos) for ts, etype, pos, _ in events if etype == "dot_on"]
    if not dot_on_events:
        return None, 0, 0

    seen = set()
    last_new_dot_ts = dot_on_events[0][0]
    for ts, pos in dot_on_events:
        if pos not in seen:
            seen.add(pos)
            last_new_dot_ts = ts

    # Find the dot_off immediately after the last new dot_on
    cal_end_log_ts = None
    for ts, etype, _, _ in events:
        if etype == "dot_off" and ts > last_new_dot_ts:
            cal_end_log_ts = ts
            break

    if cal_end_log_ts is None:
        return None, len(seen), len(dot_on_events)

    # Convert log timestamp to frame timestamp
    # offset = first_event_ts (frame time 0 ≈ first log event time)
    cal_cutoff_frame_time = cal_end_log_ts - first_event_ts

    return cal_cutoff_frame_time, len(seen), len(dot_on_events)


def _frame_timestamp(frame_name):
    """Extract timestamp (seconds) from frame filename."""
    m = re.search(r'timestamp_(\d+\.\d+)', frame_name)
    return float(m.group(1)) if m else None


def _split_by_cutoff(items, cutoff_time, get_frame_name):
    """Split items into cal/test lists based on frame timestamp cutoff.

    items: list of anything
    cutoff_time: frame timestamp cutoff (None = all cal)
    get_frame_name: function to extract frame name from item
    Returns (cal_indices, test_indices) as sets of int indices.
    """
    cal_idx = set()
    test_idx = set()
    for i, item in enumerate(items):
        fname = get_frame_name(item)
        ft = _frame_timestamp(fname) if fname else None
        if cutoff_time is None or ft is None or ft <= cutoff_time:
            cal_idx.add(i)
        else:
            test_idx.add(i)
    return cal_idx, test_idx


def _report_cal_test_stats(tag, all_distances, cal_idx, test_idx, known_distance_mm):
    """Print separate cal/test statistics."""
    cal_dists = [all_distances[i] for i in sorted(cal_idx) if i < len(all_distances) and all_distances[i] is not None]
    test_dists = [all_distances[i] for i in sorted(test_idx) if i < len(all_distances) and all_distances[i] is not None]

    if cal_dists:
        err_cal = abs(np.median(cal_dists) - known_distance_mm)
        print(f"  [{tag}] CAL  ({len(cal_dists)} frames): "
              f"median={np.median(cal_dists)/10:.1f}cm "
              f"mean={np.mean(cal_dists)/10:.1f}cm "
              f"std={np.std(cal_dists)/10:.1f}cm "
              f"err={err_cal/10:.1f}cm")
    if test_dists:
        err_test = abs(np.median(test_dists) - known_distance_mm)
        print(f"  [{tag}] TEST ({len(test_dists)} frames): "
              f"median={np.median(test_dists)/10:.1f}cm "
              f"mean={np.mean(test_dists)/10:.1f}cm "
              f"std={np.std(test_dists)/10:.1f}cm "
              f"err={err_test/10:.1f}cm")

    return cal_dists, test_dists


def calibrate_reflect_c3d_convergence(output_dir, calib, known_distance_mm=500.0,
                                       cal_cutoff_time=None, fair_cc=False):
    """Calibrate per-eye gaze bias for Reflect C3D using known fixation distance.

    Optimizes (dx_r, dy_r, dx_l, dy_l) — constant angular offsets subtracted
    from each eye's gaze_norm — such that the median convergence distance
    matches the known fixation distance. This captures kappa angle + any
    systematic camera-eye geometry bias.

    When fair_cc=True, recomputes CC using only calibration-phase frames
    (avoiding data leakage from test frames into the CC estimate).

    Constraints:
      - Horizontal symmetry: dx_r ≈ -dx_l (eyes converge symmetrically)
      - Minimize bias magnitude (regularisation)
      - Minimize variance of fixation distances across frames

    Saves calibrated results to convergence_meta_reflectc3d_cal.json.
    """
    from scipy.optimize import minimize

    out_base = Path(output_dir)

    # Load existing reflect c3d meta for CC positions
    meta_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_path.exists():
        print("  [CALIB] No reflect c3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro") is None or meta.get("left_corneal_center_lo") is None:
        print("  [CALIB] Missing corneal centers in reflect c3d meta")
        return

    right_cc = np.array(meta["right_corneal_center_ro"])
    left_cc = np.array(meta["left_corneal_center_lo"])

    # Fair CC: recompute CC from calibration-only frames
    right_cc_fair = None
    left_cc_fair = None
    if fair_cc and cal_cutoff_time is not None:
        cc_obs_r = meta.get("cc_observations_right")
        cc_obs_l = meta.get("cc_observations_left")
        if cc_obs_r:
            right_cc_fair = _recompute_median_cc_from_observations(cc_obs_r, cal_cutoff_time)
        if cc_obs_l:
            left_cc_fair = _recompute_median_cc_from_observations(cc_obs_l, cal_cutoff_time)
        if right_cc_fair is not None and left_cc_fair is not None:
            print(f"  [CALIB] Fair CC (cal-only): R=[{right_cc_fair[0]:.2f},{right_cc_fair[1]:.2f},{right_cc_fair[2]:.2f}] "
                  f"L=[{left_cc_fair[0]:.2f},{left_cc_fair[1]:.2f},{left_cc_fair[2]:.2f}]")
            print(f"  [CALIB] Standard CC:        R=[{right_cc[0]:.2f},{right_cc[1]:.2f},{right_cc[2]:.2f}] "
                  f"L=[{left_cc[0]:.2f},{left_cc[1]:.2f},{left_cc[2]:.2f}]")
        else:
            print(f"  [CALIB] Fair CC: not enough cal-only observations, using standard CC")

    cross = calib.get("cross")
    if cross is None:
        print("  [CALIB] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load pupil_3d for both eyes from seg combined
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [CALIB] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    # Collect frames with valid pupil_3d for both eyes
    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [CALIB] Only {len(valid_frames)} valid frames, need >= 3")
        return

    # Precompute raw (uncorrected) gaze for each frame
    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - r_cc_proj

        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - l_cc_proj

        raw_gazes.append((r_gaze, l_gaze))

    # Split into calibration / test
    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CALIB] {len(valid_frames)} valid frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CALIB] Split: {n_cal} cal + {n_test} test "
              f"(cutoff={cal_cutoff_time:.1f}s)")
    print(f"  [CALIB] Uncalibrated median: "
          f"{meta.get('median_fixation_mm', 0)/10:.1f} cm")

    def _convergence_distances(bias, indices=None):
        """Compute convergence distances for given frames with given bias."""
        dx_r, dy_r, dx_l, dy_l = bias
        distances = []
        ray_misses = []

        for i, (r_gaze_raw, l_gaze_raw) in enumerate(raw_gazes):
            if indices is not None and i not in indices:
                continue
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)

            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                distances.append(fix_dist)
                ray_misses.append(ray_miss)

        return distances, ray_misses

    def objective(bias):
        # Only use calibration frames for optimization
        distances, ray_misses = _convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10

        median_dist = np.median(distances)
        # Primary: match median distance to known
        dist_error = (median_dist - known_distance_mm) ** 2

        # Minimize variance (person looking at roughly fixed distance)
        variance = np.var(distances)

        # Regularisation: prefer small bias
        reg = 0.001 * np.sum(np.array(bias) ** 2)

        # Horizontal symmetry: dx_r ≈ -dx_l (symmetric vergence)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        # Vertical symmetry: dy_r ≈ dy_l (both eyes same vertical bias)
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2

        return dist_error + 0.01 * variance + reg + sym_h + sym_v

    result = minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [CALIB] Optimized gaze bias (kappa + geometry):")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")

    # Recompute convergence with calibrated bias
    calibrated_results = []
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        r_gaze_raw, l_gaze_raw = raw_gazes[i]
        r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
        l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))

                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [
                    round(float(conv_pt[k]), 2) for k in range(3)]
                entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # ---- Kappa angle estimation (decomposition method) ----
    # Total calibrated bias = kappa + geometric parallax bias.
    # Geometric bias arises because CC is offset from camera origin.
    #
    # For each frame, kappa in gaze_norm space = pupil_proj - zerokappa_proj
    # where zerokappa_proj = projection of a hypothetical pupil at
    #   P_zk = CC + |pupil-CC| * normalize(target - CC)
    # i.e. a pupil placed on the visual axis at the same distance from CC.
    #
    # H/V decomposition is in CAMERA frame (not anatomical frame).
    # Camera axes may be rotated relative to eye anatomy on glasses.
    # Total kappa magnitude is frame-independent.

    # Use median convergence point as the stable target estimate
    valid_conv_pts = [np.array(e["convergence_point"])
                      for e in calibrated_results if e.get("convergence_point")]
    if not valid_conv_pts:
        kappa_data = {}
    else:
        target_ro = np.median(valid_conv_pts, axis=0)

        kappa_right_h, kappa_right_v, kappa_right_mag = [], [], []
        kappa_left_h, kappa_left_v, kappa_left_mag = [], [], []

        # Left eye target in LO frame
        target_lo = (R_cross @ target_ro.reshape(3, 1) + T_cross).flatten()

        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            entry = calibrated_results[i]
            if entry.get("convergence_point") is None:
                continue

            # Right eye kappa (RO frame)
            cc_to_pupil_dist = float(np.linalg.norm(rp - right_cc))
            d_visual_r = target_ro - right_cc
            d_visual_r = d_visual_r / np.linalg.norm(d_visual_r)
            p_zk_r = right_cc + cc_to_pupil_dist * d_visual_r  # zero-kappa pupil

            pupil_proj_r = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            zk_proj_r = np.array([p_zk_r[0] / p_zk_r[2], p_zk_r[1] / p_zk_r[2]])
            kappa_gn_r = pupil_proj_r - zk_proj_r  # kappa in gaze_norm units

            kr_h = float(np.degrees(np.arctan(kappa_gn_r[0])))
            kr_v = float(np.degrees(np.arctan(kappa_gn_r[1])))
            kr_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_r))))
            kappa_right_h.append(kr_h)
            kappa_right_v.append(kr_v)
            kappa_right_mag.append(kr_mag)

            # Left eye kappa (LO frame)
            cc_to_pupil_dist_l = float(np.linalg.norm(lp_lo - left_cc))
            d_visual_l = target_lo - left_cc
            d_visual_l = d_visual_l / np.linalg.norm(d_visual_l)
            p_zk_l = left_cc + cc_to_pupil_dist_l * d_visual_l

            pupil_proj_l = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            zk_proj_l = np.array([p_zk_l[0] / p_zk_l[2], p_zk_l[1] / p_zk_l[2]])
            kappa_gn_l = pupil_proj_l - zk_proj_l

            kl_h = float(np.degrees(np.arctan(kappa_gn_l[0])))
            kl_v = float(np.degrees(np.arctan(kappa_gn_l[1])))
            kl_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_l))))
            kappa_left_h.append(kl_h)
            kappa_left_v.append(kl_v)
            kappa_left_mag.append(kl_mag)

            # Store per-frame kappa [H, V, magnitude] in degrees
            entry["kappa_right_deg"] = [round(kr_h, 3), round(kr_v, 3),
                                        round(kr_mag, 3)]
            entry["kappa_left_deg"] = [round(kl_h, 3), round(kl_v, 3),
                                       round(kl_mag, 3)]

        # Report kappa with noise analysis
        kappa_data = {}

        def _kappa_noise(vals):
            """Compute noise metrics for a kappa component."""
            arr = np.array(vals)
            med = float(np.median(arr))
            std = float(np.std(arr))
            mad = float(np.median(np.abs(arr - med)))
            return {
                "median": round(med, 3),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(std, 3),
                "mad": round(mad, 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "range": round(float(np.max(arr) - np.min(arr)), 3),
            }

        if kappa_right_h:
            rh = _kappa_noise(kappa_right_h)
            rv = _kappa_noise(kappa_right_v)
            rmag = _kappa_noise(kappa_right_mag)
            print(f"  [KAPPA] Right eye (camera frame):")
            print(f"    cam-H: median={rh['median']:+.2f}° "
                  f"std={rh['std']:.3f}° mad={rh['mad']:.3f}° "
                  f"range=[{rh['min']:+.2f}°, {rh['max']:+.2f}°]")
            print(f"    cam-V: median={rv['median']:+.2f}° "
                  f"std={rv['std']:.3f}° mad={rv['mad']:.3f}° "
                  f"range=[{rv['min']:+.2f}°, {rv['max']:+.2f}°]")
            print(f"    magnitude: median={rmag['median']:.2f}° "
                  f"std={rmag['std']:.3f}° "
                  f"range=[{rmag['min']:.2f}°, {rmag['max']:.2f}°]")
            kappa_data["right"] = {
                "cam_h_deg": rh, "cam_v_deg": rv, "magnitude_deg": rmag,
                "n_frames": len(kappa_right_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_right_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_right_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_right_mag],
            }

        if kappa_left_h:
            lh = _kappa_noise(kappa_left_h)
            lv = _kappa_noise(kappa_left_v)
            lmag = _kappa_noise(kappa_left_mag)
            print(f"  [KAPPA] Left eye (camera frame):")
            print(f"    cam-H: median={lh['median']:+.2f}° "
                  f"std={lh['std']:.3f}° mad={lh['mad']:.3f}° "
                  f"range=[{lh['min']:+.2f}°, {lh['max']:+.2f}°]")
            print(f"    cam-V: median={lv['median']:+.2f}° "
                  f"std={lv['std']:.3f}° mad={lv['mad']:.3f}° "
                  f"range=[{lv['min']:+.2f}°, {lv['max']:+.2f}°]")
            print(f"    magnitude: median={lmag['median']:.2f}° "
                  f"std={lmag['std']:.3f}° "
                  f"range=[{lmag['min']:.2f}°, {lmag['max']:.2f}°]")
            kappa_data["left"] = {
                "cam_h_deg": lh, "cam_v_deg": lv, "magnitude_deg": lmag,
                "n_frames": len(kappa_left_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_left_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_left_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_left_mag],
            }

        # Geometric bias verification
        print(f"  [KAPPA] Decomposition check (total_bias = kappa + geometric):")
        total_r = np.array([np.degrees(np.arctan(dx_r)),
                            np.degrees(np.arctan(dy_r))])
        if kappa_right_h:
            kappa_r_med = np.array([rh['median'], rv['median']])
            geom_r = total_r - kappa_r_med
            print(f"    Right: total=[{total_r[0]:+.2f}°, {total_r[1]:+.2f}°] "
                  f"= kappa=[{kappa_r_med[0]:+.2f}°, {kappa_r_med[1]:+.2f}°] "
                  f"+ geom=[{geom_r[0]:+.2f}°, {geom_r[1]:+.2f}°]")
        total_l = np.array([np.degrees(np.arctan(dx_l)),
                            np.degrees(np.arctan(dy_l))])
        if kappa_left_h:
            kappa_l_med = np.array([lh['median'], lv['median']])
            geom_l = total_l - kappa_l_med
            print(f"    Left:  total=[{total_l[0]:+.2f}°, {total_l[1]:+.2f}°] "
                  f"= kappa=[{kappa_l_med[0]:+.2f}°, {kappa_l_med[1]:+.2f}°] "
                  f"+ geom=[{geom_l[0]:+.2f}°, {geom_l[1]:+.2f}°]")

        if kappa_right_h or kappa_left_h:
            # Noise quality assessment
            all_stds = []
            if kappa_right_h:
                all_stds.extend([rh['std'], rv['std']])
            if kappa_left_h:
                all_stds.extend([lh['std'], lv['std']])
            max_std = max(all_stds)
            if max_std < 0.5:
                quality = "excellent (std < 0.5)"
            elif max_std < 1.0:
                quality = "good (std < 1.0)"
            elif max_std < 2.0:
                quality = "fair (std < 2.0)"
            else:
                quality = f"poor (std up to {max_std:.1f}, check outlier frames)"
            kappa_data["noise_quality"] = quality
            kappa_data["target_ro_mm"] = [round(float(v), 2) for v in target_ro]
            kappa_data["note"] = ("H/V are in camera frame, not anatomical. "
                                  "Magnitude is frame-independent.")
            print(f"  [KAPPA] Noise quality: {quality}")
            print(f"  [KAPPA] Note: H/V in camera frame. "
                  f"Magnitude is frame-independent.")
            print(f"  [KAPPA] Literature: magnitude ~5° "
                  f"(Hashemi 2010: 5.46 +/- 1.12)")

    # Stats — overall + cal/test split
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    if fix_vals:
        print(f"  [CALIB] Calibrated: {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [CALIB] ray miss: median={np.median(miss_vals):.2f}mm "
              f"mean={np.mean(miss_vals):.2f}mm")
    if ipd_vals:
        print(f"  [CALIB] IPD: median={np.median(ipd_vals):.1f}mm "
              f"mean={np.mean(ipd_vals):.1f}mm")

    # Cal/test split reporting
    if cal_cutoff_time is not None:
        all_fix = [e.get("fixation_distance_mm") for e in calibrated_results]
        _report_cal_test_stats("CALIB", all_fix, cal_idx, test_idx, known_distance_mm)

    # --- Compute calibrated variants (apply same bias to variant gaze norms) ---
    # Load reflect_c3d per-frame data for refracted pupil_3d
    reflect_per_frame = {}
    if meta.get("per_frame"):
        for pf in meta["per_frame"]:
            reflect_per_frame[pf["frame"]] = pf

    # Personal R CC (different CC, same pupil_3d)
    right_cc_pr = np.array(meta["right_corneal_center_ro_personalR"]) \
        if meta.get("right_corneal_center_ro_personalR") else None
    left_cc_pr = np.array(meta["left_corneal_center_lo_personalR"]) \
        if meta.get("left_corneal_center_lo_personalR") else None
    has_pr = right_cc_pr is not None and left_cc_pr is not None
    if has_pr:
        r_cc_pr_proj = np.array([right_cc_pr[0]/right_cc_pr[2],
                                 right_cc_pr[1]/right_cc_pr[2]])
        l_cc_pr_proj = np.array([left_cc_pr[0]/left_cc_pr[2],
                                 left_cc_pr[1]/left_cc_pr[2]])

    # Aspherical CC (different CC, same pupil_3d)
    right_cc_asph = np.array(meta["right_corneal_center_ro_asph"]) \
        if meta.get("right_corneal_center_ro_asph") else None
    left_cc_asph = np.array(meta["left_corneal_center_lo_asph"]) \
        if meta.get("left_corneal_center_lo_asph") else None
    has_asph = right_cc_asph is not None and left_cc_asph is not None
    if has_asph:
        r_cc_asph_proj = np.array([right_cc_asph[0]/right_cc_asph[2],
                                    right_cc_asph[1]/right_cc_asph[2]])
        l_cc_asph_proj = np.array([left_cc_asph[0]/left_cc_asph[2],
                                    left_cc_asph[1]/left_cc_asph[2]])

    def _conv_from_biased_gaze(r_gaze, l_gaze):
        """Compute convergence from bias-corrected gaze_norm values."""
        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)
        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) < 1e-10:
            return None
        sc = (b_val * e_val - c * d_val) / denom
        tc = (a * e_val - b_val * d_val) / denom
        if sc <= 0 or tc <= 0:
            return None
        closest_r = sc * r_dir
        closest_l = lo_origin_ro + tc * l_dir_ro
        conv_pt = (closest_r + closest_l) / 2.0
        ray_miss = float(np.linalg.norm(closest_r - closest_l))
        cam_mid = lo_origin_ro / 2.0
        fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
        return {
            'fixation_distance_mm': round(fix_dist, 2),
            'convergence_mm': round(fix_dist, 2),
            'convergence_point': [round(float(conv_pt[k]), 2) for k in range(3)],
            'ray_miss_mm': round(ray_miss, 2),
        }

    # Apply calibrated bias to each variant per frame
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        entry = calibrated_results[i]
        r_gaze_raw, l_gaze_raw = raw_gazes[i]
        rpf = reflect_per_frame.get(frame_name, {})

        # Cal Refr: refracted pupil_3d + default CC + calibrated bias
        rp_ref = rpf.get("right_pupil_3d_refracted")
        lp_ref = rpf.get("left_pupil_3d_lo_refracted")
        if rp_ref and lp_ref:
            rp_ref = np.array(rp_ref)
            lp_ref = np.array(lp_ref)
            r_gaze_ref = np.array([rp_ref[0]/rp_ref[2], rp_ref[1]/rp_ref[2]]) - r_cc_proj
            l_gaze_ref = np.array([lp_ref[0]/lp_ref[2], lp_ref[1]/lp_ref[2]]) - l_cc_proj
            r_gaze_ref_cal = r_gaze_ref - np.array([dx_r, dy_r])
            l_gaze_ref_cal = l_gaze_ref - np.array([dx_l, dy_l])
            conv_ref = _conv_from_biased_gaze(r_gaze_ref_cal, l_gaze_ref_cal)
            if conv_ref:
                entry["fixation_distance_mm_refracted"] = conv_ref['fixation_distance_mm']
                entry["convergence_mm_refracted"] = conv_ref['convergence_mm']
                entry["convergence_point_refracted"] = conv_ref['convergence_point']
                entry["ray_miss_mm_refracted"] = conv_ref['ray_miss_mm']

        # Cal PR: default pupil_3d + personalR CC + calibrated bias
        if has_pr:
            r_gaze_pr = np.array([rp[0]/rp[2], rp[1]/rp[2]]) - r_cc_pr_proj
            l_gaze_pr = np.array([lp_lo[0]/lp_lo[2], lp_lo[1]/lp_lo[2]]) - l_cc_pr_proj
            r_gaze_pr_cal = r_gaze_pr - np.array([dx_r, dy_r])
            l_gaze_pr_cal = l_gaze_pr - np.array([dx_l, dy_l])
            conv_pr = _conv_from_biased_gaze(r_gaze_pr_cal, l_gaze_pr_cal)
            if conv_pr:
                entry["fixation_distance_mm_personalR"] = conv_pr['fixation_distance_mm']
                entry["convergence_mm_personalR"] = conv_pr['convergence_mm']
                entry["convergence_point_personalR"] = conv_pr['convergence_point']
                entry["ray_miss_mm_personalR"] = conv_pr['ray_miss_mm']

        # Cal Asph: default pupil_3d + aspherical CC + calibrated bias
        if has_asph:
            r_gaze_asph = np.array([rp[0]/rp[2], rp[1]/rp[2]]) - r_cc_asph_proj
            l_gaze_asph = np.array([lp_lo[0]/lp_lo[2], lp_lo[1]/lp_lo[2]]) - l_cc_asph_proj
            r_gaze_asph_cal = r_gaze_asph - np.array([dx_r, dy_r])
            l_gaze_asph_cal = l_gaze_asph - np.array([dx_l, dy_l])
            conv_asph = _conv_from_biased_gaze(r_gaze_asph_cal, l_gaze_asph_cal)
            if conv_asph:
                entry["fixation_distance_mm_aspherical"] = conv_asph['fixation_distance_mm']
                entry["convergence_mm_aspherical"] = conv_asph['convergence_mm']
                entry["convergence_point_aspherical"] = conv_asph['convergence_point']
                entry["ray_miss_mm_aspherical"] = conv_asph['ray_miss_mm']

    # Variant stats
    cal_refr_fix = [e["fixation_distance_mm_refracted"]
                    for e in calibrated_results if e.get("fixation_distance_mm_refracted")]
    cal_pr_fix = [e["fixation_distance_mm_personalR"]
                  for e in calibrated_results if e.get("fixation_distance_mm_personalR")]
    if cal_refr_fix:
        cal_refr_miss = [e["ray_miss_mm_refracted"]
                         for e in calibrated_results if e.get("ray_miss_mm_refracted") is not None]
        print(f"  [CALIB REFR] {len(cal_refr_fix)} frames | "
              f"fixation: median={np.median(cal_refr_fix)/10:.1f}cm "
              f"ray miss: {np.median(cal_refr_miss):.2f}mm" if cal_refr_miss else "")
    if cal_pr_fix:
        cal_pr_miss = [e["ray_miss_mm_personalR"]
                       for e in calibrated_results if e.get("ray_miss_mm_personalR") is not None]
        print(f"  [CALIB PR] {len(cal_pr_fix)} frames | "
              f"fixation: median={np.median(cal_pr_fix)/10:.1f}cm "
              f"ray miss: {np.median(cal_pr_miss):.2f}mm" if cal_pr_miss else "")
    cal_asph_fix = [e["fixation_distance_mm_aspherical"]
                    for e in calibrated_results if e.get("fixation_distance_mm_aspherical")]
    if cal_asph_fix:
        cal_asph_miss = [e["ray_miss_mm_aspherical"]
                         for e in calibrated_results if e.get("ray_miss_mm_aspherical") is not None]
        print(f"  [CALIB ASPH] {len(cal_asph_fix)} frames | "
              f"fixation: median={np.median(cal_asph_fix)/10:.1f}cm "
              f"ray miss: {np.median(cal_asph_miss):.2f}mm" if cal_asph_miss else "")

    conv_meta = {
        "method": "reflection_corneal_3d_calibrated",
        "description": f"Reflect C3D calibrated to {known_distance_mm/10:.0f}cm "
                       f"(per-eye gaze bias / kappa correction)",
        "cc_estimation_method": "reflection_law",
        "personal_radius": meta.get("personal_radius"),
        "calibration": {
            "known_distance_mm": known_distance_mm,
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
        },
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_fixation_mm_refracted": round(float(np.median(cal_refr_fix)), 2) if cal_refr_fix else None,
        "median_fixation_mm_personalR": round(float(np.median(cal_pr_fix)), 2) if cal_pr_fix else None,
        "median_fixation_mm_aspherical": round(float(np.median(cal_asph_fix)), 2) if cal_asph_fix else None,
        "asphericity": meta.get("asphericity"),
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta["right_corneal_center_ro"],
        "left_corneal_center_lo": meta["left_corneal_center_lo"],
        "right_corneal_center_ro_personalR": meta.get("right_corneal_center_ro_personalR"),
        "left_corneal_center_lo_personalR": meta.get("left_corneal_center_lo_personalR"),
        "right_corneal_center_ro_asph": meta.get("right_corneal_center_ro_asph"),
        "left_corneal_center_lo_asph": meta.get("left_corneal_center_lo_asph"),
        "kappa": kappa_data if kappa_data else None,
        "cal_test_split": {
            "cal_cutoff_time": cal_cutoff_time,
            "n_cal": n_cal,
            "n_test": n_test,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
            "kappa_right_deg": e.get("kappa_right_deg"),
            "kappa_left_deg": e.get("kappa_left_deg"),
            "fixation_distance_mm_refracted": e.get("fixation_distance_mm_refracted"),
            "convergence_mm_refracted": e.get("convergence_mm_refracted"),
            "convergence_point_refracted": e.get("convergence_point_refracted"),
            "ray_miss_mm_refracted": e.get("ray_miss_mm_refracted"),
            "fixation_distance_mm_personalR": e.get("fixation_distance_mm_personalR"),
            "convergence_mm_personalR": e.get("convergence_mm_personalR"),
            "convergence_point_personalR": e.get("convergence_point_personalR"),
            "ray_miss_mm_personalR": e.get("ray_miss_mm_personalR"),
            "fixation_distance_mm_aspherical": e.get("fixation_distance_mm_aspherical"),
            "convergence_mm_aspherical": e.get("convergence_mm_aspherical"),
            "convergence_point_aspherical": e.get("convergence_point_aspherical"),
            "ray_miss_mm_aspherical": e.get("ray_miss_mm_aspherical"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    # === Fair CC: re-optimize with cal-only CC ===
    if fair_cc and right_cc_fair is not None and left_cc_fair is not None:
        r_cc_fair_proj = np.array([right_cc_fair[0] / right_cc_fair[2],
                                   right_cc_fair[1] / right_cc_fair[2]])
        l_cc_fair_proj = np.array([left_cc_fair[0] / left_cc_fair[2],
                                   left_cc_fair[1] / left_cc_fair[2]])

        fair_raw_gazes = []
        for frame_name, rp, lp_lo in valid_frames:
            r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            r_gaze = r_pupil_proj - r_cc_fair_proj
            l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            l_gaze = l_pupil_proj - l_cc_fair_proj
            fair_raw_gazes.append((r_gaze, l_gaze))

        def _fair_conv_distances(bias, indices=None):
            dx_r, dy_r, dx_l, dy_l = bias
            distances = []
            for i_f, (r_gaze_raw, l_gaze_raw) in enumerate(fair_raw_gazes):
                if indices is not None and i_f not in indices:
                    continue
                r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
                l_gaze = l_gaze_raw - np.array([dx_l, dy_l])
                r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
                r_dir = r_dir / np.linalg.norm(r_dir)
                l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
                l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
                l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
                l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)
                w0 = -lo_origin_ro
                a_v = float(np.dot(r_dir, r_dir))
                b_v = float(np.dot(r_dir, l_dir_ro))
                c_v = float(np.dot(l_dir_ro, l_dir_ro))
                d_v = float(np.dot(r_dir, w0))
                e_v = float(np.dot(l_dir_ro, w0))
                denom = a_v * c_v - b_v * b_v
                if abs(denom) < 1e-10:
                    continue
                sc = (b_v * e_v - c_v * d_v) / denom
                tc = (a_v * e_v - b_v * d_v) / denom
                if sc > 0 and tc > 0:
                    closest_r = sc * r_dir
                    closest_l = lo_origin_ro + tc * l_dir_ro
                    conv_pt = (closest_r + closest_l) / 2.0
                    cam_mid = lo_origin_ro / 2.0
                    fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                    distances.append(fix_dist)
            return distances

        def fair_objective(bias):
            distances = _fair_conv_distances(bias, cal_idx if cal_cutoff_time is not None else None)
            if len(distances) < 3:
                return 1e10
            median_dist = np.median(distances)
            dist_error = (median_dist - known_distance_mm) ** 2
            variance = np.var(distances)
            reg = 0.001 * np.sum(np.array(bias) ** 2)
            sym_h = 0.1 * (bias[0] + bias[2]) ** 2
            sym_v = 0.1 * (bias[1] - bias[3]) ** 2
            return dist_error + 0.01 * variance + reg + sym_h + sym_v

        fair_result = minimize(fair_objective, [0, 0, 0, 0], method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})
        fdx_r, fdy_r, fdx_l, fdy_l = fair_result.x
        print(f"  [CALIB FAIR] Optimized gaze bias (fair CC):")
        print(f"    Right: dx={fdx_r:.6f}, dy={fdy_r:.6f}")
        print(f"    Left:  dx={fdx_l:.6f}, dy={fdy_l:.6f}")

        # Compute fair per-frame convergence
        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            entry = calibrated_results[i]
            r_gaze_raw, l_gaze_raw = fair_raw_gazes[i]
            r_gaze = r_gaze_raw - np.array([fdx_r, fdy_r])
            l_gaze = l_gaze_raw - np.array([fdx_l, fdy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            w0 = -lo_origin_ro
            a_v = float(np.dot(r_dir, r_dir))
            b_v = float(np.dot(r_dir, l_dir_ro))
            c_v = float(np.dot(l_dir_ro, l_dir_ro))
            d_v = float(np.dot(r_dir, w0))
            e_v = float(np.dot(l_dir_ro, w0))
            denom = a_v * c_v - b_v * b_v
            if abs(denom) > 1e-10:
                sc = (b_v * e_v - c_v * d_v) / denom
                tc = (a_v * e_v - b_v * d_v) / denom
                if sc > 0 and tc > 0:
                    closest_r = sc * r_dir
                    closest_l = lo_origin_ro + tc * l_dir_ro
                    conv_pt = (closest_r + closest_l) / 2.0
                    ray_miss = float(np.linalg.norm(closest_r - closest_l))
                    cam_mid = lo_origin_ro / 2.0
                    fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                    entry["fixation_distance_mm_fair"] = round(fix_dist, 2)
                    entry["convergence_mm_fair"] = round(fix_dist, 2)
                    entry["ray_miss_mm_fair"] = round(ray_miss, 2)

        # Fair stats
        fair_fix = [e.get("fixation_distance_mm_fair") for e in calibrated_results
                    if e.get("fixation_distance_mm_fair")]
        fair_cal = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results)
                    if i in cal_idx and e.get("fixation_distance_mm_fair")]
        fair_test = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results)
                     if i in test_idx and e.get("fixation_distance_mm_fair")]
        if fair_fix:
            print(f"  [CALIB FAIR] {len(fair_fix)} frames | "
                  f"median={np.median(fair_fix)/10:.1f}cm std={np.std(fair_fix)/10:.1f}cm")
        if fair_cal:
            print(f"  [CALIB FAIR] CAL ({len(fair_cal)}): median={np.median(fair_cal)/10:.1f}cm "
                  f"err={abs(np.median(fair_cal)-known_distance_mm)/10:.1f}cm")
        if fair_test:
            print(f"  [CALIB FAIR] TEST ({len(fair_test)}): median={np.median(fair_test)/10:.1f}cm "
                  f"err={abs(np.median(fair_test)-known_distance_mm)/10:.1f}cm")

        conv_meta["fair_cc"] = {
            "right_corneal_center_ro_fair": [round(float(v), 4) for v in right_cc_fair],
            "left_corneal_center_lo_fair": [round(float(v), 4) for v in left_cc_fair],
            "bias_right_fair": [round(float(fdx_r), 6), round(float(fdy_r), 6)],
            "bias_left_fair": [round(float(fdx_l), 6), round(float(fdy_l), 6)],
            "median_fixation_mm_fair": round(float(np.median(fair_fix)), 2) if fair_fix else None,
            "cal_median_mm": round(float(np.median(fair_cal)), 2) if fair_cal else None,
            "test_median_mm": round(float(np.median(fair_test)), 2) if fair_test else None,
            "cal_err_cm": round(abs(np.median(fair_cal)-known_distance_mm)/10, 2) if fair_cal else None,
            "test_err_cm": round(abs(np.median(fair_test)-known_distance_mm)/10, 2) if fair_test else None,
            "n_cal": len(fair_cal),
            "n_test": len(fair_test),
        }

        # Add fair fields to per_frame entries
        for pf in conv_meta["per_frame"]:
            match = next((e for e in calibrated_results if e["frame"] == pf["frame"]), None)
            if match:
                pf["fixation_distance_mm_fair"] = match.get("fixation_distance_mm_fair")
                pf["convergence_mm_fair"] = match.get("convergence_mm_fair")
                pf["ray_miss_mm_fair"] = match.get("ray_miss_mm_fair")

    conv_path = out_base / "convergence_meta_reflectc3d_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CALIB] Saved to {conv_path}")

    # Save separate JSONs for calibrated variant methods
    _cal_variant_suffixes = [
        ("refracted", "Calibrated refraction-corrected (Snell's law)"),
        ("personalR", "Calibrated personalized corneal radius"),
        ("aspherical", "Calibrated aspherical cornea model (R,Q)"),
    ]
    for suffix, desc in _cal_variant_suffixes:
        variant_frames = []
        for e in conv_meta["per_frame"]:
            cp = e.get(f"convergence_point_{suffix}")
            fd = e.get(f"fixation_distance_mm_{suffix}")
            if cp and fd:
                variant_frames.append({
                    "frame": e["frame"],
                    "convergence_point": cp,
                    "fixation_distance_mm": fd,
                    "convergence_mm": e.get(f"convergence_mm_{suffix}"),
                    "ray_miss_mm": e.get(f"ray_miss_mm_{suffix}"),
                    "ipd_mm": e.get("ipd_mm"),
                    "is_calibration": e.get("is_calibration"),
                })
        if variant_frames:
            variant_meta = {
                "method": f"reflection_corneal_3d_{suffix}_calibrated",
                "description": desc,
                "parent_method": "convergence_meta_reflectc3d_cal.json",
                "calibration": conv_meta.get("calibration"),
                "median_fixation_mm": conv_meta.get(f"median_fixation_mm_{suffix}"),
                "n_frames": len(variant_frames),
                "per_frame": variant_frames,
            }
            vpath = out_base / f"convergence_meta_{suffix}_cal.json"
            with open(str(vpath), "w") as f:
                json.dump(variant_meta, f, indent=2)
            print(f"  [CALIB] Saved variant {suffix} -> {vpath} ({len(variant_frames)} frames)")




def calibrate_cor_c3d_convergence(output_dir, calib, known_distance_mm=500.0,
                                   cal_cutoff_time=None, fair_cc=False):
    """Calibrate per-eye gaze bias for COR C3D using known fixation distance.

    Same approach as calibrate_reflect_c3d_convergence() but uses COR
    (Center of Rotation) instead of CC as the ray origin.

    Optimizes (dx_r, dy_r, dx_l, dy_l) — constant angular offsets subtracted
    from each eye's gaze_norm — such that the median convergence distance
    matches the known fixation distance.

    Saves calibrated results to convergence_meta_corc3d_cal.json.
    """
    from scipy.optimize import minimize

    COR_OFFSET_MM = 5.7

    out_base = Path(output_dir)

    # Load existing COR c3d meta for COR positions
    meta_path = out_base / "convergence_meta_corc3d.json"
    if not meta_path.exists():
        print("  [CALIB-COR] No COR c3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_cor_ro") is None or meta.get("left_cor_lo") is None:
        print("  [CALIB-COR] Missing COR positions in meta")
        return

    right_cor = np.array(meta["right_cor_ro"])
    left_cor = np.array(meta["left_cor_lo"])

    # Also get CC for kappa estimation
    right_cc = np.array(meta["right_corneal_center_ro"]) if meta.get("right_corneal_center_ro") else None
    left_cc = np.array(meta["left_corneal_center_lo"]) if meta.get("left_corneal_center_lo") else None

    cross = calib.get("cross")
    if cross is None:
        print("  [CALIB-COR] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load pupil_3d for both eyes from seg combined
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [CALIB-COR] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    # Collect frames with valid pupil_3d
    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [CALIB-COR] Only {len(valid_frames)} valid frames, need >= 3")
        return

    # Precompute raw gaze using projection (same direction as RC3D since cor_proj == cc_proj)
    # But convergence uses COR as ray origin (not camera center)
    r_cor_proj = np.array([right_cor[0] / right_cor[2], right_cor[1] / right_cor[2]])
    l_cor_proj = np.array([left_cor[0] / left_cor[2], left_cor[1] / left_cor[2]])
    # Transform left COR to RO frame for convergence computation
    l_cor_ro = (R_cross.T @ (left_cor.reshape(3, 1) - T_cross)).flatten()

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - r_cor_proj

        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - l_cor_proj

        raw_gazes.append((r_gaze, l_gaze))

    # Split into calibration / test
    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CALIB-COR] {len(valid_frames)} valid frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CALIB-COR] Split: {n_cal} cal + {n_test} test "
              f"(cutoff={cal_cutoff_time:.1f}s)")
    print(f"  [CALIB-COR] Uncalibrated median: "
          f"{meta.get('median_fixation_mm', 0)/10:.1f} cm")

    def _convergence_distances(bias, indices=None):
        dx_r, dy_r, dx_l, dy_l = bias
        distances = []
        ray_misses = []

        for i, (r_gaze_raw, l_gaze_raw) in enumerate(raw_gazes):
            if indices is not None and i not in indices:
                continue
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)

            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            # Use COR positions as ray origins (not camera centers)
            w0 = right_cor - l_cor_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = right_cor + sc * r_dir
                closest_l = l_cor_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                distances.append(fix_dist)
                ray_misses.append(ray_miss)

        return distances, ray_misses

    def objective(bias):
        distances, ray_misses = _convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10
        median_dist = np.median(distances)
        dist_error = (median_dist - known_distance_mm) ** 2
        variance = np.var(distances)
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2
        return dist_error + 0.01 * variance + reg + sym_h + sym_v

    result = minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [CALIB-COR] Optimized gaze bias (kappa + geometry):")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")

    # Recompute convergence with calibrated bias
    calibrated_results = []
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        r_gaze_raw, l_gaze_raw = raw_gazes[i]
        r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
        l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        # Use COR as ray origins (must match optimizer geometry!)
        w0 = right_cor - l_cor_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = right_cor + sc * r_dir
                closest_l = l_cor_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))

                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [
                    round(float(conv_pt[k]), 2) for k in range(3)]
                entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # ---- Kappa angle estimation ----
    valid_conv_pts = [np.array(e["convergence_point"])
                      for e in calibrated_results if e.get("convergence_point")]
    kappa_data = {}
    if valid_conv_pts:
        target_ro = np.median(valid_conv_pts, axis=0)

        kappa_right_h, kappa_right_v, kappa_right_mag = [], [], []
        kappa_left_h, kappa_left_v, kappa_left_mag = [], [], []

        target_lo = (R_cross @ target_ro.reshape(3, 1) + T_cross).flatten()

        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            entry = calibrated_results[i]
            if entry.get("convergence_point") is None:
                continue

            # Right eye kappa (RO frame) — use COR as center
            cor_to_pupil_dist = float(np.linalg.norm(rp - right_cor))
            d_visual_r = target_ro - right_cor
            d_visual_r = d_visual_r / np.linalg.norm(d_visual_r)
            p_zk_r = right_cor + cor_to_pupil_dist * d_visual_r

            pupil_proj_r = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            zk_proj_r = np.array([p_zk_r[0] / p_zk_r[2], p_zk_r[1] / p_zk_r[2]])
            kappa_gn_r = pupil_proj_r - zk_proj_r

            kr_h = float(np.degrees(np.arctan(kappa_gn_r[0])))
            kr_v = float(np.degrees(np.arctan(kappa_gn_r[1])))
            kr_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_r))))
            kappa_right_h.append(kr_h)
            kappa_right_v.append(kr_v)
            kappa_right_mag.append(kr_mag)

            # Left eye kappa (LO frame)
            cor_to_pupil_dist_l = float(np.linalg.norm(lp_lo - left_cor))
            d_visual_l = target_lo - left_cor
            d_visual_l = d_visual_l / np.linalg.norm(d_visual_l)
            p_zk_l = left_cor + cor_to_pupil_dist_l * d_visual_l

            pupil_proj_l = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            zk_proj_l = np.array([p_zk_l[0] / p_zk_l[2], p_zk_l[1] / p_zk_l[2]])
            kappa_gn_l = pupil_proj_l - zk_proj_l

            kl_h = float(np.degrees(np.arctan(kappa_gn_l[0])))
            kl_v = float(np.degrees(np.arctan(kappa_gn_l[1])))
            kl_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_l))))
            kappa_left_h.append(kl_h)
            kappa_left_v.append(kl_v)
            kappa_left_mag.append(kl_mag)

            entry["kappa_right_deg"] = [round(kr_h, 3), round(kr_v, 3),
                                        round(kr_mag, 3)]
            entry["kappa_left_deg"] = [round(kl_h, 3), round(kl_v, 3),
                                       round(kl_mag, 3)]

        def _kappa_noise(vals):
            arr = np.array(vals)
            med = float(np.median(arr))
            std = float(np.std(arr))
            mad = float(np.median(np.abs(arr - med)))
            return {
                "median": round(med, 3),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(std, 3),
                "mad": round(mad, 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "range": round(float(np.max(arr) - np.min(arr)), 3),
            }

        if kappa_right_h:
            rh = _kappa_noise(kappa_right_h)
            rv = _kappa_noise(kappa_right_v)
            rmag = _kappa_noise(kappa_right_mag)
            print(f"  [KAPPA-COR] Right eye (camera frame):")
            print(f"    cam-H: median={rh['median']:+.2f}deg "
                  f"std={rh['std']:.3f}deg")
            print(f"    cam-V: median={rv['median']:+.2f}deg "
                  f"std={rv['std']:.3f}deg")
            print(f"    magnitude: median={rmag['median']:.2f}deg")
            kappa_data["right"] = {
                "cam_h_deg": rh, "cam_v_deg": rv, "magnitude_deg": rmag,
                "n_frames": len(kappa_right_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_right_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_right_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_right_mag],
            }

        if kappa_left_h:
            lh = _kappa_noise(kappa_left_h)
            lv = _kappa_noise(kappa_left_v)
            lmag = _kappa_noise(kappa_left_mag)
            print(f"  [KAPPA-COR] Left eye (camera frame):")
            print(f"    cam-H: median={lh['median']:+.2f}deg "
                  f"std={lh['std']:.3f}deg")
            print(f"    cam-V: median={lv['median']:+.2f}deg "
                  f"std={lv['std']:.3f}deg")
            print(f"    magnitude: median={lmag['median']:.2f}deg")
            kappa_data["left"] = {
                "cam_h_deg": lh, "cam_v_deg": lv, "magnitude_deg": lmag,
                "n_frames": len(kappa_left_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_left_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_left_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_left_mag],
            }

        if kappa_right_h or kappa_left_h:
            all_stds = []
            if kappa_right_h:
                all_stds.extend([rh['std'], rv['std']])
            if kappa_left_h:
                all_stds.extend([lh['std'], lv['std']])
            max_std = max(all_stds)
            if max_std < 0.5:
                quality = "excellent (std < 0.5)"
            elif max_std < 1.0:
                quality = "good (std < 1.0)"
            elif max_std < 2.0:
                quality = "fair (std < 2.0)"
            else:
                quality = f"poor (std up to {max_std:.1f})"
            kappa_data["noise_quality"] = quality
            kappa_data["target_ro_mm"] = [round(float(v), 2) for v in target_ro]
            print(f"  [KAPPA-COR] Noise quality: {quality}")

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    if fix_vals:
        print(f"  [CALIB-COR] Calibrated: {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [CALIB-COR] ray miss: median={np.median(miss_vals):.2f}mm")
    if cal_cutoff_time is not None:
        all_fix = [e.get("fixation_distance_mm") for e in calibrated_results]
        _report_cal_test_stats("CALIB-COR", all_fix, cal_idx, test_idx, known_distance_mm)

    conv_meta = {
        "method": "cor_corneal_3d_calibrated",
        "description": f"COR C3D calibrated to {known_distance_mm/10:.0f}cm "
                       f"(per-eye gaze bias / kappa correction)",
        "cc_estimation_method": "reflection_law",
        "cor_offset_mm": COR_OFFSET_MM,
        "calibration": {
            "known_distance_mm": known_distance_mm,
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
        },
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_cor_ro": [round(float(v), 4) for v in right_cor],
        "left_cor_lo": [round(float(v), 4) for v in left_cor],
        "right_corneal_center_ro": meta.get("right_corneal_center_ro"),
        "left_corneal_center_lo": meta.get("left_corneal_center_lo"),
        "kappa": kappa_data if kappa_data else None,
        "cal_test_split": {
            "cal_cutoff_time": cal_cutoff_time,
            "n_cal": n_cal,
            "n_test": n_test,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
            "kappa_right_deg": e.get("kappa_right_deg"),
            "kappa_left_deg": e.get("kappa_left_deg"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_corc3d_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CALIB-COR] Saved to {conv_path}")




def calibrate_scene_c3d_convergence(output_dir, calib, recording_dir=None,
                                     cal_cutoff_time=None, transform='poly',
                                     cc_mode='fixed'):
    """Calibrate per-eye gaze bias to minimize scene projection error.

    Unlike calibrate_reflect_c3d_convergence (which matches fixation distance),
    this method optimizes (dx_r, dy_r, dx_l, dy_l) to minimize the gaze-to-screen
    mapping residual. Each optimizer iteration:
      1. Applies trial bias -> recomputes convergence points (X,Y,Z)
      2. Groups frames by dot intervals -> median gaze_xy per dot
      3. Fits transform (gaze_xy -> screen_cm) on all cal dots
      4. Computes mean squared residual (training error)

    Args:
        transform: 'poly' for polynomial mapping, 'homography' for projective transform.

    Saves calibrated results to:
        convergence_meta_scenec3d_poly_cal.json  (transform='poly')
        convergence_meta_scenec3d_h_cal.json     (transform='homography')
    """
    import cv2
    from scipy.optimize import minimize as sp_minimize

    _cc_label = 'SL-' if cc_mode == 'sliding' else ''
    TAG = f"SCENE-{_cc_label}{'POLY' if transform == 'poly' else 'H'}"
    out_base = Path(output_dir)

    # ---- Load existing reflect c3d meta for CC positions ----
    meta_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_path.exists():
        print(f"  [{TAG}] No reflect c3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro") is None or meta.get("left_corneal_center_lo") is None:
        print(f"  [{TAG}] Missing corneal centers in reflect c3d meta")
        return

    right_cc = np.array(meta["right_corneal_center_ro"])
    left_cc = np.array(meta["left_corneal_center_lo"])

    cross = calib.get("cross")
    if cross is None:
        print(f"  [{TAG}] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # ---- Load dot timing from logs.json ----
    if recording_dir is None:
        print(f"  [{TAG}] recording_dir is required")
        return

    recording_dir = Path(recording_dir)
    logs_path = recording_dir / "logs.json"
    if not logs_path.exists():
        print(f"  [{TAG}] logs.json not found")
        return

    with open(logs_path) as f:
        logs = json.load(f)

    ws_msgs = logs.get("websocket_messages", [])

    # Find TaskStart timestamp
    task_start_ts = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        if parsed.get("eventType") == "TaskStart":
            task_start_ts = msg["system_unix_ts"]
            break
    if task_start_ts is None:
        print(f"  [{TAG}] TaskStart not found in logs")
        return

    # Extract VisualDot events (on + off) with screen positions
    dots = []
    current_dot = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        evt = parsed.get("eventType")
        ts = msg["system_unix_ts"]
        if evt == "VisualDot":
            det = parsed.get("details", {})
            mid = parsed.get("markerId", det.get("markerId"))
            current_dot = {
                "markerId": mid,
                "xCm": det.get("xCm"),
                "yCm": det.get("yCm"),
                "on_ts": ts - task_start_ts,
                "off_ts": None,
            }
            dots.append(current_dot)
        elif evt == "Event" and current_dot:
            det = parsed.get("details", {})
            if det.get("action") == "dot_off":
                current_dot["off_ts"] = ts - task_start_ts
                current_dot = None

    # Get first 9 unique dots
    seen_ids = set()
    first_dots = []
    for d in dots:
        if d["markerId"] not in seen_ids:
            seen_ids.add(d["markerId"])
            first_dots.append(d)
        if len(first_dots) == 9:
            break

    if len(first_dots) < 4:
        print(f"  [{TAG}] Only {len(first_dots)} dots found, need >= 4")
        return

    # ---- Load eye frame timestamps ----
    ri_dir = recording_dir / "ri"
    if not ri_dir.exists():
        ri_dir = Path(output_dir).parent / "ri"
    if not ri_dir.exists():
        print(f"  [{TAG}] Cannot find ri/ directory for eye frame timestamps")
        return

    eye_frames = []
    for p in sorted(ri_dir.glob("*.png")):
        m = re.search(r'timestamp_(\d+\.\d+)\.png$', p.name)
        if m:
            eye_frames.append({"name": p.name, "timestamp": float(m.group(1))})

    if not eye_frames:
        print(f"  [{TAG}] No eye frames found")
        return

    # ---- Load pupil_3d for both eyes ----
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print(f"  [{TAG}] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [{TAG}] Only {len(valid_frames)} valid frames, need >= 3")
        return

    # ---- Sliding CC setup (if cc_mode='sliding') ----
    _sl_r_cc_by_frame = _sl_l_cc_by_frame = None
    _sl_r_frames = _sl_l_frames = None
    _sl_r_global = _sl_l_global = None
    _sl_half_win = 25  # half of window_size=50
    if cc_mode == 'sliding':
        # Load reflection-law CC observations (NOT corneal3d — different CC method!)
        _sl_reflect_path = out_base / "convergence_meta_reflectc3d.json"
        if _sl_reflect_path.exists():
            with open(_sl_reflect_path) as _f:
                _sl_reflect_meta = json.load(_f)
            _sl_r_obs = _sl_reflect_meta.get("cc_observations_right")
            _sl_l_obs = _sl_reflect_meta.get("cc_observations_left")
        else:
            _sl_r_obs = _sl_l_obs = None
        if _sl_r_obs and _sl_l_obs:
            _sl_r_cc_by_frame = {o["frame"]: np.array(o["cc"]) for o in _sl_r_obs}
            _sl_l_cc_by_frame = {o["frame"]: np.array(o["cc"]) for o in _sl_l_obs}
            _sl_r_frames = sorted(_sl_r_cc_by_frame.keys())
            _sl_l_frames = sorted(_sl_l_cc_by_frame.keys())
            _sl_r_global = np.median(np.array([o["cc"] for o in _sl_r_obs]), axis=0)
            _sl_l_global = np.median(np.array([o["cc"] for o in _sl_l_obs]), axis=0)
            print(f"  [{TAG}] Sliding CC: {len(_sl_r_obs)} R obs, {len(_sl_l_obs)} L obs")
        else:
            print(f"  [{TAG}] CC observations not found, falling back to fixed CC")
            cc_mode = 'fixed'

    # Precompute raw gaze
    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        if cc_mode == 'sliding' and _sl_r_cc_by_frame is not None:
            _r_cc = _sliding_cc_for_frame(_sl_r_cc_by_frame, _sl_r_frames,
                                           _sl_r_global, frame_name, _sl_half_win, causal=True)
            _l_cc = _sliding_cc_for_frame(_sl_l_cc_by_frame, _sl_l_frames,
                                           _sl_l_global, frame_name, _sl_half_win, causal=True)
            _r_cc_proj = np.array([_r_cc[0] / _r_cc[2], _r_cc[1] / _r_cc[2]])
            _l_cc_proj = np.array([_l_cc[0] / _l_cc[2], _l_cc[1] / _l_cc[2]])
        else:
            _r_cc_proj = r_cc_proj
            _l_cc_proj = l_cc_proj
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - _r_cc_proj
        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - _l_cc_proj
        raw_gazes.append((r_gaze, l_gaze))

    frame_ts_map = {}
    for ef in eye_frames:
        frame_ts_map[ef["name"]] = ef["timestamp"]

    # ---- Convergence points for all frames given bias ----
    def _convergence_points(bias):
        dx_r, dy_r, dx_l, dy_l = bias
        result = {}
        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            r_gaze_raw, l_gaze_raw = raw_gazes[i]
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                result[frame_name] = [float(conv_pt[0]), float(conv_pt[1]), float(conv_pt[2])]
        return result

    # ---- Extract median gaze_xy per dot ----
    def _extract_dot_medians(conv_points):
        dot_medians = []
        for di, dot in enumerate(first_dots):
            if dot["on_ts"] is None or dot["off_ts"] is None:
                continue
            if dot["xCm"] is None or dot["yCm"] is None:
                continue
            margin = 0.15
            t_start = dot["on_ts"] + margin
            t_end = dot["off_ts"] - margin
            if t_end <= t_start:
                t_start = dot["on_ts"]
                t_end = dot["off_ts"]

            xy_list = []
            for ef in eye_frames:
                if t_start <= ef["timestamp"] <= t_end:
                    cp = conv_points.get(ef["name"])
                    if cp:
                        xy_list.append([cp[0], cp[1]])

            if len(xy_list) >= 3:
                xy_arr = np.array(xy_list)
                med_x = float(np.median(xy_arr[:, 0]))
                med_y = float(np.median(xy_arr[:, 1]))
                dot_medians.append({
                    "dot_idx": di,
                    "gaze_xy": [med_x, med_y],
                    "screen_cm": [dot["xCm"], dot["yCm"]],
                    "n_frames": len(xy_list),
                })
        return dot_medians

    # ---- Transform fitting functions ----
    def _fit_poly(gaze_pts, screen_pts):
        n = len(gaze_pts)
        if n < 4:
            return None
        g = np.array(gaze_pts, dtype=np.float64)
        s = np.array(screen_pts, dtype=np.float64)
        x, y = g[:, 0], g[:, 1]
        if n < 7:
            A = np.column_stack([np.ones(n), x, y])
        else:
            A = np.column_stack([np.ones(n), x, y, x * y, x**2, y**2])
        try:
            cx, _, _, _ = np.linalg.lstsq(A, s[:, 0], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, s[:, 1], rcond=None)
        except np.linalg.LinAlgError:
            return None
        return cx, cy

    def _apply_poly(coeffs_x, coeffs_y, pt):
        gx, gy = pt
        n = len(coeffs_x)
        if n == 3:
            terms = [1.0, gx, gy]
        elif n == 6:
            terms = [1.0, gx, gy, gx * gy, gx**2, gy**2]
        else:
            return None
        sx = sum(c * t for c, t in zip(coeffs_x, terms))
        sy = sum(c * t for c, t in zip(coeffs_y, terms))
        return [float(sx), float(sy)]

    def _fit_homography(gaze_pts, screen_pts):
        g = np.array(gaze_pts, dtype=np.float64)
        s = np.array(screen_pts, dtype=np.float64)
        if len(g) < 4:
            return None
        H, _ = cv2.findHomography(g, s, cv2.RANSAC, 3.0)
        return H

    def _apply_homography(H, pt):
        x, y = pt
        w = H[2][0]*x + H[2][1]*y + H[2][2]
        if abs(w) < 1e-10:
            return None
        return [float((H[0][0]*x + H[0][1]*y + H[0][2]) / w),
                float((H[1][0]*x + H[1][1]*y + H[1][2]) / w)]

    # ---- Objective function ----
    def objective(bias):
        conv_points = _convergence_points(bias)
        if len(conv_points) < 10:
            return 1e10

        dot_medians = _extract_dot_medians(conv_points)
        if len(dot_medians) < 4:
            return 1e10

        gaze_pts = [d["gaze_xy"] for d in dot_medians]
        screen_pts = [d["screen_cm"] for d in dot_medians]

        if transform == 'poly':
            poly = _fit_poly(gaze_pts, screen_pts)
            if poly is None:
                return 1e10
            cx, cy = poly
            residuals_sq = []
            for d in dot_medians:
                pred = _apply_poly(cx, cy, d["gaze_xy"])
                if pred is None:
                    continue
                dx = pred[0] - d["screen_cm"][0]
                dy = pred[1] - d["screen_cm"][1]
                residuals_sq.append(dx**2 + dy**2)
        else:
            H = _fit_homography(gaze_pts, screen_pts)
            if H is None:
                return 1e10
            residuals_sq = []
            for d in dot_medians:
                pred = _apply_homography(H, d["gaze_xy"])
                if pred is None:
                    continue
                dx = pred[0] - d["screen_cm"][0]
                dy = pred[1] - d["screen_cm"][1]
                residuals_sq.append(dx**2 + dy**2)

        if not residuals_sq:
            return 1e10

        mse = np.mean(residuals_sq)

        # Regularisation
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2

        return mse + reg + sym_h + sym_v

    print(f"  [{TAG}] {len(valid_frames)} valid frames, "
          f"{len(first_dots)} cal dots, transform={transform}")

    # Run optimizer
    result = sp_minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-8})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [{TAG}] Optimized gaze bias:")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")
    print(f"  [{TAG}] Final MSE: {result.fun:.6f} cm²")

    # ---- Recompute all convergence with calibrated bias ----
    final_conv_points = _convergence_points(result.x)
    final_dot_medians = _extract_dot_medians(final_conv_points)

    # Compute LOOCV errors (using the same transform type)
    loocv_errors = []
    if len(final_dot_medians) >= 4:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        for fold in range(len(final_dot_medians)):
            g_train = [gaze_pts_all[j] for j in range(len(gaze_pts_all)) if j != fold]
            s_train = [screen_pts_all[j] for j in range(len(screen_pts_all)) if j != fold]

            if transform == 'poly':
                model = _fit_poly(g_train, s_train)
                if model is None:
                    continue
                cx, cy = model
                pred = _apply_poly(cx, cy, gaze_pts_all[fold])
            else:
                model = _fit_homography(g_train, s_train)
                if model is None:
                    continue
                pred = _apply_homography(model, gaze_pts_all[fold])

            if pred is None:
                continue
            true = screen_pts_all[fold]
            err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
            loocv_errors.append({
                "dot_idx": final_dot_medians[fold]["dot_idx"],
                "error_cm": round(err, 2),
                "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                "true_cm": [round(true[0], 2), round(true[1], 2)],
            })
        if loocv_errors:
            mean_loocv = np.mean([e["error_cm"] for e in loocv_errors])
            print(f"  [{TAG}] LOOCV (9 dots): mean={mean_loocv:.2f} cm, "
                  f"errors={[e['error_cm'] for e in loocv_errors]}")

    # Build per-frame results
    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])

    calibrated_results = []
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        cp = final_conv_points.get(frame_name)
        entry = {
            "frame": frame_name,
            "convergence_mm": None,
            "fixation_distance_mm": None,
            "convergence_point": None,
            "ray_miss_mm": None,
            "ipd_mm": None,
            "is_calibration": i in cal_idx,
        }
        if cp:
            cam_mid = lo_origin_ro / 2.0
            fix_dist = float(np.linalg.norm(np.array(cp) - cam_mid))
            entry["fixation_distance_mm"] = round(fix_dist, 2)
            entry["convergence_mm"] = round(fix_dist, 2)
            entry["convergence_point"] = [round(cp[0], 2), round(cp[1], 2), round(cp[2], 2)]

            # Ray miss
            r_gaze_raw, l_gaze_raw = raw_gazes[i]
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])
            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)
            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) > 1e-10:
                sc = (b_val * e_val - c * d_val) / denom
                tc = (a * e_val - b_val * d_val) / denom
                if sc > 0 and tc > 0:
                    closest_r = sc * r_dir
                    closest_l = lo_origin_ro + tc * l_dir_ro
                    ray_miss = float(np.linalg.norm(closest_r - closest_l))
                    entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)

        calibrated_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]

    if fix_vals:
        print(f"  [{TAG}] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [{TAG}] ray miss: median={np.median(miss_vals):.2f}mm")

    # Rep2+ evaluation
    marker_counts = {}
    all_dots_with_rep = []
    for d in dots:
        mid = d["markerId"]
        marker_counts[mid] = marker_counts.get(mid, 0) + 1
        d_copy = dict(d)
        d_copy["repetition"] = marker_counts[mid]
        all_dots_with_rep.append(d_copy)
    rep2_only = [d for d in all_dots_with_rep if d.get("repetition", 1) > 1]

    rep2_eval = []
    if rep2_only and len(final_dot_medians) >= 4:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]

        if transform == 'poly':
            model = _fit_poly(gaze_pts_all, screen_pts_all)
        else:
            model = _fit_homography(gaze_pts_all, screen_pts_all)

        if model is not None:
            for dot in rep2_only:
                if dot["on_ts"] is None or dot["off_ts"] is None:
                    continue
                if dot["xCm"] is None or dot["yCm"] is None:
                    continue
                margin = 0.15
                t_start = dot["on_ts"] + margin
                t_end = dot["off_ts"] - margin
                if t_end <= t_start:
                    t_start = dot["on_ts"]
                    t_end = dot["off_ts"]
                xy_list = []
                for ef in eye_frames:
                    if t_start <= ef["timestamp"] <= t_end:
                        cp = final_conv_points.get(ef["name"])
                        if cp:
                            xy_list.append([cp[0], cp[1]])
                if len(xy_list) >= 3:
                    xy_arr = np.array(xy_list)
                    med_x = float(np.median(xy_arr[:, 0]))
                    med_y = float(np.median(xy_arr[:, 1]))
                    if transform == 'poly':
                        cx, cy = model
                        pred = _apply_poly(cx, cy, [med_x, med_y])
                    else:
                        pred = _apply_homography(model, [med_x, med_y])
                    if pred:
                        true = [dot["xCm"], dot["yCm"]]
                        err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
                        rep2_eval.append({
                            "markerId": dot["markerId"],
                            "repetition": dot["repetition"],
                            "error_cm": round(err, 2),
                            "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                            "true_cm": [round(true[0], 2), round(true[1], 2)],
                        })
            if rep2_eval:
                mean_rep2 = np.mean([e["error_cm"] for e in rep2_eval])
                print(f"  [{TAG}] Rep2+ ({len(rep2_eval)} dots): "
                      f"mean error={mean_rep2:.2f} cm")

    # ---- Save JSON ----
    suffix = "poly" if transform == "poly" else "h"
    _fname_prefix = "scene_slidingcc" if cc_mode == 'sliding' else "scenec3d"
    conv_meta = {
        "method": f"{_fname_prefix}_{suffix}_calibrated",
        "description": f"{'Sliding CC' if cc_mode == 'sliding' else 'Reflect C3D'} "
                       f"calibrated by minimizing scene {transform} mapping residual",
        "cc_estimation_method": "sliding_window" if cc_mode == 'sliding' else "reflection_law",
        "transform_type": transform,
        "calibration": {
            "objective": f"scene_{transform}_mse",
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
            "final_mse_cm2": round(float(result.fun), 6),
            "n_cal_dots": len(final_dot_medians),
        },
        "loocv": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in loocv_errors])), 2) if loocv_errors else None,
            "errors": loocv_errors,
        } if loocv_errors else None,
        "rep2_eval": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in rep2_eval])), 2) if rep2_eval else None,
            "n_dots": len(rep2_eval),
            "errors": rep2_eval,
        } if rep2_eval else None,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta["right_corneal_center_ro"],
        "left_corneal_center_lo": meta["left_corneal_center_lo"],
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "is_calibration": e.get("is_calibration"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / f"convergence_meta_{_fname_prefix}_{suffix}_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [{TAG}] Saved to {conv_path}")




def calibrate_joint_convergence(output_dir, calib, recording_dir=None,
                                 known_distance_mm=500.0,
                                 cal_cutoff_time=None, fair_cc=False,
                                 transform='poly', lambda_dist=1.0):
    """Joint distance + gaze calibration: optimizes (dx_r, dy_r, dx_l, dy_l)
    to simultaneously minimize gaze-to-screen mapping residual (XY) and
    fixation distance error (Z).

    Each calibration dot provides 3 constraints: X,Y on screen + Z distance.
    This gives 50% more calibration signal than gaze-only or distance-only.

    Args:
        lambda_dist: Weight for distance loss relative to gaze loss. Default 1.0.

    Saves to convergence_meta_joint_cal.json.
    """
    import cv2
    from scipy.optimize import minimize as sp_minimize

    TAG = "JOINT-D+G"
    out_base = Path(output_dir)

    # ---- Load existing reflect c3d meta for CC positions ----
    meta_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_path.exists():
        print(f"  [{TAG}] No reflect c3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro") is None or meta.get("left_corneal_center_lo") is None:
        print(f"  [{TAG}] Missing corneal centers in reflect c3d meta")
        return

    right_cc = np.array(meta["right_corneal_center_ro"])
    left_cc = np.array(meta["left_corneal_center_lo"])

    cross = calib.get("cross")
    if cross is None:
        print(f"  [{TAG}] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # ---- Load dot timing from logs.json ----
    if recording_dir is None:
        print(f"  [{TAG}] recording_dir is required")
        return

    recording_dir = Path(recording_dir)
    logs_path = recording_dir / "logs.json"
    if not logs_path.exists():
        print(f"  [{TAG}] logs.json not found")
        return

    with open(logs_path) as f:
        logs = json.load(f)

    ws_msgs = logs.get("websocket_messages", [])

    # Find TaskStart timestamp
    task_start_ts = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        if parsed.get("eventType") == "TaskStart":
            task_start_ts = msg["system_unix_ts"]
            break
    if task_start_ts is None:
        print(f"  [{TAG}] TaskStart not found in logs")
        return

    # Extract VisualDot events (on + off) with screen positions
    dots = []
    current_dot = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        evt = parsed.get("eventType")
        ts = msg["system_unix_ts"]
        if evt == "VisualDot":
            det = parsed.get("details", {})
            mid = parsed.get("markerId", det.get("markerId"))
            current_dot = {
                "markerId": mid,
                "xCm": det.get("xCm"),
                "yCm": det.get("yCm"),
                "on_ts": ts - task_start_ts,
                "off_ts": None,
            }
            dots.append(current_dot)
        elif evt == "Event" and current_dot:
            det = parsed.get("details", {})
            if det.get("action") == "dot_off":
                current_dot["off_ts"] = ts - task_start_ts
                current_dot = None

    # Get first 9 unique dots
    seen_ids = set()
    first_dots = []
    for d in dots:
        if d["markerId"] not in seen_ids:
            seen_ids.add(d["markerId"])
            first_dots.append(d)
        if len(first_dots) == 9:
            break

    if len(first_dots) < 4:
        print(f"  [{TAG}] Only {len(first_dots)} dots found, need >= 4")
        return

    # ---- Load eye frame timestamps ----
    ri_dir = recording_dir / "ri"
    if not ri_dir.exists():
        ri_dir = Path(output_dir).parent / "ri"
    if not ri_dir.exists():
        print(f"  [{TAG}] Cannot find ri/ directory for eye frame timestamps")
        return

    eye_frames = []
    for p in sorted(ri_dir.glob("*.png")):
        m = re.search(r'timestamp_(\d+\.\d+)\.png$', p.name)
        if m:
            eye_frames.append({"name": p.name, "timestamp": float(m.group(1))})

    if not eye_frames:
        print(f"  [{TAG}] No eye frames found")
        return

    # ---- Load pupil_3d for both eyes ----
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print(f"  [{TAG}] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [{TAG}] Only {len(valid_frames)} valid frames, need >= 3")
        return

    # Precompute raw gaze
    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - r_cc_proj
        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - l_cc_proj
        raw_gazes.append((r_gaze, l_gaze))

    frame_ts_map = {}
    for ef in eye_frames:
        frame_ts_map[ef["name"]] = ef["timestamp"]

    # ---- Convergence points for all frames given bias ----
    def _convergence_points(bias):
        dx_r, dy_r, dx_l, dy_l = bias
        result = {}
        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            r_gaze_raw, l_gaze_raw = raw_gazes[i]
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                result[frame_name] = [float(conv_pt[0]), float(conv_pt[1]), float(conv_pt[2])]
        return result

    # ---- Extract median gaze_xy AND gaze_z per dot ----
    def _extract_dot_medians_with_z(conv_points):
        dot_medians = []
        for di, dot in enumerate(first_dots):
            if dot["on_ts"] is None or dot["off_ts"] is None:
                continue
            if dot["xCm"] is None or dot["yCm"] is None:
                continue
            margin = 0.15
            t_start = dot["on_ts"] + margin
            t_end = dot["off_ts"] - margin
            if t_end <= t_start:
                t_start = dot["on_ts"]
                t_end = dot["off_ts"]

            xy_list = []
            z_list = []
            for ef in eye_frames:
                if t_start <= ef["timestamp"] <= t_end:
                    cp = conv_points.get(ef["name"])
                    if cp:
                        xy_list.append([cp[0], cp[1]])
                        z_list.append(cp[2])

            if len(xy_list) >= 3:
                xy_arr = np.array(xy_list)
                med_x = float(np.median(xy_arr[:, 0]))
                med_y = float(np.median(xy_arr[:, 1]))
                med_z = float(np.median(z_list))
                dot_medians.append({
                    "dot_idx": di,
                    "gaze_xy": [med_x, med_y],
                    "gaze_z": med_z,
                    "screen_cm": [dot["xCm"], dot["yCm"]],
                    "n_frames": len(xy_list),
                })
        return dot_medians

    # ---- Transform fitting functions ----
    def _fit_poly(gaze_pts, screen_pts):
        n = len(gaze_pts)
        if n < 4:
            return None
        g = np.array(gaze_pts, dtype=np.float64)
        s = np.array(screen_pts, dtype=np.float64)
        x, y = g[:, 0], g[:, 1]
        if n < 7:
            A = np.column_stack([np.ones(n), x, y])
        else:
            A = np.column_stack([np.ones(n), x, y, x * y, x**2, y**2])
        try:
            cx, _, _, _ = np.linalg.lstsq(A, s[:, 0], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, s[:, 1], rcond=None)
        except np.linalg.LinAlgError:
            return None
        return cx, cy

    def _apply_poly(coeffs_x, coeffs_y, pt):
        gx, gy = pt
        n = len(coeffs_x)
        if n == 3:
            terms = [1.0, gx, gy]
        elif n == 6:
            terms = [1.0, gx, gy, gx * gy, gx**2, gy**2]
        else:
            return None
        sx = sum(c * t for c, t in zip(coeffs_x, terms))
        sy = sum(c * t for c, t in zip(coeffs_y, terms))
        return [float(sx), float(sy)]

    # ---- Joint objective function ----
    def objective(bias):
        conv_points = _convergence_points(bias)
        if len(conv_points) < 10:
            return 1e10

        dot_medians = _extract_dot_medians_with_z(conv_points)
        if len(dot_medians) < 4:
            return 1e10

        gaze_pts = [d["gaze_xy"] for d in dot_medians]
        screen_pts = [d["screen_cm"] for d in dot_medians]

        # ── XY: gaze-to-screen mapping residual ──
        poly = _fit_poly(gaze_pts, screen_pts)
        if poly is None:
            return 1e10
        cx, cy = poly
        gaze_residuals = []
        for d in dot_medians:
            pred = _apply_poly(cx, cy, d["gaze_xy"])
            if pred is None:
                continue
            gaze_residuals.append(
                (pred[0] - d["screen_cm"][0])**2 + (pred[1] - d["screen_cm"][1])**2)
        gaze_mse = np.mean(gaze_residuals) if gaze_residuals else 1e10  # cm²

        # ── Z: distance accuracy ──
        cal_z = [d["gaze_z"] for d in dot_medians]  # median Z per dot
        median_z = np.median(cal_z)
        dist_error = (median_z - known_distance_mm) ** 2
        # Also penalize Z variance across dots (all at same distance)
        z_variance = np.var(cal_z)
        # Normalize: 1cm² gaze ≈ 100mm² distance → divide by 100
        dist_loss = (dist_error + 0.01 * z_variance) / 100.0

        # ── Regularization (same as existing) ──
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2

        return gaze_mse + lambda_dist * dist_loss + reg + sym_h + sym_v

    print(f"  [{TAG}] {len(valid_frames)} valid frames, "
          f"{len(first_dots)} cal dots, transform={transform}, lambda_dist={lambda_dist}")

    # Run optimizer
    result = sp_minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-8})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [{TAG}] Optimized gaze bias:")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")
    print(f"  [{TAG}] Final objective: {result.fun:.6f}")

    # ---- Recompute all convergence with calibrated bias ----
    final_conv_points = _convergence_points(result.x)
    final_dot_medians = _extract_dot_medians_with_z(final_conv_points)

    # Report XY and Z components separately
    if final_dot_medians:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        poly_final = _fit_poly(gaze_pts_all, screen_pts_all)
        if poly_final:
            cx_f, cy_f = poly_final
            gaze_errs = []
            for d in final_dot_medians:
                pred = _apply_poly(cx_f, cy_f, d["gaze_xy"])
                if pred:
                    err = float(np.sqrt((pred[0] - d["screen_cm"][0])**2 +
                                        (pred[1] - d["screen_cm"][1])**2))
                    gaze_errs.append(err)
            if gaze_errs:
                print(f"  [{TAG}] Gaze XY: mean_err={np.mean(gaze_errs):.2f} cm, "
                      f"MSE={np.mean([e**2 for e in gaze_errs]):.4f} cm²")

        cal_z_vals = [d["gaze_z"] for d in final_dot_medians]
        median_final_z = float(np.median(cal_z_vals))
        z_err = abs(median_final_z - known_distance_mm)
        print(f"  [{TAG}] Distance Z: median={median_final_z:.1f}mm, "
              f"error={z_err:.1f}mm, target={known_distance_mm:.0f}mm")

    # Compute LOOCV errors
    loocv_errors = []
    if len(final_dot_medians) >= 4 and poly_final:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        for fold in range(len(final_dot_medians)):
            g_train = [gaze_pts_all[j] for j in range(len(gaze_pts_all)) if j != fold]
            s_train = [screen_pts_all[j] for j in range(len(screen_pts_all)) if j != fold]

            model = _fit_poly(g_train, s_train)
            if model is None:
                continue
            cx, cy = model
            pred = _apply_poly(cx, cy, gaze_pts_all[fold])

            if pred is None:
                continue
            true = screen_pts_all[fold]
            err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
            loocv_errors.append({
                "dot_idx": final_dot_medians[fold]["dot_idx"],
                "error_cm": round(err, 2),
                "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                "true_cm": [round(true[0], 2), round(true[1], 2)],
            })
        if loocv_errors:
            mean_loocv = np.mean([e["error_cm"] for e in loocv_errors])
            print(f"  [{TAG}] LOOCV (9 dots): mean={mean_loocv:.2f} cm, "
                  f"errors={[e['error_cm'] for e in loocv_errors]}")

    # Build per-frame results
    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])

    calibrated_results = []
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        cp = final_conv_points.get(frame_name)
        entry = {
            "frame": frame_name,
            "convergence_mm": None,
            "fixation_distance_mm": None,
            "convergence_point": None,
            "ray_miss_mm": None,
            "ipd_mm": None,
            "is_calibration": i in cal_idx,
        }
        if cp:
            cam_mid = lo_origin_ro / 2.0
            fix_dist = float(np.linalg.norm(np.array(cp) - cam_mid))
            entry["fixation_distance_mm"] = round(fix_dist, 2)
            entry["convergence_mm"] = round(fix_dist, 2)
            entry["convergence_point"] = [round(cp[0], 2), round(cp[1], 2), round(cp[2], 2)]

            # Ray miss
            r_gaze_raw, l_gaze_raw = raw_gazes[i]
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])
            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)
            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) > 1e-10:
                sc = (b_val * e_val - c * d_val) / denom
                tc = (a * e_val - b_val * d_val) / denom
                if sc > 0 and tc > 0:
                    closest_r = sc * r_dir
                    closest_l = lo_origin_ro + tc * l_dir_ro
                    ray_miss = float(np.linalg.norm(closest_r - closest_l))
                    entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)

        calibrated_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]

    if fix_vals:
        print(f"  [{TAG}] {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [{TAG}] ray miss: median={np.median(miss_vals):.2f}mm")

    # Rep2+ evaluation
    marker_counts = {}
    all_dots_with_rep = []
    for d in dots:
        mid = d["markerId"]
        marker_counts[mid] = marker_counts.get(mid, 0) + 1
        d_copy = dict(d)
        d_copy["repetition"] = marker_counts[mid]
        all_dots_with_rep.append(d_copy)
    rep2_only = [d for d in all_dots_with_rep if d.get("repetition", 1) > 1]

    rep2_eval = []
    if rep2_only and len(final_dot_medians) >= 4:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]

        model = _fit_poly(gaze_pts_all, screen_pts_all)

        if model is not None:
            for dot in rep2_only:
                if dot["on_ts"] is None or dot["off_ts"] is None:
                    continue
                if dot["xCm"] is None or dot["yCm"] is None:
                    continue
                margin = 0.15
                t_start = dot["on_ts"] + margin
                t_end = dot["off_ts"] - margin
                if t_end <= t_start:
                    t_start = dot["on_ts"]
                    t_end = dot["off_ts"]
                xy_list = []
                for ef in eye_frames:
                    if t_start <= ef["timestamp"] <= t_end:
                        cp = final_conv_points.get(ef["name"])
                        if cp:
                            xy_list.append([cp[0], cp[1]])
                if len(xy_list) >= 3:
                    xy_arr = np.array(xy_list)
                    med_x = float(np.median(xy_arr[:, 0]))
                    med_y = float(np.median(xy_arr[:, 1]))
                    cx, cy = model
                    pred = _apply_poly(cx, cy, [med_x, med_y])
                    if pred:
                        true = [dot["xCm"], dot["yCm"]]
                        err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
                        rep2_eval.append({
                            "markerId": dot["markerId"],
                            "repetition": dot["repetition"],
                            "error_cm": round(err, 2),
                            "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                            "true_cm": [round(true[0], 2), round(true[1], 2)],
                        })
            if rep2_eval:
                mean_rep2 = np.mean([e["error_cm"] for e in rep2_eval])
                print(f"  [{TAG}] Rep2+ ({len(rep2_eval)} dots): "
                      f"mean error={mean_rep2:.2f} cm")

    # ---- Save JSON ----
    # Compute final gaze MSE and distance error for metadata
    final_gaze_mse = None
    final_dist_err = None
    final_median_z = None
    if final_dot_medians:
        gaze_pts_all = [d["gaze_xy"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        poly_final = _fit_poly(gaze_pts_all, screen_pts_all)
        if poly_final:
            cx_f, cy_f = poly_final
            resid_sq = []
            for d in final_dot_medians:
                pred = _apply_poly(cx_f, cy_f, d["gaze_xy"])
                if pred:
                    resid_sq.append((pred[0] - d["screen_cm"][0])**2 +
                                    (pred[1] - d["screen_cm"][1])**2)
            if resid_sq:
                final_gaze_mse = float(np.mean(resid_sq))
        cal_z_vals = [d["gaze_z"] for d in final_dot_medians]
        final_median_z = float(np.median(cal_z_vals))
        final_dist_err = abs(final_median_z - known_distance_mm)

    conv_meta = {
        "method": "joint_distance_gaze",
        "description": "Joint distance + gaze calibration (XY screen + Z distance)",
        "cc_estimation_method": "reflection_law",
        "transform_type": transform,
        "lambda_dist": lambda_dist,
        "calibration": {
            "objective": "joint_gaze_mse_plus_distance",
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
            "final_objective": round(float(result.fun), 6),
            "n_cal_dots": len(final_dot_medians),
            "known_distance_mm": known_distance_mm,
        },
        "gaze_mse_cm2": round(final_gaze_mse, 6) if final_gaze_mse is not None else None,
        "dist_error_mm": round(final_dist_err, 1) if final_dist_err is not None else None,
        "median_fixation_mm": round(final_median_z, 2) if final_median_z is not None else None,
        "loocv": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in loocv_errors])), 2) if loocv_errors else None,
            "errors": loocv_errors,
        } if loocv_errors else None,
        "rep2_eval": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in rep2_eval])), 2) if rep2_eval else None,
            "n_dots": len(rep2_eval),
            "errors": rep2_eval,
        } if rep2_eval else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta["right_corneal_center_ro"],
        "left_corneal_center_lo": meta["left_corneal_center_lo"],
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "is_calibration": e.get("is_calibration"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_joint_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [{TAG}] Saved to {conv_path}")

def calibrate_pupil_glint_poly(output_dir, calib, recording_dir=None,
                                cal_cutoff_time=None, cc_mode='fixed'):
    """Calibrate gaze by mapping pupil-glint vectors directly to screen position.

    Instead of computing convergence (ray intersection) and mapping that to screen,
    this method directly maps the 4D pupil-CC gaze vector [r_gx, r_gy, l_gx, l_gy]
    to screen coordinates via a polynomial.

    The pupil-glint vector is inherently slip-invariant because CC moves with
    the glasses when they slip.

    Optimizes 4 bias params [dx_r, dy_r, dx_l, dy_l] to minimize the polynomial
    mapping residual.

    Saves to convergence_meta_pg_poly_cal.json.
    """
    import cv2
    from scipy.optimize import minimize as sp_minimize

    TAG = "PG-POLY"
    out_base = Path(output_dir)

    # ---- Load CC ----
    meta_path = out_base / "convergence_meta_reflectc3d.json"
    if not meta_path.exists():
        print(f"  [{TAG}] No reflect c3d results, skipping")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro") is None or meta.get("left_corneal_center_lo") is None:
        print(f"  [{TAG}] Missing corneal centers")
        return

    right_cc = np.array(meta["right_corneal_center_ro"])
    left_cc = np.array(meta["left_corneal_center_lo"])

    cross = calib.get("cross")
    if cross is None:
        print(f"  [{TAG}] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # ---- Load dot timing ----
    if recording_dir is None:
        print(f"  [{TAG}] recording_dir required")
        return

    recording_dir = Path(recording_dir)
    logs_path = recording_dir / "logs.json"
    if not logs_path.exists():
        print(f"  [{TAG}] logs.json not found")
        return

    with open(logs_path) as f:
        logs = json.load(f)

    ws_msgs = logs.get("websocket_messages", [])

    task_start_ts = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        if parsed.get("eventType") == "TaskStart":
            task_start_ts = msg["system_unix_ts"]
            break
    if task_start_ts is None:
        print(f"  [{TAG}] TaskStart not found")
        return

    dots = []
    current_dot = None
    for msg in ws_msgs:
        parsed = json.loads(msg["ws_message"])
        evt = parsed.get("eventType")
        ts = msg["system_unix_ts"]
        if evt == "VisualDot":
            det = parsed.get("details", {})
            mid = parsed.get("markerId", det.get("markerId"))
            current_dot = {
                "markerId": mid,
                "xCm": det.get("xCm"),
                "yCm": det.get("yCm"),
                "on_ts": ts - task_start_ts,
                "off_ts": None,
            }
            dots.append(current_dot)
        elif evt == "Event" and current_dot:
            det = parsed.get("details", {})
            if det.get("action") == "dot_off":
                current_dot["off_ts"] = ts - task_start_ts
                current_dot = None

    seen_ids = set()
    first_dots = []
    for d in dots:
        if d["markerId"] not in seen_ids:
            seen_ids.add(d["markerId"])
            first_dots.append(d)
        if len(first_dots) == 9:
            break

    if len(first_dots) < 4:
        print(f"  [{TAG}] Only {len(first_dots)} dots, need >= 4")
        return

    # ---- Load eye frame timestamps ----
    ri_dir = recording_dir / "ri"
    if not ri_dir.exists():
        ri_dir = Path(output_dir).parent / "ri"
    if not ri_dir.exists():
        print(f"  [{TAG}] ri/ directory not found")
        return

    eye_frames = []
    for p in sorted(ri_dir.glob("*.png")):
        m = re.search(r'timestamp_(\d+\.\d+)\.png$', p.name)
        if m:
            eye_frames.append({"name": p.name, "timestamp": float(m.group(1))})

    if not eye_frames:
        print(f"  [{TAG}] No eye frames found")
        return

    # ---- Load pupil_3d ----
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print(f"  [{TAG}] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [{TAG}] Only {len(valid_frames)} valid frames")
        return

    # ---- Sliding CC setup ----
    _sl_r_cc_by_frame = _sl_l_cc_by_frame = None
    _sl_r_frames = _sl_l_frames = None
    _sl_r_global = _sl_l_global = None
    _sl_half_win = 25
    if cc_mode == 'sliding':
        # Load reflection-law CC observations (NOT corneal3d — different CC method!)
        _sl_reflect_path = out_base / "convergence_meta_reflectc3d.json"
        if _sl_reflect_path.exists():
            with open(_sl_reflect_path) as _f:
                _sl_reflect_meta = json.load(_f)
            _sl_r_obs = _sl_reflect_meta.get("cc_observations_right")
            _sl_l_obs = _sl_reflect_meta.get("cc_observations_left")
        else:
            _sl_r_obs = _sl_l_obs = None
        if _sl_r_obs and _sl_l_obs:
            _sl_r_cc_by_frame = {o["frame"]: np.array(o["cc"]) for o in _sl_r_obs}
            _sl_l_cc_by_frame = {o["frame"]: np.array(o["cc"]) for o in _sl_l_obs}
            _sl_r_frames = sorted(_sl_r_cc_by_frame.keys())
            _sl_l_frames = sorted(_sl_l_cc_by_frame.keys())
            _sl_r_global = np.median(np.array([o["cc"] for o in _sl_r_obs]), axis=0)
            _sl_l_global = np.median(np.array([o["cc"] for o in _sl_l_obs]), axis=0)
            print(f"  [{TAG}] Sliding CC: {len(_sl_r_obs)} R obs, {len(_sl_l_obs)} L obs")
        else:
            print(f"  [{TAG}] CC observations not found, using fixed CC")
            cc_mode = 'fixed'

    # Precompute raw gaze features (4D: r_gx, r_gy, l_gx, l_gy)
    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_features = []
    for frame_name, rp, lp_lo in valid_frames:
        if cc_mode == 'sliding' and _sl_r_cc_by_frame is not None:
            _r_cc = _sliding_cc_for_frame(_sl_r_cc_by_frame, _sl_r_frames,
                                           _sl_r_global, frame_name, _sl_half_win, causal=True)
            _l_cc = _sliding_cc_for_frame(_sl_l_cc_by_frame, _sl_l_frames,
                                           _sl_l_global, frame_name, _sl_half_win, causal=True)
            _r_cc_p = np.array([_r_cc[0] / _r_cc[2], _r_cc[1] / _r_cc[2]])
            _l_cc_p = np.array([_l_cc[0] / _l_cc[2], _l_cc[1] / _l_cc[2]])
        else:
            _r_cc_p = r_cc_proj
            _l_cc_p = l_cc_proj
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - _r_cc_p
        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - _l_cc_p
        raw_features.append(np.array([r_gaze[0], r_gaze[1], l_gaze[0], l_gaze[1]]))

    frame_ts_map = {ef["name"]: ef["timestamp"] for ef in eye_frames}

    # ---- Biased features for all frames given bias ----
    def _biased_features(bias):
        dx_r, dy_r, dx_l, dy_l = bias
        bias_vec = np.array([dx_r, dy_r, dx_l, dy_l])
        result = {}
        for i, (frame_name, _, _) in enumerate(valid_frames):
            feat = raw_features[i] - bias_vec
            result[frame_name] = feat
        return result

    # ---- Extract median feature per dot ----
    def _extract_dot_medians(feat_map):
        dot_medians = []
        for di, dot in enumerate(first_dots):
            if dot["on_ts"] is None or dot["off_ts"] is None:
                continue
            if dot["xCm"] is None or dot["yCm"] is None:
                continue
            margin = 0.15
            t_start = dot["on_ts"] + margin
            t_end = dot["off_ts"] - margin
            if t_end <= t_start:
                t_start = dot["on_ts"]
                t_end = dot["off_ts"]
            feat_list = []
            for ef in eye_frames:
                if t_start <= ef["timestamp"] <= t_end:
                    f = feat_map.get(ef["name"])
                    if f is not None:
                        feat_list.append(f)
            if len(feat_list) >= 3:
                feat_arr = np.array(feat_list)
                med_feat = np.median(feat_arr, axis=0)
                dot_medians.append({
                    "dot_idx": di,
                    "feature": med_feat.tolist(),
                    "screen_cm": [dot["xCm"], dot["yCm"]],
                    "n_frames": len(feat_list),
                })
        return dot_medians

    # ---- 4D polynomial fit (1st order + intercept: 5 terms) ----
    def _fit_poly4d(feat_pts, screen_pts):
        n = len(feat_pts)
        if n < 5:
            return None
        F = np.array(feat_pts, dtype=np.float64)
        S = np.array(screen_pts, dtype=np.float64)
        # Design matrix: [1, r_gx, r_gy, l_gx, l_gy]
        A = np.column_stack([np.ones(n), F])
        try:
            cx, _, _, _ = np.linalg.lstsq(A, S[:, 0], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, S[:, 1], rcond=None)
        except np.linalg.LinAlgError:
            return None
        return cx, cy

    def _apply_poly4d(cx, cy, feat):
        terms = np.array([1.0, feat[0], feat[1], feat[2], feat[3]])
        sx = float(np.dot(cx, terms))
        sy = float(np.dot(cy, terms))
        return [sx, sy]

    # ---- Objective function ----
    def objective(bias):
        feat_map = _biased_features(bias)
        dot_medians = _extract_dot_medians(feat_map)
        if len(dot_medians) < 5:
            return 1e10

        feat_pts = [d["feature"] for d in dot_medians]
        screen_pts = [d["screen_cm"] for d in dot_medians]

        poly = _fit_poly4d(feat_pts, screen_pts)
        if poly is None:
            return 1e10
        cx, cy = poly

        residuals_sq = []
        for d in dot_medians:
            pred = _apply_poly4d(cx, cy, d["feature"])
            dx = pred[0] - d["screen_cm"][0]
            dy = pred[1] - d["screen_cm"][1]
            residuals_sq.append(dx**2 + dy**2)

        if not residuals_sq:
            return 1e10

        mse = np.mean(residuals_sq)
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2
        return mse + reg + sym_h + sym_v

    print(f"  [{TAG}] {len(valid_frames)} valid frames, {len(first_dots)} cal dots, "
          f"cc_mode={cc_mode}")

    # Run optimizer
    result = sp_minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-8})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [{TAG}] Optimized bias: R=({dx_r:.6f}, {dy_r:.6f}), L=({dx_l:.6f}, {dy_l:.6f})")
    print(f"  [{TAG}] Final MSE: {result.fun:.6f} cm^2")

    # ---- Final evaluation ----
    final_feat_map = _biased_features(result.x)
    final_dot_medians = _extract_dot_medians(final_feat_map)

    # LOOCV
    loocv_errors = []
    if len(final_dot_medians) >= 5:
        feat_pts_all = [d["feature"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        for fold in range(len(final_dot_medians)):
            f_train = [feat_pts_all[j] for j in range(len(feat_pts_all)) if j != fold]
            s_train = [screen_pts_all[j] for j in range(len(screen_pts_all)) if j != fold]
            model = _fit_poly4d(f_train, s_train)
            if model is None:
                continue
            cx, cy = model
            pred = _apply_poly4d(cx, cy, feat_pts_all[fold])
            true = screen_pts_all[fold]
            err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
            loocv_errors.append({
                "dot_idx": final_dot_medians[fold]["dot_idx"],
                "error_cm": round(err, 2),
                "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                "true_cm": [round(true[0], 2), round(true[1], 2)],
            })
        if loocv_errors:
            print(f"  [{TAG}] LOOCV: mean={np.mean([e['error_cm'] for e in loocv_errors]):.2f} cm")

    # Also compute convergence points for visualization (using biased gaze + ray intersection)
    bias_vec = np.array([dx_r, dy_r, dx_l, dy_l])
    calibrated_results = []
    cal_idx, test_idx = _split_by_cutoff(valid_frames, cal_cutoff_time, lambda x: x[0])

    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        feat_biased = raw_features[i] - bias_vec
        r_gaze = feat_biased[:2]
        l_gaze = feat_biased[2:]

        entry = {
            "frame": frame_name,
            "convergence_mm": None,
            "fixation_distance_mm": None,
            "convergence_point": None,
            "ray_miss_mm": None,
            "ipd_mm": None,
            "is_calibration": i in cal_idx,
        }

        # Compute convergence for visualization
        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))

                entry["convergence_mm"] = round(fix_dist, 2)
                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [round(float(conv_pt[0]), 2),
                                               round(float(conv_pt[1]), 2),
                                               round(float(conv_pt[2]), 2)]
                entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)

        calibrated_results.append(entry)

    # Rep2+ evaluation
    marker_counts = {}
    all_dots_with_rep = []
    for d in dots:
        mid = d["markerId"]
        marker_counts[mid] = marker_counts.get(mid, 0) + 1
        d_copy = dict(d)
        d_copy["repetition"] = marker_counts[mid]
        all_dots_with_rep.append(d_copy)
    rep2_only = [d for d in all_dots_with_rep if d.get("repetition", 1) > 1]

    rep2_eval = []
    if rep2_only and len(final_dot_medians) >= 5:
        feat_pts_all = [d["feature"] for d in final_dot_medians]
        screen_pts_all = [d["screen_cm"] for d in final_dot_medians]
        model = _fit_poly4d(feat_pts_all, screen_pts_all)
        if model is not None:
            cx, cy = model
            for dot in rep2_only:
                if dot["on_ts"] is None or dot["off_ts"] is None:
                    continue
                if dot["xCm"] is None or dot["yCm"] is None:
                    continue
                margin = 0.15
                t_start = dot["on_ts"] + margin
                t_end = dot["off_ts"] - margin
                if t_end <= t_start:
                    t_start = dot["on_ts"]
                    t_end = dot["off_ts"]
                feat_list = []
                for ef in eye_frames:
                    if t_start <= ef["timestamp"] <= t_end:
                        f = final_feat_map.get(ef["name"])
                        if f is not None:
                            feat_list.append(f)
                if len(feat_list) >= 3:
                    feat_arr = np.array(feat_list)
                    med_feat = np.median(feat_arr, axis=0)
                    pred = _apply_poly4d(cx, cy, med_feat)
                    true = [dot["xCm"], dot["yCm"]]
                    err = float(np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2))
                    rep2_eval.append({
                        "markerId": dot["markerId"],
                        "repetition": dot["repetition"],
                        "error_cm": round(err, 2),
                        "pred_cm": [round(pred[0], 2), round(pred[1], 2)],
                        "true_cm": [round(true[0], 2), round(true[1], 2)],
                    })
            if rep2_eval:
                print(f"  [{TAG}] Rep2+ ({len(rep2_eval)} dots): "
                      f"mean error={np.mean([e['error_cm'] for e in rep2_eval]):.2f} cm")

    # Stats
    fix_vals = [e["fixation_distance_mm"] for e in calibrated_results
                if e["fixation_distance_mm"] is not None]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]

    # ---- Save JSON ----
    conv_meta = {
        "method": "pupil_glint_poly_calibrated",
        "description": "Direct pupil-glint vector polynomial mapping to screen",
        "cc_estimation_method": "sliding_window" if cc_mode == 'sliding' else "reflection_law",
        "cc_mode": cc_mode,
        "transform_type": "poly4d",
        "calibration": {
            "objective": "pg_poly_mse",
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
            "final_mse_cm2": round(float(result.fun), 6),
            "n_cal_dots": len(final_dot_medians),
        },
        "loocv": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in loocv_errors])), 2) if loocv_errors else None,
            "errors": loocv_errors,
        } if loocv_errors else None,
        "rep2_eval": {
            "mean_error_cm": round(float(np.mean([e["error_cm"] for e in rep2_eval])), 2) if rep2_eval else None,
            "n_dots": len(rep2_eval),
            "errors": rep2_eval,
        } if rep2_eval else None,
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta["right_corneal_center_ro"],
        "left_corneal_center_lo": meta["left_corneal_center_lo"],
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "is_calibration": e.get("is_calibration"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_pg_poly_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [{TAG}] Saved to {conv_path}")


def calibrate_corneal3d_convergence(output_dir, calib, known_distance_mm=500.0,
                                     cal_cutoff_time=None, fair_cc=False):
    """Calibrate per-eye gaze bias for Corneal 3D using known fixation distance.

    Same approach as calibrate_reflect_c3d_convergence() but uses the sphere-fit
    corneal center from convergence_meta_corneal3d.json instead of the
    reflection-law CC from convergence_meta_reflectc3d.json.

    When fair_cc=True, also computes results using cal-only CC.

    Saves calibrated results to convergence_meta_corneal3d_cal.json.
    """
    from scipy.optimize import minimize

    out_base = Path(output_dir)

    # Load existing corneal 3d meta for CC positions
    meta_path = out_base / "convergence_meta_corneal3d.json"
    if not meta_path.exists():
        print("  [CAL-C3D-S] No corneal 3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro") is None or meta.get("left_corneal_center_lo") is None:
        print("  [CAL-C3D-S] Missing corneal centers in corneal 3d meta")
        return

    right_cc = np.array(meta["right_corneal_center_ro"])
    left_cc = np.array(meta["left_corneal_center_lo"])

    # Fair CC: recompute CC from calibration-only frames
    right_cc_fair = None
    left_cc_fair = None
    if fair_cc and cal_cutoff_time is not None:
        cc_obs_r = meta.get("cc_observations_right") or _load_cc_observations(out_base, "corneal3d", "right")
        cc_obs_l = meta.get("cc_observations_left") or _load_cc_observations(out_base, "corneal3d", "left")
        if cc_obs_r:
            right_cc_fair = _recompute_median_cc_from_observations(cc_obs_r, cal_cutoff_time)
        if cc_obs_l:
            left_cc_fair = _recompute_median_cc_from_observations(cc_obs_l, cal_cutoff_time)
        if right_cc_fair is not None and left_cc_fair is not None:
            print(f"  [CAL-C3D-S] Fair CC: R=[{right_cc_fair[0]:.2f},{right_cc_fair[1]:.2f},{right_cc_fair[2]:.2f}] "
                  f"L=[{left_cc_fair[0]:.2f},{left_cc_fair[1]:.2f},{left_cc_fair[2]:.2f}]")
        else:
            print(f"  [CAL-C3D-S] Fair CC: not enough cal-only observations")

    cross = calib.get("cross")
    if cross is None:
        print("  [CAL-C3D-S] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load pupil_3d for both eyes from seg combined
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [CAL-C3D-S] Need seg combined results for both eyes")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    # Collect frames with valid pupil_3d for both eyes
    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [CAL-C3D-S] Only {len(valid_frames)} valid frames, need >= 3")
        return

    # Precompute raw (uncorrected) gaze for each frame
    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - r_cc_proj

        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - l_cc_proj

        raw_gazes.append((r_gaze, l_gaze))

    # Split into calibration / test
    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])  # x[0] is frame_name
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CAL-C3D-S] Calibrating with {len(valid_frames)} frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CAL-C3D-S] Split: {n_cal} cal + {n_test} test "
              f"(cutoff={cal_cutoff_time:.1f}s)")
    print(f"  [CAL-C3D-S] Uncalibrated median: "
          f"{meta.get('median_fixation_mm', 0)/10:.1f} cm")

    def _convergence_distances(bias, indices=None):
        """Compute convergence distances for given frames with given bias."""
        dx_r, dy_r, dx_l, dy_l = bias
        distances = []
        ray_misses = []

        for i, (r_gaze_raw, l_gaze_raw) in enumerate(raw_gazes):
            if indices is not None and i not in indices:
                continue
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)

            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                distances.append(fix_dist)
                ray_misses.append(ray_miss)

        return distances, ray_misses

    def objective(bias):
        # Only use calibration frames for optimization
        distances, ray_misses = _convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10

        median_dist = np.median(distances)
        dist_error = (median_dist - known_distance_mm) ** 2
        variance = np.var(distances)
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2

        return dist_error + 0.01 * variance + reg + sym_h + sym_v

    result = minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [CAL-C3D-S] Optimized gaze bias:")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")

    # Recompute convergence with calibrated bias
    calibrated_results = []
    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        r_gaze_raw, l_gaze_raw = raw_gazes[i]
        r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
        l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))

                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [
                    round(float(conv_pt[k]), 2) for k in range(3)]
                entry["ray_miss_mm"] = round(ray_miss, 2)

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # ---- Kappa angle estimation (decomposition method) ----
    valid_conv_pts = [np.array(e["convergence_point"])
                      for e in calibrated_results if e.get("convergence_point")]
    if not valid_conv_pts:
        kappa_data = {}
    else:
        target_ro = np.median(valid_conv_pts, axis=0)

        kappa_right_h, kappa_right_v, kappa_right_mag = [], [], []
        kappa_left_h, kappa_left_v, kappa_left_mag = [], [], []

        target_lo = (R_cross @ target_ro.reshape(3, 1) + T_cross).flatten()

        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            entry = calibrated_results[i]
            if entry.get("convergence_point") is None:
                continue

            # Right eye kappa (RO frame)
            cc_to_pupil_dist = float(np.linalg.norm(rp - right_cc))
            d_visual_r = target_ro - right_cc
            d_visual_r = d_visual_r / np.linalg.norm(d_visual_r)
            p_zk_r = right_cc + cc_to_pupil_dist * d_visual_r

            pupil_proj_r = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            zk_proj_r = np.array([p_zk_r[0] / p_zk_r[2], p_zk_r[1] / p_zk_r[2]])
            kappa_gn_r = pupil_proj_r - zk_proj_r

            kr_h = float(np.degrees(np.arctan(kappa_gn_r[0])))
            kr_v = float(np.degrees(np.arctan(kappa_gn_r[1])))
            kr_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_r))))
            kappa_right_h.append(kr_h)
            kappa_right_v.append(kr_v)
            kappa_right_mag.append(kr_mag)

            # Left eye kappa (LO frame)
            cc_to_pupil_dist_l = float(np.linalg.norm(lp_lo - left_cc))
            d_visual_l = target_lo - left_cc
            d_visual_l = d_visual_l / np.linalg.norm(d_visual_l)
            p_zk_l = left_cc + cc_to_pupil_dist_l * d_visual_l

            pupil_proj_l = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            zk_proj_l = np.array([p_zk_l[0] / p_zk_l[2], p_zk_l[1] / p_zk_l[2]])
            kappa_gn_l = pupil_proj_l - zk_proj_l

            kl_h = float(np.degrees(np.arctan(kappa_gn_l[0])))
            kl_v = float(np.degrees(np.arctan(kappa_gn_l[1])))
            kl_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_l))))
            kappa_left_h.append(kl_h)
            kappa_left_v.append(kl_v)
            kappa_left_mag.append(kl_mag)

            entry["kappa_right_deg"] = [round(kr_h, 3), round(kr_v, 3),
                                        round(kr_mag, 3)]
            entry["kappa_left_deg"] = [round(kl_h, 3), round(kl_v, 3),
                                       round(kl_mag, 3)]

        kappa_data = {}

        def _kappa_noise(vals):
            arr = np.array(vals)
            med = float(np.median(arr))
            std = float(np.std(arr))
            mad = float(np.median(np.abs(arr - med)))
            return {
                "median": round(med, 3),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(std, 3),
                "mad": round(mad, 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "range": round(float(np.max(arr) - np.min(arr)), 3),
            }

        if kappa_right_h:
            rh = _kappa_noise(kappa_right_h)
            rv = _kappa_noise(kappa_right_v)
            rmag = _kappa_noise(kappa_right_mag)
            print(f"  [CAL-C3D-S KAPPA] Right eye: "
                  f"mag={rmag['median']:.2f}° std={rmag['std']:.3f}°")
            kappa_data["right"] = {
                "cam_h_deg": rh, "cam_v_deg": rv, "magnitude_deg": rmag,
                "n_frames": len(kappa_right_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_right_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_right_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_right_mag],
            }

        if kappa_left_h:
            lh = _kappa_noise(kappa_left_h)
            lv = _kappa_noise(kappa_left_v)
            lmag = _kappa_noise(kappa_left_mag)
            print(f"  [CAL-C3D-S KAPPA] Left eye: "
                  f"mag={lmag['median']:.2f}° std={lmag['std']:.3f}°")
            kappa_data["left"] = {
                "cam_h_deg": lh, "cam_v_deg": lv, "magnitude_deg": lmag,
                "n_frames": len(kappa_left_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_left_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_left_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_left_mag],
            }

        if kappa_right_h or kappa_left_h:
            all_stds = []
            if kappa_right_h:
                all_stds.extend([rh['std'], rv['std']])
            if kappa_left_h:
                all_stds.extend([lh['std'], lv['std']])
            max_std = max(all_stds)
            if max_std < 0.5:
                quality = "excellent (std < 0.5)"
            elif max_std < 1.0:
                quality = "good (std < 1.0)"
            elif max_std < 2.0:
                quality = "fair (std < 2.0)"
            else:
                quality = f"poor (std up to {max_std:.1f})"
            kappa_data["noise_quality"] = quality
            kappa_data["target_ro_mm"] = [round(float(v), 2) for v in target_ro]

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    if fix_vals:
        print(f"  [CAL-C3D-S] Calibrated: {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [CAL-C3D-S] ray miss: median={np.median(miss_vals):.2f}mm "
              f"mean={np.mean(miss_vals):.2f}mm")

    # Cal/test split reporting
    if cal_cutoff_time is not None:
        all_fix = [e.get("fixation_distance_mm") for e in calibrated_results]
        _report_cal_test_stats("CAL-C3D-S", all_fix, cal_idx, test_idx, known_distance_mm)

    conv_meta = {
        "method": "corneal_3d_calibrated",
        "description": f"Corneal 3D (sphere fit) calibrated to {known_distance_mm/10:.0f}cm "
                       f"(per-eye gaze bias / kappa correction)",
        "cc_estimation_method": "sphere_fit",
        "calibration": {
            "known_distance_mm": known_distance_mm,
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
        },
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta["right_corneal_center_ro"],
        "left_corneal_center_lo": meta["left_corneal_center_lo"],
        "kappa": kappa_data if kappa_data else None,
        "cal_test_split": {
            "cal_cutoff_time": cal_cutoff_time,
            "n_cal": n_cal,
            "n_test": n_test,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
            "kappa_right_deg": e.get("kappa_right_deg"),
            "kappa_left_deg": e.get("kappa_left_deg"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    # === Fair CC: re-optimize with cal-only CC ===
    if fair_cc and right_cc_fair is not None and left_cc_fair is not None:
        r_cc_fair_proj = np.array([right_cc_fair[0] / right_cc_fair[2],
                                   right_cc_fair[1] / right_cc_fair[2]])
        l_cc_fair_proj = np.array([left_cc_fair[0] / left_cc_fair[2],
                                   left_cc_fair[1] / left_cc_fair[2]])

        fair_raw_gazes = []
        for frame_name, rp, lp_lo in valid_frames:
            r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            r_gaze = r_pupil_proj - r_cc_fair_proj
            l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            l_gaze = l_pupil_proj - l_cc_fair_proj
            fair_raw_gazes.append((r_gaze, l_gaze))

        def _fair_conv_c3ds(bias, indices=None):
            dx_r, dy_r, dx_l, dy_l = bias
            distances = []
            for i_f, (rg, lg) in enumerate(fair_raw_gazes):
                if indices is not None and i_f not in indices:
                    continue
                rg2 = rg - np.array([dx_r, dy_r])
                lg2 = lg - np.array([dx_l, dy_l])
                rd = np.array([rg2[0], rg2[1], 1.0])
                rd = rd / np.linalg.norm(rd)
                ld_lo = np.array([lg2[0], lg2[1], 1.0])
                ld_lo = ld_lo / np.linalg.norm(ld_lo)
                ld_ro = (R_cross.T @ ld_lo.reshape(3, 1)).flatten()
                ld_ro = ld_ro / np.linalg.norm(ld_ro)
                w0 = -lo_origin_ro
                av = float(np.dot(rd, rd))
                bv = float(np.dot(rd, ld_ro))
                cv = float(np.dot(ld_ro, ld_ro))
                dv = float(np.dot(rd, w0))
                ev = float(np.dot(ld_ro, w0))
                dn = av * cv - bv * bv
                if abs(dn) < 1e-10:
                    continue
                sc = (bv * ev - cv * dv) / dn
                tc = (av * ev - bv * dv) / dn
                if sc > 0 and tc > 0:
                    cr = sc * rd
                    cl = lo_origin_ro + tc * ld_ro
                    cp = (cr + cl) / 2.0
                    cm = lo_origin_ro / 2.0
                    distances.append(float(np.linalg.norm(cp - cm)))
            return distances

        def fair_obj_c3ds(bias):
            dists = _fair_conv_c3ds(bias, cal_idx if cal_cutoff_time is not None else None)
            if len(dists) < 3:
                return 1e10
            md = np.median(dists)
            return (md - known_distance_mm)**2 + 0.01*np.var(dists) + 0.001*np.sum(np.array(bias)**2) + \
                   0.1*(bias[0]+bias[2])**2 + 0.1*(bias[1]-bias[3])**2

        fair_res = minimize(fair_obj_c3ds, [0,0,0,0], method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})
        fdx_r, fdy_r, fdx_l, fdy_l = fair_res.x
        print(f"  [CAL-C3D-S FAIR] bias: R=[{fdx_r:.6f},{fdy_r:.6f}] L=[{fdx_l:.6f},{fdy_l:.6f}]")

        for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
            entry = calibrated_results[i]
            rg, lg = fair_raw_gazes[i]
            rg2 = rg - np.array([fdx_r, fdy_r])
            lg2 = lg - np.array([fdx_l, fdy_l])
            rd = np.array([rg2[0], rg2[1], 1.0])
            rd = rd / np.linalg.norm(rd)
            ld_lo = np.array([lg2[0], lg2[1], 1.0])
            ld_lo = ld_lo / np.linalg.norm(ld_lo)
            ld_ro = (R_cross.T @ ld_lo.reshape(3, 1)).flatten()
            ld_ro = ld_ro / np.linalg.norm(ld_ro)
            w0 = -lo_origin_ro
            av = float(np.dot(rd, rd)); bv = float(np.dot(rd, ld_ro))
            cv = float(np.dot(ld_ro, ld_ro)); dv = float(np.dot(rd, w0))
            ev = float(np.dot(ld_ro, w0)); dn = av*cv - bv*bv
            if abs(dn) > 1e-10:
                sc = (bv*ev - cv*dv)/dn; tc = (av*ev - bv*dv)/dn
                if sc > 0 and tc > 0:
                    cr = sc*rd; cl = lo_origin_ro + tc*ld_ro
                    cp = (cr+cl)/2.0; rm = float(np.linalg.norm(cr-cl))
                    cm = lo_origin_ro/2.0; fd = float(np.linalg.norm(cp-cm))
                    entry["fixation_distance_mm_fair"] = round(fd, 2)
                    entry["convergence_mm_fair"] = round(fd, 2)
                    entry["ray_miss_mm_fair"] = round(rm, 2)

        fair_fix = [e.get("fixation_distance_mm_fair") for e in calibrated_results if e.get("fixation_distance_mm_fair")]
        fair_cal = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results) if i in cal_idx and e.get("fixation_distance_mm_fair")]
        fair_test = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results) if i in test_idx and e.get("fixation_distance_mm_fair")]
        if fair_fix:
            print(f"  [CAL-C3D-S FAIR] median={np.median(fair_fix)/10:.1f}cm")
        if fair_cal:
            print(f"  [CAL-C3D-S FAIR] CAL ({len(fair_cal)}): err={abs(np.median(fair_cal)-known_distance_mm)/10:.1f}cm")
        if fair_test:
            print(f"  [CAL-C3D-S FAIR] TEST ({len(fair_test)}): err={abs(np.median(fair_test)-known_distance_mm)/10:.1f}cm")

        conv_meta["fair_cc"] = {
            "right_corneal_center_ro_fair": [round(float(v), 4) for v in right_cc_fair],
            "left_corneal_center_lo_fair": [round(float(v), 4) for v in left_cc_fair],
            "bias_right_fair": [round(float(fdx_r), 6), round(float(fdy_r), 6)],
            "bias_left_fair": [round(float(fdx_l), 6), round(float(fdy_l), 6)],
            "median_fixation_mm_fair": round(float(np.median(fair_fix)), 2) if fair_fix else None,
            "cal_median_mm": round(float(np.median(fair_cal)), 2) if fair_cal else None,
            "test_median_mm": round(float(np.median(fair_test)), 2) if fair_test else None,
            "cal_err_cm": round(abs(np.median(fair_cal)-known_distance_mm)/10, 2) if fair_cal else None,
            "test_err_cm": round(abs(np.median(fair_test)-known_distance_mm)/10, 2) if fair_test else None,
            "n_cal": len(fair_cal), "n_test": len(fair_test),
        }
        for pf in conv_meta["per_frame"]:
            match = next((e for e in calibrated_results if e["frame"] == pf["frame"]), None)
            if match:
                pf["fixation_distance_mm_fair"] = match.get("fixation_distance_mm_fair")
                pf["convergence_mm_fair"] = match.get("convergence_mm_fair")
                pf["ray_miss_mm_fair"] = match.get("ray_miss_mm_fair")

    conv_path = out_base / "convergence_meta_corneal3d_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CAL-C3D-S] Saved to {conv_path}")


def calibrate_4ray_convergence(output_dir, calib, known_distance_mm=500.0,
                               cal_cutoff_time=None):
    """Calibrate per-camera gaze bias for 4-Ray using known fixation distance.

    Optimizes 8 parameters: [dx_ro, dy_ro, dx_ri, dy_ri, dx_lo, dy_lo, dx_li, dy_li]
    — constant angular offsets subtracted from each camera's gaze_norm — such that
    the median convergence distance matches the known fixation distance.

    Symmetry constraints:
      - RO ≈ RI bias (same right eye)
      - LO ≈ LI bias (same left eye)
      - Cross-eye horizontal symmetry

    No kappa decomposition (PCCR gaze has no CC reference point).
    Reports per-camera bias in degrees.

    Saves to convergence_meta_4ray_cal.json.
    """
    from scipy.optimize import minimize

    out_base = Path(output_dir)

    # Load existing 4-ray meta
    meta_path = out_base / "convergence_meta_4ray.json"
    if not meta_path.exists():
        print("  [CAL-4RAY] No 4-ray results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    # Need all three stereo pairs for full camera chain
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    cross_pair = calib.get("cross")
    if not all([right_pair, left_pair, cross_pair]):
        print("  [CAL-4RAY] Need all 3 stereo pairs, skipping")
        return

    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)
    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)

    # Camera origins in RO frame
    cam_origins = {
        'ro': np.zeros(3),
        'ri': (-R_right.T @ T_right).flatten(),
        'lo': (-R_cross.T @ T_cross).flatten(),
        'li': (R_cross.T @ ((-R_left.T @ T_left).flatten().reshape(3, 1) - T_cross)).flatten(),
    }
    cam_rotations = {
        'ro': np.eye(3),
        'ri': R_right.T,
        'lo': R_cross.T,
        'li': R_cross.T @ R_left.T,
    }

    # Camera weights
    cam_weights = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        reproj = calib.get(cam, {}).get('reproj_err')
        if reproj and reproj['mean_px'] > 0:
            cam_weights[cam] = 1.0 / reproj['mean_px']
        else:
            cam_weights[cam] = 1.0
    w_mean = np.mean(list(cam_weights.values()))
    if w_mean > 0:
        cam_weights = {cam: w / w_mean for cam, w in cam_weights.items()}

    # Load per-camera results
    cam_results = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        rpath = out_base / cam / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                cam_results[cam] = json.load(f)

    if len(cam_results) < 2:
        print(f"  [CAL-4RAY] Need at least 2 cameras, only found {len(cam_results)}")
        return

    # Load seg combined for IPD
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame, l_seg_by_frame = {}, {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    # Precompute raw per-camera gaze norms for all frames
    n_frames = min(len(v) for v in cam_results.values())
    I3 = np.eye(3)
    cam_order = ['ro', 'ri', 'lo', 'li']

    # Collect raw gaze data per frame
    frame_raw_data = []
    for i in range(n_frames):
        frame_name = None
        raw_rays = {}
        for cam in cam_order:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")
            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue
            raw_rays[cam] = np.array(gaze_norm[:2])
        frame_raw_data.append((frame_name, raw_rays))

    # Split into calibration / test
    cal_idx, test_idx = _split_by_cutoff(
        frame_raw_data, cal_cutoff_time, lambda x: x[0])  # x[0] is frame_name
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CAL-4RAY] Calibrating with {len(frame_raw_data)} frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CAL-4RAY] Split: {n_cal} cal + {n_test} test "
              f"(cutoff={cal_cutoff_time:.1f}s)")
    print(f"  [CAL-4RAY] Uncalibrated median: "
          f"{meta.get('median_fixation_mm', 0)/10:.1f} cm")

    def _4ray_convergence_distances(bias, indices=None):
        """Compute convergence distances with per-camera bias."""
        bias_map = {
            'ro': np.array([bias[0], bias[1]]),
            'ri': np.array([bias[2], bias[3]]),
            'lo': np.array([bias[4], bias[5]]),
            'li': np.array([bias[6], bias[7]]),
        }
        distances = []

        for idx, (frame_name, raw_rays) in enumerate(frame_raw_data):
            if indices is not None and idx not in indices:
                continue
            rays = []
            for cam in cam_order:
                if cam not in raw_rays:
                    continue
                gn = raw_rays[cam] - bias_map[cam]
                d_cam = np.array([gn[0], gn[1], 1.0])
                d_cam /= np.linalg.norm(d_cam)
                d_ro = cam_rotations[cam] @ d_cam
                d_ro /= np.linalg.norm(d_ro)
                if d_ro[2] < 0.3:
                    continue
                rays.append((cam_origins[cam], d_ro, cam_weights.get(cam, 1.0)))

            if len(rays) >= 2:
                A = np.zeros((3, 3))
                b = np.zeros(3)
                for origin, direction, weight in rays:
                    d = direction.reshape(3, 1)
                    M = I3 - d @ d.T
                    A += weight * M
                    b += weight * (M @ origin)
                try:
                    P = np.linalg.solve(A, b)
                    if P[2] > 0:
                        cam_mid = np.mean([o for o, _, _ in rays], axis=0)
                        fix_dist = float(np.linalg.norm(P - cam_mid))
                        distances.append(fix_dist)
                except np.linalg.LinAlgError:
                    pass

        return distances

    def objective(bias):
        # Only use calibration frames for optimization
        distances = _4ray_convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10

        median_dist = np.median(distances)
        dist_error = (median_dist - known_distance_mm) ** 2
        variance = np.var(distances)
        reg = 0.001 * np.sum(np.array(bias) ** 2)

        # Same-eye symmetry: RO ≈ RI, LO ≈ LI
        same_eye_r_h = 0.5 * (bias[0] - bias[2]) ** 2
        same_eye_r_v = 0.5 * (bias[1] - bias[3]) ** 2
        same_eye_l_h = 0.5 * (bias[4] - bias[6]) ** 2
        same_eye_l_v = 0.5 * (bias[5] - bias[7]) ** 2

        # Cross-eye horizontal symmetry: right_h ≈ -left_h
        cross_h = 0.1 * ((bias[0] + bias[2]) / 2 + (bias[4] + bias[6]) / 2) ** 2
        # Cross-eye vertical symmetry: right_v ≈ left_v
        cross_v = 0.1 * ((bias[1] + bias[3]) / 2 - (bias[5] + bias[7]) / 2) ** 2

        return (dist_error + 0.01 * variance + reg
                + same_eye_r_h + same_eye_r_v + same_eye_l_h + same_eye_l_v
                + cross_h + cross_v)

    result = minimize(objective, [0]*8, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-6})

    opt_bias = result.x
    bias_map = {
        'ro': np.array([opt_bias[0], opt_bias[1]]),
        'ri': np.array([opt_bias[2], opt_bias[3]]),
        'lo': np.array([opt_bias[4], opt_bias[5]]),
        'li': np.array([opt_bias[6], opt_bias[7]]),
    }
    print(f"  [CAL-4RAY] Optimized per-camera gaze bias:")
    for cam in cam_order:
        b = bias_map[cam]
        print(f"    {cam.upper()}: dx={b[0]:.6f} ({np.degrees(np.arctan(b[0])):.3f}°), "
              f"dy={b[1]:.6f} ({np.degrees(np.arctan(b[1])):.3f}°)")

    # Recompute convergence with calibrated bias
    calibrated_results = []
    for i, (frame_name, raw_rays) in enumerate(frame_raw_data):
        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "n_cameras": 0, "per_camera_residual": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        rays = []
        for cam in cam_order:
            if cam not in raw_rays:
                continue
            gn = raw_rays[cam] - bias_map[cam]
            d_cam = np.array([gn[0], gn[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)
            if d_ro[2] < 0.3:
                continue
            rays.append((cam_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))

        entry["n_cameras"] = len(rays)

        if len(rays) >= 2:
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for origin, direction, _, weight in rays:
                d = direction.reshape(3, 1)
                M = I3 - d @ d.T
                A += weight * M
                b += weight * (M @ origin)

            try:
                P = np.linalg.solve(A, b)
                if P[2] > 0:
                    residuals = {}
                    for origin, direction, cam_name, _ in rays:
                        diff = P - origin
                        proj_len = np.dot(diff, direction)
                        if proj_len < 0:
                            continue
                        perp = diff - proj_len * direction
                        residuals[cam_name] = float(np.linalg.norm(perp))

                    if residuals:
                        rms_residual = float(np.sqrt(
                            np.mean([r**2 for r in residuals.values()])))
                        cam_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                        fix_dist = float(np.linalg.norm(P - cam_mid))

                        entry["fixation_distance_mm"] = round(fix_dist, 2)
                        entry["convergence_mm"] = round(fix_dist, 2)
                        entry["convergence_point"] = [
                            round(float(P[k]), 2) for k in range(3)]
                        entry["ray_miss_mm"] = round(rms_residual, 2)
                        entry["per_camera_residual"] = {
                            k: round(v, 2) for k, v in residuals.items()}
            except np.linalg.LinAlgError:
                pass

        # IPD
        if frame_name:
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if r_seg and l_seg:
                r_pupil = r_seg.get("pupil_3d")
                l_pupil = l_seg.get("pupil_3d")
                if r_pupil and l_pupil:
                    rp = np.array(r_pupil)
                    lp_lo = np.array(l_pupil)
                    lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                    ipd = float(np.linalg.norm(rp - lp_ro))
                    entry["ipd_mm"] = round(ipd, 2)
                    entry["right_pupil_3d"] = [
                        round(float(rp[k]), 4) for k in range(3)]
                    entry["left_pupil_3d_ro"] = [
                        round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    if fix_vals:
        print(f"  [CAL-4RAY] Calibrated: {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [CAL-4RAY] ray miss: median={np.median(miss_vals):.2f}mm "
              f"mean={np.mean(miss_vals):.2f}mm")

    # Cal/test split reporting
    if cal_cutoff_time is not None:
        all_fix = [e.get("fixation_distance_mm") for e in calibrated_results]
        _report_cal_test_stats("CAL-4RAY", all_fix, cal_idx, test_idx, known_distance_mm)

    conv_meta = {
        "method": "4ray_weighted_calibrated",
        "description": f"4-ray weighted calibrated to {known_distance_mm/10:.0f}cm "
                       f"(per-camera gaze bias correction)",
        "calibration": {
            "known_distance_mm": known_distance_mm,
            "bias_per_camera": {cam: [round(float(bias_map[cam][0]), 6),
                                      round(float(bias_map[cam][1]), 6)]
                                for cam in cam_order},
            "bias_per_camera_deg": {cam: [round(float(np.degrees(np.arctan(bias_map[cam][0]))), 3),
                                          round(float(np.degrees(np.arctan(bias_map[cam][1]))), 3)]
                                    for cam in cam_order},
        },
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "camera_weights": {cam: round(w, 4) for cam, w in cam_weights.items()},
        "cal_test_split": {
            "cal_cutoff_time": cal_cutoff_time,
            "n_cal": n_cal,
            "n_test": n_test,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "n_cameras": e.get("n_cameras"),
            "per_camera_residual": e.get("per_camera_residual"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    conv_path = out_base / "convergence_meta_4ray_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CAL-4RAY] Saved to {conv_path}")

    # --- Huber variant: re-solve with Huber IRLS using same calibrated gaze ---
    huber_results = []
    for i, (frame_name, raw_rays) in enumerate(frame_raw_data):
        h_entry = {"frame": frame_name, "convergence_mm": None,
                   "fixation_distance_mm": None, "convergence_point": None,
                   "ray_miss_mm": None, "ipd_mm": None,
                   "n_cameras": 0, "per_camera_residual": None,
                   "huber_weights": None, "is_calibration": i in cal_idx}
        rays = []
        for cam in cam_order:
            if cam not in raw_rays:
                continue
            gn = raw_rays[cam] - bias_map[cam]
            d_cam = np.array([gn[0], gn[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)
            if d_ro[2] < 0.3:
                continue
            rays.append((cam_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))
        h_entry["n_cameras"] = len(rays)
        if len(rays) >= 2:
            h_sol = _solve_nray_fixation(rays, huber_delta=5.0)
            if h_sol:
                h_entry["fixation_distance_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_point"] = [round(float(h_sol['P'][k]), 2) for k in range(3)]
                h_entry["ray_miss_mm"] = round(h_sol['rms_residual'], 2)
                h_entry["per_camera_residual"] = {k: round(v, 2) for k, v in h_sol['residuals'].items()}
                h_entry["huber_weights"] = {k: round(v, 4) for k, v in h_sol.get('huber_weights', {}).items()}
        # Copy IPD from standard result
        std_entry = calibrated_results[i]
        h_entry["ipd_mm"] = std_entry.get("ipd_mm")
        h_entry["right_pupil_3d"] = std_entry.get("right_pupil_3d")
        h_entry["left_pupil_3d_ro"] = std_entry.get("left_pupil_3d_ro")
        huber_results.append(h_entry)

    h_fix = [e["fixation_distance_mm"] for e in huber_results if e["fixation_distance_mm"]]
    if h_fix:
        print(f"  [CAL-4RAY HUBER] {len(h_fix)} frames | "
              f"fixation: median={np.median(h_fix)/10:.1f}cm "
              f"std={np.std(h_fix)/10:.1f}cm")
    h_meta = dict(conv_meta)
    h_meta["method"] = "huber_4ray_weighted_calibrated"
    h_meta["description"] = h_meta["description"].replace("4-ray", "Huber 4-ray")
    h_meta["huber_delta_mm"] = 5.0
    h_meta["median_fixation_mm"] = round(float(np.median(h_fix)), 2) if h_fix else None
    h_meta["mean_fixation_mm"] = round(float(np.mean(h_fix)), 2) if h_fix else None
    h_meta["std_fixation_mm"] = round(float(np.std(h_fix)), 2) if h_fix else None
    h_meta["per_frame"] = [{
        "frame": e["frame"], "convergence_mm": e["convergence_mm"],
        "fixation_distance_mm": e["fixation_distance_mm"],
        "convergence_point": e.get("convergence_point"),
        "ray_miss_mm": e.get("ray_miss_mm"), "n_cameras": e.get("n_cameras"),
        "per_camera_residual": e.get("per_camera_residual"),
        "huber_weights": e.get("huber_weights"),
        "ipd_mm": e.get("ipd_mm"), "is_calibration": e.get("is_calibration"),
    } for e in huber_results if e["fixation_distance_mm"]]
    h_path = out_base / "convergence_meta_huber4ray_cal.json"
    with open(str(h_path), "w") as f:
        json.dump(h_meta, f, indent=2)
    print(f"  [CAL-4RAY HUBER] Saved to {h_path}")


def calibrate_reflect4ray_convergence(output_dir, calib, known_distance_mm=500.0,
                                      cal_cutoff_time=None, fair_cc=False):
    """Calibrate per-camera gaze bias for Reflect 4-Ray using known fixation distance.

    Same 8-param per-camera bias as calibrate_4ray_convergence, but uses CC-based
    ray origins from convergence_meta_reflect4ray.json instead of camera origins.

    Has per-eye CC → supports kappa decomposition (same method as Cal C3D).
    When fair_cc=True, also computes results using cal-only CC.

    Saves to convergence_meta_reflect4ray_cal.json.
    """
    from scipy.optimize import minimize

    out_base = Path(output_dir)

    # Load existing reflect 4-ray meta for CC positions
    meta_path = out_base / "convergence_meta_reflect4ray.json"
    if not meta_path.exists():
        print("  [CAL-R4R] No reflect 4-ray results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    # Need all three stereo pairs
    right_pair = calib.get("right")
    left_pair = calib.get("left")
    cross_pair = calib.get("cross")
    if not all([right_pair, left_pair, cross_pair]):
        print("  [CAL-R4R] Need all 3 stereo pairs, skipping")
        return

    R_right = np.array(right_pair["R"])
    T_right = np.array(right_pair["T"]).reshape(3, 1)
    R_left = np.array(left_pair["R"])
    T_left = np.array(left_pair["T"]).reshape(3, 1)
    R_cross = np.array(cross_pair["R"])
    T_cross = np.array(cross_pair["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    cam_rotations = {
        'ro': np.eye(3),
        'ri': R_right.T,
        'lo': R_cross.T,
        'li': R_cross.T @ R_left.T,
    }

    # Camera weights
    cam_weights = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        reproj = calib.get(cam, {}).get('reproj_err')
        if reproj and reproj['mean_px'] > 0:
            cam_weights[cam] = 1.0 / reproj['mean_px']
        else:
            cam_weights[cam] = 1.0
    w_mean = np.mean(list(cam_weights.values()))
    if w_mean > 0:
        cam_weights = {cam: w / w_mean for cam, w in cam_weights.items()}

    # Get corneal centers and build ray origins
    right_cc_ro = np.array(meta["right_corneal_center_ro"]) \
        if meta.get("right_corneal_center_ro") else None
    left_cc_lo = np.array(meta["left_corneal_center_lo"]) \
        if meta.get("left_corneal_center_lo") else None

    if right_cc_ro is None or left_cc_lo is None:
        print("  [CAL-R4R] Missing corneal centers in reflect 4-ray meta")
        return

    left_cc_ro = (R_cross.T @ (left_cc_lo.reshape(3, 1) - T_cross)).flatten()

    ray_origins = {
        'ro': right_cc_ro,
        'ri': right_cc_ro,
        'lo': left_cc_ro,
        'li': left_cc_ro,
    }

    # Fair CC: recompute from cal-only observations
    right_cc_ro_fair = None
    left_cc_lo_fair = None
    if fair_cc and cal_cutoff_time is not None:
        cc_obs_r = meta.get("cc_observations_right")
        cc_obs_l = meta.get("cc_observations_left")
        if cc_obs_r:
            right_cc_ro_fair = _recompute_median_cc_from_observations(cc_obs_r, cal_cutoff_time)
        if cc_obs_l:
            left_cc_lo_fair = _recompute_median_cc_from_observations(cc_obs_l, cal_cutoff_time)
        if right_cc_ro_fair is not None and left_cc_lo_fair is not None:
            print(f"  [CAL-R4R] Fair CC available")
        else:
            print(f"  [CAL-R4R] Fair CC: not enough cal-only observations")

    # Load per-camera results
    cam_results = {}
    for cam in ['ro', 'ri', 'lo', 'li']:
        rpath = out_base / cam / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                cam_results[cam] = json.load(f)

    if len(cam_results) < 2:
        print(f"  [CAL-R4R] Need at least 2 cameras, only found {len(cam_results)}")
        return

    # Load seg combined for IPD
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    r_seg_by_frame, l_seg_by_frame = {}, {}
    if r_seg_path.exists() and l_seg_path.exists():
        with open(r_seg_path) as f:
            for e in json.load(f):
                r_seg_by_frame[e["frame"]] = e
        with open(l_seg_path) as f:
            for e in json.load(f):
                l_seg_by_frame[e["frame"]] = e

    # Precompute raw per-camera gaze norms
    n_frames = min(len(v) for v in cam_results.values())
    I3 = np.eye(3)
    cam_order = ['ro', 'ri', 'lo', 'li']

    frame_raw_data = []
    for i in range(n_frames):
        frame_name = None
        raw_rays = {}
        for cam in cam_order:
            if cam not in cam_results or i >= len(cam_results[cam]):
                continue
            r = cam_results[cam][i]
            if frame_name is None:
                frame_name = r.get("frame", f"frame_{i}")
            gaze_norm = r.get("seg_gaze_vector_norm")
            if gaze_norm is None or r.get("eye_closed"):
                continue
            raw_rays[cam] = np.array(gaze_norm[:2])
        frame_raw_data.append((frame_name, raw_rays))

    # Split into calibration / test
    cal_idx, test_idx = _split_by_cutoff(
        frame_raw_data, cal_cutoff_time, lambda x: x[0])  # x[0] is frame_name
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CAL-R4R] Calibrating with {len(frame_raw_data)} frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CAL-R4R] Split: {n_cal} cal + {n_test} test "
              f"(cutoff={cal_cutoff_time:.1f}s)")
    print(f"  [CAL-R4R] Uncalibrated median: "
          f"{meta.get('median_fixation_mm', 0)/10:.1f} cm")

    def _r4r_convergence_distances(bias, indices=None):
        """Compute convergence distances with per-camera bias, CC origins."""
        bias_map = {
            'ro': np.array([bias[0], bias[1]]),
            'ri': np.array([bias[2], bias[3]]),
            'lo': np.array([bias[4], bias[5]]),
            'li': np.array([bias[6], bias[7]]),
        }
        distances = []

        for idx, (frame_name, raw_rays) in enumerate(frame_raw_data):
            if indices is not None and idx not in indices:
                continue
            rays = []
            for cam in cam_order:
                if cam not in raw_rays:
                    continue
                gn = raw_rays[cam] - bias_map[cam]
                d_cam = np.array([gn[0], gn[1], 1.0])
                d_cam /= np.linalg.norm(d_cam)
                d_ro = cam_rotations[cam] @ d_cam
                d_ro /= np.linalg.norm(d_ro)
                if d_ro[2] < 0.3:
                    continue
                rays.append((ray_origins[cam], d_ro, cam_weights.get(cam, 1.0)))

            if len(rays) >= 2:
                A = np.zeros((3, 3))
                b = np.zeros(3)
                for origin, direction, weight in rays:
                    d = direction.reshape(3, 1)
                    M = I3 - d @ d.T
                    A += weight * M
                    b += weight * (M @ origin)
                try:
                    P = np.linalg.solve(A, b)
                    if P[2] > 0:
                        ray_mid = np.mean([o for o, _, _ in rays], axis=0)
                        fix_dist = float(np.linalg.norm(P - ray_mid))
                        distances.append(fix_dist)
                except np.linalg.LinAlgError:
                    pass

        return distances

    def objective(bias):
        # Only use calibration frames for optimization
        distances = _r4r_convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10

        median_dist = np.median(distances)
        dist_error = (median_dist - known_distance_mm) ** 2
        variance = np.var(distances)
        reg = 0.001 * np.sum(np.array(bias) ** 2)

        # Same-eye symmetry: RO ≈ RI, LO ≈ LI
        same_eye_r_h = 0.5 * (bias[0] - bias[2]) ** 2
        same_eye_r_v = 0.5 * (bias[1] - bias[3]) ** 2
        same_eye_l_h = 0.5 * (bias[4] - bias[6]) ** 2
        same_eye_l_v = 0.5 * (bias[5] - bias[7]) ** 2

        # Cross-eye horizontal symmetry
        cross_h = 0.1 * ((bias[0] + bias[2]) / 2 + (bias[4] + bias[6]) / 2) ** 2
        cross_v = 0.1 * ((bias[1] + bias[3]) / 2 - (bias[5] + bias[7]) / 2) ** 2

        return (dist_error + 0.01 * variance + reg
                + same_eye_r_h + same_eye_r_v + same_eye_l_h + same_eye_l_v
                + cross_h + cross_v)

    result = minimize(objective, [0]*8, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-6})

    opt_bias = result.x
    bias_map = {
        'ro': np.array([opt_bias[0], opt_bias[1]]),
        'ri': np.array([opt_bias[2], opt_bias[3]]),
        'lo': np.array([opt_bias[4], opt_bias[5]]),
        'li': np.array([opt_bias[6], opt_bias[7]]),
    }
    print(f"  [CAL-R4R] Optimized per-camera gaze bias:")
    for cam in cam_order:
        b = bias_map[cam]
        print(f"    {cam.upper()}: dx={b[0]:.6f} ({np.degrees(np.arctan(b[0])):.3f}°), "
              f"dy={b[1]:.6f} ({np.degrees(np.arctan(b[1])):.3f}°)")

    # Recompute convergence with calibrated bias
    calibrated_results = []
    for i, (frame_name, raw_rays) in enumerate(frame_raw_data):
        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "n_cameras": 0, "per_camera_residual": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        rays = []
        for cam in cam_order:
            if cam not in raw_rays:
                continue
            gn = raw_rays[cam] - bias_map[cam]
            d_cam = np.array([gn[0], gn[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)
            if d_ro[2] < 0.3:
                continue
            rays.append((ray_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))

        entry["n_cameras"] = len(rays)

        if len(rays) >= 2:
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for origin, direction, _, weight in rays:
                d = direction.reshape(3, 1)
                M = I3 - d @ d.T
                A += weight * M
                b += weight * (M @ origin)

            try:
                P = np.linalg.solve(A, b)
                if P[2] > 0:
                    residuals = {}
                    for origin, direction, cam_name, _ in rays:
                        diff = P - origin
                        proj_len = np.dot(diff, direction)
                        if proj_len < 0:
                            continue
                        perp = diff - proj_len * direction
                        residuals[cam_name] = float(np.linalg.norm(perp))

                    if residuals:
                        rms_residual = float(np.sqrt(
                            np.mean([r**2 for r in residuals.values()])))
                        ray_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                        fix_dist = float(np.linalg.norm(P - ray_mid))

                        entry["fixation_distance_mm"] = round(fix_dist, 2)
                        entry["convergence_mm"] = round(fix_dist, 2)
                        entry["convergence_point"] = [
                            round(float(P[k]), 2) for k in range(3)]
                        entry["ray_miss_mm"] = round(rms_residual, 2)
                        entry["per_camera_residual"] = {
                            k: round(v, 2) for k, v in residuals.items()}
            except np.linalg.LinAlgError:
                pass

        # IPD
        if frame_name:
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if r_seg and l_seg:
                r_pupil = r_seg.get("pupil_3d")
                l_pupil = l_seg.get("pupil_3d")
                if r_pupil and l_pupil:
                    rp = np.array(r_pupil)
                    lp_lo = np.array(l_pupil)
                    lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
                    ipd = float(np.linalg.norm(rp - lp_ro))
                    entry["ipd_mm"] = round(ipd, 2)
                    entry["right_pupil_3d"] = [
                        round(float(rp[k]), 4) for k in range(3)]
                    entry["left_pupil_3d_ro"] = [
                        round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # ---- Kappa angle estimation (has CC → can decompose) ----
    valid_conv_pts = [np.array(e["convergence_point"])
                      for e in calibrated_results if e.get("convergence_point")]
    kappa_data = {}
    if valid_conv_pts:
        target_ro = np.median(valid_conv_pts, axis=0)

        kappa_right_h, kappa_right_v, kappa_right_mag = [], [], []
        kappa_left_h, kappa_left_v, kappa_left_mag = [], [], []

        target_lo = (R_cross @ target_ro.reshape(3, 1) + T_cross).flatten()

        for i, (frame_name, raw_rays) in enumerate(frame_raw_data):
            entry = calibrated_results[i]
            if entry.get("convergence_point") is None:
                continue

            # Need pupil_3d from seg combined for kappa
            r_seg = r_seg_by_frame.get(frame_name)
            l_seg = l_seg_by_frame.get(frame_name)
            if not r_seg or not l_seg:
                continue
            rp_raw = r_seg.get("pupil_3d")
            lp_raw = l_seg.get("pupil_3d")
            if not rp_raw or not lp_raw:
                continue
            rp = np.array(rp_raw)
            lp_lo = np.array(lp_raw)

            # Right eye kappa (RO frame)
            cc_to_pupil_dist = float(np.linalg.norm(rp - right_cc_ro))
            d_visual_r = target_ro - right_cc_ro
            d_visual_r = d_visual_r / np.linalg.norm(d_visual_r)
            p_zk_r = right_cc_ro + cc_to_pupil_dist * d_visual_r

            pupil_proj_r = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            zk_proj_r = np.array([p_zk_r[0] / p_zk_r[2], p_zk_r[1] / p_zk_r[2]])
            kappa_gn_r = pupil_proj_r - zk_proj_r

            kr_h = float(np.degrees(np.arctan(kappa_gn_r[0])))
            kr_v = float(np.degrees(np.arctan(kappa_gn_r[1])))
            kr_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_r))))
            kappa_right_h.append(kr_h)
            kappa_right_v.append(kr_v)
            kappa_right_mag.append(kr_mag)

            # Left eye kappa (LO frame)
            cc_to_pupil_dist_l = float(np.linalg.norm(lp_lo - left_cc_lo))
            d_visual_l = target_lo - left_cc_lo
            d_visual_l = d_visual_l / np.linalg.norm(d_visual_l)
            p_zk_l = left_cc_lo + cc_to_pupil_dist_l * d_visual_l

            pupil_proj_l = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            zk_proj_l = np.array([p_zk_l[0] / p_zk_l[2], p_zk_l[1] / p_zk_l[2]])
            kappa_gn_l = pupil_proj_l - zk_proj_l

            kl_h = float(np.degrees(np.arctan(kappa_gn_l[0])))
            kl_v = float(np.degrees(np.arctan(kappa_gn_l[1])))
            kl_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_l))))
            kappa_left_h.append(kl_h)
            kappa_left_v.append(kl_v)
            kappa_left_mag.append(kl_mag)

            entry["kappa_right_deg"] = [round(kr_h, 3), round(kr_v, 3),
                                        round(kr_mag, 3)]
            entry["kappa_left_deg"] = [round(kl_h, 3), round(kl_v, 3),
                                       round(kl_mag, 3)]

        def _kappa_noise(vals):
            arr = np.array(vals)
            med = float(np.median(arr))
            std = float(np.std(arr))
            mad = float(np.median(np.abs(arr - med)))
            return {
                "median": round(med, 3),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(std, 3),
                "mad": round(mad, 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "range": round(float(np.max(arr) - np.min(arr)), 3),
            }

        if kappa_right_h:
            rh = _kappa_noise(kappa_right_h)
            rmag = _kappa_noise(kappa_right_mag)
            print(f"  [CAL-R4R KAPPA] Right eye: "
                  f"mag={rmag['median']:.2f}° std={rmag['std']:.3f}°")
            kappa_data["right"] = {
                "cam_h_deg": rh, "cam_v_deg": _kappa_noise(kappa_right_v),
                "magnitude_deg": rmag,
                "n_frames": len(kappa_right_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_right_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_right_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_right_mag],
            }

        if kappa_left_h:
            lh = _kappa_noise(kappa_left_h)
            lmag = _kappa_noise(kappa_left_mag)
            print(f"  [CAL-R4R KAPPA] Left eye: "
                  f"mag={lmag['median']:.2f}° std={lmag['std']:.3f}°")
            kappa_data["left"] = {
                "cam_h_deg": lh, "cam_v_deg": _kappa_noise(kappa_left_v),
                "magnitude_deg": lmag,
                "n_frames": len(kappa_left_h),
                "per_frame_h_deg": [round(v, 3) for v in kappa_left_h],
                "per_frame_v_deg": [round(v, 3) for v in kappa_left_v],
                "per_frame_mag_deg": [round(v, 3) for v in kappa_left_mag],
            }

        if kappa_right_h or kappa_left_h:
            all_stds = []
            if kappa_right_h:
                all_stds.extend([rh['std'], _kappa_noise(kappa_right_v)['std']])
            if kappa_left_h:
                all_stds.extend([lh['std'], _kappa_noise(kappa_left_v)['std']])
            max_std = max(all_stds)
            if max_std < 0.5:
                quality = "excellent (std < 0.5)"
            elif max_std < 1.0:
                quality = "good (std < 1.0)"
            elif max_std < 2.0:
                quality = "fair (std < 2.0)"
            else:
                quality = f"poor (std up to {max_std:.1f})"
            kappa_data["noise_quality"] = quality
            kappa_data["target_ro_mm"] = [round(float(v), 2) for v in target_ro]

    # ---- Fair CC computation ----
    fair_bias_map = None
    fair_fix, fair_cal, fair_test = [], [], []
    if fair_cc and right_cc_ro_fair is not None and left_cc_lo_fair is not None:
        left_cc_ro_fair = (R_cross.T @ (left_cc_lo_fair.reshape(3, 1) - T_cross)).flatten()
        fair_ray_origins = {
            'ro': right_cc_ro_fair, 'ri': right_cc_ro_fair,
            'lo': left_cc_ro_fair, 'li': left_cc_ro_fair,
        }

        def _r4r_conv_fair(bias, indices=None):
            bias_map_f = {
                'ro': np.array([bias[0], bias[1]]),
                'ri': np.array([bias[2], bias[3]]),
                'lo': np.array([bias[4], bias[5]]),
                'li': np.array([bias[6], bias[7]]),
            }
            distances = []
            for idx, (fn, rr) in enumerate(frame_raw_data):
                if indices is not None and idx not in indices:
                    continue
                rays = []
                for cam in cam_order:
                    if cam not in rr:
                        continue
                    gn = rr[cam] - bias_map_f[cam]
                    d_cam = np.array([gn[0], gn[1], 1.0])
                    d_cam /= np.linalg.norm(d_cam)
                    d_ro = cam_rotations[cam] @ d_cam
                    d_ro /= np.linalg.norm(d_ro)
                    if d_ro[2] < 0.3:
                        continue
                    rays.append((fair_ray_origins[cam], d_ro, cam_weights.get(cam, 1.0)))
                if len(rays) >= 2:
                    A_f = np.zeros((3, 3))
                    b_f = np.zeros(3)
                    for origin, direction, weight in rays:
                        d = direction.reshape(3, 1)
                        M = I3 - d @ d.T
                        A_f += weight * M
                        b_f += weight * (M @ origin)
                    try:
                        P = np.linalg.solve(A_f, b_f)
                        if P[2] > 0:
                            ray_mid = np.mean([o for o, _, _ in rays], axis=0)
                            distances.append(float(np.linalg.norm(P - ray_mid)))
                    except np.linalg.LinAlgError:
                        pass
            return distances

        def fair_obj_r4r(bias):
            dists = _r4r_conv_fair(bias, cal_idx if cal_cutoff_time is not None else None)
            if len(dists) < 3:
                return 1e10
            md = np.median(dists)
            dist_err = (md - known_distance_mm)**2
            var = np.var(dists)
            reg = 0.001 * np.sum(np.array(bias)**2)
            se_rh = 0.5*(bias[0]-bias[2])**2; se_rv = 0.5*(bias[1]-bias[3])**2
            se_lh = 0.5*(bias[4]-bias[6])**2; se_lv = 0.5*(bias[5]-bias[7])**2
            cx_h = 0.1*((bias[0]+bias[2])/2 + (bias[4]+bias[6])/2)**2
            cx_v = 0.1*((bias[1]+bias[3])/2 - (bias[5]+bias[7])/2)**2
            return dist_err + 0.01*var + reg + se_rh + se_rv + se_lh + se_lv + cx_h + cx_v

        fair_result = minimize(fair_obj_r4r, [0]*8, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-6})
        fair_bias = fair_result.x
        fair_bias_map = {
            'ro': np.array([fair_bias[0], fair_bias[1]]),
            'ri': np.array([fair_bias[2], fair_bias[3]]),
            'lo': np.array([fair_bias[4], fair_bias[5]]),
            'li': np.array([fair_bias[6], fair_bias[7]]),
        }
        print(f"  [CAL-R4R FAIR] bias per camera:")
        for cam in cam_order:
            fb = fair_bias_map[cam]
            print(f"    {cam.upper()}: dx={fb[0]:.6f}, dy={fb[1]:.6f}")

        # Recompute per-frame with fair CC + fair bias
        for i, (fn, rr) in enumerate(frame_raw_data):
            entry = calibrated_results[i]
            rays = []
            for cam in cam_order:
                if cam not in rr:
                    continue
                gn = rr[cam] - fair_bias_map[cam]
                d_cam = np.array([gn[0], gn[1], 1.0])
                d_cam /= np.linalg.norm(d_cam)
                d_ro = cam_rotations[cam] @ d_cam
                d_ro /= np.linalg.norm(d_ro)
                if d_ro[2] < 0.3:
                    continue
                rays.append((fair_ray_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))
            if len(rays) >= 2:
                A_f = np.zeros((3, 3))
                b_f = np.zeros(3)
                for origin, direction, _, weight in rays:
                    d = direction.reshape(3, 1)
                    M = I3 - d @ d.T
                    A_f += weight * M
                    b_f += weight * (M @ origin)
                try:
                    P = np.linalg.solve(A_f, b_f)
                    if P[2] > 0:
                        residuals_f = {}
                        for origin, direction, cam_name, _ in rays:
                            diff = P - origin
                            proj_len = np.dot(diff, direction)
                            if proj_len < 0:
                                continue
                            perp = diff - proj_len * direction
                            residuals_f[cam_name] = float(np.linalg.norm(perp))
                        if residuals_f:
                            rms_f = float(np.sqrt(np.mean([r**2 for r in residuals_f.values()])))
                            ray_mid = np.mean([o for o, _, _, _ in rays], axis=0)
                            fd = float(np.linalg.norm(P - ray_mid))
                            entry["fixation_distance_mm_fair"] = round(fd, 2)
                            entry["convergence_mm_fair"] = round(fd, 2)
                            entry["ray_miss_mm_fair"] = round(rms_f, 2)
                except np.linalg.LinAlgError:
                    pass

        fair_fix = [e.get("fixation_distance_mm_fair") for e in calibrated_results
                    if e.get("fixation_distance_mm_fair")]
        fair_cal = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results)
                    if i in cal_idx and e.get("fixation_distance_mm_fair")]
        fair_test = [e.get("fixation_distance_mm_fair") for i, e in enumerate(calibrated_results)
                     if i in test_idx and e.get("fixation_distance_mm_fair")]
        if fair_fix:
            print(f"  [CAL-R4R FAIR] median={np.median(fair_fix)/10:.1f}cm")
        if fair_cal:
            print(f"  [CAL-R4R FAIR] CAL ({len(fair_cal)}): "
                  f"err={abs(np.median(fair_cal)-known_distance_mm)/10:.1f}cm")
        if fair_test:
            print(f"  [CAL-R4R FAIR] TEST ({len(fair_test)}): "
                  f"err={abs(np.median(fair_test)-known_distance_mm)/10:.1f}cm")

    # Stats
    fix_vals = [e["fixation_distance_mm"]
                for e in calibrated_results if e["fixation_distance_mm"]]
    ipd_vals = [e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]]
    miss_vals = [e["ray_miss_mm"]
                 for e in calibrated_results if e["ray_miss_mm"] is not None]
    if fix_vals:
        print(f"  [CAL-R4R] Calibrated: {len(fix_vals)} frames | "
              f"fixation: median={np.median(fix_vals)/10:.1f}cm "
              f"mean={np.mean(fix_vals)/10:.1f}cm "
              f"std={np.std(fix_vals)/10:.1f}cm")
    if miss_vals:
        print(f"  [CAL-R4R] ray miss: median={np.median(miss_vals):.2f}mm "
              f"mean={np.mean(miss_vals):.2f}mm")

    # Cal/test split reporting
    if cal_cutoff_time is not None:
        all_fix = [e.get("fixation_distance_mm") for e in calibrated_results]
        _report_cal_test_stats("CAL-R4R", all_fix, cal_idx, test_idx, known_distance_mm)

    conv_meta = {
        "method": "reflection_4ray_weighted_calibrated",
        "description": f"Reflect 4-ray calibrated to {known_distance_mm/10:.0f}cm "
                       f"(per-camera gaze bias / kappa correction)",
        "cc_estimation_method": "reflection_law",
        "calibration": {
            "known_distance_mm": known_distance_mm,
            "bias_per_camera": {cam: [round(float(bias_map[cam][0]), 6),
                                      round(float(bias_map[cam][1]), 6)]
                                for cam in cam_order},
            "bias_per_camera_deg": {cam: [round(float(np.degrees(np.arctan(bias_map[cam][0]))), 3),
                                          round(float(np.degrees(np.arctan(bias_map[cam][1]))), 3)]
                                    for cam in cam_order},
        },
        "median_fixation_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "mean_fixation_mm": round(float(np.mean(fix_vals)), 2) if fix_vals else None,
        "std_fixation_mm": round(float(np.std(fix_vals)), 2) if fix_vals else None,
        "median_convergence_mm": round(float(np.median(fix_vals)), 2) if fix_vals else None,
        "median_ipd_mm": round(float(np.median(ipd_vals)), 2) if ipd_vals else None,
        "n_frames": len(fix_vals),
        "right_corneal_center_ro": meta.get("right_corneal_center_ro"),
        "left_corneal_center_lo": meta.get("left_corneal_center_lo"),
        "personal_radius": meta.get("personal_radius"),
        "kappa": kappa_data if kappa_data else None,
        "camera_weights": {cam: round(w, 4) for cam, w in cam_weights.items()},
        "cal_test_split": {
            "cal_cutoff_time": cal_cutoff_time,
            "n_cal": n_cal,
            "n_test": n_test,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "n_cameras": e.get("n_cameras"),
            "per_camera_residual": e.get("per_camera_residual"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
            "kappa_right_deg": e.get("kappa_right_deg"),
            "kappa_left_deg": e.get("kappa_left_deg"),
        } for e in calibrated_results if e["fixation_distance_mm"]],
    }

    # Fair CC results
    if fair_cc and fair_bias_map is not None:
        conv_meta["fair_cc"] = {
            "right_corneal_center_ro_fair": [round(float(v), 4) for v in right_cc_ro_fair],
            "left_corneal_center_lo_fair": [round(float(v), 4) for v in left_cc_lo_fair],
            "bias_per_camera_fair": {cam: [round(float(fair_bias_map[cam][0]), 6),
                                           round(float(fair_bias_map[cam][1]), 6)]
                                    for cam in cam_order},
            "median_fixation_mm_fair": round(float(np.median(fair_fix)), 2) if fair_fix else None,
            "cal_median_mm": round(float(np.median(fair_cal)), 2) if fair_cal else None,
            "test_median_mm": round(float(np.median(fair_test)), 2) if fair_test else None,
            "cal_err_cm": round(abs(np.median(fair_cal)-known_distance_mm)/10, 2) if fair_cal else None,
            "test_err_cm": round(abs(np.median(fair_test)-known_distance_mm)/10, 2) if fair_test else None,
            "n_cal": len(fair_cal), "n_test": len(fair_test),
        }
        for pf in conv_meta["per_frame"]:
            match = next((e for e in calibrated_results if e["frame"] == pf["frame"]), None)
            if match:
                pf["fixation_distance_mm_fair"] = match.get("fixation_distance_mm_fair")
                pf["convergence_mm_fair"] = match.get("convergence_mm_fair")
                pf["ray_miss_mm_fair"] = match.get("ray_miss_mm_fair")

    conv_path = out_base / "convergence_meta_reflect4ray_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CAL-R4R] Saved to {conv_path}")

    # --- Huber variant: re-solve with Huber IRLS using same calibrated gaze ---
    huber_results = []
    for i, (frame_name, raw_rays) in enumerate(frame_raw_data):
        h_entry = {"frame": frame_name, "convergence_mm": None,
                   "fixation_distance_mm": None, "convergence_point": None,
                   "ray_miss_mm": None, "ipd_mm": None,
                   "n_cameras": 0, "per_camera_residual": None,
                   "huber_weights": None, "is_calibration": i in cal_idx}
        rays = []
        for cam in cam_order:
            if cam not in raw_rays:
                continue
            gn = raw_rays[cam] - bias_map[cam]
            d_cam = np.array([gn[0], gn[1], 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ro = cam_rotations[cam] @ d_cam
            d_ro /= np.linalg.norm(d_ro)
            if d_ro[2] < 0.3:
                continue
            rays.append((ray_origins[cam], d_ro, cam, cam_weights.get(cam, 1.0)))
        h_entry["n_cameras"] = len(rays)
        if len(rays) >= 2:
            h_sol = _solve_nray_fixation(rays, huber_delta=5.0)
            if h_sol:
                h_entry["fixation_distance_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_mm"] = round(h_sol['fixation_dist'], 2)
                h_entry["convergence_point"] = [round(float(h_sol['P'][k]), 2) for k in range(3)]
                h_entry["ray_miss_mm"] = round(h_sol['rms_residual'], 2)
                h_entry["per_camera_residual"] = {k: round(v, 2) for k, v in h_sol['residuals'].items()}
                h_entry["huber_weights"] = {k: round(v, 4) for k, v in h_sol.get('huber_weights', {}).items()}
        std_entry = calibrated_results[i]
        h_entry["ipd_mm"] = std_entry.get("ipd_mm")
        huber_results.append(h_entry)

    h_fix = [e["fixation_distance_mm"] for e in huber_results if e["fixation_distance_mm"]]
    if h_fix:
        print(f"  [CAL-R4R HUBER] {len(h_fix)} frames | "
              f"fixation: median={np.median(h_fix)/10:.1f}cm "
              f"std={np.std(h_fix)/10:.1f}cm")
    h_meta = dict(conv_meta)
    h_meta["method"] = "huber_reflect4ray_calibrated"
    h_meta["huber_delta_mm"] = 5.0
    h_meta["median_fixation_mm"] = round(float(np.median(h_fix)), 2) if h_fix else None
    h_meta["per_frame"] = [{
        "frame": e["frame"], "convergence_mm": e["convergence_mm"],
        "fixation_distance_mm": e["fixation_distance_mm"],
        "convergence_point": e.get("convergence_point"),
        "ray_miss_mm": e.get("ray_miss_mm"), "n_cameras": e.get("n_cameras"),
        "per_camera_residual": e.get("per_camera_residual"),
        "huber_weights": e.get("huber_weights"),
        "ipd_mm": e.get("ipd_mm"), "is_calibration": e.get("is_calibration"),
    } for e in huber_results if e["fixation_distance_mm"]]
    h_path = out_base / "convergence_meta_huber_reflect4ray_cal.json"
    with open(str(h_path), "w") as f:
        json.dump(h_meta, f, indent=2)
    print(f"  [CAL-R4R HUBER] Saved to {h_path}")



def calibrate_physreflect_convergence(output_dir, calib, known_distance_mm=500.0,
                                       cal_cutoff_time=None, fair_cc=False):
    """Calibrate per-eye gaze bias for Physical Reflect C3D using known fixation distance.

    Same 4-param per-eye bias as calibrate_reflect_c3d_convergence, but reads from
    convergence_meta_physreflectc3d.json. Saves to convergence_meta_physreflectc3d_cal.json.
    """
    from scipy.optimize import minimize as _minimize

    out_base = Path(output_dir)

    meta_path = out_base / "convergence_meta_physreflectc3d.json"
    if not meta_path.exists():
        print("  [CAL-PHYS] No physical reflect c3d results to calibrate")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("right_corneal_center_ro_physical") is None or        meta.get("left_corneal_center_lo_physical") is None:
        print("  [CAL-PHYS] Missing corneal centers in phys reflect meta")
        return

    right_cc = np.array(meta["right_corneal_center_ro_physical"])
    left_cc = np.array(meta["left_corneal_center_lo_physical"])

    cross = calib.get("cross")
    if cross is None:
        print("  [CAL-PHYS] No cross-pair calibration")
        return
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load pupil_3d
    r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
    l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
    if not r_seg_path.exists() or not l_seg_path.exists():
        print("  [CAL-PHYS] Need seg combined results")
        return
    with open(r_seg_path) as f:
        r_seg = {e["frame"]: e for e in json.load(f)}
    with open(l_seg_path) as f:
        l_seg = {e["frame"]: e for e in json.load(f)}

    common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
    valid_frames = []
    for frame_name in common_frames:
        rp = r_seg[frame_name].get("pupil_3d")
        lp = l_seg[frame_name].get("pupil_3d")
        if rp and lp:
            valid_frames.append((frame_name, np.array(rp), np.array(lp)))

    if len(valid_frames) < 3:
        print(f"  [CAL-PHYS] Only {len(valid_frames)} valid frames, need >= 3")
        return

    r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
    l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])

    raw_gazes = []
    for frame_name, rp, lp_lo in valid_frames:
        r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
        r_gaze = r_pupil_proj - r_cc_proj
        l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
        l_gaze = l_pupil_proj - l_cc_proj
        raw_gazes.append((r_gaze, l_gaze))

    cal_idx, test_idx = _split_by_cutoff(
        valid_frames, cal_cutoff_time, lambda x: x[0])
    n_cal = len(cal_idx)
    n_test = len(test_idx)

    print(f"  [CAL-PHYS] {len(valid_frames)} valid frames, "
          f"target distance = {known_distance_mm/10:.1f} cm")
    if cal_cutoff_time is not None:
        print(f"  [CAL-PHYS] Split: {n_cal} cal + {n_test} test")

    def _convergence_distances(bias, indices=None):
        dx_r, dy_r, dx_l, dy_l = bias
        distances = []
        ray_misses = []
        for i, (r_gaze_raw, l_gaze_raw) in enumerate(raw_gazes):
            if indices is not None and i not in indices:
                continue
            r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
            l_gaze = l_gaze_raw - np.array([dx_l, dy_l])
            r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
            r_dir = r_dir / np.linalg.norm(r_dir)
            l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
            l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
            l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
            l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)
            w0 = -lo_origin_ro
            a = float(np.dot(r_dir, r_dir))
            b_val = float(np.dot(r_dir, l_dir_ro))
            c = float(np.dot(l_dir_ro, l_dir_ro))
            d_val = float(np.dot(r_dir, w0))
            e_val = float(np.dot(l_dir_ro, w0))
            denom = a * c - b_val * b_val
            if abs(denom) < 1e-10:
                continue
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                distances.append(fix_dist)
                ray_misses.append(ray_miss)
        return distances, ray_misses

    def objective(bias):
        distances, _ = _convergence_distances(bias, cal_idx if cal_cutoff_time is not None else None)
        if len(distances) < 3:
            return 1e10
        median_dist = np.median(distances)
        dist_error = (median_dist - known_distance_mm) ** 2
        variance = np.var(distances)
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2
        return dist_error + 0.01 * variance + reg + sym_h + sym_v

    result = _minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-6})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [CAL-PHYS] Optimized gaze bias:")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f} deg), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f} deg)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f} deg), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f} deg)")

    # Recompute with calibrated bias + kappa decomposition
    calibrated_results = []
    kappa_right_h, kappa_right_v, kappa_right_mag = [], [], []
    kappa_left_h, kappa_left_v, kappa_left_mag = [], [], []

    for i, (frame_name, rp, lp_lo) in enumerate(valid_frames):
        r_gaze_raw, l_gaze_raw = raw_gazes[i]
        r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
        l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

        entry = {"frame": frame_name, "convergence_mm": None,
                 "fixation_distance_mm": None, "convergence_point": None,
                 "ray_miss_mm": None, "ipd_mm": None,
                 "right_pupil_3d": None, "left_pupil_3d_ro": None,
                 "is_calibration": i in cal_idx}

        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val

        if abs(denom) > 1e-10:
            sc = (b_val * e_val - c * d_val) / denom
            tc = (a * e_val - b_val * d_val) / denom
            if sc > 0 and tc > 0:
                closest_r = sc * r_dir
                closest_l = lo_origin_ro + tc * l_dir_ro
                conv_pt = (closest_r + closest_l) / 2.0
                ray_miss = float(np.linalg.norm(closest_r - closest_l))
                cam_mid = lo_origin_ro / 2.0
                fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
                entry["fixation_distance_mm"] = round(fix_dist, 2)
                entry["convergence_mm"] = round(fix_dist, 2)
                entry["convergence_point"] = [round(float(conv_pt[k]), 2) for k in range(3)]
                entry["ray_miss_mm"] = round(ray_miss, 2)

                # Kappa decomposition
                target_ro = conv_pt
                target_lo = (R_cross @ target_ro.reshape(3, 1) + T_cross).flatten()

                # Right eye kappa
                cc_to_pupil_dist_r = float(np.linalg.norm(rp - right_cc))
                d_visual_r = target_ro - right_cc
                d_visual_r = d_visual_r / np.linalg.norm(d_visual_r)
                p_zk_r = right_cc + cc_to_pupil_dist_r * d_visual_r

                pupil_proj_r = np.array([rp[0] / rp[2], rp[1] / rp[2]])
                zk_proj_r = np.array([p_zk_r[0] / p_zk_r[2], p_zk_r[1] / p_zk_r[2]])
                kappa_gn_r = pupil_proj_r - zk_proj_r

                kr_h = float(np.degrees(np.arctan(kappa_gn_r[0])))
                kr_v = float(np.degrees(np.arctan(kappa_gn_r[1])))
                kr_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_r))))
                kappa_right_h.append(kr_h)
                kappa_right_v.append(kr_v)
                kappa_right_mag.append(kr_mag)

                # Left eye kappa
                cc_to_pupil_dist_l = float(np.linalg.norm(lp_lo - left_cc))
                d_visual_l = target_lo - left_cc
                d_visual_l = d_visual_l / np.linalg.norm(d_visual_l)
                p_zk_l = left_cc + cc_to_pupil_dist_l * d_visual_l

                pupil_proj_l = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
                zk_proj_l = np.array([p_zk_l[0] / p_zk_l[2], p_zk_l[1] / p_zk_l[2]])
                kappa_gn_l = pupil_proj_l - zk_proj_l

                kl_h = float(np.degrees(np.arctan(kappa_gn_l[0])))
                kl_v = float(np.degrees(np.arctan(kappa_gn_l[1])))
                kl_mag = float(np.degrees(np.arctan(np.linalg.norm(kappa_gn_l))))
                kappa_left_h.append(kl_h)
                kappa_left_v.append(kl_v)
                kappa_left_mag.append(kl_mag)

                entry["kappa_right_deg"] = [round(kr_h, 3), round(kr_v, 3), round(kr_mag, 3)]
                entry["kappa_left_deg"] = [round(kl_h, 3), round(kl_v, 3), round(kl_mag, 3)]

        # IPD
        lp_ro = (R_cross.T @ (lp_lo.reshape(3, 1) - T_cross)).flatten()
        ipd = float(np.linalg.norm(rp - lp_ro))
        entry["ipd_mm"] = round(ipd, 2)
        entry["right_pupil_3d"] = [round(float(rp[k]), 4) for k in range(3)]
        entry["left_pupil_3d_ro"] = [round(float(lp_ro[k]), 4) for k in range(3)]

        calibrated_results.append(entry)

    # Stats
    cal_distances, cal_misses = _convergence_distances(result.x, cal_idx if cal_cutoff_time else None)
    all_distances, all_misses = _convergence_distances(result.x)
    _report_cal_test_stats("[CAL-PHYS]", all_distances, cal_idx, test_idx, known_distance_mm)

    conv_meta = {
        "method": "physical_reflection_c3d_calibrated",
        "description": "Calibrated physical specular reflection CC + C3D gaze",
        "cc_estimation_method": "specular_reflection",
        "corneal_radius_mm": 7.8,
        "gaze_bias": {
            "right": {"dx": round(dx_r, 6), "dy": round(dy_r, 6),
                      "h_deg": round(float(np.degrees(np.arctan(dx_r))), 3),
                      "v_deg": round(float(np.degrees(np.arctan(dy_r))), 3)},
            "left": {"dx": round(dx_l, 6), "dy": round(dy_l, 6),
                     "h_deg": round(float(np.degrees(np.arctan(dx_l))), 3),
                     "v_deg": round(float(np.degrees(np.arctan(dy_l))), 3)},
        },
        "median_fixation_mm": round(float(np.median(all_distances)), 2) if all_distances else None,
        "mean_fixation_mm": round(float(np.mean(all_distances)), 2) if all_distances else None,
        "std_fixation_mm": round(float(np.std(all_distances)), 2) if all_distances else None,
        "median_ray_miss_mm": round(float(np.median(all_misses)), 2) if all_misses else None,
        "median_ipd_mm": round(float(np.median([e["ipd_mm"] for e in calibrated_results if e["ipd_mm"]])), 2)             if any(e["ipd_mm"] for e in calibrated_results) else None,
        "n_frames": len([e for e in calibrated_results if e["convergence_mm"]]),
        "right_corneal_center_ro": [round(float(v), 4) for v in right_cc],
        "left_corneal_center_lo": [round(float(v), 4) for v in left_cc],
        "cal_test_split": {
            "n_cal": n_cal,
            "n_test": n_test,
            "cutoff_time": cal_cutoff_time,
        } if cal_cutoff_time is not None else None,
        "per_frame": [{
            "frame": e["frame"],
            "convergence_mm": e["convergence_mm"],
            "fixation_distance_mm": e["fixation_distance_mm"],
            "convergence_point": e.get("convergence_point"),
            "ray_miss_mm": e.get("ray_miss_mm"),
            "ipd_mm": e.get("ipd_mm"),
            "right_pupil_3d": e.get("right_pupil_3d"),
            "left_pupil_3d_ro": e.get("left_pupil_3d_ro"),
            "is_calibration": e.get("is_calibration"),
            "kappa_right_deg": e.get("kappa_right_deg"),
            "kappa_left_deg": e.get("kappa_left_deg"),
        } for e in calibrated_results if e["convergence_mm"]],
    }

    conv_path = out_base / "convergence_meta_physreflectc3d_cal.json"
    with open(str(conv_path), "w") as f:
        json.dump(conv_meta, f, indent=2)
    print(f"  [CAL-PHYS] Saved to {conv_path}")

def _map_frames_to_dots(task_dir):
    """Map frame timestamps to their VisualDot display windows.

    Parses logs.json to extract dot on/off intervals, then for each frame
    (identified by its timestamp in the filename) finds which dot it was
    viewing during that time.

    Returns:
        dot_windows: list of dicts {position: (x,y), marker_id, on_time, off_time, pass_num}
            pass_num = which repetition of this dot position (0-based)
        unique_positions: list of (x,y) tuples in order of first appearance
    """
    logs_path = Path(task_dir) / "logs.json"
    if not logs_path.exists():
        return [], []

    with open(logs_path) as f:
        logs = json.load(f)

    first_ts = None
    raw_events = []

    for msg in logs.get("websocket_messages", []):
        parsed = json.loads(msg["ws_message"])
        ts_str = parsed.get("timestamp", "")
        if not isinstance(ts_str, str): ts_str = str(ts_str)
        if not ts_str or "T" not in ts_str:
            continue
        try:
            h, m, s = ts_str.split("T")[1].split(":")
            ts_sec = int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, IndexError):
            continue
        if first_ts is None:
            first_ts = ts_sec

        et = parsed.get("eventType", "")
        if et == "VisualDot":
            details = parsed.get("details", {})
            rel = details.get("relativePosition", {})
            raw_events.append({
                "type": "dot_on",
                "time": ts_sec - first_ts,
                "position": (rel.get("x", 0), rel.get("y", 0)),
                "marker_id": parsed.get("markerId"),
            })
        elif et == "Event" and parsed.get("details", {}).get("action") == "dot_off":
            raw_events.append({
                "type": "dot_off",
                "time": ts_sec - first_ts,
            })

    if not raw_events:
        return [], []

    # Build dot windows from on/off pairs
    dot_windows = []
    unique_positions = []
    pos_count = {}  # position -> count for pass_num
    current_dot = None

    for ev in raw_events:
        if ev["type"] == "dot_on":
            current_dot = ev
        elif ev["type"] == "dot_off" and current_dot is not None:
            pos = current_dot["position"]
            if pos not in pos_count:
                unique_positions.append(pos)
                pos_count[pos] = 0
            else:
                pos_count[pos] += 1
            dot_windows.append({
                "position": pos,
                "marker_id": current_dot["marker_id"],
                "on_time": current_dot["time"],
                "off_time": ev["time"],
                "pass_num": pos_count[pos],
            })
            current_dot = None

    return dot_windows, unique_positions


def _match_frame_to_dot(frame_name, dot_windows):
    """Match a single frame name to a dot window using its embedded timestamp.

    Returns index into dot_windows, or None if frame is outside all dot windows.
    """
    ft = _frame_timestamp(frame_name)
    if ft is None:
        return None
    for di, dw in enumerate(dot_windows):
        if dw["on_time"] <= ft <= dw["off_time"]:
            return di
    return None


def _get_spread_order(positions):
    """Get a D-optimal spread order for dot positions (maximal spatial coverage).

    Starts with the center dot, then greedily adds the dot farthest from all
    already-selected dots. This gives good coverage with any subset size.
    """
    if not positions:
        return []

    positions = list(positions)
    n = len(positions)
    if n <= 1:
        return list(range(n))

    # Start with center dot (closest to mean position)
    mean_x = sum(p[0] for p in positions) / n
    mean_y = sum(p[1] for p in positions) / n
    center_idx = min(range(n), key=lambda i: (positions[i][0] - mean_x)**2 + (positions[i][1] - mean_y)**2)

    order = [center_idx]
    remaining = set(range(n)) - {center_idx}

    while remaining:
        # Add the dot farthest from all selected dots (max min-distance)
        best_idx = max(remaining, key=lambda i: min(
            (positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2
            for j in order
        ))
        order.append(best_idx)
        remaining.remove(best_idx)

    return order


def calibrate_multi_distance(sessions, calib, method='reflectc3d', n_cal_dots=9):
    """Multi-distance calibration using data from multiple recordings.

    Optimizes gaze bias parameters using frames from multiple recordings at
    different known distances simultaneously. This is more constrained than
    single-distance calibration because the same bias must produce correct
    convergence distances across all distances.

    Args:
        sessions: list of dicts, each with:
            - output_dir: str, path to processing output (e.g. output_saccade33/)
            - distance_mm: float, known screen distance in mm
            - task_dir: str, path to recording dir (for logs.json dot events)
            - is_cal: bool, whether to use this session for calibration (vs test-only)
        calib: stereo calibration dict (needs 'cross' pair)
        method: convergence method to calibrate. Currently supports 'reflectc3d'.
        n_cal_dots: int 1-9, number of dot positions to use for calibration from each cal session.

    Returns:
        dict with calibration results, or None on failure.
    """
    from scipy.optimize import minimize

    TAG = "MULTI-DIST"

    if method != 'reflectc3d':
        print(f"  [{TAG}] Only 'reflectc3d' method is currently supported")
        return None

    cross = calib.get("cross")
    if cross is None:
        print(f"  [{TAG}] No cross-pair calibration")
        return None
    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)
    lo_origin_ro = (-R_cross.T @ T_cross).flatten()

    # Load data from each session
    session_data = []
    for sess in sessions:
        out_base = Path(sess["output_dir"])
        task_dir = Path(sess["task_dir"])
        distance_mm = float(sess["distance_mm"])
        is_cal = sess.get("is_cal", True)

        # Load reflect c3d meta for CC positions
        meta_path = out_base / "convergence_meta_reflectc3d.json"
        if not meta_path.exists():
            print(f"  [{TAG}] No reflectc3d results in {out_base}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        right_cc = meta.get("right_corneal_center_ro")
        left_cc = meta.get("left_corneal_center_lo")
        if right_cc is None or left_cc is None:
            print(f"  [{TAG}] Missing corneal centers for {out_base}")
            continue
        right_cc = np.array(right_cc)
        left_cc = np.array(left_cc)

        # Load pupil_3d from seg combined
        r_seg_path = out_base / "right_seg_combined" / "combined_results.json"
        l_seg_path = out_base / "left_seg_combined" / "combined_results.json"
        if not r_seg_path.exists() or not l_seg_path.exists():
            print(f"  [{TAG}] Missing seg combined for {out_base}")
            continue
        with open(r_seg_path) as f:
            r_seg = {e["frame"]: e for e in json.load(f)}
        with open(l_seg_path) as f:
            l_seg = {e["frame"]: e for e in json.load(f)}

        # Collect valid frames with pupil_3d for both eyes
        common_frames = sorted(set(r_seg.keys()) & set(l_seg.keys()))
        valid_frames = []
        for fn in common_frames:
            rp = r_seg[fn].get("pupil_3d")
            lp = l_seg[fn].get("pupil_3d")
            if rp and lp:
                valid_frames.append((fn, np.array(rp), np.array(lp)))

        if not valid_frames:
            print(f"  [{TAG}] No valid frames for {out_base}")
            continue

        # Compute raw gaze
        r_cc_proj = np.array([right_cc[0] / right_cc[2], right_cc[1] / right_cc[2]])
        l_cc_proj = np.array([left_cc[0] / left_cc[2], left_cc[1] / left_cc[2]])
        raw_gazes = []
        for fn, rp, lp_lo in valid_frames:
            r_pupil_proj = np.array([rp[0] / rp[2], rp[1] / rp[2]])
            r_gaze = r_pupil_proj - r_cc_proj
            l_pupil_proj = np.array([lp_lo[0] / lp_lo[2], lp_lo[1] / lp_lo[2]])
            l_gaze = l_pupil_proj - l_cc_proj
            raw_gazes.append((r_gaze, l_gaze))

        # Map frames to dot positions using timestamps
        dot_windows, unique_positions = _map_frames_to_dots(str(task_dir))
        spread_order = _get_spread_order(unique_positions)

        # Determine cal/test frame indices
        # Cal: first pass (pass_num=0) of first N dots in spread order
        cal_positions = set()
        if is_cal and unique_positions:
            n_use = min(n_cal_dots, len(unique_positions))
            for si in spread_order[:n_use]:
                cal_positions.add(unique_positions[si])

        cal_indices = set()
        test_indices = set()
        for i, (fn, _, _) in enumerate(valid_frames):
            dot_idx = _match_frame_to_dot(fn, dot_windows)
            if dot_idx is not None and is_cal:
                dw = dot_windows[dot_idx]
                if dw["position"] in cal_positions and dw["pass_num"] == 0:
                    cal_indices.add(i)
                else:
                    test_indices.add(i)
            else:
                test_indices.add(i)

        name = Path(sess["task_dir"]).name.split("__")[0]
        print(f"  [{TAG}] Session '{name}': {len(valid_frames)} frames, "
              f"distance={distance_mm/10:.0f}cm, "
              f"cal={len(cal_indices)} test={len(test_indices)} "
              f"{'[CAL SOURCE]' if is_cal else '[TEST ONLY]'}")

        session_data.append({
            "name": name,
            "distance_mm": distance_mm,
            "is_cal": is_cal,
            "valid_frames": valid_frames,
            "raw_gazes": raw_gazes,
            "right_cc": right_cc,
            "left_cc": left_cc,
            "cal_indices": cal_indices,
            "test_indices": test_indices,
            "dot_windows": dot_windows,
            # frame_dot_map removed — using _match_frame_to_dot() per frame instead
            "unique_positions": unique_positions,
            "spread_order": spread_order,
            "output_dir": str(sess["output_dir"]),
            "task_dir": str(sess["task_dir"]),
        })

    if not session_data:
        print(f"  [{TAG}] No valid sessions loaded")
        return None

    cal_sessions = [s for s in session_data if s["is_cal"]]
    if not cal_sessions:
        print(f"  [{TAG}] No calibration source sessions")
        return None

    total_cal_frames = sum(len(s["cal_indices"]) for s in cal_sessions)
    print(f"  [{TAG}] Total: {len(session_data)} sessions, "
          f"{total_cal_frames} calibration frames, "
          f"n_cal_dots={n_cal_dots}")

    def _convergence_distance_one(r_gaze_raw, l_gaze_raw, bias):
        """Compute convergence distance for a single frame with given bias."""
        dx_r, dy_r, dx_l, dy_l = bias
        r_gaze = r_gaze_raw - np.array([dx_r, dy_r])
        l_gaze = l_gaze_raw - np.array([dx_l, dy_l])

        r_dir = np.array([r_gaze[0], r_gaze[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)
        l_dir_lo = np.array([l_gaze[0], l_gaze[1], 1.0])
        l_dir_lo = l_dir_lo / np.linalg.norm(l_dir_lo)
        l_dir_ro = (R_cross.T @ l_dir_lo.reshape(3, 1)).flatten()
        l_dir_ro = l_dir_ro / np.linalg.norm(l_dir_ro)

        w0 = -lo_origin_ro
        a = float(np.dot(r_dir, r_dir))
        b_val = float(np.dot(r_dir, l_dir_ro))
        c = float(np.dot(l_dir_ro, l_dir_ro))
        d_val = float(np.dot(r_dir, w0))
        e_val = float(np.dot(l_dir_ro, w0))
        denom = a * c - b_val * b_val
        if abs(denom) < 1e-10:
            return None, None, None
        sc = (b_val * e_val - c * d_val) / denom
        tc = (a * e_val - b_val * d_val) / denom
        if sc > 0 and tc > 0:
            closest_r = sc * r_dir
            closest_l = lo_origin_ro + tc * l_dir_ro
            conv_pt = (closest_r + closest_l) / 2.0
            ray_miss = float(np.linalg.norm(closest_r - closest_l))
            cam_mid = lo_origin_ro / 2.0
            fix_dist = float(np.linalg.norm(conv_pt - cam_mid))
            return fix_dist, ray_miss, conv_pt
        return None, None, None

    def objective(bias):
        """Multi-distance objective: minimize error across all cal sessions."""
        total_error = 0.0
        n_sessions_with_data = 0

        for s in session_data:
            if not s["is_cal"] or not s["cal_indices"]:
                continue
            distances = []
            for i in s["cal_indices"]:
                r_gaze_raw, l_gaze_raw = s["raw_gazes"][i]
                fd, _, _ = _convergence_distance_one(r_gaze_raw, l_gaze_raw, bias)
                if fd is not None:
                    distances.append(fd)

            if len(distances) < 2:
                continue
            n_sessions_with_data += 1

            median_dist = np.median(distances)
            dist_error = (median_dist - s["distance_mm"]) ** 2
            variance = np.var(distances)
            total_error += dist_error + 0.01 * variance

        if n_sessions_with_data == 0:
            return 1e10

        # Regularisation
        reg = 0.001 * np.sum(np.array(bias) ** 2)
        # Symmetry constraints
        sym_h = 0.1 * (bias[0] + bias[2]) ** 2
        sym_v = 0.1 * (bias[1] - bias[3]) ** 2

        return total_error / n_sessions_with_data + reg + sym_h + sym_v

    result = minimize(objective, [0, 0, 0, 0], method='Nelder-Mead',
                      options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-6})

    dx_r, dy_r, dx_l, dy_l = result.x
    print(f"  [{TAG}] Optimized gaze bias:")
    print(f"    Right eye: dx={dx_r:.6f} ({np.degrees(np.arctan(dx_r)):.3f}°), "
          f"dy={dy_r:.6f} ({np.degrees(np.arctan(dy_r)):.3f}°)")
    print(f"    Left eye:  dx={dx_l:.6f} ({np.degrees(np.arctan(dx_l)):.3f}°), "
          f"dy={dy_l:.6f} ({np.degrees(np.arctan(dy_l)):.3f}°)")

    # Apply calibrated bias and compute metrics per session
    per_session_results = []
    all_per_frame = []
    for s in session_data:
        cal_dists = []
        test_dists = []
        all_dists = []
        session_frames = []

        for i, (fn, rp, lp_lo) in enumerate(s["valid_frames"]):
            r_gaze_raw, l_gaze_raw = s["raw_gazes"][i]
            fd, rm, cp = _convergence_distance_one(r_gaze_raw, l_gaze_raw, result.x)
            is_cal = i in s["cal_indices"]

            entry = {
                "frame": fn,
                "session": s["name"],
                "distance_mm_known": s["distance_mm"],
                "fixation_distance_mm": round(fd, 2) if fd else None,
                "convergence_mm": round(fd, 2) if fd else None,
                "ray_miss_mm": round(rm, 2) if rm else None,
                "convergence_point": [round(float(v), 3) for v in cp] if cp is not None else None,
                "is_calibration": is_cal,
            }
            session_frames.append(entry)
            all_per_frame.append(entry)

            if fd is not None:
                all_dists.append(fd)
                if is_cal:
                    cal_dists.append(fd)
                else:
                    test_dists.append(fd)

        # Per-session stats
        sr = {
            "name": s["name"],
            "distance_mm_known": s["distance_mm"],
            "is_cal": s["is_cal"],
            "n_frames": len(s["valid_frames"]),
            "n_cal": len(s["cal_indices"]),
            "n_test": len(s["test_indices"]),
        }
        if all_dists:
            sr["median_fixation_mm"] = round(float(np.median(all_dists)), 2)
            sr["std_fixation_mm"] = round(float(np.std(all_dists)), 2)
            sr["error_mm"] = round(abs(float(np.median(all_dists)) - s["distance_mm"]), 2)
        if cal_dists:
            sr["cal_median_mm"] = round(float(np.median(cal_dists)), 2)
            sr["cal_error_mm"] = round(abs(float(np.median(cal_dists)) - s["distance_mm"]), 2)
        if test_dists:
            sr["test_median_mm"] = round(float(np.median(test_dists)), 2)
            sr["test_error_mm"] = round(abs(float(np.median(test_dists)) - s["distance_mm"]), 2)

        per_session_results.append(sr)
        tag = "CAL" if s["is_cal"] else "TEST"
        print(f"  [{TAG}] {s['name']} [{tag}]: "
              f"median={sr.get('median_fixation_mm', 0)/10:.1f}cm "
              f"(known={s['distance_mm']/10:.0f}cm, "
              f"err={sr.get('error_mm', 0)/10:.1f}cm)")
        if cal_dists:
            print(f"    CAL ({len(cal_dists)} frames): "
                  f"median={np.median(cal_dists)/10:.1f}cm "
                  f"err={abs(np.median(cal_dists)-s['distance_mm'])/10:.1f}cm")
        if test_dists:
            print(f"    TEST ({len(test_dists)} frames): "
                  f"median={np.median(test_dists)/10:.1f}cm "
                  f"err={abs(np.median(test_dists)-s['distance_mm'])/10:.1f}cm")

    # Save results to ALL session output dirs so each recording can load it
    conv_meta = {
        "method": "multi_distance_reflectc3d",
        "description": f"Multi-distance calibration across {len(session_data)} sessions",
        "n_cal_dots": n_cal_dots,
        "calibration": {
            "bias_right": [round(float(dx_r), 6), round(float(dy_r), 6)],
            "bias_left": [round(float(dx_l), 6), round(float(dy_l), 6)],
            "bias_right_deg": [round(float(np.degrees(np.arctan(dx_r))), 3),
                               round(float(np.degrees(np.arctan(dy_r))), 3)],
            "bias_left_deg": [round(float(np.degrees(np.arctan(dx_l))), 3),
                              round(float(np.degrees(np.arctan(dy_l))), 3)],
        },
        "sessions": per_session_results,
        "per_frame": all_per_frame,
    }

    saved_dirs = set()
    for s in session_data:
        save_dir = Path(s["output_dir"])
        if str(save_dir) in saved_dirs:
            continue
        saved_dirs.add(str(save_dir))
        conv_path = save_dir / "convergence_meta_multidist_cal.json"
        with open(str(conv_path), "w") as f:
            json.dump(conv_meta, f, indent=2)
        print(f"  [{TAG}] Saved to {conv_path}")

    return conv_meta


def _nk(name):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def _get_output_base(base_dir, algorithm, output_prefix="output"):
    """Return the output base directory for a given algorithm."""
    if algorithm == 'contour':
        return base_dir / output_prefix
    return base_dir / f"{output_prefix}_{algorithm}"


def _run_batch_for_algorithm(algorithm, base, task_dir, crop_size,
                             seg_enabled=False, seg_algo='worldcoin',
                             led_positions=None, cameras=None, calib_dir=None,
                             output_prefix="output", screen_distance=None,
                             calibrate_only=False, convergence_only=False,
                             fair_cc=False, camera_batch='default', crop_method="blob"):
    """Run batch processing for a single algorithm. Used for parallel execution."""
    out_base = _get_output_base(base, algorithm, output_prefix)

    # Load calibration: explicit path > new JSON > old NPZ fallback
    calib = None
    effective_calib_dir = calib_dir
    if calib_dir:
        calib = load_calibration(calib_dir)
    if calib is None:
        new_calib = Path("/Users/suleymanozdel/Downloads/New Recording and Calibration/stereo_calib_2")
        if new_calib.exists():
            calib = load_calibration(new_calib)
            if calib:
                effective_calib_dir = str(new_calib)
    if calib is None:
        old_calib = task_dir / "calib_4_res"
        if old_calib.exists():
            calib = load_calibration(old_calib)
            if calib:
                effective_calib_dir = str(old_calib)
    if calib:
        print(f"[{algorithm}] Loaded stereo calibration")
    else:
        print(f"[{algorithm}] Warning: no calibration found, using raw pixel gaze")

    skip_frames = calibrate_only or convergence_only
    run_convergence = not calibrate_only or convergence_only

    if not skip_frames:
        cam_list = cameras if cameras else ["ri", "ro", "li", "lo"]
        for cam in cam_list:
            cam_input = task_dir / cam
            cam_output = out_base / cam
            if cam_input.exists():
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Processing camera: {cam.upper()}")
                print(f"{'='*60}")
                cc = calib.get(cam) if calib else None
                process_all(str(cam_input), str(cam_output),
                            crop_size=crop_size, camera=cam, cam_calib=cc,
                            algorithm=algorithm,
                            seg_enabled=seg_enabled, seg_algo=seg_algo,
                            camera_batch=camera_batch, crop_method=crop_method)
    else:
        flag = "--convergence-only" if convergence_only else "--calibrate-only"
        print(f"[{algorithm}] {flag}: skipping frame processing")

    # Combined gaze from stereo pairs (all algorithms use the same stereo triangulation)
    if calib:
        if run_convergence:
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Computing combined gaze")
            print(f"{'='*60}")
            compute_combined_gaze(str(out_base), calib, crop_size=crop_size)

            # Corneal 3D gaze first (both eyes need to exist before convergence)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Computing 3D corneal gaze (both eyes)")
            print(f"{'='*60}")
            compute_corneal_3d_gaze(str(out_base), calib, led_positions,
                                    crop_size=crop_size)

        if seg_enabled:
            if run_convergence:
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing seg combined gaze + convergence")
                print(f"{'='*60}")
                compute_combined_seg_gaze(str(out_base), calib, crop_size=crop_size)

                # Corneal 3D convergence (alternative method using stereo corneal centers)
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing corneal 3D convergence")
                print(f"{'='*60}")
                compute_corneal_3d_convergence(str(out_base), calib, crop_size=crop_size)

                # 4-ray convergence (all 4 cameras simultaneously)
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing 4-ray convergence")
                print(f"{'='*60}")
                compute_4ray_convergence(str(out_base), calib, crop_size=crop_size)

                # Huber-weighted IRLS 4-ray convergence (robust outlier rejection)
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing Huber-weighted IRLS 4-ray convergence")
                print(f"{'='*60}")
                compute_huber_4ray_convergence(str(out_base), calib, crop_size=crop_size)

                # Reflection-law 4-ray convergence (requires LED positions from calib dir)
                if effective_calib_dir:
                    led_pos_all = _load_led_positions_from_calib_dir(effective_calib_dir)
                    if led_pos_all:
                        print(f"\n{'='*60}")
                        print(f"  [{algorithm.upper()}] Computing reflection-law 4-ray convergence")
                        print(f"{'='*60}")
                        compute_reflection_4ray_convergence(
                            str(out_base), calib, led_pos_all, crop_size=crop_size)

                    # Physical reflection convergence (specular solver)
                    led_pos_physical = _load_led_positions_from_calib_dir_field(effective_calib_dir, field='point_unmirrored')
                    if led_pos_physical and led_pos_all:
                        print(f"\n{'='*60}")
                        print(f"  [{algorithm.upper()}] Computing physical reflection convergence")
                        print(f"{'='*60}")
                        compute_physical_reflection_convergence(
                            str(out_base), calib, led_pos_physical, led_pos_all, crop_size=crop_size)


                # COR (Center of Rotation) convergence
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing COR C3D convergence")
                print(f"{'='*60}")
                compute_cor_c3d_convergence(str(out_base), calib)

                # Sliding-window CC convergence (adapts to glasses slip)
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Computing Sliding-CC convergence")
                print(f"{'='*60}")
                compute_sliding_cc_convergence(str(out_base), calib)

                # === GLINT-LESS METHODS (no corneal reflection needed) ===
                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Iris-Pupil Ratio gaze")
                print("="*60)
                compute_iris_pupil_ratio_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Eyeball Model gaze")
                print("="*60)
                compute_eyeball_model_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Feature Regression gaze")
                print("="*60)
                compute_feature_regression_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Lid Geometry gaze")
                print("="*60)
                compute_lid_geometry_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Iris Landmark CR gaze")
                print("="*60)
                compute_iris_landmark_cr_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Dual Ellipse 3D gaze")
                print("="*60)
                compute_dual_ellipse_3d_gaze(str(out_base), calib, crop_size=crop_size)

                print("\n" + "="*60)
                print(f"  [{algorithm.upper()}] Computing Temporal Smooth gaze")
                print("="*60)
                compute_temporal_smooth_gaze(str(out_base), calib, crop_size=crop_size)

            # Compute cal/test split from dot events
            known_distance_mm = 600.0  # User override: 60cm screen distance
            cal_cutoff_time, n_unique_dots, n_total_dots = _compute_cal_cutoff(task_dir)
            if cal_cutoff_time is not None:
                print(f"  Cal/test split: {n_unique_dots} unique dots, "
                      f"cutoff at {cal_cutoff_time:.1f}s")

            # Calibrate Reflect C3D with known fixation distance (500mm = screen at 50cm)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Reflect C3D convergence")
            print(f"{'='*60}")
            calibrate_reflect_c3d_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            # Calibrate Corneal 3D (sphere-fit CC) with known fixation distance
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Corneal 3D convergence")
            print(f"{'='*60}")
            calibrate_corneal3d_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            # Calibrate 4-Ray with known fixation distance
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating 4-Ray convergence")
            print(f"{'='*60}")
            calibrate_4ray_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time)

            # Calibrate Reflect 4-Ray with known fixation distance
            if effective_calib_dir:
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Calibrating Reflect 4-Ray convergence")
                print(f"{'='*60}")
                calibrate_reflect4ray_convergence(
                    str(out_base), calib, known_distance_mm=known_distance_mm,
                    cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

                # Calibrate Physical Reflect C3D with known fixation distance
                print(f"\n{'='*60}")
                print(f"  [{algorithm.upper()}] Calibrating Physical Reflect C3D convergence")
                print(f"{'='*60}")
                calibrate_physreflect_convergence(
                    str(out_base), calib, known_distance_mm=known_distance_mm,
                    cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            # Calibrate COR C3D with known fixation distance
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating COR C3D convergence")
            print(f"{'='*60}")
            calibrate_cor_c3d_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time)

            # Calibrate Scene C3D — Polynomial (scene poly error minimization)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Scene C3D (Polynomial)")
            print(f"{'='*60}")
            calibrate_scene_c3d_convergence(
                str(out_base), calib, recording_dir=str(task_dir),
                cal_cutoff_time=cal_cutoff_time, transform='poly')

            # Calibrate Scene C3D — Homography (scene homography error minimization)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Scene C3D (Homography)")
            print(f"{'='*60}")
            calibrate_scene_c3d_convergence(
                str(out_base), calib, recording_dir=str(task_dir),
                cal_cutoff_time=cal_cutoff_time, transform='homography')

            # Joint Distance+Gaze Calibration
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Joint (Distance + Gaze)")
            print(f"{'='*60}")
            calibrate_joint_convergence(
                str(out_base), calib, recording_dir=str(task_dir),
                known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc,
                transform='poly', lambda_dist=1.0)


            # Calibrate Scene Sliding-CC (polynomial, per-frame CC)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Scene Sliding-CC (Polynomial)")
            print(f"{'='*60}")
            calibrate_scene_c3d_convergence(
                str(out_base), calib, recording_dir=str(task_dir),
                cal_cutoff_time=cal_cutoff_time, transform='poly',
                cc_mode='sliding')

            # Calibrate Pupil-Glint Polynomial (direct 4D -> screen mapping)
            print(f"\n{'='*60}")
            print(f"  [{algorithm.upper()}] Calibrating Pupil-Glint Polynomial")
            print(f"{'='*60}")
            calibrate_pupil_glint_poly(
                str(out_base), calib, recording_dir=str(task_dir),
                cal_cutoff_time=cal_cutoff_time, cc_mode='fixed')

            # === GLINT-LESS CALIBRATIONS ===
            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Iris-Pupil Ratio convergence")
            print("="*60)
            calibrate_iris_pupil_ratio_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Eyeball Model convergence")
            print("="*60)
            calibrate_eyeball_model_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Lid Geometry convergence")
            print("="*60)
            calibrate_lid_geometry_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Iris Landmark CR convergence")
            print("="*60)
            calibrate_iris_landmark_cr_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Dual Ellipse 3D convergence")
            print("="*60)
            calibrate_dual_ellipse_3d_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)

            print("\n" + "="*60)
            print(f"  [{algorithm.upper()}] Calibrating Temporal Smooth convergence")
            print("="*60)
            calibrate_temporal_smooth_convergence(
                str(out_base), calib, known_distance_mm=known_distance_mm,
                cal_cutoff_time=cal_cutoff_time, fair_cc=fair_cc)


    return algorithm



"""
Glint-less Gaze Estimation Methods for eye-processing pipeline.

7 new convergence/gaze methods that do NOT require corneal reflections (glints):
  1. iris_pupil_ratio       — Pupil displacement within iris → gaze direction
  2. eyeball_model          — 3D eyeball center + pupil center → gaze vector
  3. feature_regression     — Pupil + iris ellipse features → polynomial regression → gaze
  4. lid_geometry           — Eyelid openness + pupil position → gaze
  5. iris_landmark_cr       — Cross-ratio from iris landmarks (virtual reference)
  6. dual_ellipse_3d        — Pupil + iris ellipse fitting → 3D rotation → gaze
  7. temporal_smooth_gaze   — Temporal regression over pupil trajectory → smooth gaze

All methods follow the existing pattern:
  compute_*_convergence(output_dir, calib, ...)
  calibrate_*_convergence(output_dir, calib, ...)

These methods use seg_pupil_center, seg_iris_center, pupil_ellipse,
and iris_ellipse from per-frame results.json, which are extracted by
the segmentation pipeline (RITnet/Worldcoin).

Append this file's contents to eye_crop_and_glint.py.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from collections import deque


# ======================================================================
# Helper: load per-camera results
# ======================================================================
def _load_results_pair(out_base, calib, eye):
    """Load results.json for both cameras of an eye pair."""
    stereo = calib.get(eye)
    if stereo is None:
        return None, None, None, None, None

    cam1, cam2 = stereo['cam1'], stereo['cam2']
    r1_path = Path(out_base) / cam1 / "results.json"
    r2_path = Path(out_base) / cam2 / "results.json"
    if not r1_path.exists() or not r2_path.exists():
        return None, None, None, None, None

    with open(r1_path) as f:
        res1 = json.load(f)
    with open(r2_path) as f:
        res2 = json.load(f)

    return res1, res2, cam1, cam2, stereo


def _get_pupil_iris_features(r, calib_cam):
    """Extract pupil and iris features from a single frame result.
    Returns (pupil_cx, pupil_cy, iris_cx, iris_cy, pupil_w, pupil_h,
             iris_w, iris_h) in image coordinates, or None if not available.
    """
    pc = r.get("seg_pupil_center") or r.get("pupil_center")
    ic = r.get("seg_iris_center")

    if pc is None:
        return None

    pupil_cx, pupil_cy = pc[0], pc[1]
    iris_cx = ic[0] if ic else None
    iris_cy = ic[1] if ic else None

    # Ellipse parameters from seg fitting if available
    pupil_w = r.get("seg_pupil_width", r.get("pupil_radius", 10) * 2)
    pupil_h = r.get("seg_pupil_height", r.get("pupil_radius", 10) * 2)
    iris_w = r.get("seg_iris_width", 60)
    iris_h = r.get("seg_iris_height", 60)

    return (pupil_cx, pupil_cy, iris_cx, iris_cy,
            pupil_w, pupil_h, iris_w, iris_h)


def _undistort_point(pt, K, dist):
    """Undistort a single 2D point using camera intrinsics."""
    import cv2
    pts = np.array([[pt]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return float(undist[0, 0, 0]), float(undist[0, 0, 1])


def _normalize_point(pt, K):
    """Convert pixel coordinates to normalized camera coordinates."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    nx = (pt[0] - cx) / fx
    ny = (pt[1] - cy) / fy
    return nx, ny


def _triangulate_points(pt1, pt2, K1, K2, R, T, dist1=None, dist2=None):
    """Triangulate a 3D point from two 2D observations."""
    import cv2

    if dist1 is not None:
        pt1 = _undistort_point(pt1, K1, dist1)
    if dist2 is not None:
        pt2 = _undistort_point(pt2, K2, dist2)

    # Normalized coordinates
    n1 = _normalize_point(pt1, K1)
    n2 = _normalize_point(pt2, K2)

    # Projection matrices
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])

    pts1 = np.array([[pt1[0], pt1[1]]], dtype=np.float64)
    pts2 = np.array([[pt2[0], pt2[1]]], dtype=np.float64)

    pt4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pt3d = pt4d[:3] / pt4d[3]
    return pt3d.flatten()


# ======================================================================
# Method 1: Iris-Pupil Ratio
# ======================================================================
def compute_iris_pupil_ratio_gaze(output_dir, calib, crop_size=150):
    """Compute gaze using pupil displacement relative to iris center.

    Principle: When looking straight ahead, pupil center ≈ iris center.
    Gaze direction is proportional to the offset (pupil - iris) / iris_radius.

    This is the simplest glint-free method and works with any single camera.
    No corneal reflections needed — only segmentation of pupil + iris.

    For stereo: compute per-camera gaze, then triangulate for convergence.
    Saves to convergence_meta_iris_pupil_ratio.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [IRIS-PUPIL RATIO] No cross-pair calibration, skipping convergence")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    convergence_results = []

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1, dist1 = np.array(calib[cam1]['K']), np.array(calib[cam1]['dist'])
        K2, dist2 = np.array(calib[cam2]['K']), np.array(calib[cam2]['dist'])
        R = np.array(stereo['R'])
        T = np.array(stereo['T']).reshape(3, 1)

        # Per-camera gaze output
        gaze_dir = out_base / f"{eye_side}_iris_pupil_ratio"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        per_eye_results = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "ipr_gaze_norm": None,
                "ipr_gaze_deg": None,
            }

            # Get pupil + iris from both cameras
            feat1 = _get_pupil_iris_features(r1, calib.get(cam1))
            feat2 = _get_pupil_iris_features(r2, calib.get(cam2))

            gaze_vecs = []

            for feat, K, dist, cam_name in [(feat1, K1, dist1, cam1), (feat2, K2, dist2, cam2)]:
                if feat is None or feat[2] is None:  # need iris center
                    continue

                pupil_cx, pupil_cy, iris_cx, iris_cy, pw, ph, iw, ih = feat

                # Normalized displacement: (pupil - iris) / iris_size
                if iw > 5 and ih > 5:
                    dx = (pupil_cx - iris_cx) / (iw / 2.0)
                    dy = (pupil_cy - iris_cy) / (ih / 2.0)

                    # Clamp to reasonable range
                    dx = max(-1.0, min(1.0, dx))
                    dy = max(-1.0, min(1.0, dy))

                    # Convert to gaze angles (empirical scale ~30 deg max)
                    gaze_h = np.degrees(np.arcsin(dx * 0.5))  # horizontal
                    gaze_v = np.degrees(np.arcsin(dy * 0.5))  # vertical

                    gaze_vecs.append((gaze_h, gaze_v, cam_name))

            if gaze_vecs:
                # Average gaze from available cameras
                avg_h = np.mean([g[0] for g in gaze_vecs])
                avg_v = np.mean([g[1] for g in gaze_vecs])
                entry["ipr_gaze_deg"] = [round(float(avg_h), 3), round(float(avg_v), 3)]
                entry["ipr_gaze_norm"] = [round(float(np.tan(np.radians(avg_h))), 6),
                                           round(float(np.tan(np.radians(avg_v))), 6)]

            per_eye_results.append(entry)

        # Save per-eye results
        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

    # Compute convergence from stereo gaze directions
    r_gaze_path = out_base / "right_iris_pupil_ratio" / "gaze_results.json"
    l_gaze_path = out_base / "left_iris_pupil_ratio" / "gaze_results.json"

    if r_gaze_path.exists() and l_gaze_path.exists():
        with open(r_gaze_path) as f:
            r_gaze = json.load(f)
        with open(l_gaze_path) as f:
            l_gaze = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_gaze, l_gaze, R_cross, T_cross, "ipr_gaze_norm")

        conv_path = out_base / "convergence_meta_iris_pupil_ratio.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)
        n_valid = sum(1 for c in conv_results if c.get("convergence_point") is not None)
        print(f"  [IRIS-PUPIL RATIO] Convergence: {n_valid}/{len(conv_results)} valid frames")


# ======================================================================
# Method 2: Eyeball Model-Based
# ======================================================================
def compute_eyeball_model_gaze(output_dir, calib, crop_size=150,
                                eyeball_radius_mm=12.0):
    """Compute gaze using 3D eyeball model + pupil center triangulation.

    Principle: Fit a 3D eyeball sphere from stereo pupil observations.
    Gaze = normalize(pupil_3d - eyeball_center_3d).

    Like corneal3d but uses the pupil-only eyeball model (no glints/CC needed).
    The eyeball center is estimated from the median of stereo pupil positions
    across all frames, then per-frame gaze = pupil_3d - eyeball_center.

    Saves to convergence_meta_eyeball_model.json.
    """
    out_base = Path(output_dir)
    import cv2 as cv

    cross = calib.get("cross")
    if cross is None:
        print("  [EYEBALL MODEL] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    eye_gaze_data = {}  # {eye_side: [(frame, gaze_vec_3d), ...]}

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1 = np.array(calib[cam1]['K'])
        K2 = np.array(calib[cam2]['K'])
        dist1 = np.array(calib[cam1]['dist'])
        dist2 = np.array(calib[cam2]['dist'])
        R = np.array(stereo['R'])
        T = np.array(stereo['T']).reshape(3, 1)

        # Step 1: Triangulate pupil centers for all valid frames
        pupil_3d_points = []
        frame_indices = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            pc1 = r1.get("seg_pupil_center") or r1.get("pupil_center")
            pc2 = r2.get("seg_pupil_center") or r2.get("pupil_center")

            if pc1 is None or pc2 is None:
                continue
            if r1.get("eye_closed") or r2.get("eye_closed"):
                continue

            try:
                pt3d = _triangulate_points(
                    (pc1[0], pc1[1]), (pc2[0], pc2[1]),
                    K1, K2, R, T, dist1, dist2)

                # Sanity check: reject extreme depths
                if 10 < np.linalg.norm(pt3d) < 500:
                    pupil_3d_points.append(pt3d)
                    frame_indices.append(i)
            except Exception:
                continue

        if len(pupil_3d_points) < 10:
            print(f"  [EYEBALL MODEL] {eye_side}: only {len(pupil_3d_points)} valid pupil 3D points, skipping")
            continue

        pupil_3d_arr = np.array(pupil_3d_points)

        # Step 2: Estimate eyeball center as the point that minimizes
        # distance variance to all pupil positions (sphere fitting)
        center_init = np.median(pupil_3d_arr, axis=0)

        def sphere_cost(c):
            dists = np.linalg.norm(pupil_3d_arr - c, axis=1)
            return np.var(dists)  # Minimize variance of distances

        result = minimize(sphere_cost, center_init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 0.01})
        eyeball_center = result.x
        dists = np.linalg.norm(pupil_3d_arr - eyeball_center, axis=1)
        estimated_radius = np.median(dists)

        print(f"  [EYEBALL MODEL] {eye_side}: center={eyeball_center.round(2)}, "
              f"radius={estimated_radius:.2f}mm, "
              f"from {len(pupil_3d_points)} frames")

        # Step 3: Compute per-frame gaze vectors
        gaze_dir = out_base / f"{eye_side}_eyeball_model"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        per_eye_results = []
        gaze_list = []

        for idx, (frame_i, pupil_3d) in enumerate(zip(frame_indices, pupil_3d_arr)):
            frame_name = res1[frame_i].get("frame", f"frame_{frame_i}")

            gaze_vec = pupil_3d - eyeball_center
            gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)

            # Convert to yaw/pitch degrees
            yaw = np.degrees(np.arctan2(gaze_vec[0], gaze_vec[2]))
            pitch = np.degrees(np.arcsin(np.clip(-gaze_vec[1], -1, 1)))

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "ebm_gaze_vec": [round(float(gaze_vec[0]), 6),
                                  round(float(gaze_vec[1]), 6),
                                  round(float(gaze_vec[2]), 6)],
                "ebm_gaze_deg": [round(float(yaw), 3), round(float(pitch), 3)],
                "ebm_gaze_norm": [round(float(np.tan(np.radians(yaw))), 6),
                                   round(float(np.tan(np.radians(pitch))), 6)],
                "pupil_3d": [round(float(pupil_3d[0]), 2),
                             round(float(pupil_3d[1]), 2),
                             round(float(pupil_3d[2]), 2)],
            }
            per_eye_results.append(entry)
            gaze_list.append((frame_i, gaze_vec))

        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        # Save eyeball model parameters
        with open(gaze_dir / "eyeball_model.json", 'w') as f:
            json.dump({
                "eyeball_center": eyeball_center.tolist(),
                "estimated_radius_mm": float(estimated_radius),
                "n_frames_used": len(pupil_3d_points),
            }, f, indent=2)

        eye_gaze_data[eye_side] = gaze_list

    # Step 4: Compute binocular convergence
    if "right" in eye_gaze_data and "left" in eye_gaze_data:
        r_gaze_path = out_base / "right_eyeball_model" / "gaze_results.json"
        l_gaze_path = out_base / "left_eyeball_model" / "gaze_results.json"

        with open(r_gaze_path) as f:
            r_results = json.load(f)
        with open(l_gaze_path) as f:
            l_results = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_results, l_results, R_cross, T_cross, "ebm_gaze_norm")

        conv_path = out_base / "convergence_meta_eyeball_model.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)
        n_valid = sum(1 for c in conv_results if c.get("convergence_point") is not None)
        print(f"  [EYEBALL MODEL] Convergence: {n_valid}/{len(conv_results)} valid frames")


# ======================================================================
# Method 3: Feature Regression (Polynomial)
# ======================================================================
def compute_feature_regression_gaze(output_dir, calib, crop_size=150, poly_degree=2):
    """Compute gaze using polynomial regression from pupil+iris features.

    Features (per camera):
      - Pupil center (x, y) normalized by image size
      - Iris center (x, y) normalized
      - Pupil/iris size ratio
      - Pupil eccentricity (w/h ratio)
      - Pupil-iris offset (dx, dy) normalized by iris size

    Total: 9 features per camera → polynomial expansion → gaze (yaw, pitch)

    Requires calibration phase with known gaze targets.
    Saves to convergence_meta_feature_regression.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [FEATURE REGRESSION] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1 = np.array(calib[cam1]['K'])
        K2 = np.array(calib[cam2]['K'])

        gaze_dir = out_base / f"{eye_side}_feature_regression"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        # Extract features from all frames
        all_features = []
        valid_indices = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            feat1 = _get_pupil_iris_features(r1, calib.get(cam1))
            feat2 = _get_pupil_iris_features(r2, calib.get(cam2))

            if feat1 is None or feat2 is None:
                continue

            # Build feature vector
            fv = _build_feature_vector(feat1, feat2, K1, K2, crop_size)
            if fv is not None:
                all_features.append(fv)
                valid_indices.append(i)

        if len(all_features) < 20:
            print(f"  [FEATURE REGRESSION] {eye_side}: only {len(all_features)} valid frames, skipping")
            continue

        features_arr = np.array(all_features)

        # For now, use iris-pupil ratio as "pseudo gaze" since we don't have
        # calibration targets in TEyeD. The regression will be calibrated later.
        # Save features for later calibration.
        per_eye_results = []
        for idx, feat_idx in enumerate(valid_indices):
            r1 = res1[feat_idx]
            frame_name = r1.get("frame", f"frame_{feat_idx}")

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "features": features_arr[idx].tolist(),
                "feature_dim": len(features_arr[idx]),
            }
            per_eye_results.append(entry)

        with open(gaze_dir / "feature_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        print(f"  [FEATURE REGRESSION] {eye_side}: extracted {len(all_features)} "
              f"feature vectors ({features_arr.shape[1]}D)")


def _build_feature_vector(feat1, feat2, K1, K2, crop_size):
    """Build a normalized feature vector from two camera observations."""
    features = []

    for feat, K in [(feat1, K1), (feat2, K2)]:
        if feat is None:
            return None

        pcx, pcy, icx, icy, pw, ph, iw, ih = feat

        # Normalize by focal length for camera-independent features
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Pupil center (normalized)
        features.append((pcx - cx) / fx)
        features.append((pcy - cy) / fy)

        # Iris center (normalized) - use pupil if iris unavailable
        if icx is not None and icy is not None:
            features.append((icx - cx) / fx)
            features.append((icy - cy) / fy)
            # Pupil-iris offset
            features.append((pcx - icx) / max(iw, 1))
            features.append((pcy - icy) / max(ih, 1))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Size features
        features.append(pw / max(iw, 1))  # pupil/iris size ratio
        features.append(pw / max(ph, 1) if ph > 0 else 1.0)  # pupil eccentricity
        features.append(iw / fx)  # iris apparent size (depth proxy)

    return features


# ======================================================================
# Method 4: Lid Geometry
# ======================================================================
def compute_lid_geometry_gaze(output_dir, calib, crop_size=150):
    """Compute gaze using eyelid aperture + pupil position.

    Principle: The visible portion of the pupil relative to the eyelid
    opening provides information about vertical gaze direction.
    The horizontal position of the pupil within the palpebral fissure
    indicates horizontal gaze.

    Uses: pupil_center, eye_closed, closed_confidence, seg mask (lid region)
    Saves to convergence_meta_lid_geometry.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [LID GEOMETRY] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1 = np.array(calib[cam1]['K'])
        K2 = np.array(calib[cam2]['K'])

        gaze_dir = out_base / f"{eye_side}_lid_geometry"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        per_eye_results = []
        n_frames = min(len(res1), len(res2))

        # Collect all pupil positions to compute the eye box (range of motion)
        all_pupil_cx1, all_pupil_cy1 = [], []
        for i in range(n_frames):
            pc1 = res1[i].get("seg_pupil_center") or res1[i].get("pupil_center")
            if pc1 and not res1[i].get("eye_closed"):
                all_pupil_cx1.append(pc1[0])
                all_pupil_cy1.append(pc1[1])

        if len(all_pupil_cx1) < 20:
            print(f"  [LID GEOMETRY] {eye_side}: too few open-eye frames, skipping")
            continue

        # Compute eye box (range of pupil motion) for normalization
        cx_min, cx_max = np.percentile(all_pupil_cx1, 5), np.percentile(all_pupil_cx1, 95)
        cy_min, cy_max = np.percentile(all_pupil_cy1, 5), np.percentile(all_pupil_cy1, 95)
        cx_range = max(cx_max - cx_min, 1.0)
        cy_range = max(cy_max - cy_min, 1.0)
        cx_mid = (cx_min + cx_max) / 2
        cy_mid = (cy_min + cy_max) / 2

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "lid_gaze_deg": None,
                "lid_gaze_norm": None,
                "openness": None,
            }

            pc1 = r1.get("seg_pupil_center") or r1.get("pupil_center")
            if pc1 is None or r1.get("eye_closed"):
                per_eye_results.append(entry)
                continue

            # Normalize pupil position to [-1, 1] within eye box
            norm_x = (pc1[0] - cx_mid) / (cx_range / 2)
            norm_y = (pc1[1] - cy_mid) / (cy_range / 2)
            norm_x = max(-1.0, min(1.0, norm_x))
            norm_y = max(-1.0, min(1.0, norm_y))

            # Openness from closed_confidence (1.0 = fully open, 0.0 = closed)
            openness = 1.0 - r1.get("closed_confidence", 0.5)
            entry["openness"] = round(float(openness), 3)

            # Map to gaze angles (30 deg max range typical)
            gaze_h = norm_x * 25.0  # degrees
            gaze_v = norm_y * 20.0  # degrees (smaller vertical range)

            # Weight vertical component by openness (squinting → looking up/down)
            if openness < 0.5:
                gaze_v *= (1.0 + (0.5 - openness))

            entry["lid_gaze_deg"] = [round(float(gaze_h), 3), round(float(gaze_v), 3)]
            entry["lid_gaze_norm"] = [round(float(np.tan(np.radians(gaze_h))), 6),
                                       round(float(np.tan(np.radians(gaze_v))), 6)]

            per_eye_results.append(entry)

        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        n_valid = sum(1 for r in per_eye_results if r["lid_gaze_deg"] is not None)
        print(f"  [LID GEOMETRY] {eye_side}: {n_valid}/{n_frames} valid frames")

    # Compute convergence
    r_gaze_path = out_base / "right_lid_geometry" / "gaze_results.json"
    l_gaze_path = out_base / "left_lid_geometry" / "gaze_results.json"

    if r_gaze_path.exists() and l_gaze_path.exists():
        with open(r_gaze_path) as f:
            r_gaze = json.load(f)
        with open(l_gaze_path) as f:
            l_gaze = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_gaze, l_gaze, R_cross, T_cross, "lid_gaze_norm")

        conv_path = out_base / "convergence_meta_lid_geometry.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)


# ======================================================================
# Method 5: Iris Landmark Cross-Ratio
# ======================================================================
def compute_iris_landmark_cr_gaze(output_dir, calib, crop_size=150):
    """Compute gaze using cross-ratio from iris boundary landmarks.

    Principle: The iris boundary provides 4+ natural reference points
    (top, bottom, left, right of iris). The cross-ratio of pupil center
    with respect to these points is invariant to perspective, similar to
    how traditional PCCR uses glint positions.

    Uses seg_iris_center, seg_pupil_center, and iris boundary from seg mask.
    Saves to convergence_meta_iris_landmark_cr.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [IRIS LANDMARK CR] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1 = np.array(calib[cam1]['K'])
        K2 = np.array(calib[cam2]['K'])
        dist1 = np.array(calib[cam1]['dist'])
        dist2 = np.array(calib[cam2]['dist'])

        gaze_dir = out_base / f"{eye_side}_iris_landmark_cr"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        per_eye_results = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1, r2 = res1[i], res2[i]
            frame_name = r1.get("frame", f"frame_{i}")

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "icr_gaze_deg": None,
                "icr_gaze_norm": None,
            }

            feat1 = _get_pupil_iris_features(r1, calib.get(cam1))
            if feat1 is None or feat1[2] is None:
                per_eye_results.append(entry)
                continue

            pcx, pcy, icx, icy, pw, ph, iw, ih = feat1

            # Virtual reference points: iris boundary (top, right, bottom, left)
            # These act as "virtual glints" for cross-ratio computation
            ir_top = (icx, icy - ih / 2)
            ir_right = (icx + iw / 2, icy)
            ir_bottom = (icx, icy + ih / 2)
            ir_left = (icx - iw / 2, icy)

            # Cross-ratio of pupil wrt iris boundary
            # Horizontal CR: (P - L) / (R - L)
            cr_h = (pcx - ir_left[0]) / max(ir_right[0] - ir_left[0], 1) - 0.5
            # Vertical CR: (P - T) / (B - T)
            cr_v = (pcy - ir_top[1]) / max(ir_bottom[1] - ir_top[1], 1) - 0.5

            # Undistort using camera intrinsics for better accuracy
            try:
                pc_undist = _undistort_point((pcx, pcy), K1, dist1)
                ic_undist = _undistort_point((icx, icy), K1, dist1)

                pc_norm = _normalize_point(pc_undist, K1)
                ic_norm = _normalize_point(ic_undist, K1)

                # Undistorted cross-ratio
                cr_h_undist = pc_norm[0] - ic_norm[0]
                cr_v_undist = pc_norm[1] - ic_norm[1]

                # Scale by iris size in normalized coords for comparability
                ir_right_undist = _undistort_point(ir_right, K1, dist1)
                ir_right_norm = _normalize_point(ir_right_undist, K1)
                iris_scale = abs(ir_right_norm[0] - ic_norm[0])

                if iris_scale > 0.001:
                    gaze_h = np.degrees(np.arctan(cr_h_undist / iris_scale * 0.5))
                    gaze_v = np.degrees(np.arctan(cr_v_undist / iris_scale * 0.5))
                else:
                    gaze_h = cr_h * 30.0
                    gaze_v = cr_v * 25.0

            except Exception:
                gaze_h = cr_h * 30.0
                gaze_v = cr_v * 25.0

            entry["icr_gaze_deg"] = [round(float(gaze_h), 3), round(float(gaze_v), 3)]
            entry["icr_gaze_norm"] = [round(float(np.tan(np.radians(gaze_h))), 6),
                                       round(float(np.tan(np.radians(gaze_v))), 6)]

            per_eye_results.append(entry)

        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        n_valid = sum(1 for r in per_eye_results if r["icr_gaze_deg"] is not None)
        print(f"  [IRIS LANDMARK CR] {eye_side}: {n_valid}/{n_frames} valid frames")

    # Convergence
    r_gaze_path = out_base / "right_iris_landmark_cr" / "gaze_results.json"
    l_gaze_path = out_base / "left_iris_landmark_cr" / "gaze_results.json"

    if r_gaze_path.exists() and l_gaze_path.exists():
        with open(r_gaze_path) as f:
            r_gaze = json.load(f)
        with open(l_gaze_path) as f:
            l_gaze = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_gaze, l_gaze, R_cross, T_cross, "icr_gaze_norm")

        conv_path = out_base / "convergence_meta_iris_landmark_cr.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)


# ======================================================================
# Method 6: Dual Ellipse 3D Rotation
# ======================================================================
def compute_dual_ellipse_3d_gaze(output_dir, calib, crop_size=150):
    """Compute gaze from 3D eyeball rotation estimated via dual ellipse fitting.

    Principle: The pupil and iris are concentric circles in 3D. Their
    projections as ellipses encode the 3D orientation of the eye.
    By comparing pupil and iris ellipse parameters, we can estimate
    the rotation of the eyeball relative to the camera axis.

    Uses: pupil ellipse (cx, cy, w, h, angle) + iris ellipse → 3D rotation
    Saves to convergence_meta_dual_ellipse_3d.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [DUAL ELLIPSE 3D] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    for eye_side in ["right", "left"]:
        res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
        if res1 is None:
            continue

        K1 = np.array(calib[cam1]['K'])
        K2 = np.array(calib[cam2]['K'])

        gaze_dir = out_base / f"{eye_side}_dual_ellipse_3d"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        per_eye_results = []
        n_frames = min(len(res1), len(res2))

        for i in range(n_frames):
            r1 = res1[i]
            frame_name = r1.get("frame", f"frame_{i}")

            entry = {
                "frame": frame_name,
                "eye": eye_side,
                "de3d_gaze_deg": None,
                "de3d_gaze_norm": None,
            }

            feat1 = _get_pupil_iris_features(r1, calib.get(cam1))
            if feat1 is None or feat1[2] is None:
                per_eye_results.append(entry)
                continue

            pcx, pcy, icx, icy, pw, ph, iw, ih = feat1

            if pw < 3 or ph < 3 or iw < 5 or ih < 5:
                per_eye_results.append(entry)
                continue

            # Estimate 3D eye orientation from ellipse parameters
            # When looking straight at camera: pupil and iris are circular
            # When looking to the side: they become elliptical
            # Eccentricity of the iris ellipse encodes viewing angle

            # Iris eccentricity → tilt angle
            iris_ratio = min(iw, ih) / max(iw, ih)  # 1.0 = circular, <1 = tilted

            # Compute tilt direction from iris ellipse orientation
            # The minor axis of the iris ellipse points toward the gaze direction
            # Use pupil-iris offset for direction, eccentricity for magnitude

            # Gaze direction from pupil displacement + eccentricity
            dx = (pcx - icx) / (iw / 2) if iw > 5 else 0
            dy = (pcy - icy) / (ih / 2) if ih > 5 else 0

            # Eccentricity amplifies the gaze estimate
            tilt_angle = np.degrees(np.arccos(np.clip(iris_ratio, 0, 1)))

            # Combined: displacement direction + eccentricity magnitude
            disp_mag = np.sqrt(dx**2 + dy**2)
            if disp_mag > 0.001:
                gaze_h = dx / disp_mag * tilt_angle if tilt_angle > 1 else dx * 30.0
                gaze_v = dy / disp_mag * tilt_angle if tilt_angle > 1 else dy * 25.0
            else:
                gaze_h = 0.0
                gaze_v = 0.0

            # Pupil size ratio as depth cue (pupil appears smaller when looking away)
            pupil_ratio = min(pw, ph) / max(pw, ph)
            pupil_tilt = np.degrees(np.arccos(np.clip(pupil_ratio, 0, 1)))

            # Average of iris-based and pupil-based estimates
            if pupil_tilt > 1:
                gaze_h_p = dx / max(disp_mag, 0.001) * pupil_tilt
                gaze_v_p = dy / max(disp_mag, 0.001) * pupil_tilt
                gaze_h = 0.6 * gaze_h + 0.4 * gaze_h_p
                gaze_v = 0.6 * gaze_v + 0.4 * gaze_v_p

            entry["de3d_gaze_deg"] = [round(float(gaze_h), 3), round(float(gaze_v), 3)]
            entry["de3d_gaze_norm"] = [round(float(np.tan(np.radians(gaze_h))), 6),
                                        round(float(np.tan(np.radians(gaze_v))), 6)]

            per_eye_results.append(entry)

        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        n_valid = sum(1 for r in per_eye_results if r["de3d_gaze_deg"] is not None)
        print(f"  [DUAL ELLIPSE 3D] {eye_side}: {n_valid}/{n_frames} valid frames")

    # Convergence
    r_gaze_path = out_base / "right_dual_ellipse_3d" / "gaze_results.json"
    l_gaze_path = out_base / "left_dual_ellipse_3d" / "gaze_results.json"

    if r_gaze_path.exists() and l_gaze_path.exists():
        with open(r_gaze_path) as f:
            r_gaze = json.load(f)
        with open(l_gaze_path) as f:
            l_gaze = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_gaze, l_gaze, R_cross, T_cross, "de3d_gaze_norm")

        conv_path = out_base / "convergence_meta_dual_ellipse_3d.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)


# ======================================================================
# Method 7: Temporal Smoothed Gaze
# ======================================================================
def compute_temporal_smooth_gaze(output_dir, calib, crop_size=150,
                                  window_size=5, method='kalman'):
    """Compute temporally smoothed gaze from pupil trajectory.

    Principle: Raw pupil position is noisy. By modeling the pupil trajectory
    with a Kalman filter or moving average, we get smoother and more accurate
    gaze estimates. This is especially useful during fixations.

    Combines iris_pupil_ratio (base method) with temporal filtering.
    Saves to convergence_meta_temporal_smooth.json.
    """
    out_base = Path(output_dir)

    cross = calib.get("cross")
    if cross is None:
        print("  [TEMPORAL SMOOTH] No cross-pair calibration, skipping")
        return

    R_cross = np.array(cross["R"])
    T_cross = np.array(cross["T"]).reshape(3, 1)

    for eye_side in ["right", "left"]:
        # First check if iris_pupil_ratio results exist
        ipr_path = out_base / f"{eye_side}_iris_pupil_ratio" / "gaze_results.json"
        if not ipr_path.exists():
            # Fall back to computing from raw features
            res1, res2, cam1, cam2, stereo = _load_results_pair(out_base, calib, eye_side)
            if res1 is None:
                continue
            # Use raw pupil positions as base signal
            base_gaze = _compute_raw_pupil_gaze(res1, res2, calib, cam1, cam2, eye_side)
        else:
            with open(ipr_path) as f:
                base_gaze = json.load(f)

        if not base_gaze:
            continue

        gaze_dir = out_base / f"{eye_side}_temporal_smooth"
        gaze_dir.mkdir(parents=True, exist_ok=True)

        # Apply temporal smoothing
        if method == 'kalman':
            smoothed = _kalman_smooth_gaze(base_gaze, "ipr_gaze_deg")
        else:
            smoothed = _moving_avg_smooth_gaze(base_gaze, "ipr_gaze_deg", window_size)

        per_eye_results = []
        for i, entry in enumerate(base_gaze):
            new_entry = {
                "frame": entry.get("frame"),
                "eye": eye_side,
                "ts_gaze_deg": None,
                "ts_gaze_norm": None,
            }

            if i < len(smoothed) and smoothed[i] is not None:
                gaze_h, gaze_v = smoothed[i]
                new_entry["ts_gaze_deg"] = [round(float(gaze_h), 3), round(float(gaze_v), 3)]
                new_entry["ts_gaze_norm"] = [round(float(np.tan(np.radians(gaze_h))), 6),
                                              round(float(np.tan(np.radians(gaze_v))), 6)]

            per_eye_results.append(new_entry)

        with open(gaze_dir / "gaze_results.json", 'w') as f:
            json.dump(per_eye_results, f, indent=2)

        n_valid = sum(1 for r in per_eye_results if r["ts_gaze_deg"] is not None)
        print(f"  [TEMPORAL SMOOTH] {eye_side}: {n_valid}/{len(per_eye_results)} valid frames")

    # Convergence
    r_gaze_path = out_base / "right_temporal_smooth" / "gaze_results.json"
    l_gaze_path = out_base / "left_temporal_smooth" / "gaze_results.json"

    if r_gaze_path.exists() and l_gaze_path.exists():
        with open(r_gaze_path) as f:
            r_gaze = json.load(f)
        with open(l_gaze_path) as f:
            l_gaze = json.load(f)

        conv_results = _compute_convergence_from_gaze(
            r_gaze, l_gaze, R_cross, T_cross, "ts_gaze_norm")

        conv_path = out_base / "convergence_meta_temporal_smooth.json"
        with open(conv_path, 'w') as f:
            json.dump(conv_results, f, indent=2)


# ======================================================================
# Shared helpers for glint-less methods
# ======================================================================
def _kalman_smooth_gaze(gaze_results, key="ipr_gaze_deg"):
    """Apply Kalman filter to gaze sequence."""
    smoothed = []

    # State: [gaze_h, gaze_v, vel_h, vel_v]
    x = np.zeros(4)
    P = np.eye(4) * 10.0

    # Process noise
    Q = np.eye(4)
    Q[:2, :2] *= 0.5   # position noise
    Q[2:, 2:] *= 2.0   # velocity noise

    # Measurement noise
    R_meas = np.eye(2) * 1.0

    # State transition (constant velocity model)
    dt = 1.0
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt

    # Observation matrix
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    initialized = False

    for entry in gaze_results:
        gaze = entry.get(key)
        if gaze is None:
            smoothed.append(None)
            continue

        z = np.array(gaze[:2])

        if not initialized:
            x[:2] = z
            initialized = True
            smoothed.append((float(z[0]), float(z[1])))
            continue

        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        y = z - H @ x
        S = H @ P @ H.T + R_meas
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        smoothed.append((float(x[0]), float(x[1])))

    return smoothed


def _moving_avg_smooth_gaze(gaze_results, key="ipr_gaze_deg", window=5):
    """Apply moving average to gaze sequence."""
    smoothed = []
    buffer_h = deque(maxlen=window)
    buffer_v = deque(maxlen=window)

    for entry in gaze_results:
        gaze = entry.get(key)
        if gaze is None:
            smoothed.append(None)
            continue

        buffer_h.append(gaze[0])
        buffer_v.append(gaze[1])

        smoothed.append((float(np.mean(buffer_h)), float(np.mean(buffer_v))))

    return smoothed


def _compute_raw_pupil_gaze(res1, res2, calib, cam1, cam2, eye_side):
    """Compute basic pupil-iris ratio gaze from raw results."""
    results = []
    n_frames = min(len(res1), len(res2))

    for i in range(n_frames):
        r1 = res1[i]
        frame_name = r1.get("frame", f"frame_{i}")

        entry = {"frame": frame_name, "eye": eye_side, "ipr_gaze_deg": None}

        feat = _get_pupil_iris_features(r1, calib.get(cam1))
        if feat is not None and feat[2] is not None:
            pcx, pcy, icx, icy, pw, ph, iw, ih = feat
            if iw > 5 and ih > 5:
                dx = (pcx - icx) / (iw / 2.0)
                dy = (pcy - icy) / (ih / 2.0)
                dx = max(-1.0, min(1.0, dx))
                dy = max(-1.0, min(1.0, dy))
                gaze_h = np.degrees(np.arcsin(dx * 0.5))
                gaze_v = np.degrees(np.arcsin(dy * 0.5))
                entry["ipr_gaze_deg"] = [float(gaze_h), float(gaze_v)]

        results.append(entry)
    return results


def _compute_convergence_from_gaze(r_gaze, l_gaze, R_cross, T_cross, gaze_key):
    """Compute stereo convergence from per-eye gaze estimates.

    Uses the same approach as existing convergence methods:
    intersect right and left eye gaze rays in 3D space.
    """
    # Right eye origin in cross-pair frame
    lo_origin = (-R_cross.T @ T_cross).flatten()

    convergence_results = []
    n_frames = min(len(r_gaze), len(l_gaze))

    for i in range(n_frames):
        rg = r_gaze[i]
        lg = l_gaze[i]

        entry = {
            "frame": rg.get("frame", f"frame_{i}"),
            "convergence_point": None,
            "convergence_distance_mm": None,
            "right_gaze": rg.get(gaze_key),
            "left_gaze": lg.get(gaze_key),
        }

        rg_norm = rg.get(gaze_key)
        lg_norm = lg.get(gaze_key)

        if rg_norm is None or lg_norm is None:
            convergence_results.append(entry)
            continue

        # Convert normalized gaze to 3D ray directions
        # Right eye: ray from origin along gaze direction
        r_dir = np.array([rg_norm[0], rg_norm[1], 1.0])
        r_dir = r_dir / np.linalg.norm(r_dir)

        # Left eye: ray from lo_origin, gaze in left camera frame → rotate to right frame
        l_dir_local = np.array([lg_norm[0], lg_norm[1], 1.0])
        l_dir_local = l_dir_local / np.linalg.norm(l_dir_local)
        l_dir = R_cross.T @ l_dir_local  # Transform to right eye frame

        # Closest point between two rays (right at origin, left at lo_origin)
        # Using the standard ray-ray closest approach formula
        r_origin = np.zeros(3)
        l_origin = lo_origin

        w0 = r_origin - l_origin
        a = np.dot(r_dir, r_dir)
        b = np.dot(r_dir, l_dir)
        c = np.dot(l_dir, l_dir)
        d = np.dot(r_dir, w0)
        e = np.dot(l_dir, w0)

        denom = a * c - b * b
        if abs(denom) < 1e-10:
            convergence_results.append(entry)
            continue

        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom

        # Convergence point = midpoint of closest approach
        closest_r = r_origin + sc * r_dir
        closest_l = l_origin + tc * l_dir
        midpoint = (closest_r + closest_l) / 2.0

        # Only accept if both rays point forward (positive Z)
        if sc > 0 and tc > 0:
            dist = np.linalg.norm(midpoint)
            entry["convergence_point"] = [round(float(midpoint[0]), 2),
                                           round(float(midpoint[1]), 2),
                                           round(float(midpoint[2]), 2)]
            entry["convergence_distance_mm"] = round(float(dist), 2)

        convergence_results.append(entry)

    return convergence_results


# ======================================================================
# Calibration methods for glint-less gaze
# ======================================================================
def calibrate_iris_pupil_ratio_convergence(output_dir, calib,
                                            known_distance_mm=500.0,
                                            cal_cutoff_time=None, fair_cc=False):
    """Calibrate iris-pupil ratio convergence with known fixation distance.

    Applies a scale + bias correction to the raw convergence distances
    using calibration dots at known screen distance.
    """
    _calibrate_generic_convergence(
        output_dir, calib,
        "iris_pupil_ratio", "ipr",
        known_distance_mm, cal_cutoff_time, fair_cc)


def calibrate_eyeball_model_convergence(output_dir, calib,
                                         known_distance_mm=500.0,
                                         cal_cutoff_time=None, fair_cc=False):
    """Calibrate eyeball model convergence with known fixation distance."""
    _calibrate_generic_convergence(
        output_dir, calib,
        "eyeball_model", "ebm",
        known_distance_mm, cal_cutoff_time, fair_cc)


def calibrate_lid_geometry_convergence(output_dir, calib,
                                        known_distance_mm=500.0,
                                        cal_cutoff_time=None, fair_cc=False):
    """Calibrate lid geometry convergence with known fixation distance."""
    _calibrate_generic_convergence(
        output_dir, calib,
        "lid_geometry", "lid",
        known_distance_mm, cal_cutoff_time, fair_cc)


def calibrate_iris_landmark_cr_convergence(output_dir, calib,
                                            known_distance_mm=500.0,
                                            cal_cutoff_time=None, fair_cc=False):
    """Calibrate iris landmark CR convergence with known fixation distance."""
    _calibrate_generic_convergence(
        output_dir, calib,
        "iris_landmark_cr", "icr",
        known_distance_mm, cal_cutoff_time, fair_cc)


def calibrate_dual_ellipse_3d_convergence(output_dir, calib,
                                           known_distance_mm=500.0,
                                           cal_cutoff_time=None, fair_cc=False):
    """Calibrate dual ellipse 3D convergence with known fixation distance."""
    _calibrate_generic_convergence(
        output_dir, calib,
        "dual_ellipse_3d", "de3d",
        known_distance_mm, cal_cutoff_time, fair_cc)


def calibrate_temporal_smooth_convergence(output_dir, calib,
                                           known_distance_mm=500.0,
                                           cal_cutoff_time=None, fair_cc=False):
    """Calibrate temporal smooth convergence with known fixation distance."""
    _calibrate_generic_convergence(
        output_dir, calib,
        "temporal_smooth", "ts",
        known_distance_mm, cal_cutoff_time, fair_cc)


def _calibrate_generic_convergence(output_dir, calib, method_name, prefix,
                                    known_distance_mm, cal_cutoff_time, fair_cc):
    """Generic calibration for any convergence method.

    Reads convergence_meta_{method_name}.json, splits into cal/test,
    fits scale+bias to match known distance during calibration phase,
    and applies correction to test phase.

    Saves to convergence_meta_{method_name}_cal.json.
    """
    out_base = Path(output_dir)
    conv_path = out_base / f"convergence_meta_{method_name}.json"

    if not conv_path.exists():
        print(f"  [CAL {method_name.upper()}] No raw convergence, skipping")
        return

    with open(conv_path) as f:
        conv_data = json.load(f)

    # Determine cal/test split
    cal_frames = []
    test_frames = []

    for i, entry in enumerate(conv_data):
        dist = entry.get("convergence_distance_mm")
        if dist is None:
            continue

        if cal_cutoff_time is not None:
            # Use timestamp-based split
            frame_name = entry.get("frame", "")
            # Parse timestamp from frame name (e.g., frame_5_timestamp_1.234.png)
            ts_match = re.search(r"timestamp_([\d]+\.\d+)", frame_name)
            if ts_match:
                frame_time = float(ts_match.group(1))
            else:
                fn_match = re.search(r"frame_(\d+)", frame_name)
                frame_num = int(fn_match.group(1)) if fn_match else i
                frame_time = frame_num / 30.0

            if frame_time <= cal_cutoff_time:
                cal_frames.append((i, dist))
            else:
                test_frames.append((i, dist))
        else:
            # Use first 40% as calibration
            if i < len(conv_data) * 0.4:
                cal_frames.append((i, dist))
            else:
                test_frames.append((i, dist))

    if len(cal_frames) < 5:
        print(f"  [CAL {method_name.upper()}] Only {len(cal_frames)} cal frames, need >= 5")
        return

    # Fit scale + bias: predicted_distance = scale * raw_distance + bias
    cal_dists = np.array([d for _, d in cal_frames])
    target = known_distance_mm

    # Simple: scale = target / median(cal_distances)
    median_cal = np.median(cal_dists)
    if median_cal > 0:
        scale = target / median_cal
    else:
        scale = 1.0
    bias = 0.0

    # Apply calibration to all frames
    cal_results = []
    for entry in conv_data:
        new_entry = dict(entry)
        raw_dist = entry.get("convergence_distance_mm")
        if raw_dist is not None:
            cal_dist = raw_dist * scale + bias
            new_entry["convergence_distance_mm_cal"] = round(float(cal_dist), 2)
            new_entry["convergence_distance_mm_raw"] = raw_dist

            # Also calibrate convergence point
            cp = entry.get("convergence_point")
            if cp is not None:
                cp_arr = np.array(cp)
                if np.linalg.norm(cp_arr) > 0:
                    cal_cp = cp_arr * scale
                    new_entry["convergence_point_cal"] = [round(float(cal_cp[0]), 2),
                                                           round(float(cal_cp[1]), 2),
                                                           round(float(cal_cp[2]), 2)]
        cal_results.append(new_entry)

    # Save calibrated results
    cal_path = out_base / f"convergence_meta_{method_name}_cal.json"
    with open(cal_path, 'w') as f:
        json.dump(cal_results, f, indent=2)

    # Report stats
    test_dists = [e.get("convergence_distance_mm_cal") for e in cal_results
                  if e.get("convergence_distance_mm_cal") is not None]
    if test_dists:
        print(f"  [CAL {method_name.upper()}] scale={scale:.4f}, "
              f"cal median={median_cal:.1f}mm → {target:.1f}mm, "
              f"all frames median={np.median(test_dists):.1f}mm")
    else:
        print(f"  [CAL {method_name.upper()}] No valid distances after calibration")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", nargs="?", default=None)
    p.add_argument("-o", "--output_dir", default=None)
    p.add_argument("-c", "--crop_size", type=int, default=150)
    p.add_argument("--camera", default="ri", help="Camera ID: ri, ro, li, lo")
    p.add_argument("--batch", action="store_true",
                   help="Process all 4 cameras from task_4_saccade/")
    p.add_argument("--algorithm", default="contour",
                   choices=ALGORITHMS + ['all'],
                   help="Pupil detection algorithm (default: contour). "
                        "'all' runs every algorithm in parallel.")
    p.add_argument("--seg", action="store_true",
                   help="Enable eye region segmentation overlay (debug only).")
    p.add_argument("--seg-algo", default="worldcoin",
                   choices=SEG_ALGORITHMS,
                   help="Segmentation algorithm: worldcoin (neural net, default) or classical (Otsu-based).")
    p.add_argument("--led-calib", default=None,
                   help="Path to full_calibration_data.pkl with IR LED 3D positions")
    p.add_argument("--calib-dir", default=None,
                   help="Path to calibration directory (JSON or NPZ). Default: auto-detect new calibration.")
    p.add_argument("--task-dir", default=None,
                   help="Input directory with camera subdirs (ri/, ro/, li/, lo/). Default: task_new_recording/")
    p.add_argument("--output-prefix", default="output_new",
                   help="Output directory prefix (default: output_new). Contour uses prefix/, others use prefix_algo/.")
    p.add_argument("--camera-batch", default="default",
                   help="Camera hardware batch: default, v1, v2")
    p.add_argument("--crop-method", default="blob",
                   choices=["blob", "seg_ritnet", "seg_worldcoin", "isophote"],
                   help="Crop centering method")
    p.add_argument("--cameras", default=None,
                   help="Comma-separated camera list for batch (e.g. ri,ro). Default: all 4.")
    p.add_argument("--screen-distance", type=float, default=None,
                   help="Known screen distance in mm for calibration (default: 500mm)")
    p.add_argument("--calibrate-only", action="store_true",
                   help="Skip frame processing, only re-run calibration")
    p.add_argument("--convergence-only", action="store_true",
                   help="Skip frame processing, re-run convergence + calibration")
    p.add_argument("--fair-cc", action="store_true",
                   help="Use cal-only CC for fair evaluation (no test data leakage)")
    a = p.parse_args()

    if a.seg:
        print(f"[SEG] Segmentation enabled, algorithm: {a.seg_algo}")

    # Parse camera list
    cam_list = [c.strip() for c in a.cameras.split(",")] if a.cameras else None
    if cam_list:
        print(f"[CAMERAS] Processing only: {', '.join(c.upper() for c in cam_list)}")

    # Load LED positions if provided
    led_positions = None
    if a.led_calib:
        led_positions = load_led_positions(a.led_calib)

    if a.batch:
        base = Path(__file__).parent
        task_dir = Path(a.task_dir) if a.task_dir else base / "task_new_recording"

        if a.algorithm == 'all':
            # Run all algorithms in parallel
            algos_to_run = ALGORITHMS[:]
            print(f"Running {len(algos_to_run)} algorithms in parallel: {algos_to_run}")
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(algos_to_run)) as executor:
                futures = {
                    executor.submit(_run_batch_for_algorithm, algo, base, task_dir, a.crop_size,
                                    a.seg, a.seg_algo, led_positions, cam_list, a.calib_dir,
                                    a.output_prefix, a.screen_distance,
                                    a.calibrate_only, a.convergence_only,
                                    a.fair_cc, a.camera_batch, crop_method=a.crop_method): algo
                    for algo in algos_to_run
                }
                for future in concurrent.futures.as_completed(futures):
                    algo = futures[future]
                    try:
                        future.result()
                        print(f"\n*** {algo.upper()} completed ***")
                    except Exception as e:
                        print(f"\n*** {algo.upper()} FAILED: {e} ***")
        else:
            _run_batch_for_algorithm(a.algorithm, base, task_dir, a.crop_size,
                                     a.seg, a.seg_algo, led_positions, cam_list, a.calib_dir,
                                     a.output_prefix, a.screen_distance,
                                     a.calibrate_only, a.convergence_only,
                                     a.fair_cc, a.camera_batch, crop_method=a.crop_method)
    else:
        od = a.output_dir or (a.input_dir.rstrip("/") + "_output")
        process_all(a.input_dir, od, crop_size=a.crop_size, camera=a.camera,
                    algorithm=a.algorithm,
                    seg_enabled=a.seg, seg_algo=a.seg_algo,
                    crop_method=a.crop_method)