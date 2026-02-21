"""
Enhanced floor-plan processing with OpenCV + OCR.

Pipeline
────────
1.  Binarise (Otsu).
2.  Thickness analysis – distance-transform to separate *thick* structures
    (walls) from *thin* ones (furniture outlines, door arcs, text, etc.).
3.  Wall extraction – morphological opening → skeleton → HoughLinesP → merge.
4.  Door detection – HoughCircles on the thin mask near wall gaps.
5.  Staircase detection – rectangular region with dense parallel lines.
6.  Room-label OCR – pytesseract (optional; graceful fallback).
7.  Furniture detection – contour analysis on the thin mask, classified by
    room context and shape heuristics.
8.  Room / floor detection – flood-fill white regions in the sealed wall mask.
"""

import cv2
import numpy as np
import uuid
import math
from typing import Any

# Optional OCR ─────────────────────────────────────────────────────────
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


# ═══════════════════════════════════════════════════════════════════════
# Constants / look-up tables
# ═══════════════════════════════════════════════════════════════════════

ROOM_KEYWORDS: dict[str, str] = {
    "kitchen": "kitchen", "dining": "kitchen",
    "bedroom": "bedroom", "bed room": "bedroom", "master": "bedroom",
    "living": "living", "lounge": "living", "family": "living",
    "bathroom": "bathroom", "bath": "bathroom", "toilet": "bathroom",
    "wc": "bathroom", "restroom": "bathroom",
    "laundry": "laundry", "utility": "laundry",
    "garage": "garage",
    "office": "office", "study": "office",
    "hallway": "hallway", "corridor": "hallway", "foyer": "hallway",
    "closet": "closet", "wardrobe": "closet", "storage": "closet",
}

# (label, geometry, color, height, width_ratio, depth_ratio)
# width/depth are fractions of the *contour* bounding-box
FURNITURE_TEMPLATES: dict[str, list[tuple]] = {
    "bedroom": [
        ("Bed",        "box", "#a8c4d4", 0.35, 1.0, 1.0),
        ("Nightstand", "box", "#8b6f47", 0.40, 1.0, 1.0),
        ("Wardrobe",   "box", "#6b5b4e", 0.90, 1.0, 1.0),
    ],
    "living": [
        ("Sofa",         "box", "#6b8f71", 0.40, 1.0, 1.0),
        ("Coffee Table", "box", "#8b6f47", 0.30, 1.0, 1.0),
        ("TV Unit",      "box", "#5a5a5a", 0.50, 1.0, 1.0),
    ],
    "kitchen": [
        ("Kitchen Counter", "box", "#c4b5a0", 0.90, 1.0, 1.0),
        ("Dining Table",    "box", "#8b6f47", 0.75, 1.0, 1.0),
        ("Chair",           "box", "#8b6f47", 0.45, 1.0, 1.0),
    ],
    "bathroom": [
        ("Bathtub",   "box",      "#e0e0e0", 0.35, 1.0, 1.0),
        ("Washbasin", "cylinder", "#ffffff", 0.80, 1.0, 1.0),
        ("Toilet",    "cylinder", "#ffffff", 0.40, 1.0, 1.0),
    ],
    "laundry": [
        ("Washer", "box", "#d0d0d0", 0.85, 1.0, 1.0),
        ("Dryer",  "box", "#c0c0c0", 0.85, 1.0, 1.0),
    ],
    "office": [
        ("Desk",  "box", "#8b6f47", 0.75, 1.0, 1.0),
        ("Chair", "box", "#5a5a5a", 0.45, 1.0, 1.0),
    ],
    "unknown": [
        ("Furniture", "box", "#9e9e9e", 0.50, 1.0, 1.0),
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def extract_scene_from_image(image_bytes: bytes) -> dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return _empty_scene("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 10.0 / max(w, h)

    # 1. Binarise
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)

    # 2. Thickness-based separation
    thick_mask, thin_mask, thickness_thr = _separate_by_thickness(binary, w, h)

    # 3. Walls
    walls = _extract_walls(thick_mask, w, h, scale)

    # 4. Doors & Windows
    doors, windows = _detect_doors_and_windows(thin_mask, thick_mask, walls, w, h, scale)

    # 5. Staircases
    staircases = _detect_staircases(thin_mask, w, h, scale)

    # 6. Room labels via OCR
    room_labels = _detect_room_labels(gray, w, h, scale)

    # 7. Room (floor) detection
    sealed = _seal_wall_gaps(thick_mask, w, h)
    floors_internal = _detect_rooms(sealed, w, h, scale, room_labels)

    # 8. Furniture detection (needs internal floor metadata)
    furniture = _detect_furniture(thin_mask, thick_mask, floors_internal, room_labels,
                                  staircases, doors + windows, w, h, scale)

    objects = doors + windows + staircases + furniture

    # Strip internal keys (_contour, _bbox, etc.) before JSON serialization
    floors_clean = [
        {k: v for k, v in f.items() if not k.startswith("_")}
        for f in floors_internal
    ]

    return {
        "metadata": {"name": "Imported Floor Plan", "units": "meters", "source": "opencv"},
        "walls": walls,
        "floors": floors_clean,
        "objects": objects,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Thickness separation
# ═══════════════════════════════════════════════════════════════════════


def _separate_by_thickness(
    binary: np.ndarray, w: int, h: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Use distance-transform to find thick vs thin features.

    Thick = walls (drawn with heavy strokes).
    Thin  = furniture outlines, door arcs, text, annotations.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Adaptive threshold: median of all non-zero distances
    nz = dist[dist > 0]
    if nz.size == 0:
        return binary, np.zeros_like(binary), 3.0

    # Walls are typically 2-4× thicker than furniture lines
    # We use the 65th percentile as the boundary – walls sit above it
    thickness_thr = float(np.percentile(nz, 65))
    thickness_thr = max(thickness_thr, 2.0)

    thick_mask = np.zeros_like(binary)
    thick_mask[dist >= thickness_thr] = 255
    # Re-close the thick mask to fill wall interiors
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick_mask = cv2.morphologyEx(thick_mask, cv2.MORPH_CLOSE, k, iterations=3)
    thick_mask = cv2.dilate(thick_mask, k, iterations=1)

    thin_mask = cv2.bitwise_and(binary, cv2.bitwise_not(thick_mask))
    return thick_mask, thin_mask, thickness_thr


# ═══════════════════════════════════════════════════════════════════════
# 3. Wall extraction
# ═══════════════════════════════════════════════════════════════════════


def _extract_walls(
    thick_mask: np.ndarray, w: int, h: int, scale: float,
) -> list[dict]:
    dim = max(w, h)
    min_length = int(dim * 0.03)
    max_gap = int(dim * 0.02)

    # Multi-scale directional opening to clean the thick mask
    combined = np.zeros_like(thick_mask)
    for pct in (0.05, 0.025, 0.015):
        ext = max(int(dim * pct), 3)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (ext, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ext))
        combined = cv2.bitwise_or(
            combined,
            cv2.bitwise_or(
                cv2.morphologyEx(thick_mask, cv2.MORPH_OPEN, hk),
                cv2.morphologyEx(thick_mask, cv2.MORPH_OPEN, vk),
            ),
        )
    combined = _remove_small_components(combined, 0.001, (w, h))
    jk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, jk, iterations=1)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )

    # Distance transform for per-wall thickness
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)

    skeleton = _skeletonize(combined)
    raw_lines = cv2.HoughLinesP(
        skeleton, 1, np.pi / 180, 30,
        minLineLength=min_length, maxLineGap=max_gap,
    )
    if raw_lines is None:
        return []

    median_half = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 3.0
    merged = _merge_lines(raw_lines, merge_distance=median_half * 2.5)

    walls = []
    for x1, y1, x2, y2 in merged:
        seg_len = np.hypot(x2 - x1, y2 - y1)
        if seg_len < min_length * 0.8:
            continue
        thickness = _measure_thickness_along(dist, x1, y1, x2, y2) * scale
        thickness = max(round(thickness, 2), 0.08)
        walls.append({
            "id": f"w_{uuid.uuid4().hex[:6]}",
            "start": {"x": round(x1 * scale, 2), "y": 0, "z": round(y1 * scale, 2)},
            "end":   {"x": round(x2 * scale, 2), "y": 0, "z": round(y2 * scale, 2)},
            "height": 2.8,
            "thickness": thickness,
            "color": "#e8e0d4",
        })
    return walls


# ═══════════════════════════════════════════════════════════════════════
# 4. Door & Window detection
# ═══════════════════════════════════════════════════════════════════════


def _detect_doors_and_windows(
    thin_mask: np.ndarray,
    thick_mask: np.ndarray,
    walls: list[dict],
    w: int, h: int,
    scale: float,
) -> tuple[list[dict], list[dict]]:
    """
    Find gaps between wall endpoints.
      - Outer-wall gaps → WINDOWS (rendered as glass rectangles in the
        middle third of the wall height).
      - Inner-wall gaps with an arc in the thin mask → DOORS.
    Returns (doors, windows).
    """
    if not walls:
        return [], []

    dim = max(w, h)
    min_gap = dim * 0.015
    max_gap = dim * 0.12
    wall_height = 2.8  # must match wall extraction

    # 1. Collect endpoints in pixel space
    endpoints: list[tuple[float, float, str]] = []
    for wl in walls:
        sx = wl["start"]["x"] / scale
        sz = wl["start"]["z"] / scale
        ex = wl["end"]["x"] / scale
        ez = wl["end"]["z"] / scale
        endpoints.append((sx, sz, wl["id"]))
        endpoints.append((ex, ez, wl["id"]))

    # 2. Classify outer vs inner walls using convex hull
    outer_wall_ids = _find_outer_wall_ids(walls, scale)

    # 3. Find gap pairs
    gap_candidates: list[tuple[float, float, float, float, float, bool]] = []
    used_pairs: set[tuple[str, str]] = set()

    for i, (x1, y1, wid1) in enumerate(endpoints):
        for j, (x2, y2, wid2) in enumerate(endpoints):
            if j <= i or wid1 == wid2:
                continue
            pair_key = tuple(sorted([wid1, wid2]))
            if pair_key in used_pairs:
                continue

            d = np.hypot(x1 - x2, y1 - y2)
            if min_gap < d < max_gap:
                # Is this gap on the outer perimeter?
                is_outer = (wid1 in outer_wall_ids and wid2 in outer_wall_ids)

                if is_outer:
                    # Outer gap → window (no arc check needed)
                    gap_candidates.append((x1, y1, x2, y2, d, True))
                    used_pairs.add(pair_key)
                else:
                    # Inner gap → door only if arc is present
                    if thin_mask is not None and np.any(thin_mask):
                        if _verify_arc_at_gap(thin_mask, x1, y1, x2, y2, d, w, h):
                            gap_candidates.append((x1, y1, x2, y2, d, False))
                            used_pairs.add(pair_key)

    # 4. Deduplicate & build objects
    doors: list[dict] = []
    windows: list[dict] = []
    used_gaps = [False] * len(gap_candidates)

    for i, (x1, y1, x2, y2, d, is_window) in enumerate(gap_candidates):
        if used_gaps[i]:
            continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for j in range(i + 1, len(gap_candidates)):
            if used_gaps[j]:
                continue
            ox = (gap_candidates[j][0] + gap_candidates[j][2]) / 2
            oy = (gap_candidates[j][1] + gap_candidates[j][3]) / 2
            if np.hypot(cx - ox, cy - oy) < max_gap * 0.5:
                used_gaps[j] = True

        gap_width = d * scale
        angle = math.atan2(y2 - y1, x2 - x1)

        if is_window:
            # Window: spans middle third of wall height
            win_height = wall_height / 3
            win_y = wall_height / 2  # centre of middle third
            windows.append({
                "id": f"obj_{uuid.uuid4().hex[:8]}",
                "type": "window",
                "label": "Window",
                "position": {
                    "x": round(cx * scale, 2),
                    "y": round(win_y, 2),
                    "z": round(cy * scale, 2),
                },
                "rotation": {"x": 0, "y": -round(angle, 3), "z": 0},
                "scale": {
                    "x": round(gap_width, 2),
                    "y": round(win_height, 2),
                    "z": 0.06,
                },
                "color": "#b8d8e8",
                "geometry": "box",
            })
        else:
            doors.append({
                "id": f"obj_{uuid.uuid4().hex[:8]}",
                "type": "door",
                "label": "Door",
                "position": {
                    "x": round(cx * scale, 2),
                    "y": round(gap_width / 2, 2),
                    "z": round(cy * scale, 2),
                },
                "rotation": {"x": 0, "y": -round(angle, 3), "z": 0},
                "scale": {
                    "x": round(gap_width, 2),
                    "y": round(gap_width, 2),
                    "z": 0.05,
                },
                "color": "#8b6f47",
                "geometry": "box",
            })

    return doors, windows


def _find_outer_wall_ids(walls: list[dict], scale: float) -> set[str]:
    """
    Determine which walls are on the building perimeter by computing the
    convex hull of all wall midpoints.  A wall whose midpoint is within
    a tolerance of the hull boundary is classified as 'outer'.
    """
    if len(walls) < 3:
        return {w["id"] for w in walls}

    # Collect midpoints in pixel space
    pts = []
    ids = []
    for wl in walls:
        mx = (wl["start"]["x"] + wl["end"]["x"]) / 2 / scale
        mz = (wl["start"]["z"] + wl["end"]["z"]) / 2 / scale
        pts.append([mx, mz])
        ids.append(wl["id"])

    pts_arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(pts_arr, returnPoints=True)

    # Distance threshold: walls within this distance of the hull → outer
    all_coords = []
    for wl in walls:
        all_coords.append([wl["start"]["x"] / scale, wl["start"]["z"] / scale])
        all_coords.append([wl["end"]["x"] / scale, wl["end"]["z"] / scale])
    all_arr = np.array(all_coords)
    span = max(all_arr[:, 0].max() - all_arr[:, 0].min(),
               all_arr[:, 1].max() - all_arr[:, 1].min())
    tolerance = span * 0.05

    hull_contour = hull.reshape(-1, 1, 2).astype(np.float32)

    outer_ids: set[str] = set()
    for i, (mx, mz) in enumerate(pts):
        dist = abs(cv2.pointPolygonTest(hull_contour, (mx, mz), True))
        if dist < tolerance:
            outer_ids.add(ids[i])

    return outer_ids


def _verify_arc_at_gap(
    thin_mask: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    gap_dist: float,
    w: int, h: int,
) -> bool:
    """
    Check if there are enough arc-like pixels in the thin mask near the gap.
    Sample a quarter-circle arc from one endpoint and see if pixels are set.
    """
    radius = gap_dist
    cx, cy = x1, y1  # hinge at endpoint 1
    # Try arc sweeping from the direction of endpoint 2
    base_angle = math.atan2(y2 - y1, x2 - x1)

    arc_pixel_count = 0
    n_samples = 20
    for i in range(n_samples):
        # Sample a 90° arc (quarter circle)
        theta = base_angle + (math.pi / 2) * (i / (n_samples - 1))
        px = int(cx + radius * math.cos(theta))
        py = int(cy + radius * math.sin(theta))
        if 0 <= px < w and 0 <= py < h:
            if thin_mask[py, px] > 0:
                arc_pixel_count += 1

    # Also try the other hinge point and opposite sweep direction
    cx2, cy2 = x2, y2
    base_angle2 = math.atan2(y1 - y2, x1 - x2)
    arc_pixel_count2 = 0
    for i in range(n_samples):
        theta = base_angle2 - (math.pi / 2) * (i / (n_samples - 1))
        px = int(cx2 + radius * math.cos(theta))
        py = int(cy2 + radius * math.sin(theta))
        if 0 <= px < w and 0 <= py < h:
            if thin_mask[py, px] > 0:
                arc_pixel_count2 += 1

    best = max(arc_pixel_count, arc_pixel_count2)
    # Need at least 30% of sampled arc points to be set
    return best >= n_samples * 0.3


# ═══════════════════════════════════════════════════════════════════════
# 5. Staircase detection
# ═══════════════════════════════════════════════════════════════════════


def _detect_staircases(
    thin_mask: np.ndarray, w: int, h: int, scale: float,
) -> list[dict]:
    """
    Staircases are shown as a rectangular region filled with many parallel
    lines at regular spacing.  Detect by finding contour rectangles whose
    interior has high density of short parallel Hough lines.
    """
    if thin_mask is None or not np.any(thin_mask):
        return []

    dim = max(w, h)
    # Find large-ish contours in thin mask that could be staircase regions
    contours, _ = cv2.findContours(thin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.01
    max_area = (w * h) * 0.25
    staircases: list[dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.boundingRect(cnt)
        rx, ry, rw, rh = rect
        aspect = max(rw, rh) / max(min(rw, rh), 1)

        # Staircases are roughly rectangular (aspect 1–3)
        if aspect > 4:
            continue

        # Extract the region
        roi = thin_mask[ry:ry + rh, rx:rx + rw]
        roi_lines = cv2.HoughLinesP(
            roi, 1, np.pi / 180, 15,
            minLineLength=int(min(rw, rh) * 0.3),
            maxLineGap=5,
        )
        if roi_lines is None:
            continue

        # Count lines grouped by angle (look for parallel clusters)
        angles = []
        for ln in roi_lines:
            lx1, ly1, lx2, ly2 = ln[0]
            ang = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)) % 180
            angles.append(ang)

        if len(angles) < 4:
            continue

        # Check if many lines share a similar angle (within 10°)
        angles_arr = np.array(angles)
        best_count = 0
        for a in angles_arr:
            diffs = np.abs(angles_arr - a)
            diffs = np.minimum(diffs, 180 - diffs)
            count = np.sum(diffs < 10)
            best_count = max(best_count, count)

        # Need at least 4 parallel lines to call it a staircase
        if best_count < 4:
            continue

        # Generate step objects
        cx = (rx + rw / 2) * scale
        cz = (ry + rh / 2) * scale
        sw = rw * scale
        sd = rh * scale

        n_steps = best_count
        step_height = 2.4 / n_steps  # total staircase height ~2.4m

        for step_i in range(n_steps):
            y_pos = step_i * step_height + step_height / 2
            # Steps progress along the longer dimension
            if rw >= rh:
                frac = step_i / max(n_steps - 1, 1)
                step_x = (rx + rw * frac) * scale
                step_z = cz
                step_w = sw / n_steps * 0.9
                step_d = sd * 0.95
            else:
                frac = step_i / max(n_steps - 1, 1)
                step_x = cx
                step_z = (ry + rh * frac) * scale
                step_w = sw * 0.95
                step_d = sd / n_steps * 0.9

            staircases.append({
                "id": f"obj_{uuid.uuid4().hex[:8]}",
                "type": "staircase",
                "label": f"Step {step_i + 1}",
                "position": {"x": round(step_x, 2),
                             "y": round(y_pos, 2),
                             "z": round(step_z, 2)},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": round(step_w, 2),
                          "y": round(step_height * 0.9, 2),
                          "z": round(step_d, 2)},
                "color": "#b0a090",
                "geometry": "box",
            })

        break  # one staircase per plan for now

    return staircases


# ═══════════════════════════════════════════════════════════════════════
# 6. Room-label OCR
# ═══════════════════════════════════════════════════════════════════════


def _detect_room_labels(
    gray: np.ndarray, w: int, h: int, scale: float,
) -> list[dict]:
    """
    Use pytesseract to find text labels and their bounding boxes.
    Returns [{text, type, cx, cy, box}] in **pixel** coordinates.
    """
    if not HAS_OCR:
        return []

    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                          config="--psm 11")
    except Exception:
        return []

    labels: list[dict] = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt or len(txt) < 2:
            continue
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        if conf < 40:
            continue

        bx, by, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        cx_px = bx + bw / 2
        cy_px = by + bh / 2

        room_type = _classify_room_text(txt)
        if room_type:
            labels.append({
                "text": txt,
                "type": room_type,
                "cx": cx_px,
                "cy": cy_px,
                "box": (bx, by, bw, bh),
            })

    # Deduplicate: merge labels of the same type that are nearby
    merged: list[dict] = []
    used = set()
    for i, lab in enumerate(labels):
        if i in used:
            continue
        group = [lab]
        used.add(i)
        for j, other in enumerate(labels):
            if j in used:
                continue
            if other["type"] == lab["type"]:
                dist = np.hypot(lab["cx"] - other["cx"], lab["cy"] - other["cy"])
                if dist < max(w, h) * 0.15:
                    group.append(other)
                    used.add(j)
        avg_cx = np.mean([g["cx"] for g in group])
        avg_cy = np.mean([g["cy"] for g in group])
        merged.append({
            "text": " ".join(g["text"] for g in group),
            "type": group[0]["type"],
            "cx": float(avg_cx),
            "cy": float(avg_cy),
        })

    return merged


def _classify_room_text(text: str) -> str | None:
    lower = text.lower()
    for keyword, room_type in ROOM_KEYWORDS.items():
        if keyword in lower:
            return room_type
    return None


# ═══════════════════════════════════════════════════════════════════════
# 7. Room / floor detection
# ═══════════════════════════════════════════════════════════════════════


def _seal_wall_gaps(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    gap = max(int(max(w, h) * 0.015), 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    sealed = cv2.dilate(mask, k, iterations=2)
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, ck, iterations=2)
    return sealed


def _detect_rooms(
    sealed_mask: np.ndarray, w: int, h: int, scale: float,
    room_labels: list[dict],
) -> list[dict]:
    inv = cv2.bitwise_not(sealed_mask)
    ok = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, ok, iterations=1)

    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    total_area = w * h
    min_area = total_area * 0.008
    max_area = total_area * 0.85

    room_colors = {
        "bedroom":  "#c0c9d4",
        "living":   "#d4c9b8",
        "kitchen":  "#c9d4c0",
        "bathroom": "#c0d4d4",
        "laundry":  "#d4d4c0",
        "office":   "#d4c0d4",
        "hallway":  "#d8d4cc",
        "garage":   "#c8c8c8",
        "closet":   "#ddd8d0",
        "unknown":  "#d0ccc4",
    }

    floors: list[dict] = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        # Find which room label falls inside this contour
        room_type = "unknown"
        room_name = "Room"
        for lab in room_labels:
            if cv2.pointPolygonTest(contour, (lab["cx"], lab["cy"]), False) >= 0:
                room_type = lab["type"]
                room_name = lab["text"].title()
                break

        # Fallback: guess by relative area
        if room_type == "unknown":
            area_ratio = area / total_area
            room_type = _guess_room_type_by_size(area_ratio)
            room_name = room_type.title()

        vertices = [
            {"x": round(pt[0][0] * scale, 2), "y": 0, "z": round(pt[0][1] * scale, 2)}
            for pt in approx
        ]
        color = room_colors.get(room_type, "#d0ccc4")

        floors.append({
            "id": f"f_{uuid.uuid4().hex[:6]}",
            "vertices": vertices,
            "color": color,
            "material": "concrete",
            "room_type": room_type,
            "room_name": room_name,
            # Store pixel-space bbox for furniture placement
            "_bbox": cv2.boundingRect(contour),
            "_contour": contour,
        })

    return floors


def _guess_room_type_by_size(area_ratio: float) -> str:
    """Fallback guess based on room area as fraction of total image."""
    if area_ratio > 0.25:
        return "living"
    if area_ratio > 0.12:
        return "bedroom"
    if area_ratio > 0.06:
        return "kitchen"
    if area_ratio > 0.03:
        return "bathroom"
    return "closet"


# ═══════════════════════════════════════════════════════════════════════
# 8. Furniture detection
# ═══════════════════════════════════════════════════════════════════════


def _detect_furniture(
    thin_mask: np.ndarray,
    thick_mask: np.ndarray,
    floors: list[dict],
    room_labels: list[dict],
    staircases: list[dict],
    doors: list[dict],
    w: int, h: int,
    scale: float,
) -> list[dict]:
    """
    Find contours in the thin mask, filter out doors / staircases,
    then classify remaining shapes by the room they fall in.
    """
    if thin_mask is None or not np.any(thin_mask):
        return []

    # Clean thin mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thin_mask, cv2.MORPH_CLOSE, k, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = w * h
    min_furn_area = total_area * 0.001   # very small → noise / text
    max_furn_area = total_area * 0.08    # too big → probably not furniture

    # Bounding boxes of already-detected staircases & doors (in pixels)
    excluded_centres: list[tuple[float, float]] = []
    for obj in staircases + doors:
        px = obj["position"]["x"] / scale
        pz = obj["position"]["z"] / scale
        excluded_centres.append((px, pz))

    furniture: list[dict] = []
    # Track how many items per room type to cycle through templates
    room_item_idx: dict[str, int] = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_furn_area or area > max_furn_area:
            continue

        # Skip if it overlaps an already-detected staircase / door
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cnt_cx = M["m10"] / M["m00"]
        cnt_cy = M["m01"] / M["m00"]

        skip = False
        for ecx, ecy in excluded_centres:
            if np.hypot(cnt_cx - ecx, cnt_cy - ecy) < max(w, h) * 0.04:
                skip = True
                break
        if skip:
            continue

        # Determine which room this furniture belongs to
        room_type = "unknown"
        for fl in floors:
            cnt_fl = fl.get("_contour")
            if cnt_fl is not None:
                if cv2.pointPolygonTest(cnt_fl, (cnt_cx, cnt_cy), False) >= 0:
                    room_type = fl.get("room_type", "unknown")
                    break

        # Pick a template for this room
        templates = FURNITURE_TEMPLATES.get(room_type, FURNITURE_TEMPLATES["unknown"])
        idx = room_item_idx.get(room_type, 0)
        template = templates[idx % len(templates)]
        room_item_idx[room_type] = idx + 1

        label, geom, color, obj_height, w_ratio, d_ratio = template

        # Size from bounding rect
        bx, by, bw, bh = cv2.boundingRect(cnt)
        obj_w = bw * scale * w_ratio
        obj_d = bh * scale * d_ratio

        # Clamp to reasonable sizes
        obj_w = max(min(obj_w, 3.0), 0.2)
        obj_d = max(min(obj_d, 3.0), 0.2)

        # Determine circularity → use cylinder for round objects
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity > 0.75:
            geom = "cylinder"

        furniture.append({
            "id": f"obj_{uuid.uuid4().hex[:8]}",
            "type": "furniture",
            "label": label,
            "position": {
                "x": round(cnt_cx * scale, 2),
                "y": round(obj_height / 2, 2),
                "z": round(cnt_cy * scale, 2),
            },
            "rotation": {"x": 0, "y": 0, "z": 0},
            "scale": {
                "x": round(obj_w, 2),
                "y": round(obj_height, 2),
                "z": round(obj_d, 2),
            },
            "color": color,
            "geometry": geom,
        })

    return furniture


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _remove_small_components(
    mask: np.ndarray, min_ratio: float, wh: tuple[int, int],
) -> np.ndarray:
    min_area = wh[0] * wh[1] * min_ratio
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    skel = np.zeros_like(mask)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()
    while True:
        eroded = cv2.erode(temp, el)
        opened = cv2.dilate(eroded, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(temp, opened))
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def _merge_lines(lines: np.ndarray, merge_distance: float = 15.0) -> list:
    raw = [tuple(l[0]) for l in lines]
    used: set[int] = set()
    merged: list[tuple] = []

    for i, (x1, y1, x2, y2) in enumerate(raw):
        if i in used:
            continue
        ang_i = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        group = [(x1, y1), (x2, y2)]
        used.add(i)

        for j, (ax, ay, bx, by) in enumerate(raw):
            if j in used:
                continue
            ang_j = np.degrees(np.arctan2(by - ay, bx - ax)) % 180
            ad = abs(ang_i - ang_j)
            if ad > 90:
                ad = 180 - ad
            if ad > 5:
                continue
            if _perp_dist(x1, y1, x2, y2, ax, ay, bx, by) > merge_distance:
                continue
            if _overlap(x1, y1, x2, y2, ax, ay, bx, by, merge_distance):
                group.extend([(ax, ay), (bx, by)])
                used.add(j)

        xs = [p[0] for p in group]
        ys = [p[1] for p in group]
        if ang_i <= 45 or ang_i >= 135:
            avg_y = int(np.mean(ys))
            merged.append((min(xs), avg_y, max(xs), avg_y))
        else:
            avg_x = int(np.mean(xs))
            merged.append((avg_x, min(ys), avg_x, max(ys)))
    return merged


def _perp_dist(x1, y1, x2, y2, ax, ay, bx, by) -> float:
    dx, dy = x2 - x1, y2 - y1
    ln = np.hypot(dx, dy)
    if ln < 1e-6:
        mx, my = (ax + bx) / 2, (ay + by) / 2
        return float(np.hypot(mx - x1, my - y1))
    nx, ny = -dy / ln, dx / ln
    mx, my = (ax + bx) / 2, (ay + by) / 2
    return float(abs(nx * (mx - x1) + ny * (my - y1)))


def _overlap(x1, y1, x2, y2, ax, ay, bx, by, tol: float) -> bool:
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    if ang <= 45 or ang >= 135:
        lo1, hi1 = min(x1, x2), max(x1, x2)
        lo2, hi2 = min(ax, bx), max(ax, bx)
    else:
        lo1, hi1 = min(y1, y2), max(y1, y2)
        lo2, hi2 = min(ay, by), max(ay, by)
    return lo1 <= hi2 + tol and lo2 <= hi1 + tol


def _measure_thickness_along(
    dist_map: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    n_samples: int = 15,
) -> float:
    h, w = dist_map.shape
    vals = []
    for t in np.linspace(0, 1, n_samples):
        px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
        if 0 <= px < w and 0 <= py < h and dist_map[py, px] > 0:
            vals.append(dist_map[py, px])
    return float(np.median(vals)) * 2 if vals else 0.15


def _empty_scene(msg: str):
    return {
        "metadata": {"name": msg, "units": "meters", "source": "opencv"},
        "walls": [], "floors": [], "objects": [],
    }
