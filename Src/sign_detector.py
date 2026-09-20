import cv2
import numpy as np


def detect_sign_candidates(frame, debug=False):
    """
    Detect traffic-sign regions using OpenCV HSV color segmentation,
    RGB color dominance verification, and geometric shape/solidity/circularity filtering.

    Robustly detects:
    - Solid signs (STOP, No Entry, Blue mandatory circles)
    - Ring/border signs (Speed limits 20-120, Yield, No Passing)
    - Warning & Priority signs (Caution, Priority Road, Curves)

    Rejects random background clutter, skin tones, clothing, wooden furniture, and shadows.
    Returns sorted list of bounding boxes: [(x, y, w, h), ...]
    """

    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame.astype(np.float32))

    # -------------------------------------------------------------
    # 1. VIVID RED MASK (STOP, Speed limits, Yield, No Entry, Danger)
    # Traffic Red: High saturation, high red dominance over G and B
    # -------------------------------------------------------------
    red1 = cv2.inRange(
        hsv,
        np.array([0, 110, 70]),
        np.array([10, 255, 255])
    )
    red2 = cv2.inRange(
        hsv,
        np.array([168, 110, 70]),
        np.array([180, 255, 255])
    )
    red_hsv = red1 | red2
    red_rgb = (r > 90) & (r > g + 25) & (r > b + 25)
    red_mask = red_hsv & (red_rgb.astype(np.uint8) * 255)

    # -------------------------------------------------------------
    # 2. VIVID BLUE MASK (Mandatory direction and roundabout signs)
    # Traffic Blue: High saturation, high blue dominance over R and G
    # -------------------------------------------------------------
    blue_hsv = cv2.inRange(
        hsv,
        np.array([95, 90, 60]),
        np.array([130, 255, 255])
    )
    blue_rgb = (b > 95) & (b > r + 30) & (b > g + 15)
    blue_mask = blue_hsv & (blue_rgb.astype(np.uint8) * 255)

    # -------------------------------------------------------------
    # 3. VIVID YELLOW MASK (Warning signs, Priority road)
    # Traffic Yellow: High saturation, high brightness, low blue
    # -------------------------------------------------------------
    yellow_hsv = cv2.inRange(
        hsv,
        np.array([18, 120, 110]),
        np.array([32, 255, 255])
    )
    yellow_rgb = (r > 120) & (g > 95) & (r - b > 40) & (b < 100)
    yellow_mask = yellow_hsv & (yellow_rgb.astype(np.uint8) * 255)

    # Combine all sign color masks
    mask = red_mask | blue_mask | yellow_mask

    # Morphological noise removal & contour closing
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Find external contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # Sign area constraints (min 700 px, max 40% of camera frame)
        if area < 700 or area > (h * w * 0.40):
            continue

        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < 32 or ch < 32:
            continue

        # Aspect Ratio Filter: compact shapes (circles, triangles, octagons, squares)
        aspect_ratio = cw / float(ch)
        if aspect_ratio < 0.65 or aspect_ratio > 1.50:
            continue

        # Solidity Filter: convexity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.70:
            continue

        # Circularity Filter: compactness check
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.25:
            continue

        # Color Density Filter: percentage of colored sign pixels inside bounding box
        roi_mask = mask[cy:cy+ch, cx:cx+cw]
        color_density = np.count_nonzero(roi_mask) / float(cw * ch)
        if color_density < 0.16:
            continue

        # Quality score based on area, solidity, circularity, and color density
        shape_score = solidity * circularity
        color_score = color_density
        final_score = area * shape_score * color_score
        fill_ratio = area / float(cw * ch)

        candidates.append((cx, cy, cw, ch, final_score, area, aspect_ratio, fill_ratio, circularity, shape_score, color_score))

    # Sort strongest candidate first
    candidates.sort(key=lambda item: item[4], reverse=True)

    if debug and candidates:
        print(f"\n[SignDetector Debug] NUMBER OF CANDIDATES FOUND = {len(candidates)}")
        for idx, c in enumerate(candidates):
            print(
                f" Candidate #{idx+1}: x={c[0]}, y={c[1]}, width={c[2]}, height={c[3]}, "
                f"area={c[5]:.1f}, aspect_ratio={c[6]:.2f}, fill_ratio={c[7]:.2f}, "
                f"circularity={c[8]:.2f}, shape_score={c[9]:.2f}, color_score={c[10]:.2f}, "
                f"final_score={c[4]:.1f}"
            )
        sel = candidates[0]
        print(f"[SignDetector Debug] SELECTED CANDIDATE = bbox=({sel[0]}, {sel[1]}, {sel[2]}, {sel[3]}), score={sel[4]:.1f}")

    return [(c[0], c[1], c[2], c[3]) for c in candidates]