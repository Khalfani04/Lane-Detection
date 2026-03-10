"""
Lane Detection using Image Segmentation
========================================
Technique: Canny Edge Detection + Hough Line Transform + Color Masking
Libraries: OpenCV, NumPy, Matplotlib, Pillow, Tkinter
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from PIL import Image


def load_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Road Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        exit()
    image = np.array(Image.open(file_path).convert("RGB"))
    print(f"Loaded: {file_path}  |  Size: {image.shape[1]}x{image.shape[0]}")
    return image


def get_yellow_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    yellow = cv2.dilate(yellow, np.ones((5, 5), np.uint8))
    return yellow


def get_gray_edges(image, h):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges[:int(h * 0.5)] = 0
    return edges


def get_color_mask_display(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow_raw = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    _, white = cv2.threshold(lab[:, :, 0], 200, 255, cv2.THRESH_BINARY)
    white = cv2.bitwise_and(white, cv2.bitwise_not(yellow_raw))
    return cv2.bitwise_or(yellow_raw, white)


def hough(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                             minLineLength=20, maxLineGap=100)
    segs = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if x1 == x2: continue
            slope = (y2-y1)/(x2-x1)
            if abs(slope) < 0.4 or abs(slope) > 3.0: continue
            segs.append((x1, y1, x2, y2))
    return segs


def cluster_by_gap(segs, w):
    if not segs:
        return []
    segs_sorted = sorted(segs, key=lambda s: (s[0]+s[2])/2)
    xs = [(s[0]+s[2])/2 for s in segs_sorted]
    gaps = [(xs[i+1]-xs[i], i) for i in range(len(xs)-1)]
    if not gaps:
        return [segs_sorted]
    threshold = max(w * 0.10, 30)
    split_points = sorted([i+1 for gap, i in gaps if gap > threshold])
    if not split_points:
        split_points = [max(gaps, key=lambda g: g[0])[1] + 1]
    clusters, prev = [], 0
    for sp in split_points:
        clusters.append(segs_sorted[prev:sp])
        prev = sp
    clusters.append(segs_sorted[prev:])
    return [c for c in clusters if c]


def is_yellow(segs, yellow_mask):
    hits, total = 0, 0
    for x1, y1, x2, y2 in segs:
        n = max(abs(x2-x1), abs(y2-y1), 1)
        for t in np.linspace(0, 1, min(n, 20)):
            x = int(np.clip(x1 + t*(x2-x1), 0, yellow_mask.shape[1]-1))
            y = int(np.clip(y1 + t*(y2-y1), 0, yellow_mask.shape[0]-1))
            if yellow_mask[y, x] > 0:
                hits += 1
            total += 1
    return (hits / total) > 0.5 if total > 0 else False


def polyfit_coeffs(segs):
    """Return (m, b) for x = m*y + b fitted through all segment points."""
    pts = np.array([(x,y) for x1,y1,x2,y2 in segs for x,y in [(x1,y1),(x2,y2)]])
    m, b = np.polyfit(pts[:,1].astype(float), pts[:,0].astype(float), 1)
    return m, b


def find_vanishing_point(fitted_lines):
    """
    Find the average intersection point of all pairs of fitted lines.
    Each fitted line is (m, b) where x = m*y + b.
    Two lines intersect at: m1*y + b1 = m2*y + b2 → y = (b2-b1)/(m1-m2)
    """
    intersections = []
    for i in range(len(fitted_lines)):
        for j in range(i+1, len(fitted_lines)):
            m1, b1 = fitted_lines[i]
            m2, b2 = fitted_lines[j]
            if abs(m1 - m2) < 1e-6:
                continue  # parallel lines
            y_int = (b2 - b1) / (m1 - m2)
            x_int = m1 * y_int + b1
            intersections.append((x_int, y_int))

    if not intersections:
        return None
    # Use median to be robust against outliers
    vp_x = np.median([p[0] for p in intersections])
    vp_y = np.median([p[1] for p in intersections])
    return vp_x, vp_y


def run_lane_detection():
    print("=" * 50)
    print("  Lane Detection — Image Segmentation Demo")
    print("=" * 50)

    original    = load_image()
    h, w        = original.shape[:2]
    yellow_mask = get_yellow_mask(original)
    edges       = get_gray_edges(original, h)
    all_segs    = hough(edges)
    clusters    = cluster_by_gap(all_segs, w)

    print(f"Total segments: {len(all_segs)} → {len(clusters)} clusters")

    # First pass: fit all lines and find vanishing point
    fitted = []
    cluster_meta = []
    for cluster in clusters:
        xs_all = [x for x1,y1,x2,y2 in cluster for x in [x1,x2]]
        yellow = is_yellow(cluster, yellow_mask)
        touches_edge = min(xs_all) < 15 or max(xs_all) > w - 15
        extend_down  = not touches_edge
        m, b = polyfit_coeffs(cluster)
        fitted.append((m, b))
        cluster_meta.append({
            'cluster': cluster,
            'yellow': yellow,
            'extend_down': extend_down,
            'm': m, 'b': b
        })

    vp = find_vanishing_point(fitted)
    if vp:
        vp_y = int(np.clip(vp[1], 0, h - 1))
        print(f"Vanishing point: x={int(vp[0])}, y={vp_y}")
    else:
        vp_y = 0

    # Second pass: draw lines stopping at vanishing point y
    line_image = np.zeros_like(original)
    for meta in cluster_meta:
        m, b = meta['m'], meta['b']
        color = (0, 200, 255) if meta['yellow'] else (0, 255, 0)
        label = "Yellow" if meta['yellow'] else "White"

        y_bot = (h-1) if meta['extend_down'] else int(
            max(y for x1,y1,x2,y2 in meta['cluster'] for y in [y1,y2]))
        y_top = vp_y  # stop at vanishing point

        x_bot = int(np.clip(m*y_bot+b, 0, w-1))
        x_top = int(np.clip(m*y_top+b, 0, w-1))
        cv2.line(line_image, (x_bot,y_bot), (x_top,y_top), color, 8)
        print(f"  {label}: ({x_bot},{y_bot}) -> ({x_top},{y_top})")


    result_img = cv2.addWeighted(original, 0.8, line_image, 1.0, 0)
    combined   = get_color_mask_display(original)
    roi_disp   = cv2.bitwise_and(original, original, mask=combined)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Lane Detection Pipeline — Image Segmentation",
                 fontsize=16, fontweight='bold')
    stages = [
        (original,  "1. Original Image",               {}),
        (combined,  "2. Color Mask\n(White & Yellow)", {"cmap": "gray"}),
        (roi_disp,  "3. ROI Masked Image",             {}),
        (edges,     "4. Edge Detection (Canny)",       {"cmap": "gray"}),
        (line_image,"5. Detected Lane Lines",          {}),
        (result_img,"6. Final Result",                 {}),
    ]
    for ax, (img, title, kwargs) in zip(axes.flatten(), stages):
        ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    print("\nLane detection complete!")


if __name__ == "__main__":
    run_lane_detection()
