# 🚗 Lane Detection — Computer Vision Pipeline

> Detects road lane lines in images using classical computer vision techniques — no deep learning required.

---

## Overview

This project implements a full lane detection pipeline using image segmentation techniques. It identifies yellow and white lane markings, clusters detected line segments, and projects full lane lines up to a calculated vanishing point. The output is a 6-stage visualization showing every step of the pipeline.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

---

## How It Works

1. **Color Masking** — HSV-based yellow mask + LAB-based white mask isolates lane colors
2. **Edge Detection** — Gaussian blur + Canny edge detection on the grayscale image
3. **Hough Transform** — Probabilistic Hough Line Transform extracts line segments
4. **Clustering** — Segments grouped by x-position gap to identify individual lanes
5. **Vanishing Point** — Calculated from intersections of all fitted lane lines
6. **Projection** — Each lane line drawn from the bottom of frame up to the vanishing point

---

## How to Run

```bash
git clone https://github.com/Khalfani04/lane-detection
cd lane-detection
pip install -r requirements.txt
python lane_detection.py
```

A file dialog will open — select any road image (`.jpg`, `.png`, etc.) to run detection on.

---

## Output

The pipeline displays a 6-panel visualization:
| Panel | Description |
|-------|-------------|
| 1 | Original Image |
| 2 | Color Mask (White & Yellow) |
| 3 | ROI Masked Image |
| 4 | Canny Edge Detection |
| 5 | Detected Lane Lines |
| 6 | Final Result |

---

## What I Learned

- How to isolate colors in HSV and LAB color spaces for robust detection under varying lighting
- How Hough Line Transform parameters affect detection sensitivity vs. noise
- How to calculate a vanishing point geometrically from multiple line fits

---

## Contact

**Khalfani Norman** · [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/Khalfani04)
