"""01_auto_crop.py — OpenCV-based grid-photo cropper.

Extracts individual coffee-bean images from 5x5 grid photos taken inside
the studio mini setup, producing a clean `Dataset_Cropped/` directory of
224x224 JPGs ready for PyTorch's `ImageFolder`.

Pipeline (Master Coding Prompt §2):
    1.  Grayscale + Gaussian blur -> smooth pencil dots and paper texture.
    2.  Otsu's inverse threshold  -> beans = white blobs, paper = black.
    3.  Morphological opening     -> remove tiny pencil dots, keep beans.
    4.  Contour dilation          -> expand masks outward to recover edges
                                     that may have been eroded.
    5.  cv2.findContours          -> bean candidates.
    6.  Area filter (MIN..MAX)    -> drop dust AND merged-bean blobs.
    7.  Outlier pruning           -> cap to exactly expected_per_photo.
    8.  Uniform crop sizing       -> P75-based side across ALL beans in
                                     a photo, so inconsistent mask sizes
                                     cannot produce clipped crops.
    9.  Centroid + bbox blended   -> perceptual-centre crop placement.
   10.  Crop with WHITE padding   -> studio background colour.
   11.  Resize to TARGET_SIZE     -> deterministic CNN input.
   12.  Save as
        <output>/<class>/<class>_<global_seq>.jpg

Usage (from repo root):
    py scripts/01_auto_crop.py
    py scripts/01_auto_crop.py --debug
    py scripts/01_auto_crop.py --input-dir Dataset --output-dir Dataset_Cropped

The script is idempotent — already-cropped output files are skipped, so
you can safely re-run it as new raw photos arrive.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2
import numpy as np
from tqdm import tqdm


class BeanBox(NamedTuple):
    """Bounding rect + centroid for one detected bean."""
    x: int
    y: int
    w: int
    h: int
    cx: int  # centroid x (from image moments — true visual centre)
    cy: int  # centroid y

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Map raw input folders -> normalised output class names.
# Master Coding Prompt §2.7 calls for `Defect/` and `Non-Defect/`, but per the
# user's preference (and chapter-3.tex line 87) we use lowercase + underscore.
CLASS_MAP: dict[str, str] = {
    "Biji Kopi Defect": "defect",
    "Biji Kopi Non-Defect": "non_defect",
}

VALID_EXT: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
JPG_QUALITY = 95
PADDING_COLOUR_BGR: tuple[int, int, int] = (255, 255, 255)  # white studio bg


@dataclass
class CropConfig:
    """Tunable parameters for the cropping pipeline."""

    input_dir: Path
    output_dir: Path
    min_area: int = 1_000          # px^2 — Master Prompt §2.4
    max_area_ratio: float = 3.0    # reject blobs > 3× median area (merged beans)
    margin_pct: float = 0.18       # 18 % padding — enterprise-grade safety margin
    target_size: int = 224         # CNN input resolution
    expected_per_photo: int = 25   # 5x5 grid; flagged in QC report if mismatch
    debug: bool = False            # save contour-overlay images
    blur_kernel: int = 5           # Gaussian blur (odd)
    morph_kernel: int = 3          # Morphological opening kernel size
    morph_iter: int = 1            # Morphological opening iterations
    dilate_iter: int = 2           # dilate mask before contour extraction


# -----------------------------------------------------------------------------
# Core image-processing helpers
# -----------------------------------------------------------------------------
def detect_bean_contours(image_bgr: np.ndarray, cfg: CropConfig
                         ) -> list[BeanBox]:
    """Return list of BeanBox for every detected bean.

    Each BeanBox contains the hull bounding rect (x, y, w, h) AND the
    contour centroid (cx, cy) computed from image moments.  The centroid
    is the bean's true visual mass-centre and is used for crop centering
    so beans appear properly centred in the output image.

    Boxes are returned sorted top-to-bottom, then left-to-right, so that
    output filenames have a stable, predictable suffix (`_00` is top-left
    bean, `_24` is bottom-right bean in a perfect 5x5 grid).

    When more candidates than ``expected_per_photo`` are found, the extras
    are pruned by removing the blobs whose area deviates most from the
    median — these are the most likely to be non-bean artefacts (pencil
    marks, shadows, paper texture, etc.).

    Improvements over the original tight-bbox approach:
      - Dilation after opening recovers bean edges lost to morphological
        erosion, preventing the crop from clipping bean sides.
      - Convex hull is used instead of raw contour for the bounding rect,
        giving a more faithful envelope around irregularly-shaped beans.
      - A max-area filter rejects blobs that are likely two merged beans.
      - Centroid-based centering ensures the bean sits at the visual
        centre of the crop, not just the bounding-box midpoint.
      - Outlier pruning caps the result to exactly expected_per_photo.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)

    # Beans are darker than the white paper -> THRESH_BINARY_INV makes beans
    # bright (255) on a black background, which is what findContours expects.
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Opening = erosion followed by dilation. Removes tiny pencil dots and
    # dust without merging adjacent beans.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel,
                            iterations=cfg.morph_iter)

    # Dilation: expand the mask outward so that the contour fully covers the
    # bean edge pixels that opening may have shaved off.  This is the key fix
    # for the "bean side getting cut" problem.
    if cfg.dilate_iter > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=cfg.dilate_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # --- First pass: collect candidate areas for median computation --------
    candidate_areas: list[float] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= cfg.min_area:
            candidate_areas.append(a)

    median_area = float(np.median(candidate_areas)) if candidate_areas else 0.0
    max_area = median_area * cfg.max_area_ratio if median_area > 0 else float("inf")

    # --- Second pass: build BeanBoxes using convex hull + centroid ---------
    # Also store each candidate's contour area for outlier pruning later.
    boxes: list[BeanBox] = []
    areas: list[float] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area:
            continue
        if area > max_area:
            # Likely two adjacent beans merged into one blob — skip.
            continue

        # Use the convex hull for the bounding rect.  The hull wraps the
        # contour's outermost points, producing a slightly larger (but more
        # accurate) envelope that is less likely to clip curved bean edges.
        hull = cv2.convexHull(c)
        bx, by, bw, bh = cv2.boundingRect(hull)

        # Compute centroid from image moments of the *original* contour.
        # This gives the true visual mass-centre, which is more accurate
        # than the bounding-box midpoint for asymmetric bean shapes.
        M = cv2.moments(c)
        if M["m00"] > 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            # Fallback to bbox midpoint if moments fail (degenerate contour).
            centroid_x = bx + bw // 2
            centroid_y = by + bh // 2

        boxes.append(BeanBox(bx, by, bw, bh, centroid_x, centroid_y))
        areas.append(area)

    # --- Outlier pruning: cap to exactly expected_per_photo ----------------
    # When more candidates than expected are found, the extras are almost
    # certainly non-bean artefacts (pencil marks, shadows, dust clumps).
    # We rank every candidate by how far its area deviates from the median
    # and drop the worst outliers until we have exactly the expected count.
    if len(boxes) > cfg.expected_per_photo and median_area > 0:
        # Pair each box with its absolute area deviation from median.
        scored = sorted(
            zip(boxes, areas),
            key=lambda pair: abs(pair[1] - median_area),
        )
        # Keep the expected_per_photo candidates closest to the median.
        boxes = [b for b, _ in scored[:cfg.expected_per_photo]]

    # Stable order: row-major (top->bottom, then left->right).
    # Bucket y into rows by quantising to ~half the median bean height; this
    # tolerates beans whose bounding-box tops are slightly off.
    if boxes:
        median_h = int(np.median([b.h for b in boxes]))
        row_bucket = max(median_h // 2, 1)
        boxes.sort(key=lambda b: (b.y // row_bucket, b.x))

    return boxes


def compute_uniform_crop_side(boxes: list[BeanBox],
                              cfg: CropConfig) -> int:
    """Compute a single, uniform crop side length for ALL beans in a photo.

    Using each bean's own ``max(w, h)`` for its crop is fragile: if Otsu /
    dilation under-segments one bean, that bean's crop will be too small
    and its edges will be clipped.  Instead, we derive a *uniform* side
    from the statistical distribution of all beans in the same photo:

        uniform_side = P75(max(w, h) per bean) × (1 + margin_pct)

    Why P75 (75th percentile)?
      - More generous than the median → covers most natural size variation.
      - More robust than the max → not swayed by one abnormally large blob.

    A per-bean safety floor guarantees that any individual bean whose
    ``max(w, h)`` exceeds the P75 still receives a large-enough crop
    (handled inside ``crop_square_with_padding``).
    """
    if not boxes:
        return cfg.target_size
    dims = [max(b.w, b.h) for b in boxes]
    p75 = int(np.percentile(dims, 75))
    return int(p75 * (1.0 + cfg.margin_pct))


def crop_square_with_padding(image_bgr: np.ndarray,
                             bean: BeanBox,
                             cfg: CropConfig,
                             uniform_side: int) -> np.ndarray:
    """Crop a square region centred on the bean's perceptual centre.

    Parameters
    ----------
    uniform_side : int
        The standard crop side length for this photo, computed once by
        ``compute_uniform_crop_side`` and shared across all beans.
        Ensures every bean gets the same generous crop window regardless
        of how well its individual mask captured its true extent.

    Centering strategy:
        The crop centre is a 50/50 blend of two complementary estimates:
          - Moments centroid (cx, cy): the centre-of-mass of the mask.
          - Bbox midpoint: equidistant from the contour's visual extents.
        Averaging the two gives a perceptually balanced centre.

    Per-bean safety floor:
        If this specific bean is larger than the uniform side (rare, but
        possible with size-inconsistent beans), the crop is expanded to
        fit this bean plus margin.  This guarantees zero clipping.
    """
    # Per-bean safety floor: never crop smaller than THIS bean requires.
    bean_side = int(max(bean.w, bean.h) * (1.0 + cfg.margin_pct))
    side = max(uniform_side, bean_side)
    half = side // 2

    # Bbox geometric midpoint (equidistant from visual extents).
    bbox_cx = bean.x + bean.w // 2
    bbox_cy = bean.y + bean.h // 2

    # Perceptual centre: blend centroid (mass) with midpoint (extent).
    cx = (bean.cx + bbox_cx) // 2
    cy = (bean.cy + bbox_cy) // 2

    # Safety clamp: ensure the full bounding box sits inside the crop.
    cx = max(cx, bean.x + bean.w - half)  # crop covers bean right edge
    cx = min(cx, bean.x + half)           # crop covers bean left edge
    cy = max(cy, bean.y + bean.h - half)  # crop covers bean bottom edge
    cy = min(cy, bean.y + half)           # crop covers bean top edge

    pad = side  # generous; covers the worst case (bean at photo edge)
    padded = cv2.copyMakeBorder(
        image_bgr, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=PADDING_COLOUR_BGR,
    )

    # Centre coordinates inside the padded image.
    pcx, pcy = cx + pad, cy + pad
    x0, y0 = pcx - half, pcy - half
    x1, y1 = x0 + side, y0 + side

    crop = padded[y0:y1, x0:x1]
    crop = cv2.resize(crop, (cfg.target_size, cfg.target_size),
                      interpolation=cv2.INTER_AREA)
    return crop


def draw_debug_overlay(image_bgr: np.ndarray,
                       boxes: list[BeanBox],
                       cfg: CropConfig,
                       uniform_side: int) -> np.ndarray:
    """Return a copy of `image_bgr` with detected boxes + index numbers drawn.

    Uses the same uniform_side as ``crop_square_with_padding`` so the cyan
    crop rectangles exactly match what is actually saved to disk.

    Draws:
      - Green rectangle  : convex-hull bounding box
      - Cyan rectangle   : actual crop region (uniform side, perceptual centre)
      - Magenta crosshair: perceptual centre (blend of centroid + bbox mid)
    """
    overlay = image_bgr.copy()
    for idx, bean in enumerate(boxes):
        # Green: raw bounding box from convex hull
        cv2.rectangle(overlay,
                      (bean.x, bean.y),
                      (bean.x + bean.w, bean.y + bean.h),
                      (0, 255, 0), 3)

        # Compute the same perceptual centre used by crop_square_with_padding.
        pcx = (bean.cx + bean.x + bean.w // 2) // 2
        pcy = (bean.cy + bean.y + bean.h // 2) // 2

        # Per-bean safety floor (mirrors crop_square_with_padding).
        bean_side = int(max(bean.w, bean.h) * (1.0 + cfg.margin_pct))
        side = max(uniform_side, bean_side)
        half = side // 2

        # Cyan: the actual crop region (uniform side, perceptual centre)
        cv2.rectangle(overlay,
                      (pcx - half, pcy - half),
                      (pcx + half, pcy + half),
                      (255, 255, 0), 2)

        # Magenta crosshair: perceptual centre
        cross = 8
        cv2.line(overlay,
                 (pcx - cross, pcy),
                 (pcx + cross, pcy),
                 (255, 0, 255), 2)
        cv2.line(overlay,
                 (pcx, pcy - cross),
                 (pcx, pcy + cross),
                 (255, 0, 255), 2)

        cv2.putText(
            overlay, str(idx), (bean.x, max(0, bean.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
        )
    return overlay


# -----------------------------------------------------------------------------
# Per-photo processing
# -----------------------------------------------------------------------------
def iter_input_photos(input_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield (photo_path, output_class) pairs for every supported photo."""
    for raw_class, out_class in CLASS_MAP.items():
        class_dir = input_dir / raw_class
        if not class_dir.is_dir():
            print(f"[warn] missing input class folder: {class_dir}", file=sys.stderr)
            continue
        for photo in sorted(class_dir.iterdir()):
            if photo.suffix.lower() in VALID_EXT:
                yield photo, out_class


def process_photo(photo_path: Path, out_class: str, cfg: CropConfig,
                  start_idx: int) -> dict:
    """Crop one source photo, write outputs, return a stats dict for QC.

    Parameters
    ----------
    start_idx : int
        The 1-based global sequence number for the first bean in this
        photo.  Beans are named ``<class>_<seq:04d>.jpg`` where seq
        counts continuously across all photos of the same class.
    """
    image_bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {
            "photo": photo_path.name,
            "class": out_class,
            "detected": 0,
            "saved": 0,
            "skipped_existing": 0,
            "status": "READ_FAILED",
        }

    boxes = detect_bean_contours(image_bgr, cfg)
    out_dir = cfg.output_dir / out_class
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute a single, uniform crop side for ALL beans in this photo.
    # This is the key to enterprise-grade robustness: even if one bean's
    # mask is slightly under-segmented, it still receives a generous crop
    # window derived from the whole-photo statistics.
    uniform_side = compute_uniform_crop_side(boxes, cfg)

    saved = 0
    skipped_existing = 0
    for idx, bean in enumerate(boxes):
        seq = start_idx + idx
        out_name = f"{out_class}_{seq:04d}.jpg"
        out_path = out_dir / out_name
        if out_path.exists():
            skipped_existing += 1
            continue
        crop = crop_square_with_padding(image_bgr, bean, cfg, uniform_side)
        cv2.imwrite(str(out_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
        saved += 1

    if cfg.debug:
        # Store debug overlays OUTSIDE the output dir so that PyTorch's
        # ImageFolder does not pick up _debug as an extra class.
        debug_dir = cfg.output_dir.parent / (cfg.output_dir.name + "_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = draw_debug_overlay(image_bgr, boxes, cfg, uniform_side)
        cv2.imwrite(
            str(debug_dir / f"{out_class}__{photo_path.stem}_overlay.jpg"),
            overlay,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )

    if len(boxes) == cfg.expected_per_photo:
        status = "OK"
    elif len(boxes) > cfg.expected_per_photo:
        status = "OVER_DETECTION"
    else:
        status = "UNDER_DETECTION"

    return {
        "photo": photo_path.name,
        "class": out_class,
        "detected": len(boxes),
        "saved": saved,
        "skipped_existing": skipped_existing,
        "status": status,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> CropConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Crop 5x5 grid coffee-bean photos into individual 224x224 images "
            "for PyTorch ImageFolder consumption."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("Dataset"),
                        help="root folder containing raw 'Biji Kopi Defect' "
                             "and 'Biji Kopi Non-Defect' subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("Dataset_Cropped"),
                        help="destination root for cropped class subfolders")
    parser.add_argument("--min-area", type=int, default=1_000,
                        help="minimum contour area in px^2 (Master Prompt §2.4)")
    parser.add_argument("--max-area-ratio", type=float, default=3.0,
                        help="reject blobs larger than this × median area "
                             "(likely merged beans)")
    parser.add_argument("--margin-pct", type=float, default=0.18,
                        help="square-box padding as a fraction of the uniform "
                             "crop side (default 18%% for enterprise-grade safety)")
    parser.add_argument("--dilate-iter", type=int, default=2,
                        help="dilation iterations to expand mask before contour "
                             "extraction (recovers bean edges lost to opening)")
    parser.add_argument("--target-size", type=int, default=224,
                        help="output image resolution (square)")
    parser.add_argument("--expected-per-photo", type=int, default=25,
                        help="beans per photo; mismatches flagged in QC CSV")
    parser.add_argument("--debug", action="store_true",
                        help="save contour-overlay images under <output>/_debug/")
    args = parser.parse_args()

    return CropConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        margin_pct=args.margin_pct,
        target_size=args.target_size,
        expected_per_photo=args.expected_per_photo,
        debug=args.debug,
        dilate_iter=args.dilate_iter,
    )


def write_qc_report(records: list[dict], output_dir: Path) -> Path:
    """Persist all per-photo stats to `_qc_review.csv`.

    Photos with status != OK should be visually inspected by the user
    (open them in `<input-dir>` and the corresponding overlay in
    `<output-dir>/_debug/`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / "_qc_review.csv"
    fieldnames = ["photo", "class", "detected", "saved",
                  "skipped_existing", "status"]
    with qc_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return qc_path


def print_summary(records: list[dict], cfg: CropConfig) -> None:
    """Print a human-readable summary to stdout."""
    by_class: dict[str, dict[str, int]] = {}
    for r in records:
        c = r["class"]
        agg = by_class.setdefault(
            c, {"photos": 0, "detected": 0, "saved": 0, "skipped": 0, "issues": 0}
        )
        agg["photos"] += 1
        agg["detected"] += r["detected"]
        agg["saved"] += r["saved"]
        agg["skipped"] += r["skipped_existing"]
        if r["status"] != "OK":
            agg["issues"] += 1

    print("\n=== Auto-Crop Summary ===")
    for c, agg in by_class.items():
        target = agg["photos"] * cfg.expected_per_photo
        print(
            f"  [{c:>10}] photos={agg['photos']:>3}  "
            f"detected={agg['detected']:>5}/{target:<5}  "
            f"new_saved={agg['saved']:>5}  "
            f"already_existed={agg['skipped']:>5}  "
            f"flagged={agg['issues']:>3}"
        )
    print()


def main() -> int:
    cfg = parse_args()

    if not cfg.input_dir.is_dir():
        print(f"[error] input directory not found: {cfg.input_dir}", file=sys.stderr)
        return 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    photos = list(iter_input_photos(cfg.input_dir))
    if not photos:
        print(f"[error] no photos found under {cfg.input_dir}/", file=sys.stderr)
        return 1

    print(f"Processing {len(photos)} photo(s) from {cfg.input_dir}/ -> "
          f"{cfg.output_dir}/  (debug={cfg.debug})")

    # Per-class global sequence counters (1-based).
    class_counters: dict[str, int] = {}

    records: list[dict] = []
    for photo_path, out_class in tqdm(photos, desc="cropping", unit="photo"):
        start_idx = class_counters.get(out_class, 1)
        record = process_photo(photo_path, out_class, cfg, start_idx)
        records.append(record)
        # Advance the counter by the number of beans detected in this photo.
        class_counters[out_class] = start_idx + record["detected"]

    qc_path = write_qc_report(records, cfg.output_dir)
    print_summary(records, cfg)
    print(f"QC report -> {qc_path}")
    flagged = [r for r in records if r["status"] != "OK"]
    if flagged:
        print(f"[note] {len(flagged)} photo(s) need manual review "
              f"(detected != {cfg.expected_per_photo}). See the CSV above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
