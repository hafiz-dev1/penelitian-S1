"""Buat figur untuk Sub-bab 4.1.4 (karakteristik visual) dan analisis
kasus salah klasifikasi (4.3.4), plus statistik visual sederhana.

Output figur disimpan ke folder figure/ KEDUA template (A4 + UNESCO):
  - karakteristik_biji_kopi.png  : montase contoh defect vs non-defect
  - sampel_salah_klasifikasi.png : 2 FN + 2 FP paling representatif

Statistik (fraksi piksel gelap & kecerahan rerata) dicetak ke stdout untuk
mendasari narasi, dan disimpan sebagai reports/visual_stats.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset_Cropped"
ROUND4 = ROOT / "Hasil Penelitian" / "Round 4 (2000 and Freeze)"
REPORTS = ROUND4 / "reports"

FIG_DIRS = [
    ROOT / "Latex-TA-IF-ITERA" / "Latex-TA-IF-ITERA-main" / "Template TA 2025 - Versi A4" / "figure",
    ROOT / "Latex-TA-IF-ITERA" / "Latex-TA-IF-ITERA-main" / "figure",
]

DARK_THRESH = 60  # piksel grayscale < 60 dianggap "gelap" (bercak/lubang/hitam)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def dark_fraction(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    return float((gray < DARK_THRESH).mean())


def brightness(arr: np.ndarray) -> float:
    return float(arr.mean())


def class_stats(folder: Path) -> tuple[float, float]:
    dfs, brs = [], []
    for fp in sorted(folder.glob("*.jpg")):
        a = load_rgb(fp)
        dfs.append(dark_fraction(a))
        brs.append(brightness(a))
    return float(np.mean(dfs)), float(np.mean(brs))


def pick_diverse_defect(n: int = 5) -> list[Path]:
    """Pilih contoh defect yang beragam berdasarkan fraksi piksel gelap."""
    files = sorted((DATA / "defect").glob("*.jpg"))
    df = np.array([dark_fraction(load_rgb(f)) for f in files])
    order = np.argsort(df)
    # ambil di persentil menyebar untuk menangkap ragam cacat
    picks_q = np.linspace(0.10, 0.95, n)
    idxs = [order[int(q * (len(order) - 1))] for q in picks_q]
    return [files[i] for i in idxs]


def pick_clean_non_defect(n: int = 5) -> list[Path]:
    """Pilih contoh non-defect paling bersih (fraksi gelap terendah)."""
    files = sorted((DATA / "non_defect").glob("*.jpg"))
    df = np.array([dark_fraction(load_rgb(f)) for f in files])
    order = np.argsort(df)  # paling bersih dulu
    # ambil menyebar di paruh terbersih agar tetap variatif tapi mulus
    idxs = [order[i] for i in np.linspace(0, len(order) // 2, n).astype(int)]
    return [files[i] for i in idxs]


def montage(defect_paths, nondefect_paths, out_paths):
    n = max(len(defect_paths), len(nondefect_paths))
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for col in range(n):
        for row, paths, label in [(0, defect_paths, "Defect"), (1, nondefect_paths, "Non-defect")]:
            ax = axes[row, col]
            ax.imshow(load_rgb(paths[col]))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold")
    fig.tight_layout()
    for op in out_paths:
        fig.savefig(op, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[FIG] montase karakteristik ->", out_paths[0].name)


def misclassified_figure(items, out_paths):
    """items: list of (path, title)."""
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (path, title) in zip(axes, items):
        ax.imshow(load_rgb(path))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    for op in out_paths:
        fig.savefig(op, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[FIG] sampel salah klasifikasi ->", out_paths[0].name)


def main() -> None:
    print("Menghitung statistik kelas (mungkin perlu beberapa detik)...")
    d_df, d_br = class_stats(DATA / "defect")
    nd_df, nd_br = class_stats(DATA / "non_defect")
    print(f"  DEFECT     : dark_frac={d_df:.4f} | brightness={d_br:.1f}")
    print(f"  NON-DEFECT : dark_frac={nd_df:.4f} | brightness={nd_br:.1f}")

    # ---- Montase karakteristik ----
    dpicks = pick_diverse_defect(5)
    ndpicks = pick_clean_non_defect(5)
    print("Defect montase:", [p.name for p in dpicks])
    print("Non-defect montase:", [p.name for p in ndpicks])
    montage(dpicks, ndpicks, [d / "karakteristik_biji_kopi.png" for d in FIG_DIRS])

    # ---- Sampel salah klasifikasi (representatif): 2 FN + 2 FP ----
    # FN paling meyakinkan + FP paling meyakinkan (model paling 'pede' tapi salah)
    rep = [
        ("defect_0513.jpg", "FN: asli Defect\nprediksi Non-defect (0,978)"),
        ("defect_0980.jpg", "FN: asli Defect\nprediksi Non-defect (0,954)"),
        ("non_defect_0144.jpg", "FP: asli Non-defect\nprediksi Defect (0,942)"),
        ("non_defect_0577.jpg", "FP: asli Non-defect\nprediksi Defect (0,937)"),
    ]
    items = []
    stat_rows = []
    for fname, title in rep:
        cls = "defect" if fname.startswith("defect") else "non_defect"
        p = DATA / cls / fname
        a = load_rgb(p)
        items.append((p, title))
        stat_rows.append({
            "file": fname, "class": cls,
            "dark_frac": round(dark_fraction(a), 4),
            "brightness": round(brightness(a), 1),
        })
    misclassified_figure(items, [d / "sampel_salah_klasifikasi.png" for d in FIG_DIRS])

    print("\nStatistik citra misclassified representatif:")
    for r in stat_rows:
        print(f"  {r['file']:<22} {r['class']:<11} dark={r['dark_frac']:.4f} bright={r['brightness']:.1f}")

    with open(REPORTS / "visual_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "dark_frac", "brightness"])
        w.writerow(["class_defect_avg", round(d_df, 4), round(d_br, 1)])
        w.writerow(["class_non_defect_avg", round(nd_df, 4), round(nd_br, 1)])
        for r in stat_rows:
            w.writerow([r["file"], r["dark_frac"], r["brightness"]])
    print("\nStatistik tersimpan: reports/visual_stats.csv")


if __name__ == "__main__":
    main()
