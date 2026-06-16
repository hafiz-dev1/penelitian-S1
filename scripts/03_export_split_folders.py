"""Materialkan split train/val/test menjadi folder fisik.

Folder fisik diturunkan PERSIS dari ``split_indices.json`` kanonik (Round 4)
sehingga komposisi tiap subset identik dengan yang dipakai pada eksperimen
(test = 108 defect + 92 non-defect, dst.). Tidak ada pengacakan ulang.

Struktur keluaran (default ``Dataset_Split/`` di akar repo):

    Dataset_Split/
      train/ defect/ (793)   non_defect/ (807)
      val/   defect/ (99)    non_defect/ (101)
      test/  defect/ (108)   non_defect/ (92)

Tujuan: reprodusibilitas anti-salah. Penguji cukup memuat tiap folder dengan
``torchvision.datasets.ImageFolder`` tanpa bergantung pada urutan berkas
maupun internal ``random_split`` yang dapat berbeda antar lingkungan.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "Hasil Penelitian" / "Round 4 (2000 and Freeze)" / "reports" / "split_indices.json"
DATA = ROOT / "Dataset_Cropped"
OUT = ROOT / "Dataset_Split"


def main() -> None:
    split = json.loads(SPLIT.read_text())
    samples = split["samples"]               # path absolut saat training (Colab)
    class_to_idx = split["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    subsets = {
        "train": split["train_indices"],
        "val": split["val_indices"],
        "test": split["test_indices"],
    }

    if OUT.exists():
        shutil.rmtree(OUT)

    manifest_rows = []
    counts: dict[str, dict[str, int]] = {}
    missing = 0
    for subset, indices in subsets.items():
        counts[subset] = {"defect": 0, "non_defect": 0}
        for gi in indices:
            # tentukan kelas dari path tersimpan (lebih robust dari label int)
            saved_path = samples[gi]
            cls = "defect" if "/defect/" in saved_path.replace("\\", "/") else "non_defect"
            fname = Path(saved_path).name
            src = DATA / cls / fname
            if not src.exists():
                missing += 1
                print(f"  [WARN] sumber tidak ditemukan: {src}")
                continue
            dst_dir = OUT / subset / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / fname)
            counts[subset][cls] += 1
            manifest_rows.append({"subset": subset, "class": cls, "file": fname})

    # Manifest CSV untuk audit/jejak
    man_path = OUT / "split_manifest.csv"
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "class", "file"])
        w.writeheader()
        w.writerows(manifest_rows)

    print("\n================ HASIL EXPORT ================")
    total = 0
    for subset in ("train", "val", "test"):
        d, nd = counts[subset]["defect"], counts[subset]["non_defect"]
        sub_total = d + nd
        total += sub_total
        print(f"  {subset:<5}: defect={d:<4} non_defect={nd:<4} total={sub_total}")
    print(f"  TOTAL: {total} citra (missing={missing})")
    print(f"\nFolder fisik : {OUT}")
    print(f"Manifest     : {man_path}")


if __name__ == "__main__":
    main()
