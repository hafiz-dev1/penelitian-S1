"""Identifikasi dan ekstrak citra yang salah diprediksi oleh model final.

Model final = ResNet50 dengan strategi full fine-tuning (Round 4).
Skrip ini memuat ulang split test yang persis sama dari ``split_indices.json``,
menjalankan inferensi, lalu:

1. Mencetak daftar lengkap citra yang salah diprediksi (FN dan FP).
2. Menyalin tiap citra misclassified ke ``reports/misclassified/`` dengan nama
   yang menjelaskan (true vs pred + confidence).
3. Menulis ``reports/misclassified_resnet50__full.csv`` (audit lengkap).

Bersifat read-only terhadap dataset; hanya menambah berkas di folder reports.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import utils  # noqa: E402
from utils import IMG_SIZE, build_model, get_transforms  # noqa: E402

ROUND4 = ROOT / "Hasil Penelitian" / "Round 4 (2000 and Freeze)"
CKPT = ROUND4 / "checkpoints" / "best_model_resnet50__full.pth"
SPLIT = ROUND4 / "reports" / "split_indices.json"
DATA_ROOT = ROOT / "Dataset_Cropped"
OUT_DIR = ROUND4 / "reports" / "misclassified"


def main() -> None:
    device = torch.device("cpu")
    print(f"IMG_SIZE={IMG_SIZE} | device={device}")

    split = json.loads(SPLIT.read_text())
    class_to_idx = split["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("class_to_idx:", class_to_idx)

    full = datasets.ImageFolder(root=str(DATA_ROOT), transform=get_transforms(train=False))

    # Validasi urutan basename sama persis dengan saat training.
    saved = [Path(p).name for p in split["samples"]]
    current = [Path(p).name for p, _ in full.samples]
    if saved != current:
        raise SystemExit("[ERROR] Urutan basename ImageFolder tidak cocok dengan split_indices.json")
    print(f"Validasi basename lolos ({len(current)} citra).")

    test_idx = split["test_indices"]
    test_ds = Subset(full, test_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model("resnet50", pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Checkpoint dimuat: best_epoch={ckpt['best_epoch']} val_acc={ckpt['best_val_acc']}")

    rows = []
    pos = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for i in range(x.size(0)):
                global_idx = test_idx[pos]
                path = full.samples[global_idx][0]
                true_lbl = int(y[i])
                pred_lbl = int(preds[i])
                conf = float(probs[i, pred_lbl])
                rows.append({
                    "file": Path(path).name,
                    "true": idx_to_class[true_lbl],
                    "pred": idx_to_class[pred_lbl],
                    "correct": true_lbl == pred_lbl,
                    "conf_pred": round(conf, 4),
                    "p_defect": round(float(probs[i, class_to_idx["defect"]]), 4),
                    "src_path": path,
                })
                pos += 1

    mis = [r for r in rows if not r["correct"]]
    fn = [r for r in mis if r["true"] == "defect"]   # defect diprediksi non_defect
    fp = [r for r in mis if r["true"] == "non_defect"]  # non_defect diprediksi defect

    print("\n================ RINGKASAN ================")
    print(f"Total test  : {len(rows)}")
    print(f"Benar       : {sum(r['correct'] for r in rows)}")
    print(f"Salah       : {len(mis)}  (FN={len(fn)} | FP={len(fp)})")

    print("\n--- FALSE NEGATIVE (defect lolos jadi non_defect) ---")
    for r in fn:
        print(f"  {r['file']:<28} pred={r['pred']:<11} conf={r['conf_pred']:.4f} p_defect={r['p_defect']:.4f}")
    print("\n--- FALSE POSITIVE (non_defect tertuduh defect) ---")
    for r in fp:
        print(f"  {r['file']:<28} pred={r['pred']:<11} conf={r['conf_pred']:.4f} p_defect={r['p_defect']:.4f}")

    # Ekstrak gambar misclassified (citra ASLI sebelum normalisasi).
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in mis:
        tag = "FN" if r["true"] == "defect" else "FP"
        dst = OUT_DIR / f"{tag}__{r['file'].replace('.jpg','')}__true-{r['true']}_pred-{r['pred']}_conf{r['conf_pred']:.3f}.jpg"
        shutil.copy2(r["src_path"], dst)
    print(f"\n{len(mis)} citra misclassified disalin ke: {OUT_DIR}")

    # Audit CSV
    import csv
    csv_path = ROUND4 / "reports" / "misclassified_resnet50__full.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true", "pred", "correct", "conf_pred", "p_defect"])
        w.writeheader()
        for r in mis:
            w.writerow({k: r[k] for k in ["file", "true", "pred", "correct", "conf_pred", "p_defect"]})
    print(f"Audit CSV: {csv_path}")


if __name__ == "__main__":
    main()
