# `scripts/` — Pipeline Klasifikasi Biner Biji Kopi

Implementasi PyTorch untuk Tugas Akhir
*"Klasifikasi Biji Kopi Robusta Lampung Cacat dan Non-cacat Berdasarkan
Cacat Fisik Menggunakan Pendekatan Deep Learning"*.

Pipeline tiga fase yang dijalankan berurutan:

| Fase | Berkas | Tujuan |
|------|--------|--------|
| 1 | `01_auto_crop.py` | Memotong foto grid 5×5 menjadi citra biji individual 224×224 |
| 2 | `02_prepare_split.ipynb` → `02a` / `02b` / `02c` → `02d` | Membagi dataset deterministik lalu melatih MobileNetV3-Large, ResNet50, dan Swin-Tiny |
| 3 | `03_evaluate_models.ipynb` | Memuat ulang *checkpoint* terbaik dan memproduksi metrik final |

Modul `utils.py` di-*import* secara otomatis oleh seluruh *notebook* fase 2 dan 3.
Urutan model yang dipakai konsisten: **mobilenet_v3_large → resnet50 → swin_t**.

---

## Pemilihan Lingkungan Eksekusi

| | Lokal (Windows) | Google Colab |
|---|---|---|
| **Kapan dipakai** | Punya GPU CUDA + Python ≤ 3.12 | Tidak punya GPU lokal, atau Python 3.13+ (CUDA wheels belum tersedia) |
| **Fase 1 (crop)** | `python scripts/01_auto_crop.py` | Jalankan lokal saja (CPU cukup, tidak butuh GPU) |
| **Fase 2 (training)** | Buka *notebook* via Jupyter Lab/Notebook | Buka *notebook* di Colab → Runtime → GPU |
| **Fase 3 (evaluasi)** | Buka `03_evaluate_models.ipynb` lewat Jupyter | Buka *notebook* di Colab → Runtime → GPU |
| **Lokasi keluaran** | `checkpoints/` dan `reports/` di *root* repo | `MyDrive/TA/checkpoints/` dan `MyDrive/TA/reports/` |

> **Catatan:** *notebook* mendeteksi sendiri lingkungan Colab vs lokal melalui
> `IN_COLAB = 'google.colab' in sys.modules` lalu menyesuaikan *path* secara
> otomatis. Tidak perlu modifikasi manual.

---

## Persiapan

### Lokal (Windows)

```powershell
# 1. Pasang PyTorch dengan CUDA (butuh Python <= 3.12)
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Pasang dependensi tambahan
py -m pip install -r requirements.txt
```

### Google Colab

1. Kompresi dataset hasil *crop*:
   ```powershell
   Compress-Archive Dataset_Cropped Dataset_Cropped.zip
   ```
2. Unggah `Dataset_Cropped.zip` dan `scripts/utils.py` ke Google Drive
   (mis. `MyDrive/TA/`).
3. Buka `.ipynb` di Colab → Runtime → Change runtime type → **GPU (T4 atau lebih baik)**.
4. Klik *Run all*. Dependensi dipasang otomatis dari sel pertama *notebook*.

---

## Fase 1 — Cropping (`01_auto_crop.py`)

> Selalu dijalankan lokal — tidak butuh GPU.

```powershell
python scripts/01_auto_crop.py --debug
```

### Pipeline 12 langkah

1. *Grayscale* + Gaussian blur untuk meredam tekstur kertas dan tanda pensil.
2. Otsu *inverse threshold* sehingga biji berwarna putih dan latar hitam.
3. *Morphological opening* untuk menghilangkan titik kecil non-biji.
4. *Contour dilation* memperluas *mask* agar tepi biji tidak terpotong.
5. `cv2.findContours` mendaftar kandidat biji.
6. Filter area MIN..MAX membuang debu sekaligus *blob* hasil dua biji menempel
   (ditolak bila luas > 3× median).
7. *Outlier pruning* memangkas hingga tepat `expected_per_photo` biji.
8. *Uniform crop sizing* berbasis P75 sisi — semua biji satu foto memakai
   ukuran *crop* yang sama, jadi tidak ada biji terpotong.
9. *Centroid + bbox blended* (50/50) untuk penempatan *crop* yang
   secara visual seimbang.
10. *Crop* dengan *padding* putih sesuai latar studio.
11. *Resize* ke `target_size` agar masukan CNN deterministik.
12. Simpan sebagai `<output>/<class>/<class>_<seq>.jpg`
    (mis. `defect_0001.jpg`).

### Keluaran

```
Dataset_Cropped/
├── defect/
│   ├── defect_0001.jpg        ← 224×224, penomoran 1-based berurutan
│   ├── defect_0002.jpg
│   ├── ...
│   └── defect_1000.jpg
├── non_defect/
│   ├── non_defect_0001.jpg
│   ├── ...
│   └── non_defect_1000.jpg
├── _debug/                    ← contour overlay (hanya jika --debug)
└── _qc_review.csv             ← log audit deteksi per foto
```

**Total**: 2.000 citra (1.000 *defect* + 1.000 *non-defect*).

### Argumen CLI

| Argumen | Default | Tujuan |
|------|---------|---------|
| `--input-dir` | `Dataset` | *Root* foto mentah grid |
| `--output-dir` | `Dataset_Cropped` | Lokasi citra biji individual |
| `--min-area` | `1000` | Luas kontur minimum (px²) — buang titik pensil |
| `--max-area-ratio` | `3.0` | Tolak *blob* lebih besar dari N× median (biji menempel) |
| `--margin-pct` | `0.18` | *Padding crop* sebagai fraksi sisi (18%) |
| `--dilate-iter` | `2` | Iterasi *dilation* untuk memperluas *mask* |
| `--target-size` | `224` | Resolusi akhir citra (kotak) |
| `--expected-per-photo` | `25` | Jumlah biji per foto; kelebihan deteksi otomatis dipangkas |
| `--debug` | mati | Simpan *overlay contour* di `<output>/_debug/` |

Skrip bersifat **idempoten** — *file* yang sudah ada akan dilewati,
sehingga aman dijalankan ulang seiring datangnya foto baru.

---

## Fase 2 — Training

| Notebook | Tujuan |
|----------|--------|
| `02_prepare_split.ipynb` | Hasilkan pembagian deterministik 80/10/10 → `reports/split_indices.json` |
| `02a_train_mobilenet_v3.ipynb` | Latih **mobilenet_v3_large** |
| `02b_train_resnet50.ipynb` | Latih **resnet50** |
| `02c_train_swin_tiny.ipynb` | Latih **swin_t** |
| `02d_compare_training.ipynb` | Agregasi hasil → kurva komparasi + tabel ringkasan |

### Urutan Eksekusi

```
02_prepare_split.ipynb           ← wajib pertama (hasil split_indices.json)
        │
        ├── 02a_train_mobilenet_v3.ipynb ─┐
        ├── 02b_train_resnet50.ipynb     ─┼─ urutan bebas / paralel
        └── 02c_train_swin_tiny.ipynb    ─┘
                                           │
                              02d_compare_training.ipynb   ← setelah ketiganya
                                           │
                              03_evaluate_models.ipynb     ← Fase 3
```

1. **Split dulu** — `02_prepare_split.ipynb` wajib jalan sebelum *notebook* training.
2. **Training (urutan bebas)** — tiga *notebook* training saling independen,
   bahkan boleh paralel di sesi Colab terpisah.
3. **Komparasi** — `02d_compare_training.ipynb` membutuhkan tiga
   `history_<name>.csv` dan `timing_<name>.json`.
4. **Evaluasi** — `03_evaluate_models.ipynb` membutuhkan `split_indices.json`
   dan tiga *checkpoint*.

### Hyperparameter Default

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `batch_size` | `64` (MobileNet/ResNet) / `48` (Swin-T) | Lebih kecil pada Swin-T karena *attention* memori-intensif |
| `epochs` | `25` | Jumlah maksimum *epoch* |
| `learning_rate` | `1e-4` | AdamW dengan *cosine annealing* |
| `weight_decay` | `1e-4` | Regularisasi AdamW |
| `patience` | `5` | *Early stopping* setelah 5 *epoch* tanpa peningkatan val_acc |
| `seed` | `42` | Deterministik di seluruh *random source* |

### Artefak Per Model

Tiap *notebook* training (`02a`, `02b`, `02c`) — di Colab maupun lokal —
menghasilkan empat *file* (di mana `<name>` adalah `mobilenet_v3_large`,
`resnet50`, atau `swin_t`):

| Artefak | Path | Deskripsi |
|---------|------|-----------|
| Checkpoint | `checkpoints/best_model_<name>.pth` | *State dict* + metadata *epoch* terbaik |
| History | `reports/history_<name>.csv` | Loss, accuracy, lr per *epoch* |
| Curves | `reports/curves_<name>.png` | Plot dua-panel loss/accuracy |
| Timing | `reports/timing_<name>.json` | Durasi training + info *epoch* terbaik |

`02d_compare_training.ipynb` menghasilkan dua artefak lintas-model:

| Artefak | Path | Deskripsi |
|---------|------|-----------|
| Comparison | `reports/curves_comparison.png` | Val loss + val accuracy, satu garis per model |
| Summary | `reports/training_summary.csv` | model, best_epoch, best_val_acc, train_minutes |

---

## Fase 3 — Evaluasi

```bash
# Lokal
jupyter notebook scripts/03_evaluate_models.ipynb

# Colab
# Buka 03_evaluate_models.ipynb -> Runtime -> GPU -> Run all
```

Kedua *path* memakai *test split* yang sama persis (lewat `split_indices.json`).

### Keluaran

```
reports/
├── confusion_matrix_<model>.png      ← heatmap Seaborn
├── classification_report_<model>.csv ← P/R/F1 per kelas
├── inference_benchmark.csv           ← ms/citra (mean, p50, p95)
├── final_comparison.csv              ← satu tabel ringkas untuk Bab IV
└── best_model.txt                    ← pemenang
```

Pemenang ditentukan berdasarkan **F1-score tertinggi pada kelas `defect`**
(kelas positif/anomali pada konteks *quality control*).

---

## Reproducibility

Semua *random source* dikunci dengan seed `42` lewat `utils.seed_everything`.
*Hyperparameter* tinggal di satu *dict* (di tiap *notebook*) sehingga
re-run dengan dataset + seed yang sama menghasilkan metrik dalam
toleransi numerik yang sangat kecil.

---

## Pengembangan Mendatang

Lihat `Skripsi_Split_MD/15_FW_PENGEMBANGAN_MENDATANG.md` (sumber utama,
Bahasa Indonesia) atau `15_FW_PENGEMBANGAN_MENDATANG.en.md` (mirror Inggris)
untuk 14 *add-on* modular: *Stratified split*, Grad-CAM, *benchmark* CPU vs
GPU, *k-fold cross-validation*, *partial unfreeze*, *multi-seed runs*, SNI
*multi-class*, *Cohen's Kappa multi-annotator*, *edge deployment* INT8,
*domain adaptation*, *self-supervised pre-training*, *hyperspectral imaging*,
dan lain-lain. Sengaja **tidak** dimasukkan ke pipeline utama agar lingkup
skripsi tetap fokus.
