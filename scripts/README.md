# `scripts/` -- Pipeline Perancangan Model Klasifikasi Biji Kopi

Implementasi PyTorch untuk Tugas Akhir
*"Perancangan Model CNN Berbasis Transfer Learning untuk Klasifikasi Biner
Cacat Fisik Biji Kopi Robusta Lampung"*.

Penelitian ini **merancang model** klasifikasi biner (*defect* vs *non-defect*)
berbasis *transfer learning* dengan menelusuri **ruang desain = 2 backbone ×
2 strategi pelatihan = 4 konfigurasi**:

- **Backbone:** `mobilenet_v3_large`, `resnet50`.
- **Strategi:** `full` (full fine-tuning) dan `freeze` (feature extraction /
  *backbone* dibekukan).

Konfigurasi yang dilatih: **`mobilenet_v3_large__full`,
`mobilenet_v3_large__freeze`, `resnet50__full`, `resnet50__freeze`**
(`utils.CONFIG_NAMES`). Checkpoint dinamai `best_model_<config>.pth`.

Pipeline berurutan:

| Fase | Berkas | Tujuan |
|------|--------|--------|
| 1 | `01_auto_crop.py` | Memotong foto grid 5×5 menjadi citra biji individual 224×224 |
| 2a | `02_prepare_split.ipynb` | Membagi dataset 80:10:10 deterministik → `reports/split_indices.json` |
| 2b | `03_export_split_folders.py` | Memateralkan indeks split menjadi folder fisik `Dataset_Split/` |
| 2c | `04_train_freezing.ipynb` | Melatih 4 konfigurasi (2 backbone × {full, freeze}) |
| 3 | `05_evaluate_models.ipynb` | Memuat ulang *checkpoint* terbaik tiap konfigurasi dan memproduksi metrik final |

Analisis pendukung Bab IV (opsional, dijalankan setelah evaluasi):

| Berkas | Tujuan |
|--------|--------|
| `06_inspect_misclassified.py` | Mengekstrak citra yang salah diprediksi model rancangan akhir (ResNet50 *full*) |
| `07_make_characteristic_figures.py` | Membangun montase karakteristik visual biji *defect* vs *non-defect* |

Modul `utils.py` di-*import* otomatis oleh seluruh *notebook* fase 2 dan 3.

---

## Pemilihan Lingkungan Eksekusi

| | Lokal (Windows) | Google Colab |
|---|---|---|
| **Kapan dipakai** | Punya GPU CUDA + Python ≤ 3.12 | Tidak punya GPU lokal, atau Python 3.13+ |
| **Fase 1 (crop)** | `python scripts/01_auto_crop.py` | Jalankan lokal saja (CPU cukup) |
| **Fase 2 (training)** | Buka *notebook* via Jupyter Lab/Notebook | Buka *notebook* di Colab → Runtime → GPU |
| **Fase 3 (evaluasi)** | Buka `05_evaluate_models.ipynb` lewat Jupyter | Buka *notebook* di Colab → Runtime → GPU |
| **Lokasi keluaran** | `checkpoints/` dan `reports/` di *root* repo | `MyDrive/.../checkpoints/` dan `reports/` |

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
2. Unggah `Dataset_Cropped.zip` dan `scripts/utils.py` ke Google Drive.
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
6. Filter area MIN..MAX membuang debu sekaligus *blob* dua biji menempel
   (ditolak bila luas > 3× median).
7. *Outlier pruning* memangkas hingga tepat `expected_per_photo` biji.
8. *Uniform crop sizing* berbasis P75 sisi — semua biji satu foto memakai
   ukuran *crop* yang sama.
9. *Centroid + bbox blended* (50/50) untuk penempatan *crop* seimbang.
10. *Crop* dengan *padding* putih sesuai latar studio.
11. *Resize* ke `target_size` agar masukan CNN deterministik.
12. Simpan sebagai `<output>/<class>/<class>_<seq>.jpg`
    (mis. `defect_0001.jpg`).

### Keluaran

```
Dataset_Cropped/
├── defect/        defect_0001.jpg ... defect_1000.jpg     (224×224)
├── non_defect/    non_defect_0001.jpg ... non_defect_1000.jpg
├── _debug/        contour overlay (hanya jika --debug)
└── _qc_review.csv log audit deteksi per foto
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
| `--expected-per-photo` | `25` | Jumlah biji per foto; kelebihan deteksi dipangkas |
| `--debug` | mati | Simpan *overlay contour* di `<output>/_debug/` |

Skrip bersifat **idempoten** — *file* yang sudah ada dilewati, sehingga aman
dijalankan ulang seiring datangnya foto baru.

---

## Fase 2 — Split dan Pelatihan Ruang Desain

### Urutan Eksekusi

```
02_prepare_split.ipynb        ← wajib pertama (hasil reports/split_indices.json)
        │
03_export_split_folders.py    ← materialkan indeks → Dataset_Split/{train,val,test}/
        │
04_train_freezing.ipynb      ← latih 4 konfigurasi (2 backbone × {full, freeze})
        │
05_evaluate_models.ipynb      ← Fase 3
```

1. **Split dulu** — `02_prepare_split.ipynb` membuat pembagian 80:10:10
   deterministik (`seed=42`) dan menyimpannya ke `reports/split_indices.json`.
2. **Materialkan folder fisik (opsional tapi disarankan)** —
   `03_export_split_folders.py` membaca indeks tersebut lalu menyalin berkas
   ke `Dataset_Split/{train,val,test}/{defect,non_defect}/`. Komposisi
   persis: train 793/807, val 99/101, test 108/92. Ini menjamin penguji
   memperoleh subset identik tanpa bergantung pada urutan berkas maupun
   internal `random_split`.
3. **Latih ruang desain** — `04_train_freezing.ipynb` melatih keempat
   konfigurasi sekaligus dengan *hyperparameter* seragam.

### Hyperparameter Default

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `batch_size` | `64` | Sama untuk kedua *backbone* |
| `epochs` | `25` | Jumlah maksimum *epoch* |
| `learning_rate` | `1e-4` | AdamW dengan *cosine annealing* |
| `weight_decay` | `1e-4` | Regularisasi AdamW |
| `patience` | `5` | *Early stopping* setelah 5 *epoch* tanpa peningkatan val_acc |
| `seed` | `42` | Deterministik di seluruh *random source* |

Pada strategi `freeze`, hanya *classifier head* yang dilatih; seluruh
parameter *backbone* dibekukan (`requires_grad=False`). Pada strategi `full`,
seluruh parameter diperbarui.

### Artefak Per Konfigurasi

`04_train_freezing.ipynb` menghasilkan, untuk tiap `<config>` (mis.
`resnet50__full`):

| Artefak | Path | Deskripsi |
|---------|------|-----------|
| Checkpoint | `checkpoints/best_model_<config>.pth` | *State dict* + metadata *epoch* terbaik |
| History | `reports/history_<config>.csv` | Loss, accuracy, lr per *epoch* |
| Curves | `reports/curves_<config>.png` | Plot dua-panel loss/accuracy |
| Timing | `reports/timing_<config>.json` | Durasi training + info *epoch* terbaik |

Plus dua artefak lintas-konfigurasi:

| Artefak | Path | Deskripsi |
|---------|------|-----------|
| Comparison | `reports/curves_comparison.png` | Val loss + val accuracy, satu garis per konfigurasi |
| Summary | `reports/training_summary_freezing.csv` | config, backbone, strategy, best_epoch, best_val_acc, train_secs |

---

## Fase 3 — Evaluasi

```bash
# Lokal
jupyter notebook scripts/05_evaluate_models.ipynb

# Colab: buka 05_evaluate_models.ipynb -> Runtime -> GPU -> Run all
```

Memakai *test split* yang sama persis (lewat `split_indices.json`).

### Keluaran

```
reports/
├── confusion_matrix_<config>.png      ← heatmap Seaborn
├── classification_report_<config>.csv ← P/R/F1 per kelas
├── inference_benchmark.csv            ← ms/citra (mean, p50, p95)
├── final_comparison.csv               ← satu tabel ringkas untuk Bab IV
└── best_model.txt                     ← konfigurasi terbaik
```

Pemenang ditentukan berdasarkan **F1-score tertinggi pada kelas `defect`**
(kelas positif/anomali pada konteks *quality control*). Pada eksperimen ini
model rancangan akhir adalah **`resnet50__full`** (akurasi uji 97%,
*recall defect* 98,15%).

---

## Analisis Pendukung (opsional)

```bash
python scripts/06_inspect_misclassified.py        # citra salah-prediksi model final
python scripts/07_make_characteristic_figures.py  # montase karakteristik + statistik visual
```

- `06_inspect_misclassified.py` memuat `best_model_resnet50__full.pth`,
  menjalankan inferensi pada *test set*, lalu mengeluarkan daftar dan salinan
  citra yang salah diprediksi (FN dan FP) ke `reports/misclassified/`.
- `07_make_characteristic_figures.py` membangun montase contoh biji *defect*
  vs *non-defect* dan menghitung statistik visual sederhana (fraksi piksel
  gelap, kecerahan) untuk pembahasan kualitatif Bab IV.

---

## Reproducibility

Semua *random source* dikunci dengan seed `42` lewat `utils.seed_everything`.
Pembagian *train/val/test* (80:10:10) deterministik, disimpan di
`reports/split_indices.json`, dan dapat dimaterialkan menjadi folder fisik
(`Dataset_Split/`) lewat `03_export_split_folders.py`. Re-run dengan dataset +
seed yang sama menghasilkan metrik dalam toleransi numerik yang sangat kecil.

---

## Pengembangan Mendatang

Beberapa arah lanjutan yang sengaja **tidak** dimasukkan ke pipeline utama agar
lingkup skripsi tetap fokus: *stratified split*, Grad-CAM, *benchmark* CPU vs
GPU, *k-fold cross-validation*, *partial unfreeze*, *multi-seed runs*, SNI
*multi-class*, *multi-annotator agreement*, dan *edge deployment* INT8.
