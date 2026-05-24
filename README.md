# Klasifikasi Biji Kopi Robusta Lampung -- Deep Learning Pipeline

Pipeline PyTorch untuk klasifikasi biner cacat fisik biji kopi Robusta Lampung
(*defect* vs *non-defect*) menggunakan tiga arsitektur deep learning:
**ResNet50**, **MobileNetV3-Large**, dan **Swin-Tiny** (Vision Transformer).

Repositori ini adalah implementasi kode pendamping Tugas Akhir berjudul
*"Klasifikasi Biji Kopi Robusta Lampung Cacat dan Non-cacat Berdasarkan Cacat Fisik Menggunakan Pendekatan Deep Learning"* -- Hafiz Amrullah, NIM 119140177,
Program Studi Teknik Informatika, Institut Teknologi Sumatera, 2026.

## Hasil Utama (Round-3, dataset 2.000 citra)

| Model              | Test Accuracy | Recall (defect) | F1 (defect) | Inference (ms) |
|--------------------|--------------:|----------------:|------------:|---------------:|
| **ResNet50**       |   **98.00 %** |     **99.07 %** |     0.980   |             ~9 |
| Swin-Tiny          |     97.00 %   |       97.20 %   |     0.970   |            ~13 |
| MobileNetV3-Large  |     94.50 %   |       95.30 %   |     0.945   |             ~5 |

Pemenang: **ResNet50** (akurasi tertinggi + recall *defect* tertinggi).

## Quickstart

```bash
# 1. Clone
git clone https://github.com/hafiz-dev1/penelitian-S1.git
cd penelitian-S1

# 2. Pasang dependensi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Siapkan dataset (`Dataset_Cropped/` siap-pakai dari Google Drive)
#    Atau jalankan crop dari foto mentah:
python scripts/01_auto_crop.py

# 4. Training (Jupyter / Colab)
jupyter notebook scripts/02_prepare_split.ipynb        # split deterministik
jupyter notebook scripts/02a_train_mobilenet_v3.ipynb  # MobileNetV3-Large
jupyter notebook scripts/02b_train_resnet50.ipynb      # ResNet50
jupyter notebook scripts/02c_train_swin_tiny.ipynb     # Swin-Tiny
jupyter notebook scripts/02d_compare_training.ipynb    # agregasi

# 5. Evaluasi
jupyter notebook scripts/03_evaluate_models.ipynb
```

> Untuk eksekusi di Colab, langsung buka `.ipynb` lewat menu Colab dan pilih
> Runtime -> GPU (T4 atau lebih baik). *Notebook* mendeteksi sendiri
> lingkungan dan menyesuaikan *path*.

## Tutorial Penggunaan

Tutorial detail tiap fase (parameter, urutan eksekusi, daftar artefak
keluaran, perbedaan eksekusi lokal vs Colab) ada di
**[`scripts/README.md`](scripts/README.md)**. Ringkasan tiga fase:

### Fase 1 -- Cropping (`scripts/01_auto_crop.py`)

Memotong foto grid 5x5 menjadi citra biji individual 224x224 menggunakan
OpenCV (Otsu threshold + contour detection). Dijalankan lokal, tidak butuh
GPU. Skrip bersifat *idempoten*; foto baru bisa ditambah kapan saja.

### Fase 2 -- Training (5 notebook)

Tiga model dilatih independen dengan AdamW + Cosine Annealing,
*early stopping* `patience=5`, AMP aktif. Hyperparameter default:
`epochs=25`, `batch_size=64` (48 untuk Swin-T), `lr=1e-4`, `seed=42`.
Tiap *notebook* menyimpan *checkpoint* terbaik + history CSV + kurva
training PNG. *Notebook* `02d` mengagregasi hasil ketiganya.

### Fase 3 -- Evaluasi (`scripts/03_evaluate_models.ipynb`)

Memuat tiga *checkpoint*, menjalankan inferensi pada *test set* yang sama,
dan menghasilkan *confusion matrix*, *classification report* per kelas,
*inference benchmark* (ms/citra), serta tabel komparasi final untuk Bab IV.

## Struktur Repository

```
penelitian-S1/
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- .gitignore
`-- scripts/
    +-- README.md                       (tutorial detail Phase 1-3)
    +-- requirements.txt
    +-- 01_auto_crop.py                 (pre-processing OpenCV)
    +-- utils.py                        (modul helper bersama)
    +-- 02_prepare_split.ipynb
    +-- 02a_train_mobilenet_v3.ipynb
    +-- 02b_train_resnet50.ipynb
    +-- 02c_train_swin_tiny.ipynb
    +-- 02d_compare_training.ipynb
    `-- 03_evaluate_models.ipynb
```

## Yang TIDAK ada di repo ini

| Aset | Alasan | Lokasi alternatif |
|------|--------|-------------------|
| Dataset citra (raw + cropped, ~180 MB) | Ukuran besar, ber-hak cipta data primer | Google Drive (link via permintaan) |
| Bobot model `.pth` (~636 MB total 3 ronde) | Lewat limit 100 MB GitHub | [GitHub Releases](https://github.com/hafiz-dev1/penelitian-S1/releases) atau Google Drive |
| Source LaTeX skripsi | Internal pembimbing | Repositori internal |
| Paper jurnal referensi | Hak cipta IEEE/Elsevier | Lihat `references.bib` di skripsi |

## Reproducibility

Semua *random source* di-*seed* dengan `42` via `utils.seed_everything`.
Pembagian *train/val/test* (80:10:10) deterministik dan disimpan di
`reports/split_indices.json`. Re-run dengan dataset + *seed* yang sama
mereproduksi metrik dalam toleransi numerik yang sangat kecil.

## Hardware Pengujian

- **Lokal:** Intel i7-9750H + NVIDIA GTX 1650 Max-Q (4 GB VRAM)
- **Colab:** NVIDIA Tesla T4 (16 GB VRAM)

## Lisensi

[MIT License](LICENSE) -- bebas digunakan dengan atribusi.

## Acknowledgments

- **Imam Ekowicaksono, S.Si., M.Si.** selaku Dosen Pembimbing Tugas Akhir.
- **Hafiz Budi Firmansyah, S.Kom., M.Sc., Ph.D.** & **	I Wayan Wiprayoga Wisesa, S.Kom., M.Kom.** selaku Dosen Penguji.
- Program Studi Teknik Informatika, **Institut Teknologi Sumatera**.
