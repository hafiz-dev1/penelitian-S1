"""Generate placeholder JPG untuk Lampiran C (dummy sebelum foto asli tersedia)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def make_placeholder(out_path: Path, title: str, subtitle: str,
                     size=(1200, 900), bg="#eef2f5", border="#33415c") -> None:
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)

    # Border tebal
    draw.rectangle([10, 10, size[0] - 11, size[1] - 11],
                   outline=border, width=6)
    # Cross hatch ringan agar terasa "placeholder"
    for x in range(0, size[0], 60):
        draw.line([(x, 0), (x + size[1], size[1])],
                  fill="#dfe6ec", width=1)

    # Pilih font: cari arial, fallback ke default
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    font_title = font_sub = None
    for name in candidates:
        try:
            font_title = ImageFont.truetype(name, 56)
            font_sub = ImageFont.truetype(name, 32)
            break
        except OSError:
            continue
    if font_title is None:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    def draw_centered(text: str, y: int, font) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        x = (size[0] - w) // 2
        draw.text((x, y), text, fill="#1a2a3a", font=font)

    draw_centered("[ DUMMY PLACEHOLDER ]", 240, font_sub)
    draw_centered(title, 360, font_title)
    draw_centered(subtitle, 460, font_sub)
    draw_centered("ganti dengan foto asli sebelum sidang",
                  size[1] - 120, font_sub)

    img.save(out_path, "JPEG", quality=85, optimize=True)
    print(f"  -> {out_path}")


def main() -> None:
    items = [
        ("lampiran_setup_studio.jpg",
         "Setup Studio Mini",
         "studio + smartphone + lampu LED + kertas matte"),
        ("lampiran_scrcpy_screenshot.jpg",
         "Mirroring Layar via scrcpy",
         "tampilan laptop saat sesi pemotretan grid 5x5"),
        ("lampiran_sesi_pemotretan.jpg",
         "Sesi Pemotretan Biji Kopi",
         "pola matriks 5x5, jarak antar biji +-2,5 cm"),
        ("toples_defect_1000.jpg",
         "Toples 1000 Biji Defect",
         "hasil sortasi mandiri kelas defect"),
        ("toples_non_defect_1000.jpg",
         "Toples 1000 Biji Non-Defect",
         "hasil sortasi mandiri kelas non-defect"),
    ]

    base = Path(r"d:\_GitHub\TA-Vibe-Coding\Latex-TA-IF-ITERA\Latex-TA-IF-ITERA-main")
    targets = [
        base / "figure",
        base / "Template TA 2025 - Versi A4" / "figure",
    ]

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        print(f"[{target}]")
        for fname, title, sub in items:
            make_placeholder(target / fname, title, sub)


if __name__ == "__main__":
    main()
