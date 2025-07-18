import os
import re

def rename_chili_images(directory_path):
    # Format nama yang didukung:
    # - left_chili_10_cropped_20250619_1700.jpg → left_chili_10.jpg
    # - right_chili_detection_order_full_20250619_1700.jpg → right_full.jpg

    pattern_cropped = re.compile(
        r'^(left|right)_chili_(\d+)_cropped_\d{8}_\d{4}\.(jpg|jpeg|png)$', re.IGNORECASE)
    pattern_full = re.compile(
        r'^(left|right)_chili_detection_order_full_\d{8}_\d{4}\.(jpg|jpeg|png)$', re.IGNORECASE)

    renamed_count = 0

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if not os.path.isfile(filepath):
            continue

        match_cropped = pattern_cropped.fullmatch(filename)
        match_full = pattern_full.fullmatch(filename)

        if match_cropped:
            arah = match_cropped.group(1)
            nomor = match_cropped.group(2)
            ekstensi = match_cropped.group(3)
            new_name = f"{arah}_chili_{nomor}.{ekstensi.lower()}"
        elif match_full:
            arah = match_full.group(1)
            ekstensi = match_full.group(2)  # ✅ hanya ada 2 grup: arah dan ekstensi
            new_name = f"{arah}_full.{ekstensi.lower()}"
        else:
            print(f"[SKIP] Format tidak cocok: {filename}")
            continue

        new_path = os.path.join(directory_path, new_name)

        if os.path.exists(new_path):
            print(f"[SKIP] Sudah ada: {new_name}")
            continue

        os.rename(filepath, new_path)
        print(f"[OK] Renamed: {filename} → {new_name}")
        renamed_count += 1

    print(f"\n✅ Total berhasil diubah: {renamed_count}")

# Contoh pemakaian:
rename_chili_images("Storage/2025-06-23_23-30-03")
