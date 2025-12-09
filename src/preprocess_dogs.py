import os
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm


def parse_bbox(annotation_path: Path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Stanford Dogs XML format
    obj = root.find("object")
    if obj is None:
        return None

    bbox = obj.find("bndbox")
    if bbox is None:
        return None

    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    return xmin, ymin, xmax, ymax


def main():
    data_root = Path("data")
    images_root = data_root / "dogs" / "Images"       # adjust if your folder is named differently
    ann_root = data_root / "dogs" / "Annotation"      # from annotations.tar
    out_root = data_root / "dogs_cropped"

    out_root.mkdir(parents=True, exist_ok=True)

    # Walk over all annotation files
    ann_files = list(ann_root.glob("**/*.xml"))
    print(f"Found {len(ann_files)} annotation files")

    kept = 0
    skipped = 0

    for ann_path in tqdm(ann_files):
        bbox = parse_bbox(ann_path)
        if bbox is None:
            skipped += 1
            continue

        xmin, ymin, xmax, ymax = bbox

        # The image has the same stem, different extension, and lives in Images/
        # e.g. Annotation/n02085620_10074/n02085620_10074.xml
        #      Images/n02085620_10074.jpg
        img_stem = ann_path.stem  # e.g. n02085620_10074
        # Search for a matching image in Images (jpg or jpeg)
        candidates = []
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_root / f"{img_stem}{ext}"
            if candidate.exists():
                candidates.append(candidate)

        if not candidates:
            skipped += 1
            continue

        img_path = candidates[0]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        w, h = img.size

        # Clamp bbox to image bounds (defensive)
        xmin_clamped = max(0, min(xmin, w - 1))
        xmax_clamped = max(0, min(xmax, w))
        ymin_clamped = max(0, min(ymin, h - 1))
        ymax_clamped = max(0, min(ymax, h))

        if xmax_clamped <= xmin_clamped or ymax_clamped <= ymin_clamped:
            skipped += 1
            continue

        # Optionally: ignore tiny boxes (dog too small)
        box_area = (xmax_clamped - xmin_clamped) * (ymax_clamped - ymin_clamped)
        img_area = w * h
        if box_area < 0.1 * img_area:
            # dog is tiny in the image; skip to keep dataset clean
            skipped += 1
            continue

        cropped = img.crop((xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped))

        # Make it square by padding with background (simple approach)
        cw, ch = cropped.size
        side = max(cw, ch)
        square = Image.new("RGB", (side, side), (0, 0, 0))
        offset_x = (side - cw) // 2
        offset_y = (side - ch) // 2
        square.paste(cropped, (offset_x, offset_y))

        # Resize down to something like 128x128; later the dataset will downscale to 64 if needed
        square = square.resize((128, 128), Image.BILINEAR)

        out_path = out_root / f"{img_stem}.jpg"
        square.save(out_path, quality=95)

        kept += 1

    print(f"Done. Kept {kept} cropped dog images, skipped {skipped} problematic ones.")
    print(f"Cropped images stored in: {out_root}")


if __name__ == "__main__":
    main()
