import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_bbox(annotation_path: Path):
    """
    Try to parse a Stanford Dogs annotation file and return (xmin, ymin, xmax, ymax).
    Returns None if the file isn't a valid annotation.
    """
    try:
        tree = ET.parse(annotation_path)
    except Exception:
        return None

    root = tree.getroot()
    if root.tag != "annotation":
        return None

    obj = root.find("object")
    if obj is None:
        return None

    bbox = obj.find("bndbox")
    if bbox is None:
        return None

    try:
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
    except Exception:
        return None

    return xmin, ymin, xmax, ymax


def main():
    data_root = Path("data")
    dogs_root = data_root / "dogs"

    images_root = dogs_root / "Images"      # dog photos from images.tar
    ann_root = dogs_root / "Annotation"     # annotation files from annotation.tar

    if not images_root.exists():
        print(f"Images folder not found at: {images_root}")
        return

    if not ann_root.exists():
        print(f"Annotation folder not found at: {ann_root}")
        return

    # Grab ALL files under Annotation/, regardless of extension
    all_ann_files = [p for p in ann_root.rglob("*") if p.is_file()]
    print(f"Found {len(all_ann_files)} files under {ann_root}")

    if not all_ann_files:
        print("No files found in Annotation/. Double-check extraction of annotation.tar.")
        return

    out_root = data_root / "dogs_cropped"
    out_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for ann_path in tqdm(all_ann_files):
        bbox = parse_bbox(ann_path)
        if bbox is None:
            skipped += 1
            continue

        xmin, ymin, xmax, ymax = bbox

        # Example:
        #   ann_path = data/dogs/Annotation/n02085620-Chihuahua/n02085620_7
        #   breed_dir = "n02085620-Chihuahua"
        breed_dir = ann_path.parent.name
        img_stem = ann_path.stem  # e.g. "n02085620_7"

        # Image is in Images/<breed_dir>/<img_stem>.<ext>
        img_path = None
        for ext in [".jpg", ".jpeg", ".JPEG", ".png"]:
            candidate = images_root / breed_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        w, h = img.size

        # Clamp bbox to image bounds
        xmin_c = max(0, min(xmin, w - 1))
        xmax_c = max(0, min(xmax, w))
        ymin_c = max(0, min(ymin, h - 1))
        ymax_c = max(0, min(ymax, h))

        if xmax_c <= xmin_c or ymax_c <= ymin_c:
            skipped += 1
            continue

        # Crop to the bounding box
        cropped = img.crop((xmin_c, ymin_c, xmax_c, ymax_c))

        # Make it square by padding (dog roughly centered)
        cw, ch = cropped.size
        side = max(cw, ch)
        square = Image.new("RGB", (side, side), (0, 0, 0))
        offset_x = (side - cw) // 2
        offset_y = (side - ch) // 2
        square.paste(cropped, (offset_x, offset_y))

        # Resize to 128x128; dataset will later scale to 64x64 if needed
        square = square.resize((128, 128), Image.BILINEAR)

        out_path = out_root / f"{img_stem}.jpg"
        square.save(out_path, quality=95)

        kept += 1

    print(f"Done. Kept {kept} cropped dog images, skipped {skipped} files.")
    print(f"Cropped images stored in: {out_root}")


if __name__ == "__main__":
    main()
