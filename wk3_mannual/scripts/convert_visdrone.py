from pathlib import Path
from PIL import Image


def read_dataset(dataset_root: Path) -> list[str]:
    img_dir = dataset_root / "images"
    names = [p.stem for p in sorted(img_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Found {len(names)} images")
    return names


def get_image_size(dataset_root: Path, name: str) -> tuple[int, int]:
    img_dir = dataset_root / "images"
    for suffix in [".jpg", ".jpeg", ".png"]:
        img_path = img_dir / (name + suffix)
        if img_path.exists():
            img_w, img_h = Image.open(img_path).size
            return img_w, img_h
    raise FileNotFoundError(f"No image found for {name}")


def coordinate_convert(annotation_path: Path, img_w: int, img_h: int) -> list[str]:
    yolo_lines = []
    for line in annotation_path.read_text().splitlines():
        if not line.strip():
            continue
        x, y, w, h, score, cat, trunc, occl = map(int, line.split(","))
        if cat == 0:
            continue
        if w == 0 or h == 0:
            continue
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        class_id = cat - 1
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")
    return yolo_lines


def main() -> None:
    dataset_root = Path("/mnt/d/intern_dataset/VisDrone2019-DET-val")
    ann_dir = dataset_root / "annotations"
    dst_label_dir = dataset_root / "labels"
    dst_label_dir.mkdir(exist_ok=True)

    names = read_dataset(dataset_root)

    converted_count = 0
    for name in names:
        ann_file = ann_dir / (name + ".txt")
        if not ann_file.exists():
            print(f"Warning: no annotation for {name}, skipping")
            continue

        img_w, img_h = get_image_size(dataset_root, name)
        yolo_lines = coordinate_convert(ann_file, img_w, img_h)

        (dst_label_dir / (name + ".txt")).write_text("\n".join(yolo_lines))
        converted_count += 1

    print(f"Done, converted {converted_count} files -> {dst_label_dir}")


if __name__ == "__main__":
    main()
