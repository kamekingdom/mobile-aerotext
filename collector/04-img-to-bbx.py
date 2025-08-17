from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ===== パラメータ =====
ROOT_DIR: Path = Path("data")   # データセットのルート
OUT_DIR: Path  = Path("bbx")    # 出力ルート
IMG_EXTS: Sequence[str] = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
MIN_COMPONENT_AREA: int = 25     # 連結成分の最小面積
DRAW_THICKNESS: int = 2          # 矩形線幅

# 「一回り」余白の設定：相対 + 絶対のハイブリッド
PAD_FRAC: float = 0.08           # 相対余白率（各方向）。例：0.08 => 幅・高とも +16% 拡張
MIN_PAD_PX: int = 6              # 絶対余白の下限（各方向の最小ピクセル）

BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def load_as_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = np.maximum(gray, alpha)
        return gray.astype(np.uint8)
    elif img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img.astype(np.uint8)


def binarize_white_on_black(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def connected_components_boxes(bw: np.ndarray, min_area: int) -> List[BBox]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes: List[BBox] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def merge_boxes(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("No foreground components detected.")
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def pad_box(box: BBox, img_w: int, img_h: int, pad_frac: float, min_pad_px: int) -> BBox:
    """
    BBox を左右・上下に拡張後、画像境界でクリップする。
    各方向の拡張量は max(相対余白, 絶対余白下限)。
    """
    x1, y1, x2, y2 = box
    bw: float = float(x2 - x1)
    bh: float = float(y2 - y1)

    # 各方向への拡張量（ピクセル）
    dx: float = max(bw * pad_frac, float(min_pad_px))
    dy: float = max(bh * pad_frac, float(min_pad_px))

    # 拡張
    px1: float = x1 - dx
    py1: float = y1 - dy
    px2: float = x2 + dx
    py2: float = y2 + dy

    # 画像境界でクリップ（整数化は floor/ceil 後に行う）
    px1_i: int = max(0, int(np.floor(px1)))
    py1_i: int = max(0, int(np.floor(py1)))
    px2_i: int = min(img_w, int(np.ceil(px2)))
    py2_i: int = min(img_h, int(np.ceil(py2)))

    # 幅・高さが 1px 未満にならないよう補正
    if px2_i <= px1_i:
        if px1_i > 0:
            px1_i -= 1
        else:
            px2_i = min(img_w, px1_i + 1)
    if py2_i <= py1_i:
        if py1_i > 0:
            py1_i -= 1
        else:
            py2_i = min(img_h, py1_i + 1)

    return (px1_i, py1_i, px2_i, py2_i)


def yolo_line(class_id: int, box: BBox, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"


def iter_images(dirpath: Path, exts: Sequence[str]):
    lower_exts = {e.lower() for e in exts}
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower() in lower_exts:
            yield p


def draw_box_on_gray(gray: np.ndarray, box: BBox, thickness: int = 2) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = box
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return bgr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_root = OUT_DIR / "labels"
    preview_root = OUT_DIR / "preview"
    labels_root.mkdir(exist_ok=True)
    preview_root.mkdir(exist_ok=True)

    class_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    class_map: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}

    with open(OUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        for cls in sorted(class_map.keys(), key=lambda k: class_map[k]):
            f.write(f"{cls}\n")

    all_items: List[Tuple[int, Path, Path]] = []
    for d in class_dirs:
        cid = class_map[d.name]
        for img_path in iter_images(d, IMG_EXTS):
            all_items.append((cid, d, img_path))

    pbar = tqdm(total=len(all_items), desc="Processing images", unit="img")
    for class_id, class_dir, img_path in all_items:
        gray = load_as_gray(img_path)
        bw = binarize_white_on_black(gray)
        boxes = connected_components_boxes(bw, min_area=MIN_COMPONENT_AREA)
        try:
            box = merge_boxes(boxes)
        except ValueError:
            pbar.update(1)
            continue

        h, w = bw.shape[:2]
        padded_box = pad_box(box, img_w=w, img_h=h, pad_frac=PAD_FRAC, min_pad_px=MIN_PAD_PX)
        line = yolo_line(class_id, padded_box, w, h)

        out_class_dir = labels_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        with open(out_class_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.write(line + "\n")

        prev = draw_box_on_gray(gray, padded_box, thickness=DRAW_THICKNESS)
        indiv_dir = preview_root / class_dir.name
        indiv_dir.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", prev)[1].tofile(indiv_dir / f"{img_path.stem}_preview.png")

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()

