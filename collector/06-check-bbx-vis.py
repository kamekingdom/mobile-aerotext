# Author: kamekingdom (2025-08-18)
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import cv2, numpy as np, random

def draw_yolo_box(img: np.ndarray, line: str) -> None:
    h, w = img.shape[:2]
    parts = line.strip().split()
    if len(parts) != 5: return
    _, cx, cy, bw, bh = map(float, parts)
    x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
    x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
    cv2.rectangle(img, (max(0,x1),max(0,y1)), (min(w-1,x2),min(h-1,y2)), (0,255,255), 2)

def main() -> None:
    root = Path("composed")
    img_dir, lbl_dir = root/"data", root/"bbx"
    imgs: List[Path] = sorted(img_dir.glob("*.png"))
    random.shuffle(imgs)
    for p in imgs[:10]:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        lbl = lbl_dir / (p.stem + ".txt")
        if img is None or not lbl.exists(): continue
        with open(lbl, "r", encoding="utf-8") as f:
            for ln in f: draw_yolo_box(img, ln)
        cv2.imshow("check", img); cv2.waitKey(300)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
