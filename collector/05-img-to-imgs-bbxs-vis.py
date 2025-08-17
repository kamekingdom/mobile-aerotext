# Author: kamekingdom (2025-08-18)
from __future__ import annotations
"""
単純合成器（厳密なYOLO→画素復元／クラス限定マッチ／進捗表示／自動リトライ）
- 各画像の YOLO ラベル (cx,cy,w,h) を“左上=floor・右下=ceil”で画素矩形に復元し、その矩形をそのまま切り出して貼付。
- ラベル探索は bbx/<class>/<stem>.txt のみを採用（rglobは使わない）し、stem衝突による取り違えを封じる。
- 配置は非重なり条件を満たすまでランダム配置を試行。失敗時は全スプライトを段階的に縮小して再試行。
- 進捗は tqdm と簡易ログで表示。
依存: opencv-python, numpy, tqdm
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ==========================
# パラメータ
# ==========================
DATA_IN_DIR: Path = Path("data")           # 画像: data/<class>/*.png 等
BBX_IN_DIR:  Path = Path("bbx")            # ラベル: bbx/<class>/<stem>.txt（固定）
OUT_DIR:     Path = Path("composed")

K_COMPOSE: int = 3                         # 1 枚に貼る個数
N_SAMPLES: int = 100
CANVAS_H: int = 256
CANVAS_W: int = 256

IMG_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

BORDER_MARGIN_PX: int = 6                  # キャンバス境界からの枠
GAP_BETWEEN_BOXES_PX: int = 6              # 相互距離（非交差の安全域）
LABEL_MIN_SIZE_PX: int = 3                 # 小さすぎるラベルを棄却

# 失敗時の全体スケール（順に試す）
GLOBAL_RESCALE_TRIALS: Sequence[float] = (1.00, 0.92, 0.85, 0.78, 0.72, 0.66, 0.60)

# 丸め誤差の保険（各辺を外側に拡げるピクセル数。0〜1推奨）
CROP_EDGE_EXPAND_PX: int = 1

# 再現性
SEED: int = 20250818
CACHE_FILE: str = "index_cache.json"
FORCE_REINDEX: bool = False

# 表示
TQDM_ASCII: bool = True

# ==========================
# 型
# ==========================
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class Pair:
    img: str  # 画像絶対パス
    lbl: str  # ラベル絶対パス

# ==========================
# ユーティリティ
# ==========================
def rng_seed(seed: int) -> None:
    random.seed(seed)

def list_class_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

def iter_images(dirpath: Path, exts: Sequence[str]) -> Iterable[Path]:
    lower = {e.lower() for e in exts}
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower() in lower:
            yield p

def find_label_for_image_strict(bbx_root: Path, class_name: str, stem: str) -> Optional[Path]:
    """
    ラベル配置のバリエーションを吸収:
      1) bbx/<class>/<stem>.txt
      2) bbx/labels/<class>/<stem>.txt  （あなたの作成スクリプトの既定）
    ※ rglob は使わない（stem衝突防止）
    """
    cand1 = bbx_root / class_name / f"{stem}.txt"
    if cand1.exists():
        return cand1
    cand2 = bbx_root / "labels" / class_name / f"{stem}.txt"
    if cand2.exists():
        return cand2
    return None

def load_image_bgr(path: Path) -> np.ndarray:
    """
    OpenCVで読み込み（EXIF Orientation を解釈しない点をラベル生成時と合わせる）。
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def read_yolo_labels(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    1行=  cid cx cy w h（正規化, 0..1）
    """
    if not txt_path.exists():
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:])
            out.append((cid, cx, cy, w, h))
    return out

# —— 重要：YOLO→画素復元（左上=floor, 右下=ceil, 外接）——
def yolo_to_pixels_strict(
    item: Tuple[int, float, float, float, float],
    img_w: int, img_h: int,
    edge_expand_px: int = 0
) -> Tuple[int, BBox]:
    cid, cx, cy, w, h = item
    fx1 = (cx - w/2.0) * img_w
    fx2 = (cx + w/2.0) * img_w
    fy1 = (cy - h/2.0) * img_h
    fy2 = (cy + h/2.0) * img_h
    # 左上を floor、右下を ceil（OpenCV の半開区間スライスと整合）
    x1 = math.floor(fx1 + 1e-9) - edge_expand_px
    y1 = math.floor(fy1 + 1e-9) - edge_expand_px
    x2 = math.ceil (fx2 - 1e-9) + edge_expand_px
    y2 = math.ceil (fy2 - 1e-9) + edge_expand_px
    # 境界クリップ（[0,W]×[0,H] の半開区間）
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    if x2 <= x1: x2 = min(img_w, x1 + 1)
    if y2 <= y1: y2 = min(img_h, y1 + 1)
    return cid, (x1, y1, x2, y2)

def crop_rect(bgr: np.ndarray, rect: BBox) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return bgr[y1:y2, x1:x2].copy()

def expand(b: BBox, e: int) -> BBox:
    x1, y1, x2, y2 = b
    return (x1 - e, y1 - e, x2 + e, y2 + e)

def is_overlap(a: BBox, b: BBox, gap: int) -> bool:
    ax1, ay1, ax2, ay2 = expand(a, gap)
    bx1, by1, bx2, by2 = expand(b, gap)
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def place_nonoverlap(W: int, H: int, sw: int, sh: int, existing: List[BBox], border: int, gap: int) -> Optional[BBox]:
    xmin, ymin = border, border
    xmax, ymax = W - border - sw, H - border - sh
    if xmax < xmin or ymax < ymin:
        return None
    for _ in range(600):
        x1 = random.randint(xmin, xmax)
        y1 = random.randint(ymin, ymax)
        cand = (x1, y1, x1 + sw, y1 + sh)
        if all(not is_overlap(cand, e, gap) for e in existing):
            return cand
    return None

def yolo_line(class_id: int, box: BBox, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box
    bw = x2 - x1; bh = y2 - y1
    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"

# ==========================
# インデックス（キャッシュ）
# ==========================
def build_or_load_index(
    data_dir: Path, bbx_dir: Path, out_dir: Path, force: bool
) -> Tuple[Dict[str, int], Dict[str, List[Pair]]]:
    cache_path = out_dir / CACHE_FILE
    if (not force) and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            class_map: Dict[str, int] = {k: int(v) for k, v in cache["class_map"].items()}
            pool_raw = cache["pool"]; pool: Dict[str, List[Pair]] = {}
            valid = True
            for cname, plist in pool_raw.items():
                pairs: List[Pair] = []
                for it in plist:
                    ip, lp = Path(it["img"]), Path(it["lbl"])
                    if ip.exists() and lp.exists():
                        # ここでもクラス直下の txt であることを保証
                        if Path(lp).parent.name == cname:
                            pairs.append(Pair(str(ip), str(lp)))
                if not pairs:
                    valid = False; break
                pool[cname] = pairs
            if valid and class_map:
                print("[INFO] Loaded index cache.", flush=True)
                return class_map, pool
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}", flush=True)

    class_dirs = list_class_dirs(data_dir)
    if not class_dirs:
        raise RuntimeError(f"画像側のクラスフォルダが見つかりません: {data_dir}")

    class_map: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    pool: Dict[str, List[Pair]] = {}

    with tqdm(class_dirs, desc="Indexing classes", unit="class", ascii=TQDM_ASCII) as t_classes:
        for d in t_classes:
            cname = d.name
            pairs: List[Pair] = []
            imgs = list(iter_images(d, IMG_EXTS))
            with tqdm(imgs, desc=f"  {cname}", unit="img", leave=False, ascii=TQDM_ASCII) as t_imgs:
                for img_path in t_imgs:
                    lbl = find_label_for_image_strict(bbx_dir, cname, img_path.stem)
                    if lbl is not None:
                        pairs.append(Pair(str(img_path.resolve()), str(lbl.resolve())))
            if not pairs:
                raise RuntimeError(f"クラス {cname}: (画像, ラベル) 対応が見つかりません（bbx/{cname}/<stem>.txt を確認）")
            pool[cname] = pairs

    to_save = {"class_map": class_map,
               "pool": {c: [{"img": p.img, "lbl": p.lbl} for p in plist] for c, plist in pool.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Index cache saved: {cache_path}", flush=True)
    return class_map, pool

# ==========================
# メイン
# ==========================
def main() -> None:
    rng_seed(SEED)

    data_dir = DATA_IN_DIR.resolve()
    bbx_dir  = BBX_IN_DIR.resolve()
    out_dir  = OUT_DIR.resolve()
    (out_dir/"data").mkdir(parents=True, exist_ok=True)
    (out_dir/"bbx").mkdir(parents=True, exist_ok=True)
    (out_dir/"viz").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] DATA_IN_DIR = {data_dir}", flush=True)
    print(f"[INFO] BBX_IN_DIR  = {bbx_dir}", flush=True)
    print(f"[INFO] OUT_DIR     = {out_dir}", flush=True)

    # Index
    class_map, pool = build_or_load_index(data_dir, bbx_dir, out_dir, FORCE_REINDEX)

    # classes.txt
    with open(out_dir/"classes.txt", "w", encoding="utf-8") as f:
        for cname in sorted(class_map.keys(), key=lambda k: class_map[k]):
            f.write(f"{cname}\n")

    pad_len = len(str(max(1, N_SAMPLES)))
    saved = 0

    with tqdm(range(1, N_SAMPLES+1), desc="Composing", unit="img", ascii=TQDM_ASCII) as pbar:
        for idx in pbar:
            chosen = random.sample(list(pool.keys()), K_COMPOSE)
            crops: List[Tuple[str, int, np.ndarray, Tuple[int,int]]] = []  # (cname, cid, crop_bgr, (w,h))

            # ---- 切り出し ----
            for cname in chosen:
                pair = random.choice(pool[cname])
                img_path, txt_path = Path(pair.img), Path(pair.lbl)
                img = load_image_bgr(img_path)
                H, W = img.shape[:2]                 # 注意: (H,W)
                labels = read_yolo_labels(txt_path)
                if not labels:
                    crops = []; break
                item = random.choice(labels)         # その画像から1 bbox
                cid, rect = yolo_to_pixels_strict(item, img_w=W, img_h=H, edge_expand_px=CROP_EDGE_EXPAND_PX)
                crop = crop_rect(img, rect)
                h, w = crop.shape[:2]
                if min(w, h) < LABEL_MIN_SIZE_PX:
                    crops = []; break
                crops.append((cname, cid, crop, (w, h)))
                pbar.set_postfix_str(f"cut:{cname}@{img_path.stem} rect={rect}")

            if len(crops) != K_COMPOSE:
                continue

            # ---- 配置（全体縮小のリトライ付き）----
            placed = False
            for gscale in GLOBAL_RESCALE_TRIALS:
                canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                placed_rects: List[BBox] = []
                final_labels: List[Tuple[BBox, int]] = []

                scaled: List[Tuple[str, int, np.ndarray, Tuple[int,int]]] = []
                for cname, cid, crop, (w, h) in crops:
                    nw, nh = w, h
                    if abs(gscale - 1.0) > 1e-6 or w > CANVAS_W or h > CANVAS_H:
                        # まず gscale、さらにキャンバス境界も考慮して縮小
                        sf_fit = min((CANVAS_W - 2*BORDER_MARGIN_PX) / max(1, w),
                                     (CANVAS_H - 2*BORDER_MARGIN_PX) / max(1, h), 1.0)
                        s = gscale * sf_fit
                        nw = max(1, int(round(w * s)))
                        nh = max(1, int(round(h * s)))
                        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    scaled.append((cname, cid, crop, (nw, nh)))

                ok = True
                for cname, cid, crop, (w, h) in scaled:
                    pos = place_nonoverlap(CANVAS_W, CANVAS_H, w, h, placed_rects,
                                           border=BORDER_MARGIN_PX, gap=GAP_BETWEEN_BOXES_PX)
                    if pos is None:
                        ok = False; break
                    x1, y1, x2, y2 = pos
                    canvas[y1:y2, x1:x2] = crop
                    placed_rects.append(pos)
                    if (x2-x1) < LABEL_MIN_SIZE_PX or (y2-y1) < LABEL_MIN_SIZE_PX:
                        ok = False; break
                    final_labels.append((pos, cid))

                if ok and len(final_labels) == K_COMPOSE:
                    # ---- 保存 ----
                    stem = "_".join(sorted([c for (c, _, _, _) in crops])) + f"_{str(idx).zfill(pad_len)}"
                    out_img = out_dir/"data"/f"{stem}.png"
                    out_txt = out_dir/"bbx"/f"{stem}.txt"
                    out_viz = out_dir/"viz"/f"{stem}.png"
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    out_txt.parent.mkdir(parents=True, exist_ok=True)
                    out_viz.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imencode(".png", canvas)[1].tofile(str(out_img))
                    with open(out_txt, "w", encoding="utf-8") as f:
                        for (box, cid) in final_labels:
                            f.write(yolo_line(cid, box, CANVAS_W, CANVAS_H) + "\n")

                    # 可視化（ラベル枠と安全枠）
                    viz = canvas.copy()
                    for (x1, y1, x2, y2), _cid in final_labels:
                        cv2.rectangle(viz, (x1, y1), (x2, y2), (0,255,255), 2)
                    cv2.rectangle(viz, (BORDER_MARGIN_PX, BORDER_MARGIN_PX),
                                  (CANVAS_W-BORDER_MARGIN_PX, CANVAS_H-BORDER_MARGIN_PX),
                                  (120,120,120), 1)
                    cv2.imencode(".png", viz)[1].tofile(str(out_viz))

                    saved += 1
                    pbar.set_postfix_str(f"saved={saved} scale={gscale:.2f}")
                    placed = True
                    break

            if not placed:
                tqdm.write(f"[WARN] サンプル {idx}: 生成失敗（全スケールで非交差配置できず）")

    print(f"[INFO] Done. saved={saved}", flush=True)
    print(f"[INFO] OUT data: {out_dir/'data'}", flush=True)
    print(f"[INFO] OUT bbx : {out_dir/'bbx'}", flush=True)
    print(f"[INFO] OUT viz : {out_dir/'viz'}", flush=True)

if __name__ == "__main__":
    main()
