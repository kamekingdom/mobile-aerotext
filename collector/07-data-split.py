# Author: kamekingdom (2025-08-18)
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import shutil, random

@dataclass
class SplitCfg:
    src_root: Path          # composed_simple
    out_root: Path          # datasets/aerotext
    train_ratio: float = 0.9
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def read_classes(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def main() -> None:
    cfg = SplitCfg(src_root=Path("composed"),
                   out_root=Path("datasets/aerotext"),
                   train_ratio=0.9)
    img_src, lbl_src = cfg.src_root/"data", cfg.src_root/"bbx"
    names = read_classes(cfg.src_root/"classes.txt")

    # 出力構造: datasets/aerotext/{images,labels}/{train,val}
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        (cfg.out_root/sub).mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_src.iterdir() if p.suffix.lower() in cfg.exts])
    random.seed(20250818); random.shuffle(imgs)
    n_train = int(len(imgs) * cfg.train_ratio)
    splits = [("train", imgs[:n_train]), ("val", imgs[n_train:])]

    for split_name, items in splits:
        for ip in items:
            lp = lbl_src / (ip.stem + ".txt")
            if not lp.exists():  # ラベル欠損はスキップ
                continue
            shutil.copy2(ip, cfg.out_root/f"images/{split_name}"/ip.name)
            shutil.copy2(lp, cfg.out_root/f"labels/{split_name}"/(ip.stem + ".txt"))

    # data.yaml 生成（相対パスでも絶対パスでも可）
    yaml = [
        f"path: {cfg.out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"names: {names}",
    ]
    (cfg.out_root/"data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print("[INFO] Wrote", (cfg.out_root/"data.yaml"))

if __name__ == "__main__":
    main()
