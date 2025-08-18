# Author: kamekingdom (2025-08-18)
from __future__ import annotations
"""
全自動：データ準備（train/val 分割 & data.yaml 生成）→ 学習 → テスト評価（mAP/Precision/Recall）

想定入出力:
- 入力(合成結果): SOURCE_ROOT/
    data/  *.png|jpg|...
    bbx/   *.txt (YOLO形式)
    classes.txt  (行番号 = class id)
- 出力(学習用データセット): DATASET_ROOT/  （自動生成）
    images/{train,val}/
    labels/{train,val}/
    data.yaml
- 学習成果物: runs_*/<RUN_NAME>/
- テスト評価: TEST_ROOT/ の構造を自動認識（images/labels または data/bbx）

依存:
    pip install ultralytics==8.* opencv-python numpy
"""

import os
import shutil
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==========================
# 固定パラメータ（ここだけ編集）
# ==========================
@dataclass
class Config:
    # --- データ準備 ---
    SOURCE_ROOT: Path = Path("composed")          # 合成器の出力（data/, bbx/, classes.txt）
    DATASET_ROOT: Path = Path("datasets/aerotext")       # 学習用に作るルート
    TRAIN_RATIO: float = 0.9                             # train:val の比
    IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    # --- 学習 ---
    MODEL_INIT: str = "yolov8n.pt"                       # 初期重み（無ければ yolov8n.pt に自動フォールバック）
    IMG_SIZE: int = 640
    EPOCHS: int = 100
    BATCH: int = 16
    DEVICE: Optional[str] = None                         # None: 自動判定 / "cpu" / "0" 等
    WORKERS: Optional[int] = None                        # None: 自動(CPU=0, GPU=4)
    PROJECT: str = "runs_aerotext"
    RUN_NAME: str = "yolov8n-baseline"
    AMP_FOR_CPU: bool = False                            # CPU では AMP を無効に（推奨）

    # --- テスト評価 ---
    TEST_ROOT: Path = Path("composed_simple")            # テスト用フォルダのルート
    EVAL_IMGSZ: int = 640
    EVAL_CONF: float = 0.001
    EVAL_IOU: float = 0.50
    EVAL_PROJECT: str = "runs_aerotext_eval"
    EVAL_NAME: str = "eval_yolov8"

CFG = Config()


# ==========================
# ユーティリティ
# ==========================
def choose_device(user_device: Optional[str]) -> str:
    """ユーザ指定を尊重しつつ、CUDA 非対応なら 'cpu' に自動フォールバック。"""
    if user_device is not None:
        if user_device.lower() == "cpu":
            return "cpu"
        return "cpu" if not torch.cuda.is_available() else user_device
    return "0" if torch.cuda.is_available() else "cpu"


def read_classes(path: Path) -> List[str]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise RuntimeError(f"classes.txt が空です: {path}")
    return names


def ensure_dirs(paths: Sequence[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def detect_image_label_dirs(root: Path, image_exts: Sequence[str]) -> Tuple[Path, Path]:
    """
    (images_dir, labels_dir) を root 配下から推定して返す。
    優先順:
      1) images/, labels/
      2) data/,   bbx/
    """
    cands = [
        (root / "images", root / "labels"),
        (root / "data",   root / "bbx"),
    ]
    for img_dir, lbl_dir in cands:
        if img_dir.exists() and lbl_dir.exists():
            imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in image_exts]
            lbls = list(lbl_dir.glob("*.txt"))
            if imgs and lbls:
                return img_dir, lbl_dir
    raise RuntimeError(
        f"テスト/ソースの構造を認識できませんでした: {root}\n"
        f"  想定A: {root}/images/*.png と {root}/labels/*.txt\n"
        f"  想定B: {root}/data/*.png   と {root}/bbx/*.txt"
    )


def write_data_yaml(out_yaml: Path, dataset_root: Path, names: List[str]) -> None:
    """Ultralytics data.yaml（train/val は dataset_root 下の固定パス）を作る。"""
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"names: {names}",
        ""
    ]
    out_yaml.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote data.yaml -> {out_yaml}")


# ==========================
# データ準備（train/val 分割）
# ==========================
def prepare_dataset(cfg: Config) -> Path:
    """
    SOURCE_ROOT (data/, bbx/, classes.txt) → DATASET_ROOT へ 9:1 分割コピー。
    既に data.yaml がある場合は再生成しない。
    戻り値: data.yaml のパス
    """
    data_yaml = cfg.DATASET_ROOT / "data.yaml"
    if data_yaml.exists():
        print(f"[INFO] data.yaml が既に存在します。スキップ: {data_yaml}")
        return data_yaml

    # 入力確認
    src_imgs_dir, src_lbls_dir = detect_image_label_dirs(cfg.SOURCE_ROOT, cfg.IMAGE_EXTS)
    classes_txt = cfg.SOURCE_ROOT / "classes.txt"
    if not classes_txt.exists():
        raise FileNotFoundError(f"classes.txt が見つかりません: {classes_txt}")
    names = read_classes(classes_txt)

    # 出力構造
    img_tr = cfg.DATASET_ROOT / "images/train"
    img_vl = cfg.DATASET_ROOT / "images/val"
    lbl_tr = cfg.DATASET_ROOT / "labels/train"
    lbl_vl = cfg.DATASET_ROOT / "labels/val"
    ensure_dirs([img_tr, img_vl, lbl_tr, lbl_vl])

    # 分割
    imgs = sorted([p for p in src_imgs_dir.iterdir() if p.suffix.lower() in cfg.IMAGE_EXTS])
    random.seed(20250818)
    random.shuffle(imgs)
    n_tr = int(len(imgs) * cfg.TRAIN_RATIO)
    splits = [("train", imgs[:n_tr]), ("val", imgs[n_tr:])]

    copied = 0
    for split_name, items in splits:
        for ip in items:
            lp = src_lbls_dir / f"{ip.stem}.txt"
            if not lp.exists():
                # ラベル欠損はスキップ
                continue
            if split_name == "train":
                shutil.copy2(ip, img_tr / ip.name)
                shutil.copy2(lp, lbl_tr / f"{ip.stem}.txt")
            else:
                shutil.copy2(ip, img_vl / ip.name)
                shutil.copy2(lp, lbl_vl / f"{ip.stem}.txt")
            copied += 1
    print(f"[INFO] Copied {copied} (image,label) pairs into {cfg.DATASET_ROOT}")

    # data.yaml
    write_data_yaml(data_yaml, cfg.DATASET_ROOT, names)
    # classes.txt も控えで置いておく（任意）
    (cfg.DATASET_ROOT / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return data_yaml


# ==========================
# 学習
# ==========================
def train_yolo(cfg: Config, data_yaml: Path) -> Path:
    """Ultralytics YOLOv8 で学習し、best.pt のパスを返す。"""
    device = choose_device(cfg.DEVICE)
    workers = (0 if device == "cpu" else 4) if cfg.WORKERS is None else cfg.WORKERS

    # 初期重みの存在を確認（見つからなければ yolov8n.pt）
    model_init = cfg.MODEL_INIT
    if not Path(model_init).exists() and not any(model_init.startswith(p) for p in ["yolov8", "gelan-"]):
        print(f"[WARN] 初期重み {model_init} が見つかりません。'yolov8n.pt' を使用します。")
        model_init = "yolov8n.pt"

    print(f"[INFO] device={device}  cuda_available={torch.cuda.is_available()}  workers={workers}")
    print(f"[INFO] model={model_init}  data={data_yaml}  imgsz={cfg.IMG_SIZE}  epochs={cfg.EPOCHS}  batch={cfg.BATCH}")
    model = YOLO(model_init)

    model.train(
        data=str(data_yaml),
        imgsz=cfg.IMG_SIZE,
        epochs=cfg.EPOCHS,
        batch=cfg.BATCH,
        device=device,
        workers=workers,
        project=cfg.PROJECT,
        name=cfg.RUN_NAME,
        cos_lr=True,
        amp=(device != "cpu") and cfg.AMP_FOR_CPU is False,  # CPU ではAMP無効
        deterministic=True,
        verbose=True,
    )

    # best.pt の推定パス
    save_dir = Path(cfg.PROJECT) / cfg.RUN_NAME
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt が見つかりませんでした: {best}")
    print(f"[INFO] Training done. best weights: {best}")
    return best


# ==========================
# テスト評価
# ==========================
def write_temp_eval_yaml(tmp_yaml: Path, dataset_root: Path, images_rel: str, names: List[str]) -> None:
    """
    'val' スロットにテスト画像ディレクトリ（相対）を割り当てる data.yaml を動的に作成。
    """
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train   # unused",
        f"val: {images_rel}",
        f"names: {names}",
        ""
    ]
    tmp_yaml.write_text("\n".join(lines), encoding="utf-8")


def run_eval_on_folder(cfg: Config, model_pt: Path) -> Dict[str, Any]:
    """
    TEST_ROOT を評価する。TEST_ROOT は以下のいずれかの構造：
      - images/, labels/
      - data/,   bbx/
    """
    if not cfg.TEST_ROOT.exists():
        raise FileNotFoundError(f"TEST_ROOT が見つかりません: {cfg.TEST_ROOT}")

    names_src = (cfg.DATASET_ROOT / "classes.txt"
                 if (cfg.DATASET_ROOT / "classes.txt").exists()
                 else cfg.SOURCE_ROOT / "classes.txt")
    if not names_src.exists():
        raise FileNotFoundError(f"classes.txt が見つかりません: {names_src}")
    names = read_classes(names_src)

    images_dir, labels_dir = detect_image_label_dirs(cfg.TEST_ROOT, cfg.IMAGE_EXTS)

    # data.yaml（テストを val に割当）
    dataset_root = images_dir.parent
    images_rel = images_dir.name  # "images" or "data"
    tmp_yaml = cfg.TEST_ROOT / "_temp_eval_data.yaml"
    write_temp_eval_yaml(tmp_yaml, dataset_root, images_rel, names)

    device = choose_device(cfg.DEVICE)
    workers = (0 if device == "cpu" else 4) if cfg.WORKERS is None else cfg.WORKERS

    print(f"[INFO] Start evaluation on: {images_dir} (labels at {labels_dir})")
    model = YOLO(str(model_pt))
    results = model.val(
        data=str(tmp_yaml),
        imgsz=cfg.EVAL_IMGSZ,
        conf=cfg.EVAL_CONF,
        iou=cfg.EVAL_IOU,
        batch=cfg.BATCH,
        device=device,
        workers=workers,
        project=cfg.EVAL_PROJECT,
        name=cfg.EVAL_NAME,
        save_json=True,
        save_txt=True,
        plots=True,
    )

    # 主要指標（Ultralytics 8.x の互換的取得）
    rd = getattr(results, "results_dict", {})
    summary = {
        "metrics/mAP50-95": float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", float("nan")))),
        "metrics/mAP50": float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", float("nan")))),
        "metrics/precision": float(rd.get("metrics/precision(B)", float("nan"))),
        "metrics/recall": float(rd.get("metrics/recall(B)", float("nan"))),
        "imgsz": cfg.EVAL_IMGSZ,
        "conf": cfg.EVAL_CONF,
        "iou":  cfg.EVAL_IOU,
    }
    print("[INFO] Evaluation summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    print(f"[INFO] Artifacts saved under: {results.save_dir}")
    return {"save_dir": str(results.save_dir), "summary": summary}


# ==========================
# メイン実行
# ==========================
def main() -> None:
    # 1) データ準備（必要なら）
    data_yaml = prepare_dataset(CFG)

    # 2) 学習
    best_pt = train_yolo(CFG, data_yaml)

    # 3) テスト評価
    run_eval_on_folder(CFG, best_pt)


if __name__ == "__main__":
    main()
