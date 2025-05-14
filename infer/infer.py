#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hiragana-YOLO evaluator
  - CER, WER
  - latency (ms), per-phase speed, CPU usage (%)
  - confusion matrix + label table (PNG)
2025-05-14
"""

import os, re, time, shutil, statistics
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import japanize_matplotlib      # 日本語フォント
import pandas as pd

try:
    import psutil
except ImportError as e:
    raise ImportError("psutil が未インストールです。`pip install psutil` で導入してください") from e

# ---------- グローバル記録用 ---------- #
preprocess_times:   List[float] = []   # ms
inference_times:    List[float] = []   # ms
postprocess_times:  List[float] = []   # ms
latency_times_ms:   List[float] = []   # ms (end-to-end)
cpu_usages_proc:    List[float] = []   # %
# ------------------------------------- #

# ========= 文字列距離・誤り率 ========= #
def normalize_label(label: str) -> str:
    parts = label.split('_')
    parts.sort()
    return '_'.join(parts)

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1, start=1):
        current_row = [i]
        for j, c2 in enumerate(s2, start=1):
            ins = previous_row[j] + 1
            dele = current_row[j-1] + 1
            sub = previous_row[j-1] + (c1 != c2)
            current_row.append(min(ins, dele, sub))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(true_labels: List[str], pred_labels: List[str]) -> float:
    total_chars = sum(len(t) for t in true_labels)
    total_errors = sum(levenshtein_distance(t, p)
                       for t, p in zip(true_labels, pred_labels))
    return total_errors / total_chars if total_chars else 0.0

def calculate_wer(true_labels: List[str], pred_labels: List[str]) -> float:
    total_words = len(true_labels)
    total_errors = sum(t != p for t, p in zip(true_labels, pred_labels))
    return total_errors / total_words if total_words else 0.0
# ===================================== #

# ========= 推論・可視化系 ============== #
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def perform_inference(model: YOLO, image_path: str) -> Tuple[np.ndarray, list]:
    """推論を実行し速度統計を記録して戻す"""
    # 対象プロセス
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)              # baseline

    start_t = time.perf_counter()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"画像読込に失敗: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)                     # ─── YOLO inference ───
    end_t = time.perf_counter()

    latency_ms = (end_t - start_t) * 1_000
    latency_times_ms.append(latency_ms)

    cpu_usage = proc.cpu_percent(interval=None)  # % (since baseline)
    cpu_usages_proc.append(cpu_usage)

    if results:
        spd = results[0].speed                   # dict[str,float] in ms
        preprocess_times.append(spd.get('preprocess', np.nan))
        inference_times.append(spd.get('inference',  np.nan))
        postprocess_times.append(spd.get('postprocess', np.nan))

    return img_rgb, results
# ======================================= #

# ========= 評価メトリクス＆出力 ========= #
def plot_confusion_matrix_and_labels(cm: np.ndarray,
                                     labels: List[str],
                                     cm_path: str,
                                     label_path: str) -> None:
    fig = plt.figure(figsize=(20, 15))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, ticks, rotation=45, ha='right')
    plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, int(cm[i, j]),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"混合行列を保存: {cm_path}")

    # ラベル表
    df = pd.DataFrame({"Index": ticks, "Label": labels})
    fig2, ax = plt.subplots(figsize=(10, len(labels) * .5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    fig2.savefig(label_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"ラベル表を保存: {label_path}")

def extract_label_from_filename(filename: str) -> str | None:
    m = re.match(r'([a-z_]+)_\d+', filename, flags=re.I)
    return m.group(1) if m else None
# ======================================= #

# ------------- メイン評価ループ --------- #
def calculate_accuracy(model: YOLO,
                       image_folder: str,
                       class_names: List[str]) -> float:
    y_true, y_pred, y_miss = [], [], []
    correct = total = 0

    image_folder = Path(image_folder)
    temp_folder = image_folder / "temp_png"
    temp_folder.mkdir(exist_ok=True)

    for f in sorted(image_folder.iterdir()):
        if f.is_dir():
            continue
        ext = f.suffix.lower()
        # JPG → PNG に統一
        if ext == ".jpg":
            img = cv2.imread(str(f))
            if img is None:
                print(f"読み込み失敗: {f}")
                continue
            img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            f_png = temp_folder / (f.stem + ".png")
            cv2.imwrite(str(f_png), img)
            image_path = f_png
        elif ext == ".png":
            image_path = f
            img = cv2.imread(str(image_path))
            img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            continue

        truth_raw = extract_label_from_filename(f.stem)
        if truth_raw is None:
            continue

        _, results = perform_inference(model, str(image_path))

        # 推論結果をまとめて標準化
        pred_labels = []
        for res in results:
            for box in res.boxes:
                pred_labels.append(class_names[int(box.cls[0])])
        pred_raw = '_'.join(pred_labels)

        truth = normalize_label(truth_raw)
        pred  = normalize_label(pred_raw)

        y_true.append(truth)
        y_pred.append(pred)

        if truth == pred:
            correct += 1
        else:
            y_miss.append(f"{truth_raw} → {pred_raw}")

        total += 1

    # --------------- 統計計算 ---------------- #
    accuracy = correct / total if total else 0.0
    cer = calculate_cer(y_true, y_pred)
    wer = calculate_wer(y_true, y_pred)

    # 表示
    print(f"\n==== 精度指標 ====")
    print(f" Accuracy          : {accuracy:.4%}")
    print(f" CER               : {cer:.4%}")
    print(f" WER               : {wer:.4%}")
    print(f" 誤分類サンプル数  : {len(y_miss)} / {total}")

    # 混同行列
    ulabels = list(dict.fromkeys(y_true))   # 固定順序
    cm = confusion_matrix(y_true, y_pred, labels=ulabels)
    plot_confusion_matrix_and_labels(cm,
                                     labels=ulabels,
                                     cm_path="./confusion_matrix.png",
                                     label_path="./labels_table.png")
    print(classification_report(y_true, y_pred, labels=ulabels,
                                zero_division=0))

    # --------------- 時間・CPU統計 ---------- #
    def _summary(arr, name: str) -> None:
        if not arr:
            print(f"{name}: N/A")
            return

        # NaN を除去（speed キーが欠落したケース対策）
        arr_clean = [x for x in arr if not np.isnan(x)]
        if not arr_clean:
            print(f"{name}: N/A")
            return

        mean   = statistics.mean(arr_clean)
        sd     = statistics.stdev(arr_clean) if len(arr_clean) > 1 else 0.0
        median = statistics.median(arr_clean)
        p95    = np.percentile(arr_clean, 95)

        print(f"{name}: mean={mean:.2f}  sd={sd:.2f}  "
            f"median={median:.2f}  p95={p95:.2f}")


    print("\n==== レイテンシ / システム負荷 ====")
    _summary(preprocess_times,  "Preprocess (ms)")
    _summary(inference_times,   "Inference  (ms)")
    _summary(postprocess_times, "Postprocess(ms)")
    _summary(latency_times_ms,  "End-to-End (ms)")
    _summary(cpu_usages_proc,   "CPU usage  (%)")

    # 後片付け
    shutil.rmtree(temp_folder, ignore_errors=True)
    return accuracy
# ---------------------------------------- #

def main() -> None:
    model = load_model("./model/weights/best.pt")
    class_names = [
        'a','i','u','e','o',
        'ka','ki','ku','ke','ko',
        'sa','shi','su','se','so',
        'ta','chi','tsu','te','to',
        'na','ni','nu','ne','no',
        'ha','hi','fu','he','ho',
        'ma','mi','mu','me','mo',
        'ya','yu','yo','ra','ri',
        'ru','re','ro','wa','wo','n'
    ]
    acc = calculate_accuracy(model, "test_images/", class_names)
    print(f"\n--- 最終 Accuracy: {acc:.4%} ---")

if __name__ == "__main__":
    torch.set_grad_enabled(False)   # 推論のみ
    main()
