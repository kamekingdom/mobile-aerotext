# Author: kamekingdom (2025-08-15)
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os
import typing as T

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import ttkbootstrap as tb
from PIL import Image, ImageTk
from ttkbootstrap.constants import *

# かな・ローマ字の定義（例: train_comb.py にて定義）
# combinations_romaji_result: list[tuple[str, str]] = [(かな, ローマ字), ...]
from train_comb import combinations_romaji_result


# =========================
# 定数
# =========================

IMG_SIZE: int = 256                 # 出力画像サイズ（正方）
STROKE_THICKNESS: int = 6           # 描画線幅
DISTANCE_THRESHOLD: float = 0.02    # 指先サンプリング間引き（正規化距離）

# カメラ表示ウィンドウ（大きめ表示）
CAM_WINDOW_NAME: str = "Simple Collector (YOLO)"
CAM_WINDOW_W: int = 1280
CAM_WINDOW_H: int = 720

# YOLO 互換の保存先（Ultralytics 等の標準構成）
YOLO_ROOT: str = "./yolo_dataset"
YOLO_SPLIT: str = "train"           # "train" / "val" / "test" を必要に応じて切替

# ユーザーIDとクラス定義の保存先
USERS_MAP_PATH: str = os.path.join(YOLO_ROOT, "users_map.txt")
CLASSES_TXT_PATH: str = os.path.join(YOLO_ROOT, "classes.txt")

# UI フォント
FONT_TITLE: tuple[str, int] = ("Arial", 18)
FONT_TEXT: tuple[str, int] = ("Arial", 14)
FONT_LIST: tuple[str, int] = ("Arial", 14)
FONT_BTN: tuple[str, int, str] = ("Arial", 13, "bold")


# =========================
# ユーティリティ
# =========================

def ensure_yolo_dirs() -> None:
    """YOLO ディレクトリ(images/labels/<split>)を作成しておく。"""
    os.makedirs(os.path.join(YOLO_ROOT, "images", YOLO_SPLIT), exist_ok=True)
    os.makedirs(os.path.join(YOLO_ROOT, "labels", YOLO_SPLIT), exist_ok=True)


def load_user_map(path: str = USERS_MAP_PATH) -> dict[str, int]:
    """users_map.txt（形式: '01 username'）を読み込み、{username: id} を返す。"""
    if not os.path.exists(path):
        return {}
    mp_: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                mp_[" ".join(parts[1:])] = int(parts[0])
    return mp_


def save_user_map(user_map: dict[str, int], path: str = USERS_MAP_PATH) -> None:
    """{username: id} を 'id username' 形式で保存する。"""
    lines = [f"{uid:02d} {name}\n" for name, uid in sorted(user_map.items(), key=lambda x: x[1])]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_or_create_user_id(user_name: str) -> int:
    """既存ユーザーならIDを返し、無ければ新規採番して保存する。"""
    ensure_yolo_dirs()
    mp_ = load_user_map()
    if user_name in mp_:
        return mp_[user_name]
    used = set(mp_.values())
    new_id = 1
    while new_id in used:
        new_id += 1
    mp_[user_name] = new_id
    save_user_map(mp_)
    return new_id


def _list_user_numbers_in_label(label: str, user_id: int) -> list[int]:
    """
    images/<split>/<label>/IDxx_*.png の「番号」整数を列挙して返す。
    """
    img_dir = os.path.join(YOLO_ROOT, "images", YOLO_SPLIT, label)
    if not os.path.isdir(img_dir):
        return []
    prefix = f"{user_id:02d}_"
    nums: list[int] = []
    for p in glob.glob(os.path.join(img_dir, f"{prefix}*.png")):
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)  # "01_12"
        try:
            n = int(stem.split("_", 1)[1])
            nums.append(n)
        except Exception:
            pass
    return sorted(set(nums))


def next_seq_for_user_in_label(label: str, user_id: int) -> int:
    """
    指定ラベル内で、そのユーザーIDの次番号を返す（MEX：欠番を最優先）。
    例）使用済 = [1,2,4] → 次は 3
    """
    used = _list_user_numbers_in_label(label, user_id)
    if not used:
        return 1
    expect = 1
    for n in used:
        if n == expect:
            expect += 1
        elif n > expect:
            break
    return expect


def count_user_label_samples(label: str, user_id: int) -> int:
    """YOLO 構造の images/<split>/<label> における当該ユーザーの枚数を数える。"""
    img_dir = os.path.join(YOLO_ROOT, "images", YOLO_SPLIT, label)
    if not os.path.isdir(img_dir):
        return 0
    prefix = f"{user_id:02d}_"
    return len(glob.glob(os.path.join(img_dir, f"{prefix}*.png")))


def romaji_to_kana(roma: str) -> str:
    """表示用にローマ字→かなを引く（最初に一致したもの）。なければローマ字を返す。"""
    for kana, r in combinations_romaji_result:
        if r == roma:
            return kana
    return roma


# =========================
# 五十音順キー & ラベルリスト
# =========================

def gojuon_key(kana: str) -> tuple[int, int]:
    """
    五十音順のキー（行→段）を返す。未登録は末尾。
    清音（あ〜ん）の基本並び。拗音・濁音拡張は必要に応じて追記。
    """
    rows = [
        "あいうえお",
        "かきくけこ",
        "さしすせそ",
        "たちつてと",
        "なにぬねの",
        "はひふへほ",
        "まみむめも",
        "やゆよ",
        "らりるれろ",
        "わを",
        "ん",
    ]
    row_map: dict[str, tuple[int, int]] = {}
    for r_idx, row in enumerate(rows):
        for c_idx, ch in enumerate(row):
            row_map[ch] = (r_idx, c_idx)
    return row_map.get(kana, (len(rows), 0))


# =========================
# クラス管理（classes.txt）
# =========================

def load_or_init_classes(path: str = CLASSES_TXT_PATH) -> list[str]:
    """
    classes.txt を読み込む。無ければ combinations_romaji_result のローマ字を初期投入。
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    uniq: list[str] = []
    seen: set[str] = set()
    for kana, roma in combinations_romaji_result:
        if roma not in seen:
            seen.add(roma)
            uniq.append(roma)
    with open(path, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(r + "\n")
    return uniq


def get_class_id(label: str, classes: list[str], path: str = CLASSES_TXT_PATH) -> int:
    """label(romaji) を classes に登録して class_id を返す。未登録なら追記。"""
    if label in classes:
        return classes.index(label)
    with open(path, "a", encoding="utf-8") as f:
        f.write(label + "\n")
    classes.append(label)
    return len(classes) - 1


# =========================
# YOLO 正規化／保存
# =========================

def yolo_normalize_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int
) -> tuple[float, float, float, float]:
    """
    画素座標 -> YOLO 正規化 (x_center, y_center, width, height) in [0,1]
    """
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w - 1))
    ymax = max(0, min(ymax, h - 1))
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def write_yolo_label(txt_path: str, class_id: int,
                     xc: float, yc: float, bw: float, bh: float) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def save_sample_yolo(
    image_gray: np.ndarray,
    label: str,
    user_id: int,
    bbox_xyxy: tuple[int, int, int, int],
    classes: list[str],
    classes_txt_path: str = CLASSES_TXT_PATH,
    img_size: int = IMG_SIZE,
    split: str = YOLO_SPLIT,
) -> tuple[str, str]:
    """
    YOLO 構造で保存し、(image_path, label_path) を返す。
      画像 : yolo_dataset/images/<split>/<label>/<ID2桁>_<seq>.png
      ラベル: yolo_dataset/labels/<split>/<label>/<ID2桁>_<seq>.txt
    連番は「ラベル×ユーザーID」単位で MEX 採番する。
    """
    assert image_gray.dtype == np.uint8
    ensure_yolo_dirs()

    # クラスID
    cid = get_class_id(label, classes, classes_txt_path)

    # MEX 採番
    seq = next_seq_for_user_in_label(label, user_id)

    # パス
    img_dir = os.path.join(YOLO_ROOT, "images", split, label)
    lbl_dir = os.path.join(YOLO_ROOT, "labels", split, label)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    base = f"{user_id:02d}_{seq}"
    image_path = os.path.join(img_dir, f"{base}.png")
    label_path = os.path.join(lbl_dir, f"{base}.txt")

    # 画像保存
    cv2.imwrite(image_path, image_gray)

    # YOLO ラベル保存
    xmin, ymin, xmax, ymax = bbox_xyxy
    xc, yc, bw, bh = yolo_normalize_bbox(xmin, ymin, xmax, ymax, img_size, img_size)
    write_yolo_label(label_path, cid, xc, yc, bw, bh)

    return image_path, label_path


# =========================
# GUI（ユーザー名・ラベル選択）
# =========================

def get_user_name() -> str:
    """
    ユーザー名を GUI で選択／入力して返す。
    ※ ここでは ./data を一切作らず、USERS_MAP_PATH（yolo_dataset 配下）のみを利用する。
    """
    user_map = load_user_map()
    existing_users = [name for name, _id in sorted(user_map.items(), key=lambda x: x[1])]

    root = tb.Window(themename="superhero")
    root.title("ユーザーネーム選択")
    selected_username = tk.StringVar(value="")
    frame = tb.Frame(root, padding=14)
    frame.pack(fill=tk.BOTH, expand=True)

    tb.Label(frame, text="既存のユーザーネームを選択するか、新しい名前を入力してください：",
             font=FONT_TITLE).pack(pady=8)

    if existing_users:
        lb = tk.Listbox(frame, height=6, width=36, font=FONT_LIST)
        for u in existing_users:
            lb.insert(tk.END, u)
        lb.pack(pady=6)

        def sel_exist() -> None:
            if lb.curselection():
                selected_username.set(lb.get(lb.curselection()))
                root.destroy()

        tb.Button(frame, text="選択", command=sel_exist, bootstyle=SUCCESS).pack(pady=6)

    tb.Label(frame, text="新しい名前:", font=FONT_TEXT).pack(pady=4)
    ent = tb.Entry(frame, width=32, font=FONT_TEXT)
    ent.pack(pady=4)

    def sel_new() -> None:
        name = ent.get().strip()
        if not name:
            tb.Messagebox.show_error("エラー", "名前を入力してください。", parent=root)
            return
        selected_username.set(name)
        root.destroy()

    tb.Button(frame, text="新しい名前を使用", command=sel_new, bootstyle=PRIMARY).pack(pady=6)

    root.mainloop()
    return selected_username.get()


def select_label(user_id: int) -> str:
    """
    かな（第1要素）で五十音順に並べて表示する。
    表示: 「かな (romaji)  (このユーザー: N 枚)」
    戻り値: 選択された **romaji**
    """
    # (かな, romaji) のユニーク化
    pairs: list[tuple[str, str]] = []
    seen = set()
    for kana, roma in combinations_romaji_result:
        if (kana, roma) not in seen:
            seen.add((kana, roma))
            pairs.append((kana, roma))

    # 五十音順
    pairs.sort(key=lambda kr: gojuon_key(kr[0]))

    # UI
    root = tb.Window(themename="superhero")
    root.title("ラベル選択（あいうえお順）")
    frame = tb.Frame(root, padding=14)
    frame.pack(fill=tk.BOTH, expand=True)

    tb.Label(frame, text="収集するラベルを選択してください（自動保存モード）：", font=FONT_TITLE).pack(pady=8)

    lb = tk.Listbox(frame, height=18, width=48, font=FONT_LIST)
    display_to_romaji: dict[int, str] = {}
    for kana, roma in pairs:
        cnt = count_user_label_samples(roma, user_id)
        disp = f"{kana} ({roma})   (このユーザー: {cnt} 枚)"
        display_to_romaji[lb.size()] = roma
        lb.insert(tk.END, disp)
    lb.pack(pady=8)

    selected = tk.StringVar(value="")
    btns = tb.Frame(frame)
    btns.pack(pady=6)

    def refresh_counts() -> None:
        lb.delete(0, tk.END)
        display_to_romaji.clear()
        for kana, roma in pairs:
            cnt = count_user_label_samples(roma, user_id)
            disp = f"{kana} ({roma})   (このユーザー: {cnt} 枚)"
            display_to_romaji[lb.size()] = roma
            lb.insert(tk.END, disp)

    def confirm() -> None:
        if lb.curselection():
            idx = lb.curselection()[0]
            selected.set(display_to_romaji.get(idx, ""))
        root.destroy()

    tb.Button(btns, text="更新", command=refresh_counts, bootstyle=INFO, width=10).grid(row=0, column=0, padx=6)
    tb.Button(btns, text="確認", command=confirm, bootstyle=PRIMARY, width=12).grid(row=0, column=1, padx=6)

    root.mainloop()
    return selected.get()


# =========================
# プレビュー（自動保存済）— BBox 可視化＋「追加/破棄/戻る」
# =========================

def preview_autosaved_dialog(
    img_gray: np.ndarray,
    label: str,
    user_id: int,
    bbox_xyxy: tuple[int, int, int, int],
    img_path: str,
    txt_path: str
) -> str:
    """
    直前に自動保存済みのサンプルをプレビュー。
    ボタン操作に加えてキーボードでも操作可能:
      n → "add"（同ラベルで続ける）
      d → "discard"（自動保存を削除）
      b → "back"（ラベル一覧に戻る）
    戻り値: "add" | "discard" | "back"
    """
    xmin, ymin, xmax, ymax = bbox_xyxy

    # Tk root を先に作る（PhotoImage 前）
    root = tb.Window(themename="superhero")
    root.title("プレビュー（自動保存 / n:追加, d:破棄, b:戻る）")
    frame = tb.Frame(root, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)

    # 可視化（BBox を描画して2倍表示）
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    disp = cv2.resize(vis, (img_gray.shape[1] * 2, img_gray.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    pil_img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(master=root, image=pil_img)

    # 文言
    tb.Label(frame, text=f"Label: {label} / UserID: {user_id:02d}", font=("Arial", 18)).pack(pady=8)
    tb.Label(frame, text="このサンプルは自動保存されています。不要なら「破棄(d)」で削除できます。",
             font=("Arial", 12), bootstyle=INFO).pack(pady=2)
    tb.Label(frame, text=f"画像: {os.path.basename(img_path)} / ラベル: {os.path.basename(txt_path)}",
             font=("Arial", 12)).pack(pady=2)

    img_label = tb.Label(frame, image=imgtk)
    img_label.image = imgtk  # 参照保持
    img_label.pack(pady=10)

    decision: dict[str, str] = {"act": "add"}  # デフォルトは add

    # ハンドラ
    def on_add() -> None:
        decision["act"] = "add"
        root.destroy()

    def on_discard() -> None:
        decision["act"] = "discard"
        root.destroy()

    def on_back() -> None:
        decision["act"] = "back"
        root.destroy()

    def _safe_close(act: str) -> None:
        decision["act"] = act
        try:
            root.grab_release()   # ★ 先に grab を解放
        except Exception:
            pass
        root.destroy()

    # ボタン（残す）
    btns = tb.Frame(frame); btns.pack(pady=10)
    tb.Button(btns, text="追加（同ラベルで続ける）", command=on_add,
              bootstyle=SUCCESS, width=24).grid(row=0, column=0, padx=8)
    tb.Button(btns, text="破棄（この自動保存を削除）", command=on_discard,
              bootstyle=DANGER, width=24).grid(row=0, column=1, padx=8)
    tb.Button(btns, text="戻る（ラベル一覧へ）", command=on_back,
              bootstyle=SECONDARY, width=22).grid(row=1, column=0, columnspan=2, pady=8)

    # キーバインド（n/d/b で操作）
    def _key_handler(ev: tk.Event) -> None:
        ch = (ev.char or "").lower()
        if ch == "n":
            on_add()
        elif ch == "d":
            on_discard()
        elif ch == "b":
            on_back()
        # それ以外は無視

    root.bind("<Key>", _key_handler)           # 文字キー全般
    root.bind("<KeyPress-n>", lambda e: on_add())
    root.bind("<KeyPress-d>", lambda e: on_discard())
    root.bind("<KeyPress-b>", lambda e: on_back())

    # 閉じるボタン（×）は「追加」と同等に扱う
    root.protocol("WM_DELETE_WINDOW", on_add)

    # フォーカス・モーダル化
    root.update_idletasks()
    root.grab_set()
    root.focus_force()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    root.mainloop()
    return decision["act"]

# =========================
# 収集（空書き→重心センタリング→BBOX 生成）
# =========================

def collect_one_canvas(label: str, user_id: int) -> tuple[np.ndarray | None, tuple[int,int,int,int] | None, str]:
    """
    カメラで空書きを取得し、重心センタリング＋BBOXを返す。
    映像左上に「かな (romaji)｜現在 N 枚」を表示。
    キー: f=保存してプレビューへ / q=プログラム終了
    戻り値: (image_gray, bbox_xyxy, reason)  # reason∈{"quit","exit","discard"}
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした。"); return None, None, "discard"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_WINDOW_H)
    cv2.namedWindow(CAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAM_WINDOW_NAME, CAM_WINDOW_W, CAM_WINDOW_H)

    kana_disp: str = romaji_to_kana(label)
    current_count: int = count_user_label_samples(label, user_id)
    overlay_text: str = f"{kana_disp} ({label})｜現在 {current_count} 枚"

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    points: list[tuple[int,int]] = []
    ret, frame = cap.read()
    if not ret:
        cap.release(); hands.close(); print("フレーム取得に失敗。"); return None, None, "discard"
    H, W = frame.shape[:2]
    prev: T.Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = hands.process(rgb); rgb.flags.writeable = True

        cv2.putText(frame, overlay_text, (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0), 4, cv2.LINE_AA)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                dip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                if tip.y < dip.y:
                    cur = np.array([tip.x, tip.y], dtype=np.float32)
                    if prev is None: prev = cur
                    if float(np.linalg.norm(cur - prev)) >= DISTANCE_THRESHOLD:
                        px, py = int(cur[0]*W), int(cur[1]*H)
                        points.append((px, py)); prev = cur
                    cv2.circle(frame, (int(cur[0]*W), int(cur[1]*H)), 9, (0,0,255), -1)

        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255,0,0), 3)

        cv2.imshow(CAM_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 保存してプレビューへ
            break
        if key == ord('f'):  # 即時終了
            cap.release(); cv2.destroyAllWindows(); hands.close()
            return None, None, "exit"

    cap.release(); cv2.destroyAllWindows(); hands.close()

    if len(points) < 5:
        print("点が少ないため破棄しました。"); return None, None, "discard"

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32)
    pts_norm = np.stack([pts[:,0]/W, pts[:,1]/H], axis=1)
    pts_img = np.stack([pts_norm[:,0]*(IMG_SIZE-1), pts_norm[:,1]*(IMG_SIZE-1)], axis=1)

    cx, cy = pts_img.mean(axis=0)
    tx = (IMG_SIZE-1)/2 - cx; ty = (IMG_SIZE-1)/2 - cy
    pts_img[:,0] = np.clip(pts_img[:,0]+tx, 0, IMG_SIZE-1)
    pts_img[:,1] = np.clip(pts_img[:,1]+ty, 0, IMG_SIZE-1)

    pts_i = pts_img.astype(np.int32)
    for i in range(1, len(pts_i)):
        p0 = tuple(int(x) for x in pts_i[i-1]); p1 = tuple(int(x) for x in pts_i[i])
        cv2.line(canvas, p0, p1, color=255, thickness=STROKE_THICKNESS)

    xs, ys = pts_i[:,0], pts_i[:,1]
    xmin, ymin, xmax, ymax = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(IMG_SIZE-1, xmax); ymax = min(IMG_SIZE-1, ymax)
    if xmax <= xmin: xmax = min(IMG_SIZE-1, xmin+1)
    if ymax <= ymin: ymax = min(IMG_SIZE-1, ymin+1)

    return canvas, (xmin, ymin, xmax, ymax), "quit"

# =========================
# メイン：自動保存→プレビュー（追加/破棄/戻る）
# =========================

def main() -> None:
    ensure_yolo_dirs()
    classes = load_or_init_classes()  # CLASSES_TXT_PATH を自動生成/読込

    # ユーザー選択（./data は使わない）
    user_name: str = get_user_name()
    if not user_name:
        print("ユーザーネーム未選択のため終了します。")
        return
    user_id: int = get_or_create_user_id(user_name)
    print(f"User: {user_name} -> ID={user_id:02d}")

    while True:
        # ラベル選択（五十音順）
        label: str = select_label(user_id)
        if not label:
            print("ラベル未選択のため終了します。")
            return
        print(f"Label: {label}")

        kana_disp = romaji_to_kana(label)
        display_text = f"{kana_disp} ({label})"

        # 同ラベルでの連続収集ループ
        while True:
            img, bbox_xyxy, reason = collect_one_canvas(label=label, user_id=user_id)
            if reason == "exit":
                print("ユーザー操作により終了しました。")
                return
            if img is None or bbox_xyxy is None:
                print("サンプルは保存されませんでした。")
                continue

            # ---- 自動保存（PNG + YOLO .txt）----
            img_path, txt_path = save_sample_yolo(
                image_gray=img,
                label=label,
                user_id=user_id,
                bbox_xyxy=bbox_xyxy,
                classes=classes,
                classes_txt_path=CLASSES_TXT_PATH,
                img_size=IMG_SIZE,
                split=YOLO_SPLIT,
            )

            # ---- プレビュー（自動保存済み）----
            act = preview_autosaved_dialog(img, label, user_id, bbox_xyxy, img_path, txt_path)

            if act == "discard":
                # 自動保存を取り消す（削除）
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    print(f"[Discarded] {img_path}, {txt_path}")
                except Exception as e:
                    print(f"削除時エラー: {e}")
                # 同ラベルで継続
                continue

            if act == "add":
                # 保存は保持、同ラベルで追加収集
                continue

            if act == "back":
                # ラベル一覧に戻る
                break  # 内側ループ脱出 → 外側のラベル選択へ


if __name__ == "__main__":
    main()
