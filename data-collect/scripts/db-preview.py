import tkinter as tk
import tkinter.font as tkFont
import tkinter.messagebox as messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# データベースからデータを読み取る関数
def fetch_trajectory_data_with_splits(db_path, table_name, label=None, tester=None, id=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # IDで検索
    if id:
        query = f"SELECT length, data, split_indices FROM {table_name} WHERE id=?"
        cursor.execute(query, (id,))
    # ラベルとテスター名で検索
    elif label and tester:
        query = f"SELECT length, data, split_indices FROM {table_name} WHERE label=? AND tester=?"
        cursor.execute(query, (label, tester))
    else:
        conn.close()
        return None, None

    result = cursor.fetchone()
    conn.close()

    if result:
        length, blob_data, split_blob = result
        num_floats = len(blob_data) // 4
        decoded_values = struct.unpack('f' * num_floats, blob_data)
        data_per_frame = num_floats // length
        frames = [decoded_values[i * data_per_frame:(i + 1) * data_per_frame] for i in range(length)]

        if split_blob:
            split_indices = struct.unpack(f'{len(split_blob) // 4}i', split_blob)
        else:
            split_indices = []

        return np.array(frames), split_indices
    else:
        return None, None

# テスター名をリスト化
def get_tester_list(data_dir):
    try:
        folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        return folders
    except FileNotFoundError:
        messagebox.showerror("エラー", f"ディレクトリ {data_dir} が見つかりません．")
        return []

# 編集モード
def edit_mode(data, split_indices, label, tester, db_path, table_name):
    if data is None or len(data) == 0:
        messagebox.showerror("エラー", "指定されたデータが見つかりませんでした．")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Edit Mode: Label={label}, Tester={tester}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    deleted_indices = []
    split_indices = list(split_indices)
    segments = sorted(split_indices + [0, len(data) - 1])
    current_idx = 0  # 現在選択中のインデックス

    def plot_segments():
        ax.clear()
        colors = cm.get_cmap('tab10', len(segments))
        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]
            valid_indices = [idx for idx in range(start_idx, end_idx) if idx not in deleted_indices]
            ax.plot(
                data[valid_indices, 1],
                -data[valid_indices, 2],
                marker='o',
                label=f"Segment {i + 1}",
                color=colors(i / (len(segments) - 1))
            )

        # 現在の選択点を強調
        if current_idx not in deleted_indices:
            ax.scatter(data[current_idx, 1], -data[current_idx, 2], color='red', s=100, label="Selected Point")

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.grid()
        fig.canvas.draw()

    plot_segments()

    def on_key(event):
        nonlocal deleted_indices, split_indices, current_idx, segments

        if event.key == 'left':
            current_idx = max(current_idx - 1, 0)
            plot_segments()

        elif event.key == 'right':
            current_idx = min(current_idx + 1, len(data) - 1)
            plot_segments()

        elif event.key == 'd':
            if current_idx not in deleted_indices:
                deleted_indices.append(current_idx)
            elif current_idx in deleted_indices:
                deleted_indices.remove(current_idx)
            plot_segments()

        elif event.key == ' ':
            if current_idx in split_indices:
                split_indices.remove(current_idx)
            else:
                split_indices.append(current_idx)
                split_indices.sort()
            segments = sorted(split_indices + [0, len(data) - 1])
            plot_segments()

        elif event.key == 'q' or event.key == 'esc':
            save_changes(data, split_indices, deleted_indices, label, tester, db_path, table_name)
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# データベースに変更を保存する関数
def save_changes(data, split_indices, deleted_indices, label, tester, db_path, table_name):
    updated_data = np.array([row for i, row in enumerate(data) if i not in deleted_indices])
    adjusted_split_indices = []
    for idx in split_indices:
        if idx not in deleted_indices:
            adjusted_split_indices.append(idx - sum(i < idx for i in deleted_indices))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    flattened_data = [value for row in updated_data for value in row]
    encoded_data = struct.pack(f'{len(flattened_data)}f', *flattened_data)
    encoded_splits = struct.pack(f'{len(adjusted_split_indices)}i', *adjusted_split_indices)

    cursor.execute(f"""
        UPDATE {table_name}
        SET data = ?, split_indices = ?, length = ?
        WHERE label = ? AND tester = ?
    """, (encoded_data, encoded_splits, len(updated_data), label, tester))
    conn.commit()
    conn.close()

    print(f"変更が保存されました．Label={label}, Tester={tester}")

# GUIメインウィンドウ
def main():
    db_path = "gesture_data_hg.db"
    table_name = "GestureTable"
    data_dir = "./data/"

    tester_list = get_tester_list(data_dir)

    root = tb.Window(themename="superhero")
    root.title("軌跡データプロットツール")
    root.geometry("400x300")

    frame = tb.Frame(root, padding=10)
    frame.pack(fill=BOTH, expand=True)

    tb.Label(frame, text="ラベル (任意):", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10)
    label_entry = tb.Entry(frame, width=30)
    label_entry.grid(row=0, column=1, padx=10, pady=10)

    tb.Label(frame, text="テスター名:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10)
    tester_combobox = tb.Combobox(frame, values=tester_list, width=28, state="readonly")
    tester_combobox.grid(row=1, column=1, padx=10, pady=10)
    tester_combobox.set("選択してください")

    tb.Label(frame, text="ID (任意):", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10)
    id_entry = tb.Entry(frame, width=30)
    id_entry.grid(row=2, column=1, padx=10, pady=10)

    def on_edit_button_click():
        label = label_entry.get().strip()
        tester = tester_combobox.get().strip()
        id_value = id_entry.get().strip()

        if not id_value and (not label or tester == "選択してください"):
            messagebox.showerror("エラー", "ラベルとテスター名，またはIDを正しく入力または選択してください．")
            return

        id_value = int(id_value) if id_value.isdigit() else None
        data, split_indices = fetch_trajectory_data_with_splits(db_path, table_name, label if label else None, tester if tester != "選択してください" else None, id_value)
        if data is not None:
            edit_mode(data, split_indices, label, tester, db_path, table_name)

    tb.Button(frame, text="編集", command=on_edit_button_click, bootstyle=PRIMARY).grid(row=3, column=0, columnspan=2, pady=10)
    tb.Button(frame, text="終了", command=root.destroy, bootstyle=DANGER).grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
