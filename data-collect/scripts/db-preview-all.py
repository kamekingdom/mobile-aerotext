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



def fetch_all_trajectory_data_with_splits(db_path, table_name, label=None, tester=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT id, label, tester, length, data, split_indices FROM {table_name} WHERE 1=1"
    params = []

    # ラベルが指定された場合にクエリを追加
    if label:
        query += " AND label LIKE ?"
        params.append(f"%_%")  # "_"を含む全ラベルを取得
    if tester:
        query += " AND tester = ?"
        params.append(tester)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    # Python側で厳密なラベルフィルタリング
    trajectory_data = []
    for result in results:
        id, raw_label, tester, length, blob_data, split_blob = result

        # ラベルが完全一致するものだけを残す
        if raw_label and label_contains_exact_match(raw_label, label):
            if blob_data:
                num_floats = len(blob_data) // 4
                decoded_values = struct.unpack('f' * num_floats, blob_data)
                data_per_frame = num_floats // length
                frames = [decoded_values[i * data_per_frame:(i + 1) * data_per_frame] for i in range(length)]

                if split_blob:
                    split_indices = struct.unpack(f'{len(split_blob) // 4}i', split_blob)
                else:
                    split_indices = []

                trajectory_data.append((id, raw_label, tester, np.array(frames), split_indices))

    return trajectory_data


def label_contains_exact_match(raw_label, target):
    """
    ラベルを'_'で分割し，どれかの部分がターゲット（検索文字列）に完全一致するかを確認
    """
    if not target:
        return False
    parts = raw_label.split('_')
    return target in parts






current_index = 0
trajectory_data_global = []


def plot_current_trajectory():
    if not trajectory_data_global:
        messagebox.showinfo("情報", "一致するデータが見つかりません．")
        return

    global current_index

    # 軌跡データを取得
    id, label, tester, data, split_indices = trajectory_data_global[current_index]

    # プロット
    plt.figure(figsize=(8, 6))
    plt.title(f"ID={id}, Label={label}, Tester={tester}")

    # 分割点の準備
    split_indices = list(sorted([0] + list(split_indices) + [len(data) - 1]))
    colors = cm.get_cmap('tab10', len(split_indices) - 1)

    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        plt.plot(
            data[start_idx:end_idx + 1, 1],  # X座標
            -data[start_idx:end_idx + 1, 2],  # Y座標（Y軸反転）
            marker='o',
            label=f"Segment {i + 1}",
            color=colors(i / (len(split_indices) - 1))
        )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.legend()
    plt.show(block=False)


def show_next_trajectory():
    global current_index
    if current_index < len(trajectory_data_global) - 1:
        current_index += 1
        plt.close('all')  # 現在のプロットを閉じる
        plot_current_trajectory()
    else:
        messagebox.showinfo("情報", "これが最後の軌跡です．")

def show_previous_trajectory():
    global current_index
    if current_index > 0:
        current_index -= 1
        plt.close('all')  # 現在のプロットを閉じる
        plot_current_trajectory()
    else:
        messagebox.showinfo("情報", "これが最初の軌跡です．")



# テスター名をリスト化
def get_tester_list(data_dir):
    try:
        folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        return folders
    except FileNotFoundError:
        messagebox.showerror("エラー", f"ディレクトリ {data_dir} が見つかりません．")
        return []

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

    tb.Label(frame, text="ラベル (部分一致):", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10)
    label_entry = tb.Entry(frame, width=30)
    label_entry.grid(row=0, column=1, padx=10, pady=10)

    tb.Label(frame, text="テスター名:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10)
    tester_combobox = tb.Combobox(frame, values=tester_list, width=28, state="readonly")
    tester_combobox.grid(row=1, column=1, padx=10, pady=10)
    tester_combobox.set("選択してください")

    def on_search_and_plot_button_click():
        global trajectory_data_global, current_index

        label = label_entry.get().strip()
        tester = tester_combobox.get().strip()

        if not label and tester == "選択してください":
            messagebox.showerror("エラー", "検索条件を入力してください．")
            return

        tester_value = tester if tester != "選択してください" else None
        trajectory_data_global = fetch_all_trajectory_data_with_splits(
            db_path, table_name, label if label else None, tester_value
        )
        current_index = 0

        if trajectory_data_global:
            plot_current_trajectory()
        else:
            messagebox.showinfo("情報", "一致するデータが見つかりません．")


    # 検索＆プロットボタンの追加
    tb.Button(frame, text="検索してプロット", command=on_search_and_plot_button_click, bootstyle=INFO).grid(row=2, column=0, columnspan=2, pady=10)

    tb.Button(frame, text="終了", command=root.destroy, bootstyle=DANGER).grid(row=3, column=0, columnspan=2, pady=10)
    
    tb.Button(frame, text="次へ", command=show_next_trajectory, bootstyle=SUCCESS).grid(row=4, column=0, padx=10, pady=10)
    
    tb.Button(frame, text="前へ", command=show_previous_trajectory, bootstyle=PRIMARY).grid(row=4, column=1, padx=10, pady=10)


    root.mainloop()

if __name__ == "__main__":
    main()
