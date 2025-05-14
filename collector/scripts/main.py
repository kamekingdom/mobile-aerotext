import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import tkinter as tk
import tkinter.font as tkFont
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from combination import combinations_romaji_result  # 軌跡リストの組み合わせ（ローマ字対応）
import keyboard  # キーボード操作用
import math
import sqlite3
import struct

def save_trajectory_to_db(db_path, table_name, label, tester, trajectory_data):
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブル作成
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            tester TEXT,
            length INTEGER,
            data BLOB
        )
    """)

    # 軌跡データのエンコード
    flattened_data = [value for row in trajectory_data for value in row]  # 配列をフラットにする
    encoded_data = struct.pack(f'{len(flattened_data)}f', *flattened_data)

    # データの保存
    cursor.execute(f"""
        INSERT INTO {table_name} (label, tester, length, data)
        VALUES (?, ?, ?, ?)
    """, (label, tester, len(trajectory_data), encoded_data))

    # コミットとクローズ
    conn.commit()
    conn.close()
    print(f"Trajectory data saved to database: {db_path}, Table: {table_name}")


def read_gesture_data(cursor, table_name, label, tester):
    cursor.execute(f"""
        SELECT length, data FROM {table_name}
        WHERE label=? AND tester=?
    """, (label, tester))
    result = cursor.fetchone()
    if result:
        length, blob_data = result
        num_floats = len(blob_data) // 4
        decoded_values = struct.unpack('f' * num_floats, blob_data)
        data_per_frame = num_floats // length
        frames = [decoded_values[i * data_per_frame:(i + 1) * data_per_frame] for i in range(length)]
        return frames
    else:
        return None

# GUIでユーザーネームを選択または入力する関数
def get_user_name():
    base_data_dir = "./data/"
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
    existing_users = os.listdir(base_data_dir)

    root = tb.Window(themename="superhero")
    root.title("ユーザーネーム選択")

    selected_username = tk.StringVar(value="")

    frame = tb.Frame(root, padding=10)
    frame.pack(fill=BOTH, expand=True)

    tb.Label(frame, text="既存のユーザーネームを選択するか、新しい名前を入力してください：", font=("Arial", 12)).pack(pady=5)

    if existing_users:
        listbox = tk.Listbox(frame, height=5, width=30)
        for user in existing_users:
            listbox.insert(tk.END, user)
        listbox.pack(pady=5)

        def select_existing_user():
            selected = listbox.get(listbox.curselection())
            selected_username.set(selected)
            root.destroy()

        tb.Button(frame, text="選択", command=select_existing_user, bootstyle=SUCCESS).pack(pady=5)
    else:
        tb.Label(frame, text="既存のユーザーネームはありません。新しい名前を入力してください。", font=("Arial", 10)).pack(pady=5)

    tb.Label(frame, text="新しい名前:", font=("Arial", 12)).pack(pady=5)
    new_user_entry = tb.Entry(frame, width=30)
    new_user_entry.pack(pady=5)

    def select_new_user():
        new_user = new_user_entry.get()
        if not new_user.strip():
            tb.Messagebox.show_error("エラー", "名前を入力してください。", parent=root)
            return
        selected_username.set(new_user.strip())
        root.destroy()

    tb.Button(frame, text="新しい名前を使用", command=select_new_user, bootstyle=PRIMARY).pack(pady=5)

    root.mainloop()
    return selected_username.get()


def select_combination(user_name):
    import os
    import tkinter as tk
    from tkinter import font as tkFont
    import ttkbootstrap as tb
    from ttkbootstrap.constants import PRIMARY, INFO, SUCCESS, LIGHT
    from tkinter.constants import BOTH, END, SINGLE, VERTICAL, W, NS, E

    base_dir = f"./data/{user_name}/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 動画ファイルのチェック
    completed = []
    video_files = [f for f in os.listdir(base_dir) if f.endswith('.mp4')]
    completed = [os.path.splitext(f)[0] for f in video_files]  # 拡張子を除いたファイル名がラベル名

    # 全体の組み合わせから未完了のものを抽出
    pending = [c[1] for c in combinations_romaji_result if c[1] not in completed]

    root = tb.Window(themename="superhero")
    root.title("軌跡選択")

    frame = tb.Frame(root, padding=10)
    frame.pack(fill=BOTH, expand=True)

    # カスタムフォントの設定
    custom_font = tkFont.Font(family="Arial", size=12)

    # 未検出セクション
    tb.Label(frame, text="未検出", font=("Arial", 14), bootstyle=INFO).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky=W)
    pending_search_entry = tb.Entry(frame, font=("Arial", 12), bootstyle=LIGHT)
    pending_search_entry.grid(row=1, column=0, padx=10, pady=5, sticky=W)
    tb.Button(frame, text="検索", command=lambda: search_pending(pending_search_entry.get()), bootstyle=INFO).grid(row=1, column=1, padx=0, pady=5, sticky="w")

    # 未検出のアイテム数表示
    pending_count_label = tb.Label(frame, text=f"件数: {len(pending)}", font=("Arial", 12), bootstyle=LIGHT)
    pending_count_label.grid(row=2, column=0, padx=10, pady=5, sticky=W)

    pending_listbox = tk.Listbox(frame, height=15, width=30, selectmode=SINGLE, font=custom_font)
    for item in pending:
        pending_listbox.insert(END, item)
    pending_scrollbar = tk.Scrollbar(frame, orient=VERTICAL, command=pending_listbox.yview)
    pending_listbox.configure(yscrollcommand=pending_scrollbar.set)
    pending_listbox.grid(row=3, column=0, padx=10, pady=5, sticky=W)
    pending_scrollbar.grid(row=3, column=1, sticky=NS)

    # 検出済みセクション
    tb.Label(frame, text="検出済み", font=("Arial", 14), bootstyle=SUCCESS).grid(row=0, column=2, columnspan=2, padx=10, pady=5, sticky=W)
    completed_search_entry = tb.Entry(frame, font=("Arial", 12), bootstyle=LIGHT)
    completed_search_entry.grid(row=1, column=2, padx=10, pady=5, sticky=W)
    tb.Button(frame, text="検索", command=lambda: search_completed(completed_search_entry.get()), bootstyle=SUCCESS).grid(row=1, column=3, padx=0, pady=5, sticky="w")

    # 検出済みのアイテム数表示
    completed_count_label = tb.Label(frame, text=f"件数: {len(completed)}", font=("Arial", 12), bootstyle=LIGHT)
    completed_count_label.grid(row=2, column=2, padx=10, pady=5, sticky=W)

    completed_listbox = tk.Listbox(frame, height=15, width=30, selectmode=SINGLE, font=custom_font)
    for item in completed:
        completed_listbox.insert(END, item)
    completed_scrollbar = tk.Scrollbar(frame, orient=VERTICAL, command=completed_listbox.yview)
    completed_listbox.configure(yscrollcommand=completed_scrollbar.set)
    completed_listbox.grid(row=3, column=2, padx=10, pady=5, sticky=W)
    completed_scrollbar.grid(row=3, column=3, sticky=NS)

    # リスト更新関数
    def update_listbox(listbox, original_list, search_term):
        listbox.delete(0, END)
        filtered_list = [item for item in original_list if search_term.lower() in item.lower()]
        for item in filtered_list:
            listbox.insert(END, item)
        return filtered_list

    # 未検出の検索処理
    def search_pending(search_term):
        filtered_list = update_listbox(pending_listbox, pending, search_term)
        pending_count_label.config(text=f"件数: {len(filtered_list)}")

    # 検出済みの検索処理
    def search_completed(search_term):
        filtered_list = update_listbox(completed_listbox, completed, search_term)
        completed_count_label.config(text=f"件数: {len(filtered_list)}")

    # 選択確認ボタン
    selected_combination = tk.StringVar(value="")

    def confirm_selection():
        if pending_listbox.curselection():
            selected = pending_listbox.get(pending_listbox.curselection())
            selected_combination.set(selected)
        elif completed_listbox.curselection():
            selected = completed_listbox.get(completed_listbox.curselection())
            selected_combination.set(selected)
        root.destroy()

    tb.Button(frame, text="確認", command=confirm_selection, bootstyle=PRIMARY).grid(row=4, column=0, columnspan=4, pady=10)

    root.mainloop()
    return selected_combination.get()



def segment_trajectory(trajectory_data, split_indices, selected_romaji):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import keyboard

    plt.ion()

    deleted_trajectory = []  # 削除されたフレームのインデックスを記録

    # メインウィンドウを作成し、グリッドレイアウトで分割
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig)  # 1行2列のレイアウト
    ax3d = fig.add_subplot(gs[0], projection='3d')  # 左側に3Dプロット
    ax2d = fig.add_subplot(gs[1])  # 右側に2Dプロット

    # 軌跡データの準備
    trajectory_array = np.array(trajectory_data)
    frame_indices = trajectory_array[:, 0]  # frame_index
    x_coords = trajectory_array[:, 1]      # x座標
    y_coords = -trajectory_array[:, 2]     # y座標

    # 軸範囲を初期データから計算して固定
    x_min, x_max = min(frame_indices), max(frame_indices)
    y_min, y_max = min(x_coords), max(x_coords)
    z_min, z_max = min(y_coords), max(y_coords)

    # カメラ角度の初期化
    elev, azim = 30, -60
    frame_idx = 0  # 現在のフレームインデックス

    def update_3d_plot():
        """現在の状態を3Dプロットする"""
        nonlocal elev, azim
        ax3d.clear()
        ax3d.set_title(f"3D Trajectory Segmentation - {selected_romaji}")
        ax3d.set_xlabel("Index")
        ax3d.set_ylabel("X Coordinate")
        ax3d.set_zlabel("Y Coordinate")

        # 全軌跡を灰色で表示 (deleted_trajectoryを除外)
        valid_indices = [i for i in range(len(frame_indices)) if i not in deleted_trajectory]
        ax3d.scatter(
            [frame_indices[i] for i in valid_indices],
            [x_coords[i] for i in valid_indices],
            [y_coords[i] for i in valid_indices],
            c='gray',
            s=10,
            alpha=0.5
        )

        # 現在の座標を青色で表示
        if len(frame_indices) > frame_idx:
            ax3d.scatter(frame_indices[frame_idx], x_coords[frame_idx], y_coords[frame_idx], c='blue', s=50)

        # 分割された点を赤色で表示
        for idx in split_indices:
            if idx < len(frame_indices):
                ax3d.scatter(frame_indices[idx], x_coords[idx], y_coords[idx], c='red', s=50)
                

        # 固定された軸範囲を適用
        ax3d.set_xlim(x_min - 0.1, x_max + 0.1)
        ax3d.set_ylim(y_min - 0.1, y_max + 0.1)
        ax3d.set_zlim(z_min - 0.1, z_max + 0.1)

        # 前回のカメラ角度を適用
        ax3d.view_init(elev=elev, azim=azim)

    def update_2d_plot():
        """セグメント化された座標を2Dプロットに表示"""
        ax2d.clear()
        ax2d.set_title("Segmented Trajectory - 2D View")
        ax2d.set_xlabel("X Coordinate")
        ax2d.set_ylabel("Y Coordinate")

        # 全体の軌跡データをグレーで表示
        valid_indices = [i for i in range(len(frame_indices)) if i not in deleted_trajectory]
        ax2d.plot(
            [x_coords[i] for i in valid_indices],
            [y_coords[i] for i in valid_indices],
            linestyle='-', marker='o', markersize=3, color='gray', alpha=0.5, label="Full Trajectory"
        )

        # 分割点の間隔でセグメント化されたデータを描画
        if split_indices:
            indices = sorted(split_indices + [0, len(frame_indices) - 1])
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1] + 1  # 修正: 終端を含むように範囲を調整

                # 削除されたインデックスを除外した座標データを取得
                segment_indices = [
                    idx for idx in range(start_idx, end_idx) if idx not in deleted_trajectory
                ]
                segment_x = [x_coords[idx] for idx in segment_indices]
                segment_y = [y_coords[idx] for idx in segment_indices]

                if segment_x and segment_y:  # データが残っている場合のみプロット
                    ax2d.plot(segment_x, segment_y, marker='o', label=f"Segment {i + 1}")

        # 軸の範囲を固定
        ax2d.set_xlim(min(x_coords) - 0.1, max(x_coords) + 0.1)
        ax2d.set_ylim(min(y_coords) - 0.1, max(y_coords) + 0.1)
        ax2d.legend()



    # 初回にプロットを更新
    update_2d_plot()
    update_3d_plot()

    while True:
        plt.pause(0.01)  # 短い待機時間

        # 矢印キーでフレームを操作（2Dプロットは更新しない）
        while keyboard.is_pressed('right'):
            frame_idx = min(frame_idx + 1, len(frame_indices) - 1)
            update_3d_plot()
            plt.pause(0.05)

        while keyboard.is_pressed('left'):
            frame_idx = max(frame_idx - 1, 0)
            update_3d_plot()
            plt.pause(0.05)
            
        # 矢印キーでフレームを操作（2Dプロットは更新しない）
        while keyboard.is_pressed('>'):
            frame_idx = min(frame_idx + 3, len(frame_indices) - 1)
            update_3d_plot()
            plt.pause(0.05)

        while keyboard.is_pressed('<'):
            frame_idx = max(frame_idx - 3, 0)
            update_3d_plot()
            plt.pause(0.05)

        # DeleteキーまたはDキーで削除または復元
        if keyboard.is_pressed('delete') or keyboard.is_pressed('d'):
            if frame_idx in deleted_trajectory:
                print(f"Restored point at frame {frame_idx}")
                deleted_trajectory.remove(frame_idx)
            else:
                print(f"Deleted point at frame {frame_idx}")
                deleted_trajectory.append(frame_idx)

            # プロットを更新
            update_2d_plot()
            update_3d_plot()

            # キーが離れるのを待つ
            while keyboard.is_pressed('delete') or keyboard.is_pressed('d'):
                plt.pause(0.01)


        # スペースキーで分割点を記録または削除
        if keyboard.is_pressed('space'):
            if frame_idx in split_indices:
                print(f"Split removed at frame {frame_idx}")
                split_indices.remove(frame_idx)
            else:
                print(f"Split recorded at frame {frame_idx}")
                split_indices.append(frame_idx)

            # セグメント化された座標を更新
            update_2d_plot()

            # キーが離れるのを待つ
            while keyboard.is_pressed('space'):
                plt.pause(0.01)

        # ESCキーで終了
        if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
            print("Segmentation complete. Exiting...")
            break

    # ESCで終了後に削除リストを反映
    trajectory_data = np.array(
        [row for i, row in enumerate(trajectory_data) if i not in deleted_trajectory]
    )

    # 削除されたインデックスを考慮してsplit_indicesを調整
    adjusted_split_indices = []
    deleted_count = 0
    for i in range(len(frame_indices)):
        if i in deleted_trajectory:
            deleted_count += 1
        elif i in split_indices:
            adjusted_split_indices.append(i - deleted_count)
            
    split_indices = adjusted_split_indices

    plt.ioff()
    plt.close(fig)
    return split_indices, trajectory_data



def save_trajectory_to_db(db_path, table_name, label, tester, trajectory_data, split_indices):
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブル作成
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            tester TEXT,
            length INTEGER,
            data BLOB,
            split_indices BLOB
        )
    """)

    # 既存データの削除（同じラベルとテスターが一致するもの）
    cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE label = ? AND tester = ?
    """, (label, tester))

    # 軌跡データのエンコード
    flattened_data = [value for row in trajectory_data for value in row]  # 配列をフラットにする
    encoded_data = struct.pack(f'{len(flattened_data)}f', *flattened_data)

    # 分割点のエンコード
    encoded_splits = struct.pack(f'{len(split_indices)}i', *split_indices)

    # 新しいデータの挿入
    cursor.execute(f"""
        INSERT INTO {table_name} (label, tester, length, data, split_indices)
        VALUES (?, ?, ?, ?, ?)
    """, (label, tester, len(trajectory_data), encoded_data, encoded_splits))

    # コミットとクローズ
    conn.commit()
    conn.close()
    print(f"Trajectory data saved to database: {db_path}, Table: {table_name}")

def record_trajectory(user_name, selected_romaji):
    db_path = "gesture_data_hg.db"
    table_name = "GestureTable"

    # 保存先ディレクトリの作成
    base_dir = f"./data/{user_name}/"
    os.makedirs(base_dir, exist_ok=True)
    video_file = os.path.join(base_dir, f"{selected_romaji}.mp4")

    trajectory_data = []
    split_indices = []
    prev_position = None  # 前フレームの位置
    prev_time = None      # 前フレームの時間
    distance_threshold = 0.02  # 移動距離の閾値（正規化座標で）

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    video_writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        square_size = 200
        square_adjust_y = 50
        top_left = (center_x - square_size // 2, center_y - square_size // 2 - square_adjust_y)
        bottom_right = (center_x + square_size // 2, center_y + square_size // 2 - square_adjust_y)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 現在取得中の文字列を表示
        cv2.putText(image, f"{selected_romaji}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        hand_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y, z = index_tip.x, index_tip.y, index_tip.z
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

                # 手が指している状態（指先が曲がっていない）
                if y < index_dip.y:
                    hand_detected = True
                    current_position = np.array([x, y, z])

                    # 現在の時間を取得
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()

                    # 前の位置がない場合は初期化
                    if prev_position is None:
                        prev_position = current_position
                        prev_time = current_time

                    # 距離を計算し、`distance_threshold`以上の場合に保存
                    distance = np.linalg.norm(current_position - prev_position)
                    if distance >= distance_threshold:
                        # 加速度と速度の計算
                        acceleration = [0.0, 0.0, 0.0]
                        velocity = [0.0, 0.0, 0.0]

                        if prev_time is not None:
                            delta_t = current_time - prev_time
                            velocity = (current_position - prev_position) / delta_t

                            if len(trajectory_data) > 0:
                                prev_velocity = trajectory_data[-1][4:7]  # 前フレームの速度
                                acceleration = (velocity - np.array(prev_velocity)) / delta_t

                        # 保存するデータ
                        trajectory_data.append([
                            len(trajectory_data),  # Frame Index
                            x, y, z,               # Position
                            *velocity,             # Velocity (x, y, z)
                            *acceleration          # Acceleration (x, y, z)
                        ])

                        prev_position = current_position
                        prev_time = current_time

                        # 現在の点を表示
                        cx, cy = int(x * w), int(y * h)
                        cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)

        # 軌跡を描画
        for i in range(1, len(trajectory_data)):
            start_point = (int(trajectory_data[i - 1][1] * w), int(trajectory_data[i - 1][2] * h))
            end_point = (int(trajectory_data[i][1] * w), int(trajectory_data[i][2] * h))
            cv2.line(image, start_point, end_point, (255, 0, 0), 4)

        if hand_detected:
            if video_writer is None:
                video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
            video_writer.write(frame)
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:  # ESCキーまたはQキーで終了
            break

    if video_writer is not None:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # セグメンテーションを実行
    # split_indices, trajectory_data = segment_trajectory(trajectory_data, split_indices, selected_romaji)

    # データをデータベースに保存
    save_trajectory_to_db(db_path, table_name, selected_romaji, user_name, trajectory_data, split_indices)

    print(f"Trajectory recording complete. Video saved at: {video_file}")
    print("Data saved to database.")

# メイン処理
if __name__ == "__main__":
    user_name = get_user_name()
    if not user_name:
        print("ユーザーネームが選択または入力されませんでした。プログラムを終了します。")
        exit()

    while True:
        selected_combination = select_combination(user_name)
        if not selected_combination:
            print("何も選択されませんでした。プログラムを終了します。")
            break

        print(f"選択された軌跡: {selected_combination}")
        record_trajectory(user_name, selected_combination)
        print(f"軌跡 {selected_combination} を記録しました！")