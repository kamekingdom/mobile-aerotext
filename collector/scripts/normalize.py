import sqlite3
import struct
import numpy as np
from tqdm import tqdm

def normalize_and_recalculate(db_path, new_db_path, table_name):
    """
    データベースの座標を正規化し、速度・加速度を再計算して新しいデータベースに保存
    """
    # 元のデータベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 新しいデータベースを作成
    new_conn = sqlite3.connect(new_db_path)
    new_cursor = new_conn.cursor()

    # テーブル構造をコピー
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    column_definitions = ", ".join([f"{col[1]} {col[2]}" for col in columns])
    new_cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions});")

    # データを取得
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # 各列のインデックス
    col_index = {col[1]: idx for idx, col in enumerate(columns)}

    # データの正規化と速度・加速度の再計算
    for row in tqdm(rows, desc="Processing data"):
        length = row[col_index["length"]]
        blob_data = row[col_index["data"]]

        # デコードして座標を取得
        num_floats = len(blob_data) // 4
        decoded_values = struct.unpack('f' * num_floats, blob_data)
        data_per_frame = num_floats // length
        frames = np.array([decoded_values[i * data_per_frame:(i + 1) * data_per_frame] for i in range(length)])

        # 座標列の取得（x: index 1, y: index 2, z: index 3）
        x_coords, y_coords, z_coords = frames[:, 1], frames[:, 2], frames[:, 3]

        # 最小値・最大値を計算
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        # 正規化
        x_coords = (x_coords - x_min) / (x_max - x_min) if x_max > x_min else x_coords
        y_coords = (y_coords - y_min) / (y_max - y_min) if y_max > y_min else y_coords
        z_coords = (z_coords - z_min) / (z_max - z_min) if z_max > z_min else z_coords

        # 正規化後のデータを更新
        frames[:, 1], frames[:, 2], frames[:, 3] = x_coords, y_coords, z_coords

        # 速度と加速度を再計算
        velocities = np.zeros((length, 3))  # [vx, vy, vz]
        accelerations = np.zeros((length, 3))  # [ax, ay, az]
        for i in range(1, length):
            delta_t = 1.0  # フレーム間の時間差を仮定（必要に応じて変更）
            velocities[i] = (frames[i, 1:4] - frames[i - 1, 1:4]) / delta_t
            if i > 1:
                accelerations[i] = (velocities[i] - velocities[i - 1]) / delta_t

        # 再計算した速度・加速度をフレームデータに統合
        frames = np.hstack([frames[:, :4], velocities, accelerations])

        # フラット化してエンコード
        flattened_data = frames.flatten()
        encoded_data = struct.pack(f'{len(flattened_data)}f', *flattened_data)

        # 新しいデータを挿入
        new_row = list(row)
        new_row[col_index["data"]] = encoded_data
        new_cursor.execute(f"INSERT INTO {table_name} VALUES ({','.join(['?'] * len(new_row))})", new_row)

    # コミットとクローズ
    new_conn.commit()
    conn.close()
    new_conn.close()

    print(f"Normalization and recalculation complete. New database saved to {new_db_path}")

# メイン処理
db_path = "gesture_data_hg.db"  # 元のデータベース
new_db_path = "gesture_data_normalized.db"  # 新しいデータベース
table_name = "GestureTable"  # テーブル名

normalize_and_recalculate(db_path, new_db_path, table_name)
