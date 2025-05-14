import sqlite3
import struct

# データベースのパス
db_path = "gesture_data_hg.db"

def find_invalid_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # GestureTableからすべてのIDを取得
    cursor.execute("SELECT id FROM GestureTable")
    ids = [row[0] for row in cursor.fetchall()]

    # 有効なIDの範囲
    valid_range = set(range(0, 2117))

    # 範囲外のIDを特定
    invalid_ids = [id for id in ids if id not in valid_range]

    conn.close()

    # 結果を出力
    if invalid_ids:
        print("Invalid IDs found:", invalid_ids)
    else:
        print("No invalid IDs found.")

# 実行
# find_invalid_ids(db_path)

hiragana_to_romaji = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "を": "wo",
    "ん": "n",
}

def reset_ids_based_on_labels(db_path, hiragana_to_romaji):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ラベルからIDを計算するヘルパー関数
    def calculate_id_from_label(label):
        if len(label) != 2:
            raise ValueError(f"Invalid label: {label}")
        first, second = label
        first_index = list(hiragana_to_romaji.keys()).index(first)
        second_index = list(hiragana_to_romaji.keys()).index(second)
        return first_index * 46 + second_index

    # GestureTableからすべてのレコードを取得
    cursor.execute("SELECT id, label FROM GestureTable")
    rows = cursor.fetchall()

    updated_count = 0

    for row in rows:
        current_id = row[0]
        label = row[1]

        try:
            # IDを再計算
            new_id = calculate_id_from_label(label)

            # IDが異なる場合に更新
            if current_id != new_id:
                cursor.execute(
                    "UPDATE GestureTable SET id = ? WHERE id = ?",
                    (new_id, current_id)
                )
                print(f"Updated ID for label: {label}, Old ID: {current_id}, New ID: {new_id}")
                updated_count += 1

        except Exception as e:
            print(f"Error processing label {label} with ID {current_id}: {e}")

    # 変更をコミット
    conn.commit()
    conn.close()

    print(f"Total updated records: {updated_count}")

def check_invalid_and_duplicate_labels(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # GestureTableの全ラベルを取得
    cursor.execute("SELECT label, COUNT(*) FROM GestureTable GROUP BY label HAVING COUNT(*) > 1")
    duplicates = cursor.fetchall()

    # 異常なラベル（正しいラベル形式に該当しないもの）を取得
    cursor.execute("SELECT label FROM GestureTable")
    all_labels = [row[0] for row in cursor.fetchall()]

    valid_hiragana = list(hiragana_to_romaji.keys())
    invalid_labels = [label for label in all_labels if len(label) != 3 or not all(part in valid_hiragana for part in label.split("_"))]

    conn.close()

    # 結果を表示
    if duplicates:
        print("Duplicate labels found:", duplicates)
    else:
        print("No duplicate labels found.")

    if invalid_labels:
        print("Invalid labels found:", invalid_labels)
    else:
        print("No invalid labels found.")


def investigate_and_fix_labels(db_path, hiragana_to_romaji):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 有効なひらがなのリストを作成
    valid_hiragana = list(hiragana_to_romaji.keys())
    valid_labels = {f"{hiragana_to_romaji[a]}_{hiragana_to_romaji[b]}" for a in valid_hiragana for b in valid_hiragana}

    # すべてのラベルを取得
    cursor.execute("SELECT id, label FROM GestureTable")
    rows = cursor.fetchall()

    invalid_labels = []
    corrected_labels = {}

    for record_id, label in rows:
        if label not in valid_labels:
            invalid_labels.append((record_id, label))
            # 修正案（必要に応じて手動調整）
            corrected_label = None
            if "_" in label:
                parts = label.split("_")
                if len(parts) == 2 and parts[0] in hiragana_to_romaji.values() and parts[1] in hiragana_to_romaji.values():
                    corrected_label = label  # 修正不要
                else:
                    corrected_label = None  # 適切な修正をマッピングで指定
            corrected_labels[record_id] = corrected_label

    # 結果を出力
    print(f"Invalid labels found: {invalid_labels}")
    print(f"Proposed corrections: {corrected_labels}")

    # 修正を適用
    for record_id, new_label in corrected_labels.items():
        if new_label is not None:
            cursor.execute(
                "UPDATE GestureTable SET label = ? WHERE id = ?",
                (new_label, record_id)
            )
            print(f"Updated label for ID {record_id}: {new_label}")

    conn.commit()
    conn.close()

def check_data_length(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # GestureTableの全レコード数を取得
    cursor.execute("SELECT COUNT(*) FROM GestureTable")
    total_records = cursor.fetchone()[0]

    conn.close()
    print(f"Total records in GestureTable: {total_records}")
    return total_records


def reset_correct_ids(db_path, hiragana_to_romaji):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ラベルからIDを計算する関数
    def calculate_id_from_label(label):
        parts = label.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid label format: {label}")
        first, second = parts
        valid_hiragana = list(hiragana_to_romaji.keys())
        first_index = valid_hiragana.index(next(key for key, value in hiragana_to_romaji.items() if value == first))
        second_index = valid_hiragana.index(next(key for key, value in hiragana_to_romaji.items() if value == second))
        return first_index * 46 + second_index

    # GestureTableからすべてのレコードを取得
    cursor.execute("SELECT id, label FROM GestureTable")
    rows = cursor.fetchall()

    updated_count = 0

    for current_id, label in rows:
        try:
            # IDを再計算
            new_id = calculate_id_from_label(label)

            # IDが異なる場合に更新
            if current_id != new_id:
                cursor.execute(
                    "UPDATE GestureTable SET id = ? WHERE id = ?",
                    (new_id, current_id)
                )
                print(f"Updated ID for label: {label}, Old ID: {current_id}, New ID: {new_id}")
                updated_count += 1

        except Exception as e:
            print(f"Error processing label {label} with ID {current_id}: {e}")

    # 更新をコミット
    conn.commit()
    conn.close()

    print(f"Total updated records: {updated_count}")


def find_invalid_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # GestureTableからすべてのIDを取得
    cursor.execute("SELECT id FROM GestureTable")
    ids = [row[0] for row in cursor.fetchall()]

    # 有効なIDの範囲
    valid_range = set(range(0, 2117))

    # 範囲外のIDを特定
    invalid_ids = [id for id in ids if id not in valid_range]

    conn.close()

    # 結果を出力
    if invalid_ids:
        print("Invalid IDs found:", invalid_ids)
    else:
        print("No invalid IDs found.")


# 実行
find_invalid_ids(db_path)

# 実行例
# reset_ids_based_on_labels(db_path, hiragana_to_romaji)

# 実行
# check_invalid_and_duplicate_labels(db_path)

# investigate_and_fix_labels(db_path, hiragana_to_romaji)

# 実行
# check_data_length(db_path)


# reset_correct_ids(db_path, hiragana_to_romaji)
