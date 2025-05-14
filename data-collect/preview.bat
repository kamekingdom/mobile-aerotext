@echo off
REM カレントディレクトリをスクリプトのある場所に変更
cd /d "%~dp0"

REM 仮想環境を有効化
call myenv\Scripts\activate

REM db-preview.pyを実行
python scripts/db-preview.py

REM 仮想環境を無効化
deactivate

REM ユーザー入力を待機
echo 実行が終了しました。続行するにはEnterキーを押してください。
set /p dummy=
