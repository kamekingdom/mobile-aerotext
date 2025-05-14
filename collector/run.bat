@echo off
REM カレントディレクトリをスクリプトのある場所に変更
cd /d "%~dp0"

REM 仮想環境を有効化
call myenv\Scripts\activate

REM main.pyを実行
python scripts\main.py

REM 実行終了後、仮想環境を無効化
deactivate
