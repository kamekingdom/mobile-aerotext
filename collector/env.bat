@echo off

:: 仮想環境の名前
set VENV_NAME=myenv

:: Pythonのパスを確認（環境に応じて修正してください）
set PYTHON=python

:: 仮想環境を作成
%PYTHON% -m venv %VENV_NAME%

:: 仮想環境を有効化
call %VENV_NAME%\Scripts\activate

:: requirements.txtからパッケージをインストール
pip install --upgrade pip
pip install -r ./scripts/requirements.txt

:: 完了メッセージ
echo 仮想環境 %VENV_NAME% が作成され，requirements.txt からパッケージがインストールされました。
pause
