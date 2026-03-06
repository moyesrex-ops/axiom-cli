@echo off
:: Axiom CLI Launcher
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d C:\Users\moyes\Downloads\axiom-cli
python axiom.py %*
