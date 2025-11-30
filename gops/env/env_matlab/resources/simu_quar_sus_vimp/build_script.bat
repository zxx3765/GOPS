@echo off

setlocal
REM 初始化 conda 环境
call conda activate gops
REM 检查激活是否成功
if %errorlevel% neq 0 (
    echo 错误：无法激活 conda 环境
    echo 请确保：
    echo 1. Anaconda/Miniconda 已正确安装
    echo 2. 环境名称正确
    pause
    exit /b 1
)

rem 执行第一条命令
slxpy generate

REM 执行 Python 命令
rem 执行第二条命令
call conda activate gops
REM 检查激活是否成功
if %errorlevel% neq 0 (
    echo 错误：无法激活 conda 环境
    echo 请确保：
    echo 1. Anaconda/Miniconda 已正确安装
    echo 2. 环境名称正确
    pause
    exit /b 1
)
python setup_fixbug.py build_ext --inplace
if %errorlevel% neq 0 (
    echo 构建失败
    pause
    exit /b 1
)
echo 构建完成
pause
exit
