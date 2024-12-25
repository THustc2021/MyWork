#!/bin/bash

# 要检测的 Python 程序的名称
process_name="python"

# 循环检测 Python 进程数量
while true; do
    # 获取正在运行的 Python 进程数量
    process_count=$(pgrep -fc "$process_name")

    # 检查是否少于2个 Python 进程
    if [ "$process_count" -lt 12 ]; then
        echo "Less than 2 Python processes are running. Starting a new Python program."
        # 启动新的 Python 程序
        cd examples_final
        python joaov2_laug.py --DS MUTAG &
        python joaov2_laug.py --DS PROTEINS &
        python joaov2_laug.py --DS NCI1 &
        python joaov2_laug.py --DS COLLAB

        python joaov2_laug.py --DS REDDIT-BINARY &
        python joaov2_laug.py --DS REDDIT-MULTI-5K &
        python joaov2_laug.py --DS DD &
        python joaov2_laug.py --DS IMDB-BINARY

        # 测试带后处理的
        python joaov2_laug.py --DS MUTAG --aug basic &
        python joaov2_laug.py --DS PROTEINS --aug basic &
        python joaov2_laug.py --DS NCI1 --aug basic &
        python joaov2_laug.py --DS COLLAB --aug basic

        python joaov2_laug.py --DS REDDIT-BINARY --aug basic &
        python joaov2_laug.py --DS REDDIT-MULTI-5K --aug basic &
        python joaov2_laug.py --DS DD --aug basic &
        python joaov2_laug.py --DS IMDB-BINARY --aug basic
        break
    else
        echo "12 or more Python processes are already running. Waiting"
    fi

    # 等待一段时间再进行下一次检查
    sleep 10
done
