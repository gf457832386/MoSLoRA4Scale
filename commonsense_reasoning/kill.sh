#!/bin/bash

# 获取占用显存的所有进程 ID 并杀死它们
nvidia-smi | grep -oP '(?<=\s)\d+(?=\s+CUDA)' | xargs kill -9

echo "All GPU processes have been terminated."
