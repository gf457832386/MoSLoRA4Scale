#!/bin/sh

rank=16
alpha=32
gpuid=0   #指定使用的GPU id

model_p_or_n=yahma/llama-7b-hf  #使用的模型

model_path=trained_models/moslora-r$rank-a$alpha-3e4  #保存模型的路径
results_path=results/moslora-r$rank-a$alpha-3e4 #保存评估结果的路径

mkdir -p $model_path
mkdir -p $results_path

# MoSLoRA: --use_moslora
# ScaleLoRA:--use_scalelora

CUDA_VISIBLE_DEVICES=$gpuid python -u finetune.py \    #微调模型
  --base_model $model_p_or_n \
  --data_path 'ft-training_set/commonsense_170k.json' \  #数据路径
  --output_dir $model_path \
  --batch_size 16 \  #批大小
  --micro_batch_size 4 \ #微批大小
  --num_epochs 3 \ #训练的 epoch 数、学习率、截断长度和验证集大小
  --learning_rate 3e-4 \
  --cutoff_len 256 \  
  --val_set_size 120 \
  --adapter_name lora \ 
  --lora_r $rank \
  --lora_alpha $alpha \
  --use_moslora \
  --use_scalelora \
  --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]"  # 指定模型中要应用 LoRA 的部分


for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u commonsense_evaluate.py \   #评估模型 评估时也使用了 GPU 指定，并加载微调后的 LoRA 权重
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path
done