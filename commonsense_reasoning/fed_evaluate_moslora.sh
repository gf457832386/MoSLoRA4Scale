#!/bin/sh

rank=4
alpha=32
gpuid=0
timestamp=$(date +"%m%d%H")

model_p_or_n=yahma/llama-7b-hf
model_path=trained_models/moslora-r$rank-a$alpha-3e4-GPU$gpuid-$timestamp
results_path=results/moslora-r$rank-a$alpha-3e4-GPU$gpuid-$timestamp


mkdir -p $model_path
mkdir -p $results_path
export CUDA_VISIBLE_DEVICES=$gpuid
  

for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u commonsense_evaluate.py \
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path
done