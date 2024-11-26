#!/bin/sh

rank=4
alpha=32
gpuid=3
timestamp=$(date +"%m%d%H")

model_p_or_n=yahma/llama-7b-hf
model_path=trained_models/moslora-r$rank-a$alpha-3e4-GPU$gpuid-$timestamp
results_path=results/moslora-r$rank-a$alpha-3e4-GPU$gpuid-$timestamp


mkdir -p $model_path
mkdir -p $results_path
export CUDA_VISIBLE_DEVICES=$gpuid

CUDA_VISIBLE_DEVICES=$gpuid python -u fed_finetune.py \
  --base_model $model_p_or_n \
  --data_path 'ft-training_set/commonsense_170k.json' \
  --mask_file "peft/src/peft/tuners/connectWeight${gpuid}.txt" \
  --output_dir $model_path \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r $rank \
  --lora_alpha $alpha \
  --use_moslora \
  --use_scalelora \
  --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]" \
  --fed_alg "FedAvg" \
  --num_clients 1000 \
  --train_ratio 0.2 \
  --data_partition_method "iid" \
  --dirichlet_alpha 0.5 \
  --num_rounds 10 \
  --save_model_freq 3\
  --results_path $results_path \
  --gpuid $gpuid\


  


for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u commonsense_evaluate.py \
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path \
    --round_num 10
done