import random
from typing import Dict, List
import torch
import copy
import subprocess
import transformers
#from trl import SFTTrainer
from transformers import TrainerCallback # type: ignore
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
import os
import numpy as np
import math


def run_evaluation(gpuid, model_p_or_n, model_path, results_path,round_num):
    # 定义需要评估的数据集列表
    datasets = ["ARC-Easy"]
    
    for ds in datasets:
        cmd = [
           f"CUDA_VISIBLE_DEVICES={gpuid}",
            "python", "-u", "commonsense_evaluate.py",
            "--model", "LLaMA3",
            "--adapter", "LoRA",
            "--dataset", ds,
            "--batch_size", "1",
            "--base_model", model_p_or_n,
            "--lora_weights", model_path,
            "--save_dir", results_path,
            "--round_num", str(round_num)  # 确保 round_num 为字符串
            
        ]
        # 使用 subprocess 调用命令
        subprocess.run(" ".join(cmd), shell=True)

# 禁用 Hugging Face Hub 上传和下载
os.environ["HF_HUB_OFFLINE"] = "1"


# 设置随机种子
seed = 42  # 任意固定值
random.seed(seed)


def get_random_clients(fed_args, clientnum_perround,current_round):
    # 随机从0到fed_args.num_clients-1之间抽取clientnum_perround个不同的数字
    random.seed(seed + current_round)
    clients_this_round = random.sample(range(fed_args.num_clients), clientnum_perround)

    log_file_path = os.path.join(fed_args.output_dir, "client_selection_log.txt")
    # 将抽取的客户端 ID 写入日志文件
    with open(log_file_path, 'a') as f:
        f.write(f"Round {current_round}: {clients_this_round}\n")

    return clients_this_round




def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr



def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round,fed_args):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    device = next(iter(global_dict.values())).device
    print(f"Global model device: {device}")

    # 初始化全局权重的增量
    weighted_single_weights = {key: torch.zeros_like(value).to(device) for key, value in global_dict.items()}


    for k, client in enumerate(clients_this_round):
        #先提取这个客户端的所有权重
        local_client_weight = local_dict_list[client]
        for key in local_client_weight.keys():
            print(f"key: {key}")
            single_weights = local_client_weight[key].to(device)

            if fed_args.use_moslora==True:
            

                if k==0:  #第一个客户端
                    if single_weights.shape[0] == fed_args.lora_r and "A" in key and "AB" not in key:

                        #将AB和A相乘生成新的A

                        AB_key = key.replace("lora_A", "lora_AB")
                        single_weights=torch.matmul(local_client_weight[AB_key],single_weights)
                        weighted_single_weights[key] = single_weights * (sample_num_list[client] / sample_this_round)
                        
                    # 对 B 矩阵的拼接
                    if single_weights.shape[1] == fed_args.lora_r and "B" in key and "AB" not in key:
                        weighted_single_weights[key] = single_weights * (sample_num_list[client] / sample_this_round)


                else: #后续的客户端
                    # print(f"weighted_single_weights[{key}].shape: {weighted_single_weights[key].shape}")
                    # print(f"single_weights.shape: {single_weights.shape}")

                    # 对 A 矩阵的拼接
                    if single_weights.shape[0] == fed_args.lora_r and "A" in key and "AB" not in key:

                        #将AB和A相乘生成新的A

                        AB_key = key.replace("lora_A", "lora_AB")
                        single_weights=torch.matmul(local_client_weight[AB_key],single_weights)

                        weighted_single_weights[key] = torch.cat(
                            [weighted_single_weights[key],
                            single_weights * (sample_num_list[client] / sample_this_round)],
                            dim=0
                        )
                    # 对 B 矩阵的拼接
                    if single_weights.shape[1] == fed_args.lora_r and "B" in key and "AB" not in key:
                        weighted_single_weights[key] = torch.cat(
                            [weighted_single_weights[key], single_weights* (sample_num_list[client] / sample_this_round)],
                            dim=1
                        )
            else:
                if k==0:  #第一个客户端
                    if single_weights.shape[0] == fed_args.lora_r and "A" in key and "AB" not in key:

                        weighted_single_weights[key] = single_weights * (sample_num_list[client] / sample_this_round)
                        
                    # 对 B 矩阵的拼接
                    if single_weights.shape[1] == fed_args.lora_r and "B" in key and "AB" not in key:
                        weighted_single_weights[key] = single_weights * (sample_num_list[client] / sample_this_round)


                else: #后续的客户端


                    # 对 A 矩阵的拼接
                    if single_weights.shape[0] == fed_args.lora_r and "A" in key and "AB" not in key:

                        weighted_single_weights[key] = torch.cat(
                            [weighted_single_weights[key],
                            single_weights * (sample_num_list[client] / sample_this_round)],
                            dim=0
                        )
                    # 对 B 矩阵的拼接
                    if single_weights.shape[1] == fed_args.lora_r and "B" in key and "AB" not in key:
                        weighted_single_weights[key] = torch.cat(
                            [weighted_single_weights[key], single_weights* (sample_num_list[client] / sample_this_round)],
                            dim=1
                        )

                
    # 返回加权和拼接后的结果
    return weighted_single_weights
                
                
def update_base_weights(weighted_single_weights,base_weights):
    

    # 遍历每个 LoRA 权重的键 (A, B), AB已经与A相乘成为了新的A'
    for key in weighted_single_weights.keys():
        if "lora_A" in key and "AB" not in key:
            # 对应的 B 矩阵
            b_key = key.replace("lora_A", "lora_B")

            #找到对应的基础权重的键
            base_key = key.replace("lora_A.weight", "weight")

            A_matrix = weighted_single_weights[key]
            B_matrix = weighted_single_weights[b_key]

             # 计算增量权重 ΔW = A' * B
            delta_weight = torch.matmul(B_matrix,A_matrix) 

             # 检查 ΔW 和基础权重的形状是否匹配
            if delta_weight.shape != base_weights[base_key].shape:
                raise ValueError(f"Shape mismatch: ΔW {delta_weight.shape} and base weight {base_weights[base_key].shape}")
            
            print(f"base_key: {base_key}")
            print(f"base_weights[base_key]before: {base_weights[base_key]}")

            # 更新基础权重
            learning_rate=1
            base_weights[base_key] += learning_rate*delta_weight
            print(f"delta_weight mean: {delta_weight.mean().item()}, std: {delta_weight.std().item()}, max: {delta_weight.max().item()}, min: {delta_weight.min().item()}")


            print(f"base_weights[base_key]after: {base_weights[base_key]}")

    return base_weights


#生成一个秩为r的矩阵
def generate_random_matrix(r):
    # Initialize a zero matrix of size r x r
    matrix = [[0] * r for _ in range(r)]

    # Set the diagonal elements to 1
    for i in range(r):
        matrix[i][i] = 1

    # Calculate the number of 1s to add randomly to achieve 50% 0s overall
    ones_needed = (r * r) // 2 - r

    # Create a list of non-diagonal positions
    non_diag_positions = [(i, j) for i in range(r) for j in range(r) if i != j]

    # Randomly select positions to set as 1
    selected_positions = random.sample(non_diag_positions, ones_needed)
    for i, j in selected_positions:
        matrix[i][j] = 1

    return matrix

#更新每个客户端的秩为r的mask矩阵
def update_client_configs(client_ids: List[int], lora_r: int) -> Dict[int, List[List[int]]]:
    client_configs = {}
    for client_id in client_ids:
        # # 动态生成新的 LoRA mask
        # mask = [[1 if (i * client_id) % 2 == 0 else 0 for i in range(4)] for _ in range(4)]
        #生成该客户端的训练r矩阵
        mask = generate_random_matrix(lora_r)
        client_configs[client_id] = mask
    return client_configs

def FedAvg_updateW(fed_args,model,global_dict,base_weights,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint):

    # # ===== Quantize global_dict to float16 to reduce memory usage =====
    # global_dict = {k: v.half() for k, v in global_dict.items()}
    # 初始化全局权重

    for round in tqdm(range(fed_args.num_rounds)): #轮循环
        clientnum_perround=max(int(fed_args.num_clients*fed_args.train_ratio),1)  #每轮参与训练的客户端数量
        clients_this_round = get_random_clients(fed_args, clientnum_perround,round+1)  #得到客户端id的list
        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        
        #更新这轮每个client的config
        round_configs = update_client_configs(clients_this_round,fed_args.lora_r)

        # #更新基础大模型
        # for key, value in base_weights.items():
        #     if key in model.state_dict():
        #         model.state_dict()[key].data.copy_(value)

        # 每轮初始化的局部模型列表，仅包含参与训练的客户端
        local_dict_list = [None] * fed_args.num_clients

        for client in tqdm(range(fed_args.num_clients)): 

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue  #继续下一个client



            print(f">> =====Client {client}:")

            #更换model的config里的mask
            model.config.lora_mask_client=round_configs[client]
            
            #如果该客户端需要训练，则进行以下步骤,将全局模型参数同步到局部模型里
            # set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
            set_peft_model_state_dict(model, base_weights)  
            # 检查提取的参数是否为空
            peft_state_dict = get_peft_model_state_dict(model)
            print("peft_state_dict.keys():")
            print(peft_state_dict.keys())
            if not peft_state_dict:
                print(f"Error: Failed to extract PEFT model state dictionary for client {client}")


            new_lr = cosine_learning_rate(round, fed_args.num_rounds, fed_args.learning_rate, 1e-6)      # manually schedule the learning rate


            # ===== Train local model on the client side =====
            trainer = transformers.Trainer(  #训练器配置
                model=model,
                train_dataset=train_dataloader_list[client].dataset,
                eval_dataset=eval_dataloader_list[client].dataset,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=fed_args.micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=fed_args.num_epochs,
                    learning_rate=new_lr,
                    fp16=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    evaluation_strategy="steps" if fed_args.val_set_size > 0 else "no",
                    save_strategy="steps",
                    eval_steps=200 if fed_args.val_set_size > 0 else None,
                    save_steps=200,
                    output_dir=fed_args.output_dir,
                    save_total_limit=3,
                    load_best_model_at_end=True if fed_args.val_set_size > 0 else False,
                    ddp_find_unused_parameters=False,
                    group_by_length=False,
                    report_to="wandb" if use_wandb else None,
                    # debug="underflow_overflow",
                    # report_to=None,
                    run_name=wandb_run_name if use_wandb else None,
                    #remove_unused_columns=False  # 移除未使用的列
                ),
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )

            print(f"Number of samples in train_dataset for client {client}: {len(train_dataloader_list[client].dataset)}")
            print(f"Number of samples in eval_dataset for client {client}: {len(eval_dataloader_list[client].dataset)}")




            results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            training_loss[client].append(results.training_loss)  
            

            

            # ===== Client transmits local information to server =====
            # 仅存储本轮训练后的模型差异部分
            # local_dict_list[client] = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

            # #存储新的model
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!


        # ===== Server aggregates the local models =====
        #返回拼接的多个A矩阵和B矩阵
        weighted_single_weights= global_aggregate(global_dict, local_dict_list, n_sample_list, clients_this_round,fed_args)
        #更新基础权重
        base_weights= update_base_weights(weighted_single_weights,base_weights)
        

        print("Global model parameters after aggregation:") #每轮训练完检测下global_dict是不是有更新
        for key, value in global_dict.items():
            print(f"{key}: Mean={value.mean().item()}, Std={value.std().item()}")
        
     

        print("Running post-training validation...")
        device = torch.device("cuda")
        set_peft_model_state_dict(model, base_weights)
        test_prompt = "Please choose the correct answer: Which color is the sky? answer1: blue, answer2: red"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=32)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Post-training output: {generated_text}")



        # ===== Save the model（中间检查点） =====
        if (round+1) % fed_args.save_model_freq == 0:
            model_save_path = os.path.join(fed_args.output_dir, f"checkpoint-{round+1}")
            os.makedirs(model_save_path, exist_ok=True)
            trainer.save_model(model_save_path)            
            # trainer.save_model(os.path.join(fed_args.output_dir, f"checkpoint-{round+1}"))
        
        # 保存训练损失
      
        print(f"Training loss: {training_loss}")
        np.save(os.path.join(fed_args.output_dir, "training_loss.npy"), np.array(training_loss))



         # 每训练 3 轮执行一次评估
        if (round + 1) % 1 == 0:
            # set_peft_model_state_dict(model, global_dict)
            set_peft_model_state_dict(model, base_weights)
            model.save_pretrained(fed_args.output_dir)  #保存微调模型
            print(f"== Performing evaluation after round {round + 1} ==")
            run_evaluation(fed_args.gpuid, fed_args.base_model, fed_args.output_dir, fed_args.results_path,round + 1)





    #保存最终的模型
    print("==save the final model==")
    model.save_pretrained(fed_args.output_dir)  #保存微调模型
        


   
   

    
     