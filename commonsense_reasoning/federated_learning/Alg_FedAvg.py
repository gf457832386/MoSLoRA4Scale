import random
from typing import Dict, List
import torch
import copy
import subprocess
import transformers
#from trl import SFTTrainer
from transformers import TrainerCallback
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import get_peft_model_state_dict, set_peft_model_state_dict
#from peft import get_peft_model_state_dict, set_peft_model_state_dict
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


# # 模型聚合
# def aggregate_global_model(client_weights):
#     global_dict = copy.deepcopy(client_weights[0])
    
#     # 聚合所有客户端模型权重 (这里用FedAvg)
#     for key in global_dict.keys():
#         for i in range(1, len(client_weights)):
#             global_dict[key] += client_weights[i][key]
#         global_dict[key] = torch.div(global_dict[key], len(client_weights))
    
#     return global_dict

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


# def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
#     sample_this_round = sum([sample_num_list[client] for client in clients_this_round])



#     for key in global_dict.keys():
#         global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
#     # # global_auxiliary = None  注释的是一种动量优化的avg算法
#     # for key in global_dict.keys():
#     #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
#     #     proxy_dict[key] = args.fedopt_beta1 * proxy_dict[key] + (1 - args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
#     #     global_dict[key] = global_dict[key] + proxy_dict[key]

#     return global_dict

def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    device = next(iter(global_dict.values())).device
    print(f"Global model device: {device}")
    

    aggregated_diff = {key: torch.zeros_like(value).to(device) for key, value in global_dict.items()}
    
    # 对差异部分进行加权聚合
    for client in clients_this_round:
        for key in global_dict.keys():
            local_tensor = local_dict_list[client][key].to(device)
            diff = local_tensor - global_dict[key]
            aggregated_diff[key] += diff * sample_num_list[client] / sample_this_round
    
    # 应用聚合后的差异到全局模型
    for key in global_dict.keys():
        global_dict[key] += aggregated_diff[key]

    return global_dict


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

def FedAvg(fed_args,model,global_dict,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint):

    # # ===== Quantize global_dict to float16 to reduce memory usage =====
    # global_dict = {k: v.half() for k, v in global_dict.items()}

    for round in tqdm(range(fed_args.num_rounds)): #轮循环
        clientnum_perround=max(int(fed_args.num_clients*fed_args.train_ratio),1)  #每轮参与训练的客户端数量
        clients_this_round = get_random_clients(fed_args, clientnum_perround,round+1)  #得到客户端id的list
        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        
        #更新这轮每个client的config
        round_configs = update_client_configs(clients_this_round,fed_args.lora_r)

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
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
            # 检查提取的参数是否为空
            peft_state_dict = get_peft_model_state_dict(model)
            # print("peft_state_dict.keys():")
            # print(peft_state_dict.keys())
            if not peft_state_dict:
                print(f"Error: Failed to extract PEFT model state dictionary for client {client}")

            # # 打印部分参数的信息
            # for k, v in peft_state_dict.items():
            #     print(f"Extracted parameter {k}: Mean={v.mean().item()}, Std={v.std().item()}")
            #     break  # 仅打印第一个参数，避免输出过多信息


            # sub_dataset = get_dataset_this_round(train_dataloader_list[client], round, args, script_args)      # get the required sub-dataset for this round
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, fed_args.learning_rate, 1e-6)      # manually schedule the learning rate
            # training_args = get_training_args(args, new_lr)

            # for client_id, dataloader in enumerate(train_dataloader_list):
            #     if len(dataloader.dataset) == 0:
            #         print(f"Warning: Client {client_id}'s dataset is empty!")


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
            # trainer.train(resume_from_checkpoint=resume_from_checkpoint)   #执行训练!!
            # model.save_pretrained(output_dir)  #保存微调模型!!c

            

            # ===== Client transmits local information to server =====
            # 仅存储本轮训练后的模型差异部分
            # local_dict_list[client] = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            # local_dict_list[client] = {k: v.detach().to(torch.float32).cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            local_dict_list[client] = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

            # #存储新的model
            # local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!


        # ===== Server aggregates the local models =====
        global_dict= global_aggregate(global_dict, local_dict_list, n_sample_list, clients_this_round)
        # set_peft_model_state_dict(model, global_dict)   # Update global model更新全局模型参数

        # print("Global model parameters after aggregation:") #每轮训练完检测下global_dict是不是有更新
        # for key, value in global_dict.items():
        #     print(f"{key}: Mean={value.mean().item()}, Std={value.std().item()}")
        
     

        # print("Running post-training validation...")
        # device = torch.device("cuda")
        # set_peft_model_state_dict(model, global_dict)
        # test_prompt = "Please choose the correct answer: Which color is the sky? answer1: blue, answer2: red"
        # inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        # outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=32)
        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Post-training output: {generated_text}")



        

        # ===== Save the model（中间检查点） =====
        if (round+1) % fed_args.save_model_freq == 0:
            model_save_path = os.path.join(fed_args.output_dir, f"checkpoint-{round+1}")
            os.makedirs(model_save_path, exist_ok=True)
            trainer.save_model(model_save_path)            
            # trainer.save_model(os.path.join(fed_args.output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(fed_args.output_dir, "training_loss.npy"), np.array(training_loss))





         # 每训练 3 轮执行一次评估
        if (round + 1) % 3 == 0:
            set_peft_model_state_dict(model, global_dict)
            model.save_pretrained(fed_args.output_dir)  #保存微调模型
            print(f"== Performing evaluation after round {round + 1} ==")
            run_evaluation(fed_args.gpuid, fed_args.base_model, fed_args.output_dir, fed_args.results_path,round + 1)





    #保存最终的模型
    print("==save the final model==")
    model.save_pretrained(fed_args.output_dir)  #保存微调模型
        


   
   

    
     