import random
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
from tqdm import tqdm
import os
import numpy as np
import math
from torch.nn.functional import normalize

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

def get_random_clients(fed_args, clientnum_perround,current_round):
    # 随机从0到fed_args.num_clients-1之间抽取clientnum_perround个不同的数字
    random.seed()
    clients_this_round = random.sample(range(fed_args.num_clients), clientnum_perround)


    
    log_file_path = os.path.join(fed_args.output_dir, "client_selection_log_FLoRA.txt")
    # 将抽取的客户端 ID 写入日志文件
    with open(log_file_path, 'a') as f:
        f.write(f"Round {current_round}: {clients_this_round}\n")

    return clients_this_round


def global_aggregate(global_dict, local_dict_list, n_sample_list, clients_this_round,fed_args):
    heter=False  #先默认不异构

    # Normalize weights based on number of samples
    weights_array = normalize(
        torch.tensor([n_sample_list[client_id] for client_id in clients_this_round], dtype=torch.float32),
        p=1, dim=0
    )
    print("Weights:", weights_array)

    # Initialize aggregated weights
    weighted_single_weights = None

    # Perform aggregation
    for k, client_id in enumerate(clients_this_round):
        single_weights = local_dict_list[client_id]
        
        x=0
        if k == 0:
            weighted_single_weights = single_weights
            for key in weighted_single_weights.keys():
                #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                #print(weighted_single_weights[key].shape)
                if heter:
                    x += 1
                    # if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                    #     weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                else:
                    if weighted_single_weights[key].shape[0] == fed_args.lora_r:
                        weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

        else:
            for key in single_weights.keys():

                if 'lora_AB' not in key: #对于A、B矩阵分别进行拼接
                    if heter:
                        x += 1
                        # if single_weights[key].shape[0] == local_ranks[client_id]:
                        #     new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                        #     weighted_single_weights[key] = torch.cat(new, dim=0)
                    else:
                        if single_weights[key].shape[0] == fed_args.lora_r:
                            new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                            weighted_single_weights[key] = torch.cat(new, dim=0)
                    
                    if heter:
                        print("heter")
                        # if single_weights[key].shape[1] == local_ranks[client_id]:
                        #     new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                        #     weighted_single_weights[key] = torch.cat(new, dim=1)
                    else:
                        if single_weights[key].shape[1] == fed_args.lora_r:
                            new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                            for i, tensor in enumerate(new):
                                print(f"Tensor {i} shape: {tensor.shape}")
                            weighted_single_weights[key] = torch.cat(new, dim=1)
                else: #对AB矩阵进行加权
                        weighted_single_weights[key] += single_weights[key] * weights_array[k]

    # # Save aggregated weights
    # torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))

    # Update global model
    global_dict=weighted_single_weights
    # set_peft_model_state_dict(global_dict, weighted_single_weights)
    print(global_dict)
    return global_dict
 


def FLoRA(fed_args,model,global_dict,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint):



    for round in tqdm(range(fed_args.num_rounds)): #轮循环
        clientnum_perround=max(int(fed_args.num_clients*fed_args.train_ratio),1)  #每轮参与训练的客户端数量
        clients_this_round = get_random_clients(fed_args, clientnum_perround,round+1)  #得到客户端id的list
        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")

        # 每轮初始化的局部模型列表，仅包含参与训练的客户端
        local_dict_list = [None] * fed_args.num_clients
    
    
        for client in tqdm(range(fed_args.num_clients)): 

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue  #继续下一个client
                
            print(f">> =====Client {client}:")
            #如果该客户端需要训练，则进行以下步骤,将全局模型参数同步到局部模型
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model


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
        global_dict= global_aggregate(global_dict, local_dict_list, n_sample_list, clients_this_round,fed_args)
        # set_peft_model_state_dict(model, global_dict)   # Update global model更新全局模型参数

        # print("Global model parameters after aggregation:") #每轮训练完检测下global_dict是不是有更新
        # for key, value in global_dict.items():
        #     print(f"{key}: Mean={value.mean().item()}, Std={value.std().item()}")
    

        print("Running post-training validation...")
        device = torch.device("cuda")
        set_peft_model_state_dict(model, global_dict)
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
        


  












