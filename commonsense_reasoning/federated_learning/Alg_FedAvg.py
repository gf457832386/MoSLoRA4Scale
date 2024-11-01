import random
import torch
import copy

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


# 禁用 Hugging Face Hub 上传和下载
os.environ["HF_HUB_OFFLINE"] = "1"

def get_random_clients(fed_args, clientnum_perround):
    # 随机从0到fed_args.num_clients-1之间抽取clientnum_perround个不同的数字
    clients_this_round = random.sample(range(fed_args.num_clients), clientnum_perround)
    return clients_this_round


# 模型聚合
def aggregate_global_model(client_weights):
    global_dict = copy.deepcopy(client_weights[0])
    
    # 聚合所有客户端模型权重 (这里用FedAvg)
    for key in global_dict.keys():
        for i in range(1, len(client_weights)):
            global_dict[key] += client_weights[i][key]
        global_dict[key] = torch.div(global_dict[key], len(client_weights))
    
    return global_dict

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


def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])


    for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    # # global_auxiliary = None  注释的是一种动量优化的avg算法
    # for key in global_dict.keys():
    #     delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
    #     proxy_dict[key] = args.fedopt_beta1 * proxy_dict[key] + (1 - args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
    #     global_dict[key] = global_dict[key] + proxy_dict[key]

    # return global_dict, global_auxiliary  #返回全局模型参数和全局辅助参量
    return global_dict



def FedAvg(fed_args,model,global_dict,local_dict_list,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint):

    for round in tqdm(range(fed_args.num_rounds)): #轮循环
        clientnum_perround=max(int(fed_args.num_clients*fed_args.train_ratio),1)  #每轮参与训练的客户端数量
        clients_this_round = get_random_clients(fed_args, clientnum_perround)  #得到客户端id的list
        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")

        
        for client in range(fed_args.num_clients): 

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue  #继续下一个client

            print(f">> =====Client {client}:")
            #如果该客户端需要训练，则进行以下步骤
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
            training_loss[client].append(results.training_loss)  #???
            # trainer.train(resume_from_checkpoint=resume_from_checkpoint)   #执行训练!!
            # model.save_pretrained(output_dir)  #保存微调模型!!c


            # ===== Client transmits local information to server =====
            #存储新的model
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        # ===== Server aggregates the local models =====
        global_dict= global_aggregate(global_dict, local_dict_list, n_sample_list,clients_this_round)
        set_peft_model_state_dict(model, global_dict)   # Update global model更新全局模型参数

        # ===== Save the model（中间检查点） =====
        if (round+1) % fed_args.save_model_freq == 0:
            model_save_path = os.path.join(fed_args.output_dir, f"checkpoint-{round+1}")
            os.makedirs(model_save_path, exist_ok=True)
            trainer.save_model(model_save_path)            
            # trainer.save_model(os.path.join(fed_args.output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(fed_args.output_dir, "training_loss.npy"), np.array(training_loss))

    #保存最终的模型
    print("==save the final model==")
    model.save_pretrained(fed_args.output_dir)  #保存微调模型
        


   
   

    
     