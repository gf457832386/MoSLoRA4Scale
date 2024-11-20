import os
import sys
from typing import List
import argparse
import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
from data.partition import *
from arguments import get_args, read_from_json
import copy
import os

import numpy as np
from federated_learning import *
import math
from federated_learning.Alg_FedAvg import FedAvg
from federated_learning.Alg_FLoRA import FLoRA




"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402

print(sys.argv)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with LoRA and Federated Learning")

    # Existing parameters
    parser.add_argument("--base_model", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./lora-alpaca", help="Directory to save the output")
    parser.add_argument("--adapter_name", type=str, default="lora", help="Name of the adapter")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=256, help="Cutoff length for inputs")
    parser.add_argument("--val_set_size", type=int, default=2000, help="Validation set size")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    #parser.add_argument("--lora_target_modules", type=str, nargs="*", help="Target modules for LoRA")
    parser.add_argument("--target_modules", type=str, nargs="*", help="Target modules for LoRA")
    parser.add_argument("--use_moslora", action="store_true", help="Use MoS LoRA")
    parser.add_argument("--use_scalelora", action="store_true", help="Use ScaleLoRA")
    parser.add_argument("--mask_file", type=str, default="", help="Path to mask file for LoRA")

    # New parameters to add  FL
    parser.add_argument("--fed_alg", type=str, default="FedAvg", help="federated learning method")
    parser.add_argument("--num_clients", type=int, default=100, help="num_clients")
    parser.add_argument("--train_ratio", type=float, default=0.2, help="train_ratio")
    parser.add_argument("--data_partition_method", type=str, default="iid", help="Data partition method (e.g., iid, dirichlet)")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Dirichlet alpha for non-iid partitioning")
    parser.add_argument("--num_rounds", type=int, default=10, help="Dirichlet alpha for non-iid partitioning")
    parser.add_argument("--save_model_freq", type=int, default=1, help="save the checkpoint freq")
    parser.add_argument("--results_path", type=str, default="", help="save the checkpoint freq")
    parser.add_argument("--gpuid", type=int, default=0, help="gpuid")
    # Parse arguments and return
    return parser.parse_args()

# 在全局范围内解析参数
fed_args = parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        use_moslora: bool=False,
        use_scalelora: bool=False,  #add by phoebe
        mask_file: str="",          #add by phoebe
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        use_gradient_checkpointing: bool = False,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        #FL
        fed_alg: str="",
        num_clients: int = 10,  # number of clients for federated learning
        train_ratio: float=0.2, #ratio of number for training per round
        data_partition_method: str = "iid",  # data partitioning method
        dirichlet_alpha: float = 0.5,  # Dirichlet distribution alpha for non-iid data
        num_rounds: int=10,
        save_model_freq: int =1,
        results_path:str = "",
        gpuid: int =0
        
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"use_moslora: {use_moslora}\n" #! added
        f"use_scalelora: {use_scalelora}\n" #! added
        f"mask_file: {mask_file}\n" #! added
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"fed_alg: {fed_alg}\n"  # added
        f"num_clients: {num_clients}\n"  # added
        f"train_ratio: {train_ratio}\n"  # added
        f"data_partition_method: {data_partition_method}\n"  # added
        f"dirichlet_alpha: {dirichlet_alpha}\n"  # added
        f"num_rounds: {num_rounds}\n"  # added
        f"save_model_freq: {save_model_freq}\n"  # added
        f"result_path: {results_path}\n"  # added
        f"gpuid: {gpuid}\n"  # added
    )
  
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size  #根据批量大小和微批量大小计算累积的梯度步数

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1  #ddp：判断是否使用分布式训练，ddp 为 True 时表示在多 GPU 环境中使用分布式数据并行
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ 这部分代码决定是否启用 wandb 进行训练监控，优先使用传入的 wandb_project，如果没有则检查环境变量
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    #加载基础模型，根据 load_8bit 参数，决定是否以 8 位精度加载模型，8bit 精度有助于减少显存占用。
    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    #============加载分词器==================
    if "llama2" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt,add_eos_token=True):    
        #将输入文本转换为模型可以理解的 token id，并可选择是否添加 eos_token（结束标记）
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # result = tokenizer(
        #     prompt,
        #     truncation=True,
        #     max_length=cutoff_len,
        #     padding=False,
        #     return_tensors=None,
        # )
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,  # 填充至最大长度
            #padding = 'max_length',
            return_tensors=None  # 返回 PyTorch 张量
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point): #生成训练数据时的提示（prompt），然后对提示进行分词
        full_prompt = generate_prompt(data_point)

        # 使用填充和截断将 `input_ids` 处理为一致长度
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(
                user_prompt, 
                add_eos_token=False
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # user_prompt_len = tokenized_user_prompt["input_ids"].shape[-1]

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
            
            # 检查是否生成了 input_ids
            if not tokenized_full_prompt.get("input_ids"):
                print("Error: No input_ids generated for data_point:", data_point)
            else:
                print("Generated input_ids:", tokenized_full_prompt["input_ids"])
        return tokenized_full_prompt


    #=======模型配置===========
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_use_mixer=use_moslora, #! added  #启用MoSLoRA！！！!!!
            lora_mask_file=mask_file,  #add by phoebe
            lora_use_scale=use_scalelora,  #add by phoebe
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)  #!!!加载模型和配置
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    #数据加载 
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)


    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.计算可训练参数
    
    #这一部分在mapping,切分为了训练集train_data和测试集val_data
    if val_set_size > 0:
        #按 val_set_size 参数的比例将训练数据集划分为训练集和验证集
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        # 对训练集和验证集进行随机打乱并应用分词和生成处理函数
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else: #如果没有验证集就只处理训练集
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    # #检查input_ids的生成
    # # 迭代检查 train_data 中的每个样本，确认 `input_ids` 存在且非空
    # for i, data_point in enumerate(train_data):
    #     if "input_ids" not in data_point or len(data_point["input_ids"]) == 0:
    #         print(f"Error: No input_ids in train_data sample {i}")
    #     # else:
    #     #     print(f"Sample {i} input_ids:", data_point["input_ids"])

    # # 对 val_data 进行类似检查
    # if val_data:
    #     for i, data_point in enumerate(val_data):
    #         if "input_ids" not in data_point or len(data_point["input_ids"]) == 0:
    #             print(f"Error: No input_ids in val_data sample {i}")
    #         # else:
    #         #     print(f"Sample {i} input_ids:", data_point["input_ids"])



    #=========数据集切分到客户端clients============
    # train_dataset = load_and_cache_data(fed_args, tokenizer, data_type='train')  #data_lodar文件
    # eval_dataset = load_and_cache_data(fed_args, tokenizer, data_type='dev')
    train_dataloader_list, eval_dataloader_list, n_sample_list = partition(fed_args, train_data, val_data)
    
    w_sum = sum(n_sample_list)  #计算所有客户端的数据样本总和
    weight_list = [x / w_sum for x in n_sample_list]  #计算每个客户端的数据权重
    assert math.isclose(sum(weight_list), 1)  #验证 weight_list 的总和是否接近1。
    
    #for client_idx in range(fed_args.num_clients):


    #如果不是使用 DDP 且有多个 GPU，则启用模型并行化
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        # model.is_parallelizable = True
        # model.model_parallel = True
        model.is_parallelizable = False
        model.model_parallel = False
        
    model.config.use_cache = False


    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    #========定义全局模型与局部模型==========
    print(f"准备copyglobal_dict")
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    # print(f"准备local_dict_list")
    # local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

    #========开始联邦学习============
    training_loss = [[] for i in range(fed_args.num_clients)]  #创建一个空列表，来记录该客户端在每一轮的训练损失

    # sample_input = "This is a test sentence"
    # encoded_input = tokenizer(sample_input, return_tensors="pt")

    # if "input_ids" not in encoded_input or len(encoded_input["input_ids"]) == 0:
    #     print("Error: Tokenizer did not generate input_ids correctly.")
    # else:
    #     print(f"Encoded input: {encoded_input}")

    device = torch.device("cuda")
    set_peft_model_state_dict(model, global_dict)
    test_prompt = "Please choose the correct answer: Which color is the sky? answer1: blue, answer2: red"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=32)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Post-training output: {generated_text}")

    #联邦算法选择
    if fed_args.fed_alg == "FedAvg": 
        FedAvg(fed_args,model,global_dict,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint)
    elif fed_args.fed_alg == "FLoRA":
        FLoRA(fed_args,model,global_dict,training_loss,tokenizer,train_dataloader_list, eval_dataloader_list, n_sample_list,use_wandb, gradient_accumulation_steps,wandb_run_name,resume_from_checkpoint)

    #elif补充其他联邦学习算法







def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
