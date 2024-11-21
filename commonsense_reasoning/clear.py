import torch

# 假设只使用第一个 GPU（ID 为 0）
gpu_id = 0

# 获取系统中可用的 GPU 数量
available_gpus = torch.cuda.device_count()
if gpu_id >= available_gpus:
    raise ValueError(f"GPU ID {gpu_id} is not available. Only {available_gpus} GPUs detected.")

# 设置 GPU 设备
device = torch.device(f'cuda:{gpu_id}')
dummy_tensor = torch.ones((1, 1), device=device)

# 清空缓存
torch.cuda.empty_cache()

print(f"Cleared GPU {gpu_id} memory.")

