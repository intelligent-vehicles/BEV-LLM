import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

def ddp_setup():
    mp.set_start_method('spawn', force=True)
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[INFO] Process initialized on GPU {local_rank}")

def cleanup():
    destroy_process_group()
