import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from dataset import NuScenesDataset, NuScenesCaptionDataset, NuScenesQADataset, NuScenesViewCaptionDataset, NuScenesLidarLLmCaptionDataset
from torch.distributed import init_process_group, destroy_process_group
from trainer import Trainer
from mmengine.dataset import pseudo_collate
from transformers import LlamaConfig, AutoTokenizer,BitsAndBytesConfig 
from bevllm.model.bevllm_llama import BevLLMLlamaForCausalLM 
from argparse import ArgumentParser
from peft import LoraConfig,  get_peft_model
from dotenv import load_dotenv


load_dotenv()
mp.set_start_method('spawn', force=True)


def get_model_size_in_gb(model):
    """
    Calculate the size of a PyTorch model in gigabytes.c

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    float: The size of the model in gigabytes.
    """
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Get the size of each parameter in bytes
    param_size_bytes = 4  # for float32

    # Calculate total size in bytes
    total_size_bytes = total_params * param_size_bytes

    # Convert bytes to gigabytes (1 GB = 1e9 bytes)
    total_size_gb = total_size_bytes / 1e9

    return total_size_gb

def count_parameters(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def collate_fn_no_loading(batch):
    bev = [sample["bev"] for sample in batch]
    view = [sample["view"] for sample in batch]
    #text = [sample["text"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]

    #padded_texts = pad_sequence(torch.tensor(text), batch_first=True, padding_value=128001)

    #return {"text":text, "bev": torch.cat(bev, dim=0), "view": view}
    return {"input_ids": torch.stack(input_ids, dim=0), "bev": pseudo_collate(bev), "view": view}

def collate_fn(batch):
    bev = [sample["bev"] for sample in batch]
    view = [sample["view"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    return {"input_ids": torch.stack(input_ids, dim=0),"attention_mask": torch.stack(attention_masks, dim=0), "bev": torch.cat(bev, dim=0), "view": view}

def ddp_setup():
    init_process_group(backend="nccl")
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank % num_gpus)

def cleanup():
    dist.destroy_process_group()

def load_train_objs(cache_dir):
    access_token = os.getenv("ACCESS_TOKEN")
    model_id = os.getenv("MODEL_ID")

    config = LlamaConfig.from_pretrained(model_id, token=access_token, cache_dir=cache_dir)
    config.cache_dir = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token,  cache_dir=cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    tokenizer.pad_token = tokenizer.eos_token

    #train_set = NuScenesQADataset(cache_dir + "/nuscenes/", "train", model.get_bev_config(),None,False)
    model = BevLLMLlamaForCausalLM(config,freeze_qformer=False)
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    train_set = NuScenesViewCaptionDataset("/anvme/workspace/v103fe15-bev_features/data/sets/nuscenes/", 
                                  "train", 
                                  None, 
                                  tokenizer, 
                                  load_from_file=True, 
                                  tensor_root="/anvme/workspace/v103fe15-bev_features/bev_features/train/")
    
    model.model = get_peft_model(model.model, peft_config)
    optimizer = optim.AdamW(model.parameters(), 1e-4, betas=(0.9, 0.999), weight_decay=0.05)
    print(get_model_size_in_gb(model))
    print(count_parameters(model))

    return train_set, model, optimizer, tokenizer


def prepare_dataloader(dataset: NuScenesViewCaptionDataset, batch_size: int):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      pin_memory=True, 
                      collate_fn=collate_fn, 
                      num_workers=0,
                      sampler=DistributedSampler(dataset))               

def main(save_every: int, total_epochs: int, batch_size: int, cache_dir: str,  snapshot_path: str = "snapshot_bevllm_llama_8B_nuView.pt"):
    ddp_setup()
    torch.cuda.empty_cache()
    dataset, model, optimizer,tokenizer = load_train_objs(cache_dir)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, tokenizer)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--cache_dir', type=str, required=True,help = 'save llama model' ) 
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    

    main(args.save_every, args.total_epochs, args.batch_size, args.cache_dir)
