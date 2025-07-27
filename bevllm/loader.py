
from model.bevllm_llama import BevLLMLlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer,BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model
from utils import get_model_size_in_gb, count_parameters
from typing import Tuple

def build_model(config:dict) -> Tuple[BevLLMLlamaForCausalLM, AutoTokenizer]:

    access_token = config["access_token"]
    model_id = config["model_id"]

    model_config = LlamaConfig.from_pretrained(model_id, token=access_token, cache_dir=config["cache_dir"])
    model_config.cache_dir = config["cache_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token,  cache_dir=config["cache_dir"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    tokenizer.pad_token = tokenizer.eos_token

    #train_set = NuScenesQADataset(cache_dir + "/nuscenes/", "train", model.get_bev_config(),None,False)
    model = BevLLMLlamaForCausalLM(model_config, freeze_qformer=False)
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        target_modules=config["lora_config"]["target_modules"]
    )

    model.model = get_peft_model(model.model, peft_config)
    print(f"[INFO] Model size: {get_model_size_in_gb(model):.2f} GB")
    print(f"[INFO] Trainable params: {count_parameters(model):,}")

    return model, tokenizer

