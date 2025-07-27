import argparse
import yaml
import os
from datasets.loader import load_dataset, prepare_dataloader
from bevllm.loader import build_model
from training.ddp_setup import ddp_setup, cleanup
from training.trainer import Trainer
import torch.optim as optim


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def merge_args_with_config(config, args):
    # Override YAML config with non-None CLI args
    def override(section, key, arg_value):
        if arg_value is not None:
            config[section][key] = arg_value

    # Training
    override("training", "total_epochs", args.total_epochs)
    override("training", "save_every", args.save_every)
    override("training", "batch_size", args.batch_size)
    override("training", "learning_rate", args.learning_rate)

    # General
    override("general", "cache_dir", args.cache_dir)
    override("general", "snapshot_path", args.snapshot_path)
    override("general", "mode", args.mode)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Unified BEV-LLM Trainer/Evaluator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")

    # CLI overrides
    parser.add_argument("--mode", type=str, choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--total_epochs", type=int, help="Total training epochs")
    parser.add_argument("--save_every", type=int, help="Save checkpoint every N epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--cache_dir", type=str, help="Model cache directory")
    parser.add_argument("--snapshot_path", type=str, help="Checkpoint file path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    if config["general"]["mode"] == "train":
        ddp_setup()
        dataset = load_dataset(config["dataset"])
        model, tokenizer = build_model(config["model"])
        train_data = prepare_dataloader(dataset, config["training"]["batch_size"])
        optimizer = optim.AdamW(model.parameters(),
                                config["training"]["learning_rate"],
                                config["training"]["betas"], 
                                config["training"]["weight_decay"])
        
        trainer = Trainer(model, 
                          train_data, 
                          optimizer, 
                          config["training"]["save_every"],
                          config["general"]["save_every"],
                          tokenizer)
        trainer.train(config["training"]["total_epochs"])
        cleanup
    
    elif config["general"]["mode"] == "eval":
        print (config)
