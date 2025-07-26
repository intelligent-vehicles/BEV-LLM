from base_dataset import NuScenesDataset
from utils import load_bev_map
import torch
import json

class NuScenesViewCaptionDataset(NuScenesDataset):
    """
    Dataset class for handling NuScenes data paired with view-specific captions.
    Each sample includes a BEV map, camera view, and a natural language description of the scene.

    This dataset is intended for multimodal training tasks, such as image-language models in the context
    of autonomous driving.

    Args:
        dataroot (str): Root directory containing the dataset and caption files.
        split (str): Dataset split to use, e.g., 'train', 'val'.
        cfg (dict, optional): Optional configuration dictionary (currently unused).
        Tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer for processing text data.
        load_from_file (bool, optional): Whether to load precomputed BEV tensors from disk.
        tensor_root (str, optional): Directory containing precomputed BEV tensors, required if `load_from_file` is True.
    """
    
    def __init__(self, dataroot, split, cfg=None, Tokenizer=None, load_from_file=False, tensor_root=None) -> None:
        # Path to the JSON caption file
        super.__init__(self, dataroot, split, cfg)
        self.json_file = dataroot + 'NuScenes_' + split + '_captions.json'
        self.json_data_root = dataroot
        self.split = split
        self.tokenizer = Tokenizer
        self.load_from_file = load_from_file

        # If loading BEV tensors from file, tensor_root must be provided
        if load_from_file:
            if not tensor_root:
                assert "A dataroot must be defined when loading a torch tensor"
            else:
                self.tensor_root = tensor_root

        # Load the JSON caption data
        try:        
            with open(self.json_file, "rb") as f:
                self.cap_data = json.load(f)
                print(self.cap_data)
        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        # Mapping from camera view names to integer indices
        self.views = {
            "CAM_FRONT": 0,
            "CAM_FRONT_LEFT": 1,
            "CAM_BACK_LEFT": 2,
            "CAM_BACK": 3,
            "CAM_BACK_RIGHT": 4,
            "CAM_FRONT_RIGHT": 5,
        }

        # Human-readable view descriptions for prompt generation
        self.string_conversion = {
            "CAM_BACK": "back view", 
            "CAM_BACK_LEFT": "back left view",
            "CAM_BACK_RIGHT": "back right view",
            "CAM_FRONT": "front view",
            "CAM_FRONT_RIGHT": "front right view",
            "CAM_FRONT_LEFT": "front left view"
        }

        # Pre-tokenize the captions if a tokenizer is provided
        if self.tokenizer:
            self.tokenized_dataset = self.tokenize_dataset()

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.cap_data)

    def tokenize_dataset(self):
        """
        Tokenizes the dataset using the provided tokenizer and a chat-style template.

        Returns:
            dict: A dictionary containing tokenized inputs (input_ids and attention_mask).
        """
        prompts = []
        for caption in self.cap_data:
            # Add camera orientation as context
            orientation_string = "This is the car's " + self.string_conversion[caption["view"]] + "."

            # Construct chat message format for instruction-tuned LLMs
            messages = [
                {"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                {"role": "user", "content": " <image>" + orientation_string + "Please describe the current scene."},
                {"role": "assistant", "content": caption["text"]}
            ]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        # Tokenize all prompts and return as PyTorch tensors
        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Returns:
            dict: Contains text, BEV map, camera view index, and optionally tokenized inputs and sample token.
        """
        text = self.cap_data[index]["text"]
        view = self.views[self.cap_data[index]["view"]]

        # Load the BEV map either from memory or from a precomputed tensor
        if not self.load_from_file:
            bev = super().get_bev_datasample(self.cap_data[index]["token"])
        else:
            bev = load_bev_map(self.tensor_root, self.cap_data[index]["token"])

        # Return both tokenized inputs and BEV data if tokenizer is used
        if self.tokenizer:
            input_ids = self.tokenized_dataset["input_ids"][index]
            attention_mask = self.tokenized_dataset["attention_mask"][index]
            if self.split == "val":
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bev": bev,
                    "view": view,
                    "text": text,
                    "token": self.cap_data[index]["token"]
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bev": bev,
                    "view": view
                }
        else:
            # Return only raw data if no tokenizer is used
            return {
                "text": text,
                "bev": bev,
                "view": view
            }
