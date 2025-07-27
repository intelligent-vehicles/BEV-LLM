import torch
import json
from base_dataset import NuScenesDataset
from utils import load_bev_map

class NuScenesLidarLLmCaptionDataset(NuScenesDataset):
    """
    Dataset class for handling LiDAR-based BEV representations paired with natural language 
    questions and answers, targeting instruction-tuned LLMs in autonomous driving contexts.

    Each sample includes a BEV map (generated from LiDAR), a natural language question-answer pair,
    and associated metadata like the camera view and sample token.

    Args:
        dataroot (str): Root directory containing the dataset and caption files.
        split (str): Dataset split (e.g., 'train', 'val').
        cfg (dict, optional): Optional configuration dictionary (currently unused).
        Tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer for processing the question-answer data.
        special_split (str, optional): Optional override for split name (currently unused).
        load_from_file (bool, optional): Whether to load precomputed BEV tensors from disk.
        tensor_root (str, optional): Path to the directory with precomputed BEV `.pt` files (required if `load_from_file` is True).
    """

    def __init__(self, dataroot, split, cfg=None, Tokenizer=None, load_from_file=False, tensor_root=None) -> None:
        # Path to the JSON file with QA data
        self.json_file = dataroot + 'nuCaptionLidarView_' + split + '.json'
        self.json_data_root = dataroot
        self.split = split
        self.tokenizer = Tokenizer
        self.load_from_file = load_from_file

        # If loading precomputed tensors, the root path must be provided
        if load_from_file:
            if not tensor_root:
                assert "A dataroot must be defined when loading a torch tensor"
            else:
                self.tensor_root = tensor_root

        # Load question-answer data from JSON
        try:        
            with open(self.json_file, "rb") as f:
                self.cap_data = json.load(f)
        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        # View mapping for consistency and potential downstream use
        self.views = {
            "CAM_FRONT": 0,
            "CAM_FRONT_LEFT": 1,
            "CAM_BACK_LEFT": 2,
            "CAM_BACK": 3,
            "CAM_BACK_RIGHT": 4,
            "CAM_FRONT_RIGHT": 5
        }

        # Human-readable names for views (not used directly in this class but useful for prompts)
        self.string_conversion = {
            "CAM_BACK": "back view", 
            "CAM_BACK_LEFT": "back left view",
            "CAM_BACK_RIGHT": "back right view",
            "CAM_FRONT": "front view",
            "CAM_FRONT_RIGHT": "front right view",
            "CAM_FRONT_LEFT": "front left view"
        }

        # Pre-tokenize the data if a tokenizer is provided
        if self.tokenizer:
            self.tokenized_dataset = self.tokenize_dataset()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.cap_data)

    def tokenize_dataset(self):
        """
        Applies a chat-style tokenizer to convert QA pairs into tokenized LLM prompts.

        Returns:
            dict: A dictionary containing tokenized inputs (`input_ids`, `attention_mask`).
        """
        prompts = []
        for caption in self.cap_data:
            # Construct instruction-following prompt using chat template
            messages = [
                {"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                {"role": "user", "content": " <image>" + caption["question"]},
                {"role": "assistant", "content": caption["answer"]}
            ]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        # Tokenize all prompts using the provided tokenizer
        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Returns:
            dict: Contains question, answer, BEV tensor, view name, and optionally tokenized inputs.
        """
        # Extract components from the annotation
        answer = self.cap_data[index]["answer"]
        question = self.cap_data[index]["question"]
        view = self.cap_data[index]["view"]
        token = self.cap_data[index]["sample_token"]

        # Load BEV tensor, either precomputer or raw
        if not self.load_from_file:
            bev = self.get_bev_datasample(token)
        else:
            bev = load_bev_map(self.tensor_root, token)

        if self.tokenizer:
            if self.split == "val":
                # Return full QA pair for evaluation
                return {
                    "question": question,
                    "answer": answer,
                    "bev": bev,
                    "view": view,
                    "token": token
                }
            else:
                # Return tokenized inputs for training
                input_ids = self.tokenized_dataset["input_ids"][index]
                attention_mask = self.tokenized_dataset["attention_mask"][index]

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bev": bev,
                    "view": view
                }
        else:
            # Fallback: return plain text and BEV for debugging or baseline training
            return {
                "text": answer,
                "bev": bev,
                "view": view,
                "token": token
            }
