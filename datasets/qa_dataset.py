from base_dataset import NuScenesDataset
from utils import load_bev_map
import torch
import json


class NuScenesQADataset(NuScenesDataset):
    def __init__(self, dataroot, split, cfg, tokenizer, load_from_file = False, tensor_root = None) -> None:
        super.__init__(self, dataroot, split, cfg)


        self.load_from_file = load_from_file
        self.tokenizer = tokenizer
        self.json_file = dataroot + 'NuScenes_' + split + '_questions.json'
        self.json_data_root = dataroot
        
        self.split = split
        if self.load_from_file:
            self.load_from_file = load_from_file
            if not tensor_root:
                assert "A dataroot must be defined when loading a torch tensor"
            else:
                self.tensor_root = tensor_root

        try:
            with open(self.json_file, "rb") as f:
                self.qa_data = json.load(f)
        except:
            print(f"File can't be opened: check path: {self.json_file}")

        self.qa_data = self.qa_data["questions"]

        if self.tokenizer:
            self.tokenized_dataset = self.tokenize_dataset()

    def __len__(self):
        return len(self.qa_data)
    
    def tokenize_dataset(self):
        prompts = []
        for qa in self.qa_data:
            messages = [{"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                        {"role": "user", "content": "<image>" + qa["question"] },
                        {"role": "assistant", "content": qa["answer"] }]

            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    
    def __getitem__(self, index):
        question = self.qa_data[index]["question"]
        answer = self.qa_data[index][ "answer"]
        view = self.qa_data[index]["view"]
        if not self.load_from_file:
            bev = super().get_bev_datasample(self.qa_data[index]["sample_token"])
        else:
            bev = load_bev_map(self.tensor_root, self.qa_data[index]["sample_token"])

        if self.tokenizer:
            if self.split == "val":
                messages = [{"role": "user", "content": "<image>" + question }]
                return {"text":self.tokenizer.apply_chat_template(messages, tokenize=False), 
                        "bev" : bev, 
                        "sample_token": self.qa_data[index]["sample_token"],
                        "num_hop":self.qa_data[index]["num_hop"],
                        "template_type": self.qa_data[index]["template_type"],
                        "answer": answer,
                        "view": view }
            else:
                input_ids = self.tokenized_dataset["input_ids"][index]
                attention_mask = self.tokenized_dataset["attention_mask"][index]
                return {"input_ids":input_ids, "attention_mask": attention_mask, "bev":bev, "view":view}

        else:
            return {"bev": bev, "text_input": question, "text_output": answer}
    