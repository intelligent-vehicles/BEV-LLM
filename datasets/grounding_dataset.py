import torch
import json
import pickle
from base_dataset import NuScenesDataset
from utils import load_bev_map


class NuScenesGroundingDataset(NuScenesDataset):
    def __init__(self, dataroot, split, cfg=None, Tokenizer = None, load_from_file = False, tensor_root = None) -> None:
        super.__init__(self, dataroot, split, cfg)
        self.json_file = dataroot +  split + '_bevtsr.json'
        self.json_data_root = dataroot
        self.split = split
        self.tokenizer = Tokenizer
        self.load_from_file = load_from_file

        if load_from_file:
            if not tensor_root:
                assert "A dataroot must be defined when loading a torch tensor"
            else:
                self.tensor_root = tensor_root

        try:        
            with open(self.json_file, "rb") as f:
                self.cap_data = json.load(f)
        except:
            print(f"File can't be opened: check path: {self.cap_data}")

        self.questions = [
                "How many distinct types of vehicles can you identify in the scene?",
                "What stationary urban infrastructure elements are visible in the scene?",
                "In which directions do you observe moving objects, and what types are they?",
                "How many different categories of movable objects can you spot in the environment?",
                "Are there any visible indicators of public transportation in the scene? If so, what are they?",
                "What types of lighting sources can you identify in the nighttime setting?",
                "How would you describe the distribution of parked vehicles around the observation point?",
                "Are there any visible elements that suggest the area's primary use (e.g., residential, commercial)?",
                "What types of potential interactions between different elements in the scene can you identify?",
                "How would you characterize the overall level of activity or motion in the scene?"
            ]

        self.views = {"CAM_FRONT": 0,
                      "CAM_FRONT_LEFT": 1,
                      "CAM_BACK_LEFT": 2,
                      "CAM_BACK": 3,
                      "CAM_BACK_RIGHT": 4,
                      "CAM_FRONT_RIGHT": 5,   }
        
        self.string_conversion = {"CAM_BACK": "back view", 
                            "CAM_BACK_LEFT":" back left view",
                            "CAM_BACK_RIGHT": "back right view",
                            "CAM_FRONT": "front view",
                            "CAM_FRONT_RIGHT": "front right view",
                            "CAM_FRONT_LEFT": "front left view"}
        if self.tokenizer:
            self.tokenized_dataset = self.tokenize_dataset()
   
    def __len__(self):
        return len(self.cap_data)
       
    def randomize_question(self):
        import random
        return self.questions[random.randint(0, 9)]

    def tokenize_dataset(self):
        prompts = []
        for caption in self.cap_data:
            #orientation_string = "This is the car's " + self.string_conversion[caption["view"]] + "."
            if "question" not in caption.keys():
                caption["question"] = self.randomize_question()


            messages = [{"role": "assistant", "content": "You are an assistant for describing 3D outdoor scene out of the perspective of a car. Please focus on different objects that are present in the current scene"},
                        {"role": "user", "content": " <image>" + caption["question"]},
                        {"role": "assistant", "content": caption["text"] }]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    def __getitem__(self, index):
        text = self.cap_data[index]["text"]
        #view = self.views[self.cap_data[index]["view"]]
        #token = self.cap_data[index]["sample_token"]
        if not self.load_from_file:
            bev = self.get_bev_datasample(self.cap_data[index]["sample_token"])
        else:
            bev = load_bev_map(self.tensor_root, self.cap_data[index]["sample_token"])
        if self.tokenizer:
            if self.split == "val":
                input_ids = self.tokenized_dataset["input_ids"][index]
                attention_mask = self.tokenized_dataset["attention_mask"][index]
                return{"input_ids":input_ids, "attention_mask": attention_mask, "bev": bev, "view":6, "text":text, "token": self.cap_data[index]["sample_token"] }
            else:
                input_ids = self.tokenized_dataset["input_ids"][index]
                attention_mask = self.tokenized_dataset["attention_mask"][index]

                return {"input_ids":input_ids, "attention_mask": attention_mask, "bev":bev, "view":6}
        else: 
            if self.split == "val":
                return{"text": text, "bev": bev, "view":6, "token": self.cap_data[index]["sample_token"] }
            else:
                return {"text": text, "bev":bev, "view":6}
        