from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import pickle
import json
import torch


#Dataset for nuscenes data: loads data from annotation file which is created by mmdetection
#getitem: returns the processed datasample that is necessary for BEVFusion

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, split, cfg) -> None:

        self.file_name = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.data_root = dataroot
        
        try:
            with open(self.file_name, "rb") as f:
                self.data = pickle.load(f)
        except:
            print(f"File can't be opened: check path: {self.file_name}")
        
        

        self.data = self.data["data_list"]

        self.preprocess_config = cfg
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        processed_sample = self.construct_data(self.append_dataroot_one_sample(sample))
        return {"bev":processed_sample, "token":sample["token"]}
 
    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.data_root + "samples/LIDAR_TOP/" + lidar_path

        return sample

#Dataset for nuscenes QA data: loads data from annotation file which is created by mmdetection
#getitem: returns the processed datasample that is necessary for BEVFusion and questions and answers of QA dataset
class NuScenesQADataset(Dataset):
    def __init__(self, dataroot, split, cfg, tokenizer, load_from_file = False, tensor_root = None) -> None:

        self.load_from_file = load_from_file
        self.tokenizer = tokenizer
        self.json_file = dataroot + 'NuScenes_' + split + '_questions.json'
        self.json_data_root = dataroot

        self.ann_file = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.ann_data_root = dataroot

        self.preprocess_config = cfg
        self.split = split
        if self.load_from_file:
            self.load_from_file = load_from_file
            if not tensor_root:
                assert "A dataroot must be defined when loading a torch tensor"
            else:
                self.tensor_root = tensor_root

        try:
            with open(self.ann_file, "rb") as f:
                self.ann_data = pickle.load(f)
            with open(self.json_file, "rb") as f:
                self.qa_data = json.load(f)
        except:
            print(f"File can't be opened: check path: {self.json_file}")

        self.qa_data = self.qa_data["questions"]
        self.ann_data = self.ann_data["data_list"]

        for data in self.ann_data:
            data["has_path"] = False

        if(self.tokenizer):
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


    def get_bev_datasample(self, token):   
        sample = [s for s in self.ann_data if s["token"] == token ]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        

        processed_sample = self.construct_data(sample)
        return processed_sample
    
    def __getitem__(self, index):
        question = self.qa_data[index]["question"]
        answer = self.qa_data[index][ "answer"]
        view = self.qa_data[index]["view"]
        if not self.load_from_file:
            bev = self.get_bev_datasample(self.qa_data[index]["sample_token"])
        else:
            bev = torch.load(self.tensor_root + "tensor_" + self.qa_data[index]["sample_token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
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
                #return {"input_ids":input_ids, "bev":bev}
                #messages = [{"role": "user", "content": "<image>" + question },
                #            {"role": "system", "content": answer }]
                #return {"text":self.tokenizer.apply_chat_template(messages, tokenize=False), "bev":bev, "view":view }

        else:
            return {"bev": bev, "text_input": question, "text_output": answer}
    

    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.ann_data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.ann_data_root + "samples/LIDAR_TOP/" + lidar_path

        sample["has_path"] = True

        return sample

class NuScenesCaptionDataset(Dataset):
    def __init__(self, dataroot, split, cfg, Tokenizer, load_from_file = False, tensor_root = None) -> None:
        self.json_file = dataroot + 'NuScenes_' + split + '_summaries.json'
        self.json_data_root = dataroot

        self.ann_file = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.ann_data_root = dataroot

        self.preprocess_config = cfg
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
            if not load_from_file:
                with open(self.ann_file, "rb") as f:
                    self.ann_data = pickle.load(f)

        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        if not load_from_file:
            self.ann_data = self.ann_data["data_list"]

            for data in self.ann_data:
                data["has_path"] = False

    def __len__(self):
        return len(self.cap_data)
    
    def get_bev_datasample(self, token):   
        sample = [s for s in self.ann_data if s["token"] == token ]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        processed_sample = self.construct_data(sample)
        return processed_sample
    
    def get_image_path(self, token):
        sample = [s for s in self.ann_data if s["token"] == token]
        sample = sample[0]
        paths = list()

        for key in sample["images"].keys():
            paths.append(sample["images"][key]["img_path"])


        return paths

    def __getitem__(self, index):
        text = self.cap_data[index]["text"]
        if not self.load_from_file:
            bev = self.get_bev_datasample(self.ann_data[index]["token"])
        else:
            bev = torch.load(self.tensor_root + "tensor_" + self.cap_data[index]["sample_token"] + ".pt")
            bev = bev.to("cpu")
            bev = bev.detach().clone()
        if self.tokenizer:
            if self.split == "val":
                return {"text": text, "bev" : bev, "sample_token": self.cap_data[index]["sample_token"], "image_paths": self.get_image_path(self.cap_data[index]["sample_token"])}
            else:
                messages = [{"role": "user", "content": " <image>  Describe the current scene!" },
                            {"role": "system", "content": text }]
                return {"text":self.tokenizer.apply_chat_template(messages, tokenize=False), "bev":bev}

        else:
            if self.split == "val":
                return {"text": text, "bev" : bev, "sample_token": self.cap_data[index]["sample_token"] }
            else:
                return {"text_output": text, "bev" : bev, "text_input": "Describe the current scene!" }
    

    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.ann_data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.ann_data_root + "samples/LIDAR_TOP/" + lidar_path

        sample["has_path"] = True

        return sample
    
class NuScenesViewCaptionDataset(Dataset):
    def __init__(self, dataroot, split, cfg=None, Tokenizer = None, special_split = None, load_from_file = False, tensor_root = None) -> None:
        
        self.json_file = dataroot + 'NuScenes_' + split + '_captions.json'
        if special_split:
            self.json_file = dataroot + 'NuScenes_' + split + '_captions_' + special_split + '.json'

        self.json_data_root = dataroot

        self.ann_file = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.ann_data_root = dataroot

        self.preprocess_config = cfg
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
                print(self.cap_data)
            if not load_from_file:
                print(self.ann_file)
                with open(self.ann_file, "rb") as f:
                    self.ann_data = pickle.load(f)

        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        if not load_from_file:
            self.ann_data = self.ann_data["data_list"]

            for data in self.ann_data:
                data["has_path"] = False


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
        if(self.tokenizer):
            self.tokenized_dataset = self.tokenize_dataset()
        #if load_from_file: 
            #self.bev_features = self.load_tensor()
    def load_tensor(self):
        tensor_bev = []

        for caption in self.cap_data:
            bev = torch.load(self.tensor_root + "tensor_" + caption["token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
            tensor_bev.append(bev)

    def __len__(self):
        return len(self.cap_data)
    
    def get_bev_datasample(self, token):   
        sample = [s for s in self.ann_data if s["token"] == token ]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        processed_sample = self.construct_data(sample)
        return processed_sample
    
    def get_image_path(self, token):
        sample = [s for s in self.ann_data if s["token"] == token]
        sample = sample[0]
        paths = list()

        for key in sample["images"].keys():
            paths.append(sample["images"][key]["img_path"])


        return paths
    
    def tokenize_dataset(self):
        prompts = []
        for caption in self.cap_data:
            orientation_string = "This is the car's " + self.string_conversion[caption["view"]] + "."

            messages = [{"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                        {"role": "user", "content": " <image>" + orientation_string + "Please describe the current scene." },
                        {"role": "assistant", "content": caption["text"] }]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    def __getitem__(self, index):
        text = self.cap_data[index]["text"]
        view = self.views[self.cap_data[index]["view"]]
        #token = self.cap_data[index]["sample_token"]
        if not self.load_from_file:
            bev = self.get_bev_datasample(self.cap_data[index]["token"])
        else:
            bev = torch.load(self.tensor_root + "tensor_" + self.cap_data[index]["token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
        if self.tokenizer:
            input_ids = self.tokenized_dataset["input_ids"][index]
            attention_mask = self.tokenized_dataset["attention_mask"][index]
            if self.split == "val":
                return {"input_ids":input_ids, "attention_mask": attention_mask, "bev":bev, "view":view, "text":text, "token":self.cap_data[index]["token"]}
            else:
                return {"input_ids":input_ids, "attention_mask": attention_mask, "bev":bev, "view":view}
        

        else:
            return{"text": text, "bev":bev, "view":view}


    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.ann_data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.ann_data_root + "samples/LIDAR_TOP/" + lidar_path

        sample["has_path"] = True

        return sample

class NuScenesGroundingDataset(Dataset):
    def __init__(self, dataroot, split, cfg=None, Tokenizer = None, special_split = None, load_from_file = False, tensor_root = None) -> None:
        
        self.json_file = dataroot +  split + '_bevtsr.json'
        if special_split:
            self.json_file = dataroot + 'NuScenes_' + split + '_ol' + special_split + '.json'

        self.json_data_root = dataroot

        self.ann_file = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.ann_data_root = dataroot

        self.preprocess_config = cfg
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
            if not load_from_file:
                print(self.ann_file)
                with open(self.ann_file, "rb") as f:
                    self.ann_data = pickle.load(f)

        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        if not load_from_file:
            self.ann_data = self.ann_data["data_list"]

            for data in self.ann_data:
                data["has_path"] = False

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
        if(self.tokenizer):
            self.tokenized_dataset = self.tokenize_dataset()
        #if load_from_file: 
            #self.bev_features = self.load_tensor()
    def load_tensor(self):
        tensor_bev = []

        for caption in self.cap_data:
            bev = torch.load(self.tensor_root + "tensor_" + caption["token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
            tensor_bev.append(bev)

    def __len__(self):
        return len(self.cap_data)
    
    def get_bev_datasample(self, token):   
        sample = [s for s in self.ann_data if s["token"] == token ]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        processed_sample = self.construct_data(sample)
        return processed_sample
    
    def get_image_path(self, token):
        sample = [s for s in self.ann_data if s["token"] == token]
        sample = sample[0]
        paths = list()

        for key in sample["images"].keys():
            paths.append(sample["images"][key]["img_path"])


        return paths
    
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
            bev = torch.load(self.tensor_root + "tensor_" + self.cap_data[index]["sample_token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
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
        

        #else:
        #    return{"text": text, "bev":bev, "view":6}


    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.ann_data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.ann_data_root + "samples/LIDAR_TOP/" + lidar_path

        sample["has_path"] = True

        return sample
    
class NuScenesLidarLLmCaptionDataset(Dataset):
    def __init__(self, dataroot, split, cfg=None, Tokenizer = None, special_split = None, load_from_file = False, tensor_root = None) -> None:
        
        self.json_file = dataroot + 'nuCaptionLidarView_' + split + '.json'

        self.json_data_root = dataroot

        self.ann_file = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.ann_data_root = dataroot

        self.preprocess_config = cfg
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
            if not load_from_file:
                print(self.ann_file)
                with open(self.ann_file, "rb") as f:
                    self.ann_data = pickle.load(f)

        except:
            print(f"File can't be opened: check path: {self.ann_file}")

        if not load_from_file:
            self.ann_data = self.ann_data["data_list"]

            for data in self.ann_data:
                data["has_path"] = False


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
        if(self.tokenizer):
            self.tokenized_dataset = self.tokenize_dataset()
        #if load_from_file: 
            #self.bev_features = self.load_tensor()
    def load_tensor(self):
        tensor_bev = []

        for caption in self.cap_data:
            bev = torch.load(self.tensor_root + "tensor_" + caption["token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
            tensor_bev.append(bev)

    def __len__(self):
        return len(self.cap_data)
    
    def get_bev_datasample(self, token):   
        sample = [s for s in self.ann_data if s["token"] == token ]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        processed_sample = self.construct_data(sample)
        return processed_sample
    
    def get_image_path(self, token):
        sample = [s for s in self.ann_data if s["token"] == token]
        sample = sample[0]
        paths = list()

        for key in sample["images"].keys():
            paths.append(sample["images"][key]["img_path"])


        return paths
    
    def tokenize_dataset(self):
        prompts = []
        for caption in self.cap_data:
            #orientation_string = "This is the car's " + self.string_conversion[caption["view"]] + "."

            messages = [{"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                        {"role": "user", "content": " <image>" + caption["question"] },
                        {"role": "assistant", "content": caption["answer"] }]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized_data = self.tokenizer(prompts, padding="longest", return_tensors="pt")
        return tokenized_data

    def __getitem__(self, index):
        answer = self.cap_data[index]["answer"]
        view = self.cap_data[index]["view"]
        question = self.cap_data[index]["question"]
        #token = self.cap_data[index]["sample_token"]
        if not self.load_from_file:
            bev = self.get_bev_datasample(self.cap_data[index]["sample_token"])
        else:
            bev = torch.load(self.tensor_root + "tensor_" + self.cap_data[index]["sample_token"] + ".pt").unsqueeze(0)
            bev = bev.to("cpu")
            bev = bev.detach().clone()
        if self.tokenizer:
            if self.split == "val":
                return{"question":question, "answer": answer, "bev": bev, "view":view, "token": self.cap_data[index]["sample_token"] }
            else:
                input_ids = self.tokenized_dataset["input_ids"][index]
                attention_mask = self.tokenized_dataset["attention_mask"][index]

                return {"input_ids":input_ids, "attention_mask": attention_mask, "bev":bev, "view":view}
        

        else:
            return{"text": answer, "bev":bev, "view":view, "token": self.cap_data[index]["sample_token"] }


    def construct_data(self, sample):
        test_pipeline = deepcopy(self.preprocess_config.test_dataloader.dataset.pipeline)
        test_pipeline = Compose(test_pipeline)
        box_type_3d, box_mode_3d = \
            get_box_type(self.preprocess_config.test_dataloader.dataset.box_type_3d)
    
        data_ = dict(
            lidar_points=dict(lidar_path=sample["lidar_points"]["lidar_path"]),
            images=sample["images"],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        
        if 'timestamp' in sample:
        # Using multi-sweeps need `timestamp`
            data_['timestamp'] = sample['timestamp']

        return test_pipeline(data_)


    def append_dataroot_one_sample(self, sample):
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.ann_data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.ann_data_root + "samples/LIDAR_TOP/" + lidar_path

        sample["has_path"] = True

        return sample