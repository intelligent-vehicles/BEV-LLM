from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import pickle
import json
import torch

class NuScenesDataset(Dataset):
    def __init__(self, dataroot:str, split:str, cfg:str) -> None:
        """
        Base dataset-class for the following datasets

        dataroot: Path to dataset pickle file
        split: Dataset split (train, val, test)
        cfg: Path to datapipline cfg of mmdetection3d
        
        """
        self.file_name = dataroot + 'nuscenes_infos_' + split + '.pkl'
        self.data_root = dataroot
        
        

        try:
            with open(self.file_name, "rb") as f:
                self.data = pickle.load(f)
        except:
            print(f"File can't be opened: check path: {self.file_name}")
        

        self.data = self.data["data_list"]
        self.preprocess_config = cfg
        self.token_to_sample = {sample["token"]: sample for sample in self.data}


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int) -> dict:
        """
        Function for getting a datasample based on the index

        index: Index of datasample

        returns: Dict containing processed sample and sample token
        """

        sample = self.data[index]
        processed_sample = self.get_bev_datasample(sample["token"])
        return {"bev":processed_sample, "token":sample["token"]}
    

    def get_bev_datasample(self, token):

        """
        Function for getting the data sample based on the sample token

        token: Sample token for a data sample

        returns: processed data sample

        """

        sample = self.token_to_sample[token]
        sample = sample[0]

        if sample["has_path"] == False:
            sample = self.append_dataroot_one_sample(sample)

        processed_sample = self.construct_data(sample)
        return processed_sample

    def construct_data(self, sample:dict):
        """
            Using mmdet3d pipline for preprocessing datasample

            sample: One nuScenes sample in mmdet3d specific format 
        """
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


    def append_dataroot_one_sample(self, sample:dict):
        """
        Appending full path to raw data of one sample

        sample: One nuScenes sample in mmdet3d specific format 

        """
        imgs = sample["images"]
        for img in imgs.keys():
            curr_path = imgs[img]["img_path"]
            imgs[img]["img_path"] = self.data_root + "samples/" + img + "/" + curr_path

        lidar_path = sample["lidar_points"]["lidar_path"]
        sample["lidar_points"]["lidar_path"] = self.data_root + "samples/LIDAR_TOP/" + lidar_path

        return sample