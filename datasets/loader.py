
from torch.utils.data import DataLoader, DistributedSampler
from base_dataset import NuScenesDataset
from grounding_dataset import NuScenesGroundingDataset
from nuCaption_dataset import NuScenesLidarLLmCaptionDataset
from nuView_dataset import NuScenesViewCaptionDataset
from qa_dataset import NuScenesQADataset

DATASET_REGISTRY = {
    "NuScenesGroundingDataset": NuScenesGroundingDataset,
    "NuScenesLidarLLmCaptionDataset": NuScenesLidarLLmCaptionDataset,
    "NuScenesViewCaptionDataset": NuScenesViewCaptionDataset,
    "NuScenesQADataset": NuScenesQADataset,
}


def load_dataset(config:dict) -> NuScenesDataset:
    dataset_type = config.get("type")

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type '{dataset_type}'. "
                         f"Available types: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[dataset_type]

    kwargs = {k: v for k, v in config.items() if k != "type"}

    return dataset_cls(**kwargs)    

def prepare_dataloader(dataset: NuScenesDataset, batch_size: int):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      pin_memory=True, 
                      collate_fn=dataset.get_collate_fn(), 
                      num_workers=0,
                      sampler=DistributedSampler(dataset))    

