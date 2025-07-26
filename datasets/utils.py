import torch


def load_bev_map(tensor_root, sample_token):
    bev = torch.load(tensor_root + "tensor_" + sample_token + ".pt").unsqueeze(0)
    bev = bev.to("cpu")
    bev = bev.detach().clone()
    return bev