

def get_model_size_in_gb(model):
    """
    Calculate the size of a PyTorch model in gigabytes.c

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    float: The size of the model in gigabytes.
    """
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Get the size of each parameter in bytes
    param_size_bytes = 4  # for float32

    # Calculate total size in bytes
    total_size_bytes = total_params * param_size_bytes

    # Convert bytes to gigabytes (1 GB = 1e9 bytes)
    total_size_gb = total_size_bytes / 1e9

    return total_size_gb

def count_parameters(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())