import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from config import device

def optimum_batch_size(model, input_size):
    """Determines the optimum batch size for the model.

    params:
    model (torch.nn.Module): The model to be tested
    input_size (tuple): The dimensions of the model input

    return:
    int: The estimated optimum batch size
    """
    print(f"using : {torch.device('cpu')}")
    return 64
    # todo fix this code, there is an issue with batch size finder failing
    #total_memory = torch.xpu.get_device_properties(device).total_memory
    #memory_usage_data = find_optimal_batch_size(model, input_size)
    #return estimate_batch_size(memory_usage_data, total_memory)
