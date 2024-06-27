import warnings

# ignore warnings from packages, be careful with this
warnings.filterwarnings("ignore")

import os
import random
import torch


import fnmatch
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

try:
    import psutil
except ImportError:
    print("Failed to import psutil.")
    psutil = None

def set_config(device: torch.device):
    """Set CPU and XPU configuration for torch."""
    if psutil:
        num_physical_cores = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
        print(f"OMP_NUM_THREADS set to: {num_physical_cores}")
    else:
        print("psutil not found. Unable to set OMP_NUM_THREADS.")

def set_seed(seed_value: int = 42):
    """Set all random seeds using `seed_value`."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

def device():
    device = torch.device("cpu")
    return device

# Configuration
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
device = device()
set_config(device)


'''
def set_device(device=None):
    """
    Sets the device for PyTorch. If a specific device is specified, it will be used.
    Otherwise, it will default to CPU or XPU based on availability.
    """
    if device is not None:
        print(f"Device set to {device} by user.")
        return torch.device(device) 

    # if HAS_XPU:
    #     device_count = torch.xpu.device_count()
    #     device_id = random.randint(0, int(device_count) - 1)
    #     device = f"xpu:{device_id}"
    #     print(f"XPU devices detected, using {device}")
    #     print(f"XPU device name: {torch.xpu.get_device_name(0)}")
        

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''