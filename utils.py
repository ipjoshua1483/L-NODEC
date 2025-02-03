import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime
import os

dtype = torch.float64
device = torch.device('cpu')
constraint = 2.3

def torch_to_numpy(torch_list: List[torch.tensor]) -> List[np.array]:
    np_list = []
    for item in torch_list:
        np_list.append(item.cpu().detach().numpy())
    return np_list

def simpson_rule(values, h):
    if len(values.shape) == 0:
        return values

    if len(values) % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")

    result = values[0] + values[-1] + 4 * sum(values[1:-1:2]) + 2 * sum(values[2:-2:2])
    result *= h / 3
    return result

def save_data(timestamp_name: str, data_dict: Dict):
    save_path = '.\\data\\'
    save_path_w_timestamp = save_path + timestamp_name
    if not os.path.exists(save_path_w_timestamp):
        os.makedirs(save_path_w_timestamp)
    for data_name, data in data_dict.items():
        save_name = save_path_w_timestamp + "\\" + data_name
        np.save(save_name, data)
    print("Data saved successfully")
    
    