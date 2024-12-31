'''
Author: kylinhanx kylinhanx@gmail.com
Date: 2024-12-31 16:49:59
LastEditors: kylinhanx kylinhanx@gmail.com
LastEditTime: 2024-12-31 17:38:02
FilePath: \pipeline\src\utils.py
Description: List some commonly used functions
'''
import os

import random
import time

import numpy
import torch
import yaml
import cpuinfo
import psutil
import GPUtil
from tabulate import tabulate

def read_yaml(file_path):
    '''Read yaml file

    Read yaml file and return the data in the file, 
    if the file does not exist, raise FileNotFoundError,
    if the file cannot be read, raise ValueError

    Parameters:
        file_path: str, path to the yaml file
    Returns:
        data: dict, data in the yaml file
    Raises:
        NoneFileNotFoundError: if the file does not exist
        ValueError: if the file cannot be read
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading yaml file: {e}")
    return data

def calculate_model_params(model):
    '''calculate a pytorch model params

    Calculate the total number of parameters and trainable 
    parameters in the model
    
    Parameters:
        model: torch.nn.Module, model to calculate
    Returns:
        total_params: int, total number of parameters
        trainable_params: int, total number of trainable parameters
    Raises:
        ValueError: if model is not an instance of torch.nn.Module
    '''
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def freeze_model(model):
    '''freeze all parameters in a pytorch model

    Freeze all parameters in a pytorch model, i.e., set
    requires_grad to False for all parameters

    Parameters:
        model: torch.nn.Module, model to freeze
    Returns:
        None
    Raises:
        ValueError: if model is not an instance of torch.nn.Module
    '''
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False

def setup_seed(seed):
    """setup random seed for reproducibility

    Setup random seed for reproducibility in PyTorch, Numpy and Python random

    Parameters:
        seed: int, random seed
    Returns:
        None
    Raises:
        TypeError: if seed is not an integer
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_system_info():
    '''get system information

    Get system information including CPU model, physical CPU cores,
    logical CPU cores, total memory, GPU model and GPU total memory
    print the information in a table and return the information as a dictionary

    Parameters:
        None
    Returns:
        info: dict, system information
    Raises:
        None
    '''
    # get CPU info
    cpu_info = cpuinfo.get_cpu_info()
    cpu_model = cpu_info['brand_raw']
    cpu_physical_cores = psutil.cpu_count(logical=False)  # pyhsical cores
    cpu_logical_cores = psutil.cpu_count(logical=True)    # logical cores
    # get memory info
    virtual_memory = psutil.virtual_memory()
    total_memory_gb = virtual_memory.total / (1024 ** 2)  # turn into MB
    # get GPU info
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_model = gpus[0].name
        gpu_memory_total_gb = gpus[0].memoryTotal  # turn into MB
    else:
        gpu_model = "No GPU detected"
        gpu_memory_total_gb = 0
    # create table
    table_data = [
        ["CPU Model", cpu_model],
        ["Physical CPU Cores", cpu_physical_cores],
        ["Logical CPU Cores", cpu_logical_cores],
        ["Total Memory (GB)", f"{total_memory_gb:.2f} GB"],
        ["GPU Model", gpu_model],
        ["GPU Total Memory (GB)", f"{gpu_memory_total_gb:.2f} GB"],
    ]
    # print table
    print(tabulate(table_data, headers=["Component", "Details"], tablefmt="grid"))
    # return info as dictionary
    return {
        "CPU Model": cpu_model,
        "Physical CPU Cores": cpu_physical_cores,
        "Logical CPU Cores": cpu_logical_cores,
        "Total Memory (GB)": total_memory_gb,
        "GPU Model": gpu_model,
        "GPU Total Memory (GB)": gpu_memory_total_gb,
    }