'''
Author: kylinhanx kylinhanx@gmail.com
Date: 2024-12-31 16:49:59
LastEditors: kylinhanx kylinhanx@gmail.com
LastEditTime: 2025-01-05 20:08:16
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
    '''setup random seed for reproducibility

    Setup random seed for reproducibility in PyTorch, Numpy and Python random

    Parameters:
        seed: int, random seed
    Returns:
        None
    Raises:
        TypeError: if seed is not an integer
    '''
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

def infer_time(model, test_loader):
    '''infer time for a model

    Infer time for a model on a test_loader, the model is
    set to evaluation mode and the time is measured using
    torch.no_grad()

    Parameters:
        model: torch.nn.Module, model to infer time
        test_loader: torch.utils.data.DataLoader, test data loader
    Returns:
        time_taken: float, time taken for inference
    Raises:
        ValueError: if model is not an instance of torch.nn.Module
        ValueError: if test_loader is not an instance of torch.utils.data.DataLoader
    '''
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")
    if not isinstance(test_loader, torch.utils.data.DataLoader):
        raise ValueError("test_loader must be an instance of torch.utils.data.DataLoader")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for data, _ in test_loader:
            data = data.cuda()
            model(data)
        end_time = time.time()
    time_taken = end_time - start_time
    return time_taken

def calculate_gflops(model, input_shape):
    '''calculate GFLOPs for a model

    Calculate GFLOPs for a model with a given input shape

    Parameters:
        model: torch.nn.Module, model to calculate GFLOPs
        input_shape: tuple, input shape of the model
    Returns:
        gflops: float, GFLOPs of the model
    Raises:
        ValueError: if model is not an instance of torch.nn.Module
        ValueError: if input_shape is not a tuple
    '''
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")
    if not isinstance(input_shape, tuple):
        raise ValueError("input_shape must be a tuple")
    model.eval()
    model = model.cuda()
    input_data = torch.randn(1, *input_shape).cuda()
    gflops = torch.ops.profiler.profile(model, input_data)
    return gflops

def plot_confusion_matrix(y_true, y_pred, labels, title, saved_path, normalize=False):
    '''plot confusion matrix

    Plot confusion matrix for y_true and y_pred

    Parameters:
        y_true: numpy.ndarray, true labels
        y_pred: numpy.ndarray, predicted labels
        labels: list, class labels
        title: str, title of the plot
        saved_path: str, path to save the figure
        normalize: bool, whether to normalize the confusion matrix
    Returns:
        None
    Raises:
        ValueError: if y_true is not an instance of numpy.ndarray
        ValueError: if y_pred is not an instance of numpy.ndarray
        ValueError: if labels is not a list
        ValueError: if title is not a string
        ValueError: if normalize is not a boolean
    '''
    if not isinstance(y_true, numpy.ndarray):
        raise ValueError("y_true must be an instance of numpy.ndarray")
    if not isinstance(y_pred, numpy.ndarray):
        raise ValueError("y_pred must be an instance of numpy.ndarray")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean")
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(saved_path)
    # plt.show()

def plot_loss(saved_path, loss_list, loss_labels):
    '''plot train loss

    Plot train loss

    Parameters:
        saved_path: str, path to save the figure
        loss_list: 2-dim, list, train loss
        loss_labels: 1-dim, list, labels for each loss
    Returns:
        None
    Raises:
        ValueError: if train_loss is not a list
        ValueError: if saved_path is not a string
    '''
    if not isinstance(train_loss, list):
        raise ValueError("train_loss must be a list")
    if not isinstance(saved_path, str):
        raise ValueError("saved_path must be a string")
    if len(loss_labels) != len(loss_list):
        raise ValueError("loss_labels must have the same length as loss_list")
    import matplotlib.pyplot as plt
    for i, loss in enumerate(loss_list):
        plt.plot(loss, label=loss_labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(saved_path)
    # plt.show()
    