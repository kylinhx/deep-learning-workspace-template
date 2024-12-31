
import os
import yaml

import torch
import torch.nn as nn
import numpy as np

from utils import read_yaml, setup_seed, get_system_info

def main():
    # read configuration file
    config = read_yaml("E:\\个人项目\\pipeline\\config\\default.yaml")
    # setup random seed
    setup_seed(config["seed"])
    # get system information
    sys_info =  get_system_info()


if __name__ == "__main__":
    main()