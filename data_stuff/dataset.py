import os
import pathlib

import torch
import yaml
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset
#from torch._utils import _accumulate
from typing import List,Optional,Sequence
import matplotlib.pyplot as plt
import re
from data_stuff.transforms import NormalizeTransform

class DatasetExtendConvLSTM(Dataset):
    def __init__(self, path:str, prev_steps:int, extend:int, skip_per_dir:int, overfit=0):
        """
        Initializes a Dataset for ConvLSTM training.

        Args:
            path (str): The directory path containing 'Inputs' and 'Labels' subdirectories with data files.
            prev_steps (int): Number of previous boxes to include in each data sample.
            extend (int): Number of boxes to predict (extension of the sequence).
            skip_per_dir (int): Number of cells to skip between sampled data points.
            overfit (int, optional): Determines whether the dataset is restricted to a small subset for overfitting.
                - If overfit=0, the full dataset is used for training (no overfitting).
                - If overfit>0, only the first `overfit` data samples are used, allowing for overfitting/debugging.
                Defaults to 0.
        """
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.input_indices = [value['index'] for _, value in self.info['Inputs'].items()]
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        self.inputs = []
        self.labels = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
            file_path = self.path / "Inputs" / filename
            self.inputs.append(torch.load(file_path))
            
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
            file_path = self.path / "Labels" / filename
            self.labels.append(torch.load(file_path))

        #sort so that 'RUN_2' comes before 'RUN_10'
        self.input_names = sorted(self.input_names, key=self.natural_sort_key)
        self.label_names = sorted(self.label_names, key=self.natural_sort_key)

        self.spatial_size = self.inputs[0].shape[1:]
        self.total_nr_steps = prev_steps + extend
        self.extend = extend
        self.prev_steps = prev_steps
        self.box_len:int = (self.total_nr_steps) * self.spatial_size[1]
        self.skip_per_dir:int = skip_per_dir
        self.nr_cells_per_box_in_one_dir = 64
        self.dp_per_run:int = (self.spatial_size[0] - self.box_len) // skip_per_dir +1
        self.overfit = overfit

    @property
    def input_channels(self):
        return len(self.info["Inputs"])+1

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)
        return info
    
    def __len__(self):
        return len(self.input_names) * self.dp_per_run
    
    
    def __getitem__(self,idx):
        if self.overfit > 0:
            idx = idx % self.overfit
        run_id, window_nr = self.idx_to_window(idx)

        # get inputs and temperature
        input = self.inputs[run_id]
        temp = self.labels[run_id]

        # only use specific window
        input = input[:, self.skip_per_dir*window_nr:self.skip_per_dir*window_nr+(self.box_len),:]        
        temp = temp[:, self.skip_per_dir*window_nr:self.skip_per_dir*window_nr+(self.box_len),:] 

        input, temp = input.unsqueeze(1), temp.unsqueeze(1)

        # input temp is only left portion of the window
        input_temp = torch.cat((temp[:,:,:self.prev_steps*self.nr_cells_per_box_in_one_dir], torch.zeros([1,1, self.extend*self.nr_cells_per_box_in_one_dir,self.nr_cells_per_box_in_one_dir])), dim=2)

        # slice inputs and temp
        input = torch.cat((input, input_temp), dim=0)
        input_slices = torch.tensor_split(input,self.total_nr_steps,axis=2) # produces array
        input_seq = torch.cat(input_slices, dim=1) # concatenates array elements to form a tensor
        temp_slices = torch.tensor_split(temp, self.total_nr_steps, axis=2)
        
        label = torch.cat(temp_slices, dim=1)[:,-self.extend:].reshape(1, self.extend * self.nr_cells_per_box_in_one_dir, self.nr_cells_per_box_in_one_dir)

        return input_seq, label

    def idx_to_window(self, idx):
        run_id = idx // self.dp_per_run
        window = idx % self.dp_per_run
        return run_id, window
    
    def natural_sort_key(self,s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
def get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits