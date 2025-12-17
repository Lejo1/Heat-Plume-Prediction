import torch
from torch import Generator, default_generator, randperm
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from itertools import accumulate
from typing import List, Optional, Sequence
import postprocessing.cmap_jp

from preprocessing.datasets.dataset import DatasetBasis

class DatasetExtend(DatasetBasis):
    def __init__(self, path:str, kernel_size:int, n_blocks:int, n_convs_per_block:int=1, box_size:int=128*2, front_exclude:int=32):
        self.front_exclude:int = front_exclude # TODO where is the heat pump located + buffer
        # TODO requires prediction for first box of at least box_size + front_exclude
        DatasetBasis.__init__(self, path, box_size)
        
        print(self.spatial_size, "check x, y when new generated dataset!")
        self.out_size, self.gap = self.calc_gap_out_size(kernel_size, n_blocks, n_convs_per_block)
        self.dp_per_run:int = self.calc_dp_per_run() # CHECK! for training more dps are possible - this is more for inference
        print(f"dp_per_run: {self.dp_per_run}, spatial_size: {self.spatial_size}, box_size: {self.box_size}, skip_per_dir: {self.gap}")

    def __len__(self):
        return len(self.input_names) * self.dp_per_run
    
    @property
    def input_channels(self):
        return len(self.info["Inputs"]) + 1  # add temperature as input channel
    
    def __getitem__(self, i):
        i_run, start_inputs = self.i_to_pos(i)
        start_labels = start_inputs + self.box_size - self.gap
        inputs = self.load_data(i_run, start_inputs, self.box_size, data_type="Inputs")
        temperature_input = self.load_data(i_run, start_inputs, self.box_size, data_type="Labels")
        inputs = torch.cat((inputs, temperature_input), dim=0) # add temperature as input channel
        labels = self.load_data(i_run, start_labels, self.box_size, data_type="Labels")
        return inputs, labels
    
    def i_to_pos(self, i):
        i_run = i // self.dp_per_run
        pos_id = i % self.dp_per_run
        start_inputs = self.get_start_loc(pos_id)
        return i_run, start_inputs

    def get_start_loc(self, i):
        start_inputs = self.front_exclude + i * self.gap
        return start_inputs

    def load_data(self, i_run, start, size, data_type="Inputs"):
        run_id = self.input_names[i_run]
        data_run = torch.load(self.path / data_type / run_id) # list of tensors
        data_tensor = data_run[:, start:start+size] # shape: (n_properties, box_size, spatial_size_y)
        return data_tensor

    def calc_dp_per_run(self):
        return (self.spatial_size[0] - self.front_exclude - self.box_size - self.out_size) // self.gap

    def calc_gap_out_size(self, kernel_size:int, n_blocks:int, n_convs_per_block:int):
        # gap-formula: offset = gap = zero-adding width
        # Julius' implementation: additionally: padding and stride inputs
        out_size = copy(self.box_size)
        c = kernel_size - 1 # loss per convolution
        c_block = c * n_convs_per_block # loss for all convs per block
        b_enc = lambda x: (x - c_block)//2 # conv loss per block (incl pool) (encoder)
        b_dec = lambda x: x*2 - c_block # conv loss per block (decoder)

        for _ in range(n_blocks): # encoder
            out_size = b_enc(out_size)

        out_size -= c_block # bottleneck

        for _ in range(n_blocks): # decoder
            out_size = b_dec(out_size)

        assert out_size > 0, "Output size is non-positive! Check network parameters."
        gap = (self.box_size - out_size)//2
        assert out_size >= gap, "Output size is smaller than gap! Check network parameters."
        print(f"Calculated gap: {gap} for in_size {self.box_size}, out_size {out_size}")
        return out_size, gap  # total loss divided by 2 (both sides)
    
    def insert_in_domain(self, i, domain, new_data, case:str):
            start = self.get_start_loc(i)
            if case=="Inputs":
                domain[start : start+self.box_size] = new_data # input temperature
            elif case=="Labels":
                domain[start+self.box_size-self.gap : start+2*self.box_size-self.gap] = new_data # label temperature
            elif case=="Outputs":
                domain[start+self.box_size : start+self.box_size+self.out_size] = new_data # output temperature
       
    
    def plot_full_run(self, box_i, model, break_after:int=-1):
        # plot all extend-boxes of one run
        width = self.spatial_size[1]
        collection = torch.zeros(self.spatial_size[0],width*6)
        for i in range(self.dp_per_run): #len(data)):
            inp, lab = self[box_i + i]
            self.insert_in_domain(i, collection[:,:width], inp[-1], case="Inputs")
            self.insert_in_domain(i, collection[:,width:2*width], lab[0], case="Labels")
            out = model(inp.unsqueeze(0)).squeeze().detach()
            self.insert_in_domain(i, collection[:,2*width:3*width], out, case="Outputs")
            # self.insert_in_domain(i, collection[:,:width], out, case="Outputs") # if overwrite inputs with outputs
            collection[:,3*width:4*width] = np.abs(collection[:,2*width:3*width]-collection[:,width:2*width]) # abs diff output-label
            collection[:,4*width:5*width] = np.abs(collection[:,2*width:3*width]-collection[:,:width]) # abs diff output-input
            collection[:,5*width:6*width] = np.abs(collection[:,:width]-collection[:,width:2*width]) # abs diff input-label

            plt.figure(figsize=(20,5))
            plt.title(f"Box {i}")
            plt.imshow(collection.T, vmin=0,vmax=1, cmap="jp_temperature")
            plt.yticks(ticks=np.arange(width/2, width*6.5, width), labels=["Input", "Label", "Output", "|Out-Lab|", "|Out-Inp|", "|Inp-Lab|"])
            plt.colorbar()
            plt.show()

            if i >= break_after and break_after >= 0:
                break

def random_split_extend(dataset: DatasetExtend, split: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset]:
    r"""
        Copy from torch.utils.data.dataset with adaptation to of 'blow up indices'
    """
    indices_runs = randperm(len(dataset.input_names), generator=generator).tolist()
    split_abs = torch.tensor(split) * len(dataset.input_names)
    indices_split = [indices_runs[torch.round(offset - length).int() : torch.round(offset).int()] for offset, length in zip(accumulate(split_abs), split_abs)]
    # print("runs", indices_split)

    indices_extend = []
    for indices_this_case in indices_split:
        indices_extend_case = []
        for i_run in indices_this_case:
            indices_extend_case.extend([i_run*dataset.dp_per_run + i_box for i_box in range(dataset.dp_per_run)])
        indices_extend.append(indices_extend_case)
    # print("extend", indices_extend)

    return [Subset(dataset, indices_extend[i]) for i in range(len(split))]
