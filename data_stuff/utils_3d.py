import numpy as np
import torch




def get_hp_position(tensor: torch.tensor):#intended for summed_error_pic, as error is highest at hp position
    '''
    Returns the hp position by searching for the maximum tensor value.
    Only works for 3D input data (X,Y,Z)
    '''     
    assert len(tensor.shape)==3 , f"Tensor has wrong dimension: {len(tensor.shape)} instead of 3"
    
    loc_hp = np.array(np.where(tensor == tensor.max())).squeeze()
    return loc_hp

def get_hp_position_from_input(tensor: torch.tensor, info: dict):
    '''
    Returns the hp position by reading the corresponding channel index from info
    and then searching for the maximum tensor value in this channel.
    Only works for 4D input data with first entry "channels" (C,X,Y,Z)
    '''     
    assert len(tensor.shape)==4 , f"Tensor has wrong dimension: {len(tensor.shape)} instead of 4"
    try:
        idx = info["Inputs"]["Material ID"]["index"]
    except:
        idx = info["Inputs"]["SDF"]["index"]
    loc_hp = np.array(np.where((tensor[idx] == tensor[idx].max()).cpu())).squeeze()
    
    return loc_hp

def remove_first_dim(tensor: torch.tensor, position: int) -> torch.tensor:
    '''
    Removes first non-batch dimension at given position.
    Works for 3D tensors (X,Y,Z), 4D tensors with first entry "channels" (C,X,Y,Z)
    and 5D tensors with first entry "batches" (B,C,X,Y,Z)
    '''

    if len(tensor.shape)==3: # for tensors (X,Y,Z)
        index = 0
    elif len(tensor.shape)==4: # for input data (C,X,Y,Z)
        index = 1
    elif len(tensor.shape)==5: # for batches (B,C,X,Y,Z)
        index = 2
    else:
        raise NotImplementedError( f"Tensor has wrong dimension: {len(tensor.shape)} instead of 3,4 or 5")
        
    assert position <= tensor.shape[index], f"Position {position} is out of bounds for dimension size {tensor.shape[index]}"
    tensor=tensor.select(index, position)
    tensor.squeeze(index)   

    return tensor
