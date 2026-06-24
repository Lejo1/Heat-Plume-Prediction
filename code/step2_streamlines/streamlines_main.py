from shutil import copytree
import time
from pathlib import Path
import torch

from processing.networks.unetVariants import UNetNoPad2
from step2_streamlines.streamlines_helpers import make_streamlines, save_new_datapoint, correct_args_info, extend_inputs_dims_specific, extend_inputs_dims
from utils.utils_args import load_yaml
from preprocessing.transforms import NormalizeTransform

def build_streamlines(model_path:str=None, dataset_path:Path=None, based_on_pred:bool=True, model=None, method="RK45", randomK:bool=False, **kwargs):
    ## copy ixydk files (later overwrite xyd)
    data_dir = dataset_path.parent
    dataset_name = dataset_path.name
    # origin_data_T_inout = "inputs_ixyk outputs_t for_s" # base, replace xy, add s,s_outer
    origin_data_T = data_dir/f"{dataset_name}"# {origin_data_T_inout}"
    origin_data_v = data_dir/f"{dataset_name}" # inputs_pki outputs_xy"
    destination_data_T_inout = "+ss_o"#ixydk+s_outer outputs_t"
    if based_on_pred:
            destination = data_dir/f"{dataset_name} {destination_data_T_inout} prep_with_{model_path.name} {method}"
    else:
        destination = data_dir/f"{dataset_name} {destination_data_T_inout}" 
    copytree(origin_data_T,destination, dirs_exist_ok=True)

    idx = {"vx": 3, #1,
            "vy" : 4, #2,
            "mat_ids" : 5, #0,
            "sf" : 6, #3,
            "sf_outers" : 7, #5,
            "s_seasonal" : 8 #None
            }

    norm_before = NormalizeTransform(load_yaml(destination/"info.yaml"))
    v_info_path = origin_data_v
    correct_args_info(destination, v_info_path, based_on_pred, i_s=idx["sf"], i_s_outer=idx["sf_outers"], i_s_seasonal=idx["s_seasonal"])
    norm_after = NormalizeTransform(load_yaml(destination/"info.yaml"))
    for runid in (destination / "Inputs").iterdir():
        runid = runid.name
        start_time = time.time()
        # if based_on_pred:
        #     data_in_model = torch.load(origin_data_v/"Inputs"/runid)
        #     vv = model(data_in_model.unsqueeze(0)).detach().squeeze(0)
        # else:
        #     vv = torch.load(origin_data_v/"Inputs"/runid)
        # norm_v.reverse(vv, "Inputs")
        inputs = torch.load(destination/"Inputs"/runid)
        inputs = norm_before.reverse(inputs, "Inputs")
        # inputs = extend_inputs_dims_specific(inputs)
        inputs = extend_inputs_dims(inputs, n_added_inputs=3) # add 3 channels for streamlines

        # crop inputs to match model output
        # required_size = vv.shape[1:]
        # start_pos = ((inputs.shape[1] - required_size[0])//2, (inputs.shape[2] - required_size[1])//2)
        # inputs_reduced = inputs[:, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

        # overwrite vx, vy with model output
        # inputs_reduced[idx["vx"]] = vv[0]
        # inputs_reduced[idx["vy"]] = vv[1]
        # inputs_reduced = inputs_reduced.numpy()
        inputs_reduced = inputs.numpy()

        # generate a vector field for vx, vy and visualize with arrows
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.figure(figsize=(10, 10))
        # x = np.arange(0, 320)
        # y = np.arange(0, 1280)
        # X, Y = np.meshgrid(x, y)
        # U = inputs_reduced[idx["vx"]]
        # V = inputs_reduced[idx["vy"]]
        # plt.quiver(X, Y, U, V, scale=0.00001)
        # plt.show

        # make streamlines
        streamlines_faded = make_streamlines(mat_ids=inputs_reduced[idx["mat_ids"]], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, runid=runid.split(".")[0], randomK_data=randomK, faded=True, **kwargs)

        streamlines_faded_top = make_streamlines(mat_ids=inputs_reduced[idx["mat_ids"]], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, runid=runid.split(".")[0], randomK_data=randomK, faded=False, offset=10)
        streamlines_faded_bottom = make_streamlines(mat_ids=inputs_reduced[idx["mat_ids"]], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, runid=runid.split(".")[0], randomK_data=randomK, faded=False, offset=-10)

        streamlines_seasonal = make_streamlines(mat_ids=inputs_reduced[idx["mat_ids"]], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, runid=runid.split(".")[0], randomK_data=randomK, faded=False, seasonal=True)

        # norm inputs acc. to info
        inputs_normed = norm_after(torch.tensor(inputs_reduced), "Inputs")
        inputs_normed[idx["sf"]] = streamlines_faded.unsqueeze(0)
        # inputs_normed[idx["sf_outers"]] = streamlines_faded_top.unsqueeze(0) + streamlines_faded_bottom.unsqueeze(0)
        inputs_normed[idx["sf_outers"]] = torch.max(streamlines_faded_top.unsqueeze(0), streamlines_faded_bottom.unsqueeze(0)) # TODO for exp_inputs: if max instead of sum
        inputs_normed[idx["s_seasonal"]] = streamlines_seasonal.unsqueeze(0)

        save_new_datapoint(destination, runid, inputs_normed)
        print(f"Finished {runid} after {time.time()-start_time} seconds")