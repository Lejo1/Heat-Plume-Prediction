from shutil import copytree
import time
from pathlib import Path
import torch

from processing.networks.unetVariants import UNetNoPad2
from step2_streamlines.streamlines_helpers import make_streamlines, save_new_datapoint, correct_args_info, extend_inputs_dims
from utils.utils_args import load_yaml
from preprocessing.transforms import NormalizeTransform

def build_streamlines(model_path:str=None, dataset_path:Path=None, based_on_pred:bool=True, model=None, method="RK45", randomK:bool=False, **kwargs):
    ## copy ixydk files (later overwrite xyd)
    data_dir = dataset_path.parent
    dataset_name = dataset_path.name
    origin_data_T_inout = "inputs_ixyk outputs_t for_s" # base, replace xy, add s,s_outer
    origin_data_T = data_dir/f"{dataset_name} {origin_data_T_inout}"
    origin_data_v = data_dir/f"{dataset_name} inputs_pki outputs_xy"
    destination_data_T_inout = "inputs_ixydk+s_outer outputs_t"
    if based_on_pred:
            destination = data_dir/f"{dataset_name} {destination_data_T_inout} prep_with_{model_path.name} {method}"
    else:
        destination = data_dir/f"{dataset_name} {destination_data_T_inout}" 
    copytree(origin_data_T,destination, dirs_exist_ok=True)

    idx = {"vx": 1,
            "vy" : 2,
            "sf" : 3,
            "sf_outers" : 5,
            }

    if based_on_pred:
        # load model if not given
        info_v = load_yaml(model_path / "info.yaml")
        if model is None:
            try: 
                settings_model = load_yaml(model_path / "HPS_options.yaml") 
                model = UNetNoPad2(in_channels=len(settings_model["inputs"]["values"][0]), out_channels=2, kernel_size=settings_model["kernel_size"]["values"][0], depth=settings_model["depth"]["values"][0], init_features=settings_model["init_features"]["values"][0], stride=settings_model["stride"]["values"][0], dilation=settings_model["dilation"]["values"][0], activation=settings_model["activation_fct"]["values"][0], norm=settings_model["norm"]["values"][0], repeat_inner=settings_model["repeat_inner"]["values"][0])
            except: # old
                settings_model = load_yaml(model_path / "settings.yaml")
                model = UNetNoPad2(in_channels=len(settings_model["inputs"]), out_channels=2, kernel_size=settings_model["kernel_size"], depth=settings_model["depth"], init_features=settings_model["init_features"])
            model.load(model_path)
        model.eval()
    else:
        info_v = load_yaml(origin_data_v / "info.yaml")
    norm_v = NormalizeTransform(info_v)

    norm_before = NormalizeTransform(load_yaml(destination/"info.yaml"))
    if based_on_pred:
        v_info_path = model_path
    else:
        v_info_path = origin_data_v
    correct_args_info(destination, v_info_path, based_on_pred)
    norm_after = NormalizeTransform(load_yaml(destination/"info.yaml"))

    for runid in (destination / "Inputs").iterdir():
        runid = runid.name
        start_time = time.time()
        if based_on_pred:
            data_in_model = torch.load(origin_data_v/"Inputs"/runid)
            vv = model(data_in_model.unsqueeze(0)).detach().squeeze(0)
        else:
            vv = torch.load(origin_data_v/"Labels"/runid)
        norm_v.reverse(vv, "Labels")
        inputs = torch.load(destination/"Inputs"/runid)
        norm_before.reverse(inputs, "Inputs")

        inputs = extend_inputs_dims(inputs)
        # crop inputs to match model output
        required_size = vv.shape[1:]
        start_pos = ((inputs.shape[1] - required_size[0])//2, (inputs.shape[2] - required_size[1])//2)
        inputs_reduced = inputs[:, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

        # overwrite vx, vy with model output
        inputs_reduced[idx["vx"]] = vv[0]
        inputs_reduced[idx["vy"]] = vv[1]
        inputs_reduced = inputs_reduced.numpy()

        # make streamlines
        streamlines_faded = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, randomK_data=randomK, **kwargs)

        streamlines_faded_top = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, randomK_data=randomK, offset=10)
        streamlines_faded_bottom = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, randomK_data=randomK, offset=-10)

        # norm inputs acc. to info
        inputs_normed = norm_after(torch.tensor(inputs_reduced), "Inputs")
        inputs_normed[idx["sf"]] = streamlines_faded.unsqueeze(0)
        inputs_normed[idx["sf_outers"]] = streamlines_faded_top.unsqueeze(0) + streamlines_faded_bottom.unsqueeze(0)
        # inputs_normed[idx["sf_outers"]] = torch.max(streamlines_faded_top.unsqueeze(0), streamlines_faded_bottom.unsqueeze(0)) # TODO for exp_inputs: if max instead of sum

        save_new_datapoint(destination, runid, inputs_normed)
        print(f"Finished {runid} after {time.time()-start_time} seconds")