from pathlib import Path
import torch
import matplotlib.pyplot as plt

from utils.utils_args import load_yaml, save_yaml
from preprocessing.data_init import init_data
from preprocessing.transforms import NormalizeTransform
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetNoPad2
from torch.nn import MSELoss, L1Loss, HuberLoss
from processing.losses import SSIMLoss, LinfLoss, PATLoss #, IoULoss
# from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from postprocessing.metric_connectivity import connectivityLoss

def preparation(PATH_current_data: Path, PATH_current_model: Path, PATH_destination: Path, scaling:bool=False):
    # load data for dummyK, step1 (predV)
    #TODO wont be working because inputs to init_data changed!!

    args = load_yaml(PATH_current_model / "command_line_arguments.yaml")
    args["data_prep"] = PATH_current_data
    args["model"] = PATH_current_model
    args["case"] = "test"
    args["destination"] = PATH_destination
    print(args)

    if scaling:
        args["order_data"] = [0,0]
    input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=False, ORDER_DATA=args["order_data"])
    hparams = load_yaml(PATH_current_model / "HPS_options.yaml")
    print(hparams)

    settings = {"init_features": hparams["init_features"]["values"][0], 
                "depth": hparams["depth"]["values"][0],
                "kernel_size": hparams["kernel_size"]["values"][0],
                "stride": hparams["stride"]["values"][0],
                "dilation": hparams["dilation"]["values"][0],
                "activation": hparams["activation_fct"]["values"][0],
                "norm": hparams["norm"]["values"][0],
                "repeat_inner": hparams["repeat_inner"]["values"][0],
                }
    if "ddunet" in PATH_current_model.name:
        pass
    elif "unetnopad" in PATH_current_model.name:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    elif "unet" in PATH_current_model.name or "padding" in PATH_current_model.name.lower():
        model = UNet(input_channels, output_channels, **settings)
    else:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    model.load(PATH_current_model)
    model.eval()
    
    info = load_yaml(PATH_current_model / "info.yaml")
    norm = NormalizeTransform(info)
    print(info)

    return dataloaders, model, norm, info, args, output_channels

def collect_metrics(PATH_current_data, PATH_current_model, PATH_destination, dataloaders, model, norm, info, args, output_channels):
    collected_metrics = {"model": PATH_current_model.name, "data": PATH_current_data.name, "order data": args["order_data"]}
    for case, dataloader in dataloaders.items():
        collected_metrics[case] = {}
        metrics:dict = {"MSE [phys. unit^2]": MSELoss(), "MAE [phys. unit]": L1Loss(), "Linf [phys. unit]": LinfLoss(), "Huber [phys. unit]": HuberLoss(), "SSIM": SSIMLoss()}
        if output_channels == 1: # some only make sense for Temperature predictions
            metrics["MoC [--]"], metrics["PAT0.1 [%]"], metrics["PAT1.0 [%]"] = None, PATLoss(pat_thresholds=[0.1]), PATLoss(pat_thresholds=[1])
        for metric_name, metric in metrics.items():
            print(f"Calculating {metric_name} for {case}",end=" ")
            metrics_values = []

            for batch in dataloader:
                inputs, targets = batch
                outputs = model(inputs).detach()

                inputs, targets = crop_to_output_size(inputs, targets, outputs)
                
                if metric_name in ["SSIM", "IoU"]:
                    values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])
                else:
                    # unnormalize inputs and targets
                    reverse_normalization(norm, inputs, targets, outputs)
                    # calc metrics per output channel
                    if "MoC" in metric_name:
                        if targets.shape[-3] == 1: # only applicable to Temperature output
                            values = []
                            for input, output in zip(inputs, outputs):
                                dict_connectivity = connectivityLoss(input, output, id_mat_ids=info["Inputs"]["Material ID"]["index"], threshold=10.7)
                                values.append(dict_connectivity["ratio"])
                            values = torch.mean(torch.Tensor(values))
                        else: 
                            values = torch.Tensor([0, 0]) #[torch.inf, torch.inf])
                    elif "PAT" in metric_name:
                        values = torch.mean(torch.Tensor(metric(outputs, targets).squeeze()))
                    else:
                        values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])

                metrics_values.append(values)

            assert len(dataloader) == 1, "I assumed I always have only one batch - otherwise please rethink this code"
            metrics[metric_name] = torch.mean(torch.stack(metrics_values), dim=0)# average over all batches
            print(f": average = {metrics[metric_name]}")

            collected_metrics[case][metric_name] = metrics[metric_name]
    save_yaml(collected_metrics, PATH_destination / f"metrics_paper25_{PATH_current_model.name} {PATH_current_data.name}.yaml")

def reverse_normalization(norm, inputs, labels, outputs):
    for tmp_in in inputs:
        norm.reverse(tmp_in, data_type="Inputs")
    for tmp_tar in labels:
        norm.reverse(tmp_tar, data_type="Labels")
    for tmp_out in outputs:
        norm.reverse(tmp_out, data_type="Labels")

def crop_to_output_size(inputs=None, labels=None, outputs=None):
    required_size = outputs.shape[-2:]
    start_pos = ((labels.shape[-2] - required_size[0])//2, (labels.shape[-1] - required_size[1])//2)
    if len(labels.shape) == 4:
        inputs = inputs[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    elif len(labels.shape) == 3:
        inputs = inputs[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    return inputs,labels

# ---------------------------------------- collection of all relevant paths ---------------------------------
# dummy data (+scaling)
path_dataV_dummyK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/dataset_giant_100hp_varyK inputs_pki outputs_xy")
path_dataT_dummyK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t")
path_dataT_dummyK_predV = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t prep_with_BEST_predict_v_v4")
path_dataVscaling_dummyK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/giant_double_100hp_v2 inputs_pki outputs_xy")
path_dataTscaling_dummyK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t")
path_dataTscaling_dummyK_predV = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t prep_with_BEST_predict_v_v4")
path_data_pki_T_dummyK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/dataset_giant_100hp_varyK inputs_pki outputs_t")
path_dataT_dummyK_inputsX = lambda x: Path(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dummyK_test_inputsT/dataset_giant_100hp_varyK inputs_{x} outputs_t")
# dummy model
path_modelV_dummyK = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/BEST_predict_v_v4")
path_modelT_dummyK = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/BEST_predict_T_add_s_outer")
path_modelT_dummyK_trainedOnPredV = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/predict_T_from_s_outer based_on_predict_v")
path_modelT_dummyK_inputsXXX = Path("") #TODO
path_modelV_dummyK_zeroPad = Path("") #TODO
path_modelT_dummyK_zeroPad = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/predict_T_add_s_outer_unet")
path_modelV_dummyK_trainedOn1DP = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/predict_v_1dp")
path_modelT_dummyK_trainedOn1DP = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/predict_T_add_s_outer_1dp")
path_modelT_dummyK_unet3DP = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals/naive_approach_unetnopad")
path_modelT_dummyK_inputsX = lambda x: Path(f"/scratch/sgs/pelzerja/runs/allin1/hpsManual_dummyT_inputs/hps_dummyT_inputs_{x}") #TODO
# dummy output  dir
runs_dummyK_output = Path("/scratch/sgs/pelzerja/runs/allin1_pre2025/paper24 finals")

# real data (+scaling)
path_dataV_realK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_giant_real_fixP0_0025 inputs_pki outputs_xy")
path_dataT_realK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_giant_real_fixP0_0025 inputs_ixydk+s_outer outputs_t")
path_dataT_realK_predV = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_giant_real_fixP0_0025 inputs_ixydk+s_outer outputs_t prep_with_trainV_fixP_with_best_from_varyG_moreData RK45") #Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_giant_real_fixP0_0025 inputs_ixydk+s_outer outputs_t prep_with_bestV-2-moreData-pki")
path_dataVscaling_realK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_scaling_real_fixP0_0025 inputs_pki outputs_xy")
path_dataTscaling_realK = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_scaling_real_fixP0_0025 inputs_ixydk+s_outer outputs_t")
path_dataTscaling_realK_predV = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_scaling_real_fixP0_0025 inputs_ixydk+s_outer outputs_t prep_with_trainV_fixP_with_best_from_varyG_moreData RK45") #prep_with_bestV-2-moreData-pki")
# real model
path_modelV_realK = Path("/scratch/sgs/pelzerja/runs/allin1/paper2025/realK_V/trainV_fixP_with_best_from_varyG_moreData") #bestV-2-moreData-pki")
path_modelT_realK = Path("/scratch/sgs/pelzerja/runs/allin1/paper2025/realK_T/trainT_fixP_with_best_from_varyG-V_moreData")#Path("/scratch/sgs/pelzerja/runs/allin1/bestT_moreData_trial15")
# real output dir
runs_realK_output = Path("/scratch/sgs/pelzerja/runs/allin1/paper2025")

if __name__=="__main__":

    # sets_paths_dmd = [
    #             # [path_dataV_dummyK, path_modelV_dummyK, runs_dummyK_output, False],
    #             # [path_dataT_dummyK, path_modelT_dummyK, runs_dummyK_output, False],
    #             [path_dataV_realK, path_modelV_realK, runs_realK_output, False],
    #             [path_dataT_realK, path_modelT_realK, runs_realK_output, False],
    #             # [path_dataT_dummyK_predV, path_modelT_dummyK, runs_dummyK_output, False],
    #             # [path_dataVscaling_dummyK, path_modelV_dummyK, runs_dummyK_output, True],
    #             # [path_dataTscaling_dummyK, path_modelT_dummyK, runs_dummyK_output, True],
    #             # [path_dataTscaling_dummyK_predV, path_modelT_dummyK, runs_dummyK_output, True],
    #             [path_dataVscaling_realK, path_modelV_realK, runs_realK_output, True],
    #             [path_dataTscaling_realK, path_modelT_realK, runs_realK_output, True],
    #             # [path_dataT_dummyK_predV, path_modelT_dummyK_trainedOnPredV, runs_dummyK_output, False],
    #             # [path_dataT_dummyK, path_modelT_dummyK_zeroPad, runs_dummyK_output, False],
    #             # [path_dataV_dummyK, path_modelV_dummyK_trainedOn1DP, runs_dummyK_output, False], 
    #             # [path_dataT_dummyK, path_modelT_dummyK_trainedOn1DP, runs_dummyK_output, False], # TODO full pipeline? 
    #             [path_dataT_realK_predV, path_modelT_realK, runs_realK_output, False],
    #             [path_dataTscaling_realK_predV, path_modelT_realK, runs_realK_output, True], 
    #             # [path_data_pki_T_dummyK, path_modelT_dummyK_unet3DP, runs_dummyK_output, False],
    #               ]
    

    # # # INPUTS EXPERIMENTS
    # # inputs = [
    # #         # "ixyk",
    # #         # "ixydk",
    # #         # "ixyk+s_outer",
    # #         # "ixydk+s_outer",
    # #         # "xyd+s_outer",
    # #         # "xydk+s_outer",
    # #         "ixydk+s_outer_notfaded",
    # #         "ixydk+s_outerMax", "ixydk+s_outer_RK23", "ixydk+s_outer_RK45"]

    # # sets_paths_dmd = []
    # # for input in inputs:
    # #     sets_paths_dmd += [[path_dataT_dummyK_inputsX(input), path_modelT_dummyK_inputsX(input), runs_dummyK_output, False],]

    sets_paths_dmd = [
        [path_dataVscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step1_run1"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataVscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step1_run2"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataVscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step1_run3"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataVscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step1_run4"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataVscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step1_run5"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #                   ]

    # sets_paths_dmd = [
        [path_dataTscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step3_run1"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataTscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step3_run2"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataTscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step3_run3"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataTscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step3_run4"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
        [path_dataTscaling_dummyK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/dummyK_step3_run5"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
                      ]

    # sets_paths_dmd = [
    #     [path_dataVscaling_realK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/realK_step1_run1"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #     [path_dataVscaling_realK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/realK_step1_run2"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #     [path_dataVscaling_realK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/realK_step1_run3"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #     [path_dataVscaling_realK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/realK_step1_run4"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #     [path_dataVscaling_realK, Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times/realK_step1_run5"), Path("/scratch/sgs/pelzerja/runs/allin1/runs_several_times"), True],
    #                   ]



    for path_data, path_model, path_desti, scaling in sets_paths_dmd:
        dataloaders, model, norm, info, args, output_channels = preparation(path_data, path_model, path_desti, scaling=scaling)
        collect_metrics(path_data, path_model, path_desti, dataloaders, model, norm, info, args, output_channels)
