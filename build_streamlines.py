import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import os

import utils.utils_args as ut
import preprocessing.preprocessing as prep
from processing.streamlines_helpers import *
from processing.streamlines_main import *

# script runs the following steps:
# 1. prepare data
# 2. calculate streamlines with simulated and with predicted velocity fields
# 3. visualize streamlines
# based on the best models I have so far for the realistic datasets

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_arguments_and_set_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="allin1")
    parser.add_argument("--data_raw", type=str, default="dataset_small_10dp_varyK")
    parser.add_argument("--data_prep", type=str, default=None)
    parser.add_argument("--allin1_prepro_n_case", type=str, default=None)
    parser.add_argument("--inputs", type=str, default="ixyk")
    parser.add_argument("--outputs", type=str, default="t")
    parser.add_argument("--len_box", type=int, default=64)
    parser.add_argument("--skip_per_dir", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, default="train")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--destination", type=str, default="dummy_prep_ixyk_T")
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--device", type=str, default="3")
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)
    args["device"] = "cpu"

    args["destination"] = Path(f"/scratch/sgs/pelzerja/runs/{args['problem']}") / args["destination"]
    current_destination = args["destination"]
    return args,current_destination


if __name__ == "__main__":
    datasets = ["dataset_100hp_scaling_real_fixP0_0025"] #dataset_100hp_giant_real_fixP0_0025"] #dataset_100hp_scaling_real_fixP0_0025 #dataset_giant_100hp_varyK"] #dataset_25600x12800_1dp_200hp_no_vary_inflow_no_refinement"] #,"dataset_notrefined_12800m_10dp_100hp_no_vary_inflow_RUNS523",]
    models = ["paper2025/realK_V/trainV_fixP_with_best_from_varyG_moreData"] #, bestV-2-moreData-pki"" "try_12800m_2train1val_depth6 GOODISH"] #"try_12800m_2train1val", #["val_before_train?/train_12800m_cutouts", "val_before_train?/train_12800m_cutouts1280", "val_before_train?/train_12800m_cutouts1280 GOODISH", "val_before_train?/try_12800m_2train1val GOODISH"] #, "val_before_train?/try_12800m_2train1val_depth=6_moreDP_MSE"]

    raw_name = lambda x: f"/scratch/sgs/pelzerja/datasets/allin1/{x}"
    prep_name = lambda x: f"/scratch/sgs/pelzerja/datasets_prepared/allin1/{x} inputs_ixyk outputs_t for_s" #dummyK_test_inputsT/
    if True:
        args, current_destination = parse_arguments_and_set_paths()

        for dataset_name in datasets:
            # STEP 1: prepare data
            # !python main.py --destination dummy_prep_ixyk_T
            clas = ut.load_yaml(args["destination"] / "command_line_arguments.yaml")
            args["data_raw"] = raw_name(dataset_name)
            args["data_prep"] = prep_name(dataset_name)
            for path_typed_cla in ["data_prep", "data_raw", "destination", "model"]:
                if clas[path_typed_cla] is not None:
                    clas[path_typed_cla] = Path(clas[path_typed_cla])
            args["destination"] = current_destination # just to make sure that nothing is overwritten

            ut.make_paths(args) # and check if data / model exists
            ut.save_yaml(args, args["destination"] / "command_line_arguments.yaml")

            # prepare data
            prep.preprocessing(args) # and save info.yaml in model folder
            print(current_destination)
            for based_on_pred in [True]: #False]: 
                # streamlines_all = {}
                for method in ["RK45"]: #, "RK23", "Radau"]:
                    # STEP 2: calculate streamlines with simulated and with predicted velocity fields
                    print("data", dataset_name)
                    streamlines = build_streamlines(models[0], dataset_name, based_on_pred=based_on_pred, method=method)
                    # streamlines_all[method] = streamlines

                # COMPARE STREAMLINES
                # plt.figure(figsize=(25,20))
                # for idx, method in enumerate(streamlines_all.keys()):
                #     plt.subplot(5,1,idx+1)
                #     plt.title(f"streamlines {method}")
                #     plt.imshow(streamlines_all[method].numpy(), cmap="jp_linear")
                #     plt.colorbar()
                # plt.subplot(5,1,4)
                # plt.title("streamlines rk45-rk23")
                # plt.imshow(np.abs(streamlines_all["RK45"]-streamlines_all["RK23"]).numpy(), cmap="jp_linear")
                # print("diff", np.abs(streamlines_all["RK45"]-streamlines_all["RK23"]).numpy().max())
                # plt.colorbar()
                # plt.subplot(5,1,5)
                # plt.title("streamlines rk45-radau")
                # plt.imshow(np.abs(streamlines_all["RK45"]-streamlines_all["Radau"]).numpy(), cmap="jp_linear")
                # print("diff", np.abs(streamlines_all["RK45"]-streamlines_all["Radau"]).numpy().max())
                # plt.colorbar()
                # plt.savefig(f"streamlines_comparison_{dataset_name}_basedonpred{based_on_pred}.png",dpi=400)

    if False:
        # STEP 3: visualize streamlines
        # run_id = 10 # =val?, 12=train?
        directory = "/scratch/sgs/pelzerja/datasets_prepared/allin1"
        for run_id in [0, 10, 11, 12]:
            for dataset_name in datasets:
                print("data", dataset_name)
                try:
                    labels = torch.load(f"{directory}/{dataset_name} inputs_ixydk+s_outer outputs_t/Labels/RUN_{run_id}.pt")
                    inputs_true_loaded = torch.load(f"{directory}/{dataset_name} inputs_ixydk+s_outer outputs_t/Inputs/RUN_{run_id}.pt")
                    for model_name in models:
                        model_name = model_name.split("/")[-1]
                        print("model", model_name)
                        inputs_calc_loaded = torch.load(f"{directory}/{dataset_name} inputs_ixydk+s_outer outputs_t prep_with_{model_name}/Inputs/RUN_{run_id}.pt")
                        print("shapes", inputs_true_loaded.shape, inputs_calc_loaded.shape, labels.shape)

                        required_shape = inputs_calc_loaded.shape
                        start_pos = int((inputs_true_loaded.shape[1] - required_shape[1])/2)
                        inputs_true_loaded = inputs_true_loaded[:, start_pos:start_pos+required_shape[1], start_pos:start_pos+required_shape[2]]
                        idx_compare=3
                        plt.figure(figsize=(25,20))
                        plt.subplot(3,3,1)
                        plt.title("true streamlines")
                        plt.imshow(inputs_true_loaded[idx_compare].numpy(), cmap="jp_linear")
                        plt.colorbar()
                        plt.subplot(3,3,2)
                        plt.title("calc streamlines")
                        plt.imshow(inputs_calc_loaded[idx_compare].numpy(), cmap="jp_linear")
                        plt.colorbar()
                        plt.subplot(3,3,3)
                        plt.title("abs diff")
                        plt.imshow(np.abs(inputs_true_loaded[idx_compare]-inputs_calc_loaded[idx_compare]).numpy(), cmap="jp_linear")
                        plt.colorbar()
                        plt.subplot(3,3,4)
                        plt.title("temperature")
                        plt.imshow(labels[0].numpy(), cmap="jp_temperature")
                        plt.colorbar()
                        # min_vx = min(inputs_true_loaded[1].min(), inputs_calc_loaded[1].min())
                        min_vx = inputs_true_loaded[1].min()
                        # max_vx = max(inputs_true_loaded[1].max(), inputs_calc_loaded[1].max())
                        max_vx = inputs_true_loaded[1].max()
                        plt.subplot(3,3,5)
                        plt.title("true vx")
                        plt.imshow(inputs_true_loaded[1].numpy(), cmap="jp_linear", vmin=min_vx, vmax=max_vx)
                        plt.colorbar()
                        plt.subplot(3,3,6)
                        plt.title("calc vx")
                        plt.imshow(inputs_calc_loaded[1].numpy(), cmap="jp_linear", vmin=min_vx, vmax=max_vx)
                        plt.colorbar()
                        plt.subplot(3,3,7)
                        plt.title("permeability")
                        plt.imshow(inputs_true_loaded[4].numpy(), cmap="jp_linear")
                        plt.colorbar()
                        # min_vy = min(inputs_true_loaded[2].min(), inputs_calc_loaded[2].min())
                        min_vy = inputs_true_loaded[2].min()
                        # max_vy = max(inputs_true_loaded[2].max(), inputs_calc_loaded[2].max())
                        max_vy = inputs_true_loaded[2].max()
                        plt.subplot(3,3,8)
                        plt.title("true vy")
                        plt.imshow(inputs_true_loaded[2].numpy(), cmap="jp_linear", vmin=min_vy, vmax=max_vy)
                        plt.colorbar()
                        plt.subplot(3,3,9)
                        plt.title("calc vy")
                        plt.imshow(inputs_calc_loaded[2].numpy(), cmap="jp_linear", vmin=min_vy, vmax=max_vy)
                        plt.colorbar()

                        plt.savefig(f"{directory}/{dataset_name} inputs_ixydk+s_outer outputs_t prep_with_{model_name}/run_{run_id} streamlines_comparison_rk45.png")
                        plt.show()
                except Exception as e:
                    print(e)
                    continue