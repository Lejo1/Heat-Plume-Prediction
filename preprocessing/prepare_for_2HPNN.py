import torch
from pathlib import Path
import yaml
from copy import deepcopy
from tqdm.auto import tqdm

from processing.networks.unet import UNet
from preprocessing.transforms import SignedDistanceTransform, NormalizeTransform

## prep data for 1HPNN
# step 0: ipvsgpu1: prepare domain data with 1hp model (pksi)
    # ipvsgpu1: python HPS_optuna.py --destination "/scratch/sgs/pelzerja/runs/2hp/prep_2hps_10dp" --problem 1hp
    # download data to local machine
# step 1: load domain (scaled by 1hp model)
# step 2: find hp positions
# step 3: extract 1hp boxes
# step 4: apply norming etc from 1hp model
# step 5: save prepared data
# step 5a: save info yaml, args

def recalc_sdf(inputs_sdf, pos_hp):
    return SignedDistanceTransform().sdf(inputs_sdf.detach().clone(), pos_hp)

def prep_for_1st_stage(domain_prep_dir, save_dir):
    #step 3.a
    info = yaml.safe_load((domain_prep_dir / "info.yaml").read_text())
    print(info)
    size_box = torch.tensor(info["CellsNumberPrior"])
    relative_pos_hp = torch.tensor(info["PositionHPPrior"])[:-1]
    print(size_box, relative_pos_hp)
    max_p_1hp = info["Inputs"]["Liquid Pressure [Pa]"]["max"]

    #step 1
    domain_iterdir = (domain_prep_dir / "Inputs").iterdir()
    pos_hps_all = {}
    for domain_dir in tqdm(domain_iterdir, desc="datapoints"):
        domain_inputs = torch.load(domain_dir, weights_only=False)
        domain_labels = torch.load(domain_dir.parent.parent / "Labels" / domain_dir.name, weights_only=False)

    #step 2
        pos_hps = torch.stack(torch.where(domain_inputs[3]==max(domain_inputs[3].flatten()))).T
        pos_hps_all[domain_dir.stem] = pos_hps.numpy().tolist()

    #step 3.b
        for i_hp, pos_hp in enumerate(pos_hps):
        # print(domain_inputs[0].shape)
            x_start = pos_hp[0] - relative_pos_hp[0]
            y_start = pos_hp[1] - relative_pos_hp[1]
        # print(x_start, y_start)
            if x_start < 0 or y_start < 0 or x_start+size_box[0] > domain_inputs.shape[1] or y_start+size_box[1] > domain_inputs.shape[2]:
                print(f"Box at {pos_hp} with size {size_box} out of bounds, skipping")
                continue
            box_input = deepcopy(domain_inputs[:, x_start:x_start+size_box[0], y_start:y_start+size_box[1]])
            box_label = domain_labels[:, x_start:x_start+size_box[0], y_start:y_start+size_box[1]]

        #step 4: correct inputs
            if "Material ID" in info["Inputs"].keys():
                i = info["Inputs"]["Material ID"]["index"]
                box_input[i] *= 0.0
                box_input[i, relative_pos_hp[0],relative_pos_hp[1]] = 1.0  # set material id to 1 at hp position
            if "SDF" in info["Inputs"].keys():
                i = info["Inputs"]["SDF"]["index"]
                box_input[i] = recalc_sdf(box_input[i], relative_pos_hp)
            if "Liquid Pressure [Pa]" in info["Inputs"].keys():
            # to adapt liquid pressure to 1hp model: always set maximum to 1 (=max in 1hp model) (add difference to all values)
                i = info["Inputs"]["Liquid Pressure [Pa]"]["index"]
                diff_max_p = box_input[i].max() - 1
                box_input[i] = box_input[i] - diff_max_p

        #step 5
            (save_dir / "Inputs").mkdir(parents=True, exist_ok=True)
            (save_dir / "Labels").mkdir(parents=True, exist_ok=True)
            torch.save(box_input, save_dir / "Inputs" / f"{domain_dir.stem}_HP_{i_hp}.pt")
            torch.save(box_label, save_dir / "Labels" / f"{domain_dir.stem}_HP_{i_hp}.pt")
        #step 5a
            yaml.safe_dump(info, (save_dir / "info.yaml").open("w"))
    yaml.safe_dump({"pos_hps": pos_hps_all, "rel_pos": relative_pos_hp.numpy().tolist()}, (save_dir / "pos_hps.yaml").open("w"))

def make_new_info(info, save_dir):
    info["Inputs"] = deepcopy(info["Labels"])
    info["Inputs"]["Temperature other [C]"] = deepcopy(info["Labels"]["Temperature [C]"])
    info["Inputs"]["Temperature other [C]"]["index"] = 1
    yaml.safe_dump(info, (save_dir / "info.yaml").open("w"))

## apply 1HPNN and store data for 2HPNN
# step 6: apply model 1hp
# step 7: move inputs acc. to orig pos of hp
# step 8: save final data
def prep_data_for_2nd_stage(model_1hp_dir, data_1hp_in_dir, save_dir):
    # load model
    model_1hp = UNet(in_channels=4, out_channels=1, init_features=32, kernel_size=7, depth=3)
    model_1hp.load(model_1hp_dir)
    model_1hp.eval();
    info = yaml.safe_load((model_1hp_dir / "info.yaml").read_text())

    pos_hps_all = yaml.safe_load((data_1hp_in_dir / "pos_hps.yaml").read_text())["pos_hps"]
    relative_pos_hp = torch.tensor(yaml.safe_load((data_1hp_in_dir / "pos_hps.yaml").read_text())["rel_pos"])

    #step 6
    boxes_iterdir = (data_1hp_in_dir / "Inputs").iterdir()
    for box_dir in tqdm(boxes_iterdir, desc="datapoints"):
        box_label = torch.load(data_1hp_in_dir / "Labels" / box_dir.name, weights_only=False)
        box_input1 = torch.load(box_dir, weights_only=False)
        output_tmp1 = model_1hp(box_input1.unsqueeze(0)).squeeze(0)

        box_id = box_dir.stem.split("_HP_")[0]
        i_hp = int(box_dir.stem.split("_HP_")[1])

        box2_dir = box_dir.parent / f"{box_id}_HP_{1 - i_hp}.pt"
        if box2_dir.exists() is False:
            print(f"Box {box2_dir.stem} does not exist, skipping")
            continue
        box_input2 = torch.load(box2_dir, weights_only=False)
        output_tmp2 = model_1hp(box_input2.unsqueeze(0)).squeeze(0)

        diff_pos_hps = torch.tensor(pos_hps_all[box_id][i_hp]) - torch.tensor(pos_hps_all[box_id][1 - i_hp])

        #step 7
        inputs_2hp = torch.zeros((2, output_tmp1.shape[1], output_tmp1.shape[2]))
        inputs_2hp[0] = output_tmp1[0]
        if diff_pos_hps[0] == 0:
            diff_pos_hps[0] = 1  # to avoid zero indexing
        if diff_pos_hps[1] == 0:
            diff_pos_hps[1] = 1  # to avoid zero indexing
        if diff_pos_hps[0] < 0 and diff_pos_hps[1] < 0:
            inputs_2hp[1, -diff_pos_hps[0]:, -diff_pos_hps[1]:] = output_tmp2[0, :diff_pos_hps[0], :diff_pos_hps[1]]

        if diff_pos_hps[0] > 0 and diff_pos_hps[1] > 0:
            inputs_2hp[1, :-diff_pos_hps[0], :-diff_pos_hps[1]] = output_tmp2[0, diff_pos_hps[0]:, diff_pos_hps[1]:]

        if diff_pos_hps[0] < 0 and diff_pos_hps[1] > 0:
            inputs_2hp[1, -diff_pos_hps[0]:, :-diff_pos_hps[1]] = output_tmp2[0, :diff_pos_hps[0], diff_pos_hps[1]:]

        if diff_pos_hps[0] > 0 and diff_pos_hps[1] < 0:
            inputs_2hp[1, :-diff_pos_hps[0], -diff_pos_hps[1]:] = output_tmp2[0, diff_pos_hps[0]:, :diff_pos_hps[1]]

        #step 8
        (save_dir / "Inputs").mkdir(parents=True, exist_ok=True)
        (save_dir / "Labels").mkdir(parents=True, exist_ok=True)
        torch.save(inputs_2hp, save_dir / "Inputs" / box_dir.name)
        torch.save(box_label, save_dir / "Labels" / box_dir.name)
        make_new_info(info, save_dir)
    yaml.safe_dump({"pos_hps": pos_hps_all, "rel_pos": relative_pos_hp.numpy().tolist()}, (save_dir / "pos_hps.yaml").open("w"))

def predictions_to_domain(dataset_for_2hpnn: Path, dataset_domain: Path, run_id: int, model_2hp: UNet, ins: torch.Tensor, ins2: torch.Tensor, case: str="predictions", assembly_case:str="max"):
    pos_hps = yaml.safe_load(open(dataset_for_2hpnn / f"pos_hps.yaml","r"))
    rel_pos = torch.tensor(pos_hps["rel_pos"])
    pos_hps = torch.tensor(pos_hps["pos_hps"][f"RUN_{run_id}"])

    domain_label = torch.load(dataset_domain / f"Labels/RUN_{run_id}.pt")
    domain_info = yaml.safe_load(open(dataset_domain/ f"info.yaml","r"))
    if case == "predictions":
        with torch.no_grad():
            pred = model_2hp(ins.unsqueeze(0)).squeeze(0)
            pred2 = model_2hp(ins2.unsqueeze(0)).squeeze(0)

    elif case == "merged_inputs":
        pred = (torch.max(ins[0], ins[1])).unsqueeze(0)
        pred2 = (torch.max(ins2[0], ins2[1])).unsqueeze(0)
    box_size = pred.shape[1:]

    domain_size = domain_info["CellsNumber"]
    domain_prediction = torch.zeros((2,*domain_size))
    start = pos_hps[0] - rel_pos
    start2 = pos_hps[1] - rel_pos
    domain_prediction[0,start[0]:start[0]+box_size[0], start[1]:start[1]+box_size[1]] = pred[0]
    domain_prediction[1,start2[0]:start2[0]+box_size[0], start2[1]:start2[1]+box_size[1]] = pred2[0]
    if assembly_case == "max":
        domain_prediction = torch.max(domain_prediction, dim=0).values
    elif assembly_case == "mean":
        domain_prediction = torch.mean(domain_prediction, dim=0)

    # normalize data
    norm = NormalizeTransform(domain_info)
    domain_prediction = norm.reverse(domain_prediction.unsqueeze(0), data_type="Labels").squeeze(0)
    domain_label = norm.reverse(domain_label, data_type="Labels").squeeze(0)
    return domain_prediction,domain_label

if __name__ == "__main__":
    domain_prep_dir = Path("/scratch/sgs/pelzerja/datasets_prepared/diss/2hp/dataset_2hps_inputs_pksi_1000dp")
    data_1hp_in_dir = Path("/scratch/sgs/pelzerja/datasets_prepared/diss/2hp/TMP_dataset_2hps_inputs_pksi_1000dp for_1hp_model")
    assert "pksi" in domain_prep_dir.name, "only implemented for pksi"
    prep_for_1st_stage(domain_prep_dir, data_1hp_in_dir)

    model_1hp_dir = Path("/scratch/sgs/pelzerja/runs/1hp/diss/1hpcnn_inputs_pksi (best)") # model for pksi
    save_dir = Path("/scratch/sgs/pelzerja/datasets_prepared/diss/2hp/TMP_dataset_2hps_inputs_pksi_1000dp for_2hp_model")
    prep_data_for_2nd_stage(model_1hp_dir, data_1hp_in_dir, save_dir)
