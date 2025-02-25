# Begin working
- clone the repository
- install the requirements: `pip install -r requirements.txt`
- download the raw / prepared data from darus
  - [Models and prepared data](https://doi.org/10.18419/darus-4518)
  - [1 Heatpump raw data](https://doi.org/10.18419/darus-3650)
  - [2 Heatpump raw data](https://doi.org/10.18419/darus-3652)
- set the paths in paths.yaml (see later)

## Exemplary paths.yaml file:

    ```
    default_raw_dir: /scratch/sgs/pelzerja/datasets # where the raw data is stored
    datasets_prepared_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN # where the prepared data is stored
    datasets_raw_domain_dir: /scratch/sgs/pelzerja/datasets/2hps_demonstrator_copy_of_local
    datasets_prepared_domain_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_domain
    models_1hp_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN/runs
    models_2hp_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN/runs/2hpnn
    datasets_prepared_dir_2hp: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_NN
    ```

## Main.py
- all operations are executed via main.py and the following main arguments:
  - --case: operation of current exection, e.g `train` for training a network, `iterative` for iterative application and `prep_xhp` for preparing dataset with multiple heat pumps  (default `train`)
  - --dataset_raw: name of the raw dataset saved in the raw directory specified in paths.yaml (default `dataset_2d_small_1000dp`)
  - --dataset_prep: name of the prepared dataset saved in the prep directory specified in paths.yaml (default ``)
  - --model: model name saved in models directory specified in paths.yaml (default `default`)
  - --destination: destination folder where the results of the execution are saved (default ``)

- optional arguments:
  - --device: device for model training/inference (default `cuda:0`)
  - --epochs: number of training epochs (default `10000`)
  - --inputs: make sure, they are the same as in the model (default `gksit`)
  - --visualize: visualize the results (default `False`)
  - --only_prep: flag for bypassing dataset preparation when only prep data is available (default `False`)
  - --save_inference: flag for saving inference (default `False`)
  - --problem: type of CNN for current execution (default `2stages` which is a standard U-net)
  - --notes: not used in this fork
  - --len_box: length in y-direction that the datapoints should be cut off (default `256`). Make sure, this number is less or equal to the length of the simulation run.
  - --skip_per_dir: not used in this fork
  - --architecture: which neural architecture to choose, options are `2stages`,`quad`,`parallel`. Their differences are explained in the Master's thesis.
  
The general workflow starts by training a model specialized for the single hp scenario using --case train. This model is reused to generate a two heat pump dataset using --case prepare. This dataset can then be used to train a model for the two hp scenario via --case train. This process can be repeated until the model can provide a good estimation for a global domain by using it with --case iterative.

## Training a model:
- for training you need a dataset in datasets_prepared_dir or default_raw_dir (paths.yaml)
- execute
     ```
     python main.py --dataset_prep 12HP_2500dp --epoch 8000 --architecture quad --inputs gksit --visualize True --device cuda:0 --destination unet_quad
     ```

## Generating prepared dataset with multiple heat pumps:
- you need the model in models_1hp_dir (paths.yaml) and the dataset in default_raw_dir (paths.yaml)
- the images of the dataset end up in `runs/2hpnn`, the resulting dataset in datasets_prepared_dir_2hp (paths.yaml)
- execute
```
 python main.py --dataset_raw 5HP --architecture parallel --inputs gksit --model unet_para_f64_d5_k4_2500dp --visualize True --case prepare --destination 5hp_dataset --device cpu

```

## Iterative application:
- for iterative application you need the model in models_1hp_dir (paths.yaml)  and the dataset in default_raw_dir (paths.yaml)
- ensure that the hyperparameters from the model defined by --model are the same as in e.g. networks/unet.py (depending on the architecture)
- if they are different adjust the hyperparameters in the code by hand 
- execute
  ```
  python main.py --dataset_raw 5HP --case iterative --model unet_para_f64_d5_k4_2500dp --architecture parallel --inputs gksit --destination seq_5hp_large
  ```

## Finding the results:
- resulting model (`model.pt`) + normalization parameters (info.yaml) used can be found in `runs/1hpnn/DESTINATION` with `DESTINATION` being the user defined or default name in the call of main.py
- this folder also contains visualisations if any were made during the training/inference
- prepared datasets are in datasets_prepared (paths.yaml:`datasets_prepared_dir/PROBLEM`)

# Logging your training progress:
- use command 
    ```
    tensorboard --logdir=runs/ --host localhost --port 8088
    ```
    in command line and follow the instructions - there you can log the progress of e.g. the loss graph, the lr-schedule, etc.
- if you want to change the learning rate during training (e.g. because you see no progress anymore) you can press strg+c to interrupt the training, enter a new lr and continue the training. The new lr is documented in the lr-schedule in `learning_rate_history.csv` in the models folder in runs/

# GPU support
- if you want to use the GPU, you need to install with cuda support
- check nvidia-smi for the available gpus and cuda version
- `export CUDA_VISIBLE_DEVICES=<gpu_id>` (e.g. 0)
- if the gpu is not found after suspension, try

    `sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm`

    if it does not help, you have to reboot

# important commits
- after clean up: 0ee0cb683bb48e2e00472e231dd26a99a80a4217
