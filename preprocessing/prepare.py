import shutil

from data_stuff.utils import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_paths import Paths2HP,  set_paths_2hpnn
from preprocessing.prepare_xhp_dataset import prepare_xhp_dataset

def prepare_data_and_paths(settings:SettingsTraining):
    """
    prepare datasets using raw data if necessary
    set up paths to those prepared datasets for future usage
    """
    paths: Paths2HP
    paths, settings.inputs, destination_dir = set_paths_2hpnn(settings.dataset_raw, settings.inputs,model_name=settings.model, dataset_prep = settings.dataset_prep,)
    settings.dataset_prep = paths.datasets_boxes_prep_path
    settings.dataset_prep = paths.dataset_1st_prep_path

    settings.make_destination_path(destination_dir)
    settings.save_notes()
    settings.model = paths.model_1hp_path

    if settings.case == "prepare":
        #generate prepared dataset for raw datasets with multiple heat pumps
        if settings.device != "cpu":
            print("Devices other than cpu not supported! Switching to cpu")
            settings.device = "cpu"
        prepare_xhp_dataset(paths, settings)
        return settings, paths
    else:
        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        # if test, always want to prepare because the normalization parameters have to match
        #some datasets may only be available in prepared form, eg those created through prepare xhp, so there is no raw dataset to be prepared
        if not settings.only_prep:
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")


        if settings.case == "train":
            shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
        settings.save()
        return settings, paths