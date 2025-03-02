import pathlib
from data_stuff.utils import SettingsTraining
from preprocessing.prepare_paths import Paths2HP
from postprocessing.iterative_estimation import iterative_estimation
from networks.unet import UNet
from main import run
import unittest
import shutil
import os

class testIterativeEstimation(unittest.TestCase):

    def test_iterative_estimation(self):
        pathlib.Path("unittests/dummy_destination").mkdir(parents=True, exist_ok=True)
        settings = SettingsTraining(dataset_raw='', 
                                    inputs='gksit', 
                                    device='cuda:0', 
                                    epochs=1, 
                                    destination=pathlib.Path('unittests/dummy_destination'), 
                                    dataset_prep=pathlib.Path('unittests/example_domain/5HP_1 inputs_gksit'), 
                                    case='iterative', 
                                    finetune=False, 
                                    model=pathlib.Path('unittests/example_domain/5HP_1 inputs_gksit'), 
                                    test=False, 
                                    case_2hp=False, 
                                    prepare_xhp=False, 
                                    visualize=False, 
                                    only_prep=True, 
                                    save_inference=False, 
                                    architecture='standard', 
                                    notes='', 
                                    skip_per_dir=256, 
                                    len_box=256)
        paths = Paths2HP(raw_path=pathlib.Path(''),
                        dataset_1st_prep_path=pathlib.Path(''),
                        datasets_boxes_prep_path=pathlib.Path(''),
                        dataset_model_trained_with_prep_path=pathlib.Path(''),
                        model_1hp_path=pathlib.Path('')
                        )
        model = UNet(in_channels=len(settings.inputs)).float()

        try:
            iterative_estimation(model,settings, paths)
        except Exception:
            assert False, "iterative estimation did not succeed"
        self.assertTrue(True)
        for root, dirs, files in os.walk('unittests/dummy_destination'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    
class testRun(unittest.TestCase):
    def test_run(self):
        pathlib.Path("unittests/dummy_destination").mkdir(parents=True, exist_ok=True)
        settings = SettingsTraining(dataset_raw='', 
                                    inputs='gksit', 
                                    device='cuda:0', 
                                    epochs=1, 
                                    destination=pathlib.Path('unittests/dummy_destination'), 
                                    dataset_prep=pathlib.Path('unittests/example_domain/test_10dp'), 
                                    case='train', 
                                    finetune=False, 
                                    model=pathlib.Path('unittests/example_domain/test_10dp'), 
                                    test=False, 
                                    case_2hp=False, 
                                    prepare_xhp=False, 
                                    visualize=False, 
                                    only_prep=True, 
                                    save_inference=False, 
                                    architecture='standard', 
                                    notes='', 
                                    skip_per_dir=256, 
                                    len_box=256)
        paths = Paths2HP(raw_path=pathlib.Path(''),
                        dataset_1st_prep_path=pathlib.Path(''),
                        datasets_boxes_prep_path=pathlib.Path(''),
                        dataset_model_trained_with_prep_path=pathlib.Path(''),
                        model_1hp_path=pathlib.Path('')
                        )
        with self.assertRaises(FileNotFoundError):
            run(settings, paths)
        for root, dirs, files in os.walk('unittests/dummy_destination'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

if __name__ == "__main__":
    pathlib.Path("unittests/dummy_destination").mkdir(parents=True, exist_ok=True)
    unittest.main()