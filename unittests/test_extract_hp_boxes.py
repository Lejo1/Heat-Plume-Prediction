from preprocessing.domain_classes.domain import Domain
import pathlib
from torch import tensor,where,stack,max

def test_extract_hp_boxes():
    domain = Domain(pathlib.Path('unittests/example_domain/5HP_1 inputs_gksit'), stitching_method="max")
    material_ids = domain.get_input_field_from_name("Material ID")
    
    
    #hp box so big that it does not fit in domain
    expected_size = [512,128]
    expected_pos_hps = stack(list(where(material_ids == max(material_ids))), dim=0).T
    expected_number_of_hp = len(expected_pos_hps)

    single_hps = domain.extract_hp_boxes(size_hp=tensor(expected_size))

    assert len(single_hps) == expected_number_of_hp
    assert list(single_hps[0].inputs[0].shape) == expected_size

    for hp in single_hps:
        assert hp.pos in expected_pos_hps

if __name__ == "__main__":
    test_extract_hp_boxes()