import torch
import torch.nn as nn
import yaml

from data_stuff.utils_3d import get_hp_position, get_hp_position_from_input, remove_first_dim
from networks.unet3d import UNet3d
#to start, move file into parent folder

def test_get_hp_position():
    #Fixture
    tensor=torch.zeros((5, 5, 5))
    tensor[3,2,1] = 1

    # Expected result
    expected = 3
    # Actual result
    actual=get_hp_position(tensor)[0]
    # Test
    assert actual==expected, "Result is not the expected position"

def test_get_hp_position_from_input():
    #Fixture
    with open("unittests/dummyInfo.yaml", "r") as f:
            info = yaml.safe_load(f)
    tensor = torch.Tensor([[
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[1,0,0], [0,0,0], [0,0,0]]],
        [
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,1,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]],
        [#sdf-tensor-dummy: max at hp_position (0,1,2)
        [[0,0,0], [0,0,1], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]],
        [#Material ID-tensor-dummy: max at hp_position (0,1,2)
        [[0,0,0], [0,0,1], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]]])
    
    # Expected result
    expected = 0
    # Actual result
    actual=get_hp_position_from_input(tensor,info)[0]
    # Test
    assert actual==expected, "Result is not the expected position"

def test_remove_first_dimension_3d():
    #Fixture
    #4x2x3
    tensor = torch.Tensor([
        [[0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0]],
        [[1,1,1], [1,1,1]],
        [[0,0,0], [0,0,0]]])
    
    position=2

    # Expected result
    expected = torch.ones((2,3))
    # Actual result
    actual=remove_first_dim(tensor,position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at position {position}"

def test_remove_dimension_4d():
     #Fixture
    tensor=torch.zeros((4, 2, 3, 4))
    tensor[:,0,:,:]=5
    tensor[:,1,:,:]=6

    position=0
    
    # Expected result
    expected = tensor[:,0,:,:]
    # Actual result
    actual=remove_first_dim(tensor,position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at position {position}"

def test_unet3d_init():
    depth = 2
    model = UNet3d(in_channels=2, out_channels=1, init_features=8, depth= depth, kernel_size=3).float()
    assert model is not None, "Model could not be initialized"
    
    assert all(any(isinstance(layer, nn.Conv3d) for layer in encoder) for encoder in model.encoders), "Not all encoders contain Conv3d layers"
    assert all(any(isinstance(layer, nn.Conv3d) for layer in decoder) for decoder in model.decoders), "Not all decoders contain Conv3d layers"
    assert all(isinstance(layer, nn.ConvTranspose3d) for layer in model.upconvs), "Not all Upconvs are ConvTranspose3d layers"
    assert all(isinstance(layer, nn.MaxPool3d) for layer in model.pools), "Not all Pools are MaxPool3d layers"
   
    assert len(model.encoders) == depth + 1, f"Expected {depth + 1} encoder layers, but got {len(model.encoders)}"
    assert len(model.decoders) == depth, f"Expected {depth} decoder layers, but got {len(model.decoders)}"
    assert len(model.upconvs) == depth, f"Expected {depth} upconv layers, but got {len(model.upconvs)}"

    final_conv = model.conv
    assert isinstance(final_conv, nn.Conv3d), "Final conv layer is not Conv3d"
    assert final_conv.in_channels == 8, f"Expected final conv in_channels = 8, but got {final_conv.in_channels}"
    assert final_conv.out_channels == 1, f"Expected final conv out_channels = 1, but got {final_conv.out_channels}"

def test_unet3d_output():
    model = UNet3d(in_channels=2, out_channels=1).float()
    dummy_input = torch.randn(6, 2, 8, 8, 8)  # (B, C, X, Y, Z)
    model.apply(model.weights_init)
    output = model(dummy_input)
    
    assert output is not None, "Model does not return any output"
    expected_shape = (6, 1, 8, 8, 8)
    assert output.shape == expected_shape, f"Unexpected output shape: got {output.shape}, expected {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
     
if __name__ == "__main__":
    print("tests start")
    test_get_hp_position()
    test_get_hp_position_from_input()
    test_remove_first_dimension_3d()
    test_remove_dimension_4d()
    test_unet3d_init()
    test_unet3d_output()
    print("tests finished")