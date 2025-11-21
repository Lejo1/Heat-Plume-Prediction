import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import processing.networks.unetVariants as uV
import processing.networks.unet as u

def plot_receptive_field(model:uV.UNetNoPad2, input_shape, target_position=None):
    model.eval()  # Set model to evaluation mode
    
    # Create a dummy input
    input_tensor = torch.randn(1,*input_shape, requires_grad=True)
    print(f"{input_tensor[0].shape}, {len(input_tensor)}")

    # Forward pass through the model to get the output
    output = model(input_tensor)
    print(f"{output.shape=}")

    if target_position is None:
        # Default target position is the center of the output
        target_position = (output.size(2) // 2, output.size(3) // 2)
    print(f"{target_position=}")

    # Create a gradient tensor with the same shape as the output
    grad_output = torch.zeros_like(output)
    
    # Set the gradient at the target position to 1
    grad_output[0,0, target_position[0], target_position[1]] = 1.0
    # grad_output[0,1, target_position[0], target_position[1]] = 1.0

    # Perform backward pass to get the gradient with respect to the input
    model.zero_grad()
    output.backward(grad_output)
    
    # Get the gradient with respect to the input
    print("1", input_tensor.grad.shape)
    grad_input = input_tensor.grad #torch.concatenate([t.grad[:,:] for t in input_tensor])
    print("2", grad_input.shape)
    grad_input = grad_input[0,0].detach().numpy()

    print("Non zero values:", np.count_nonzero(grad_input))
    print("Square:", np.sqrt(np.count_nonzero(grad_input)))
    print()

    # Plot the resulting gradient
    factor=1
    plt.imshow(np.abs(grad_input), cmap='hot', interpolation='nearest', vmax=np.max(grad_input / factor))
    plt.colorbar()
    plt.title(f'Receptive Field at Output Position {target_position}')
    plt.savefig("receptive_field.png", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    model = u.UNet(in_channels=3, out_channels=2, init_features=32, depth=6, kernel_size=5)
    input_shape = (3, 480, 480)  # Example input shape (channels, height, width)
    
    plot_receptive_field(model, input_shape)
