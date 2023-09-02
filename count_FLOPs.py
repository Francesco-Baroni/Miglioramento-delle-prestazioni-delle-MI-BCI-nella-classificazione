import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv3D, SeparableConv2D, Dense

# Load the fitted model
model = tf.keras.models.load_model('saved_model/EEGNet_1001_8ele')

model.summary()

# Execute one iteration of the model passing the input with te correct dimensions
sample_input = tf.random.normal((1, 16, 125, 1))
model(sample_input)

def count_flops(model):
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, (Conv2D, DepthwiseConv2D, SeparableConv2D, Conv3D)):
            # Get the shape of the weights tensor
            weights_shape = layer.get_weights()[0].shape
            
            # Check if it's a Conv2D or Conv3D layer
            if len(weights_shape) == 4:  # Conv2D
                kernel_height, kernel_width, filters_in, filters_out = weights_shape
            elif len(weights_shape) == 5:  # Conv3D
                _, kernel_height, kernel_width, filters_in, filters_out = weights_shape
            else:
                continue  # Skip other layer types
            
            # Calculate the number of operations for a convolution
            kernel_size = kernel_height * kernel_width
            flops_per_filter = kernel_size * filters_in
            flops_per_output = flops_per_filter * filters_out
            total_flops += flops_per_output * layer.output_shape[1] * layer.output_shape[2]

        elif isinstance(layer, Dense):
            # Calculate the number of operations for a dense layer
            flops_per_output = layer.input_shape[-1] * layer.units
            total_flops += flops_per_output

    return total_flops


flops = count_flops(model)
print("Number of FLOPs:", flops)
