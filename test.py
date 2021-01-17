import modules
import numpy as np

conv_layer = modules.Conv2D(in_channels=3, out_channels=10, kernel_size=(3,3), padding=1, stride=2)

img = np.random.rand(64, 3, 32, 32)

out = conv_layer(img)

print(img.shape, out.shape)