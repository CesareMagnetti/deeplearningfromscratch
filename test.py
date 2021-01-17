import modules
import utils
import numpy as np
from PIL import Image

conv_layer = modules.Conv2D(in_channels=3, out_channels=3, kernel_size=(7,7), padding=3, stride=1)
img = Image.open("./test_image.jpg")
img = np.asarray(img).transpose((2,0,1))/255
batch = np.array([img,]*2)

utils.show_batch_of_tensors(batch, ncol=1)

out = conv_layer(conv_layer(batch))

out/=out.max()
print(conv_layer.weights.data)
print("input")
print(batch)
print("output")
print(out)
utils.show_batch_of_tensors(out, ncol=1)