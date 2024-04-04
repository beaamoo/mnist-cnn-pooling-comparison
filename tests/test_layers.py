# Test cases for CNN layers

import numpy as np
from cnn.conv import Conv3x3
from cnn.maxpool import MaxPool2
from cnn.softmax import Softmax

# Convolution layer
conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
image = np.random.randn(28, 28)
out = conv.forward(image)
d_L_d_out = np.random.randn(26, 26, 8)
learn_rate = 0.005
d_L_d_input = conv.backprop(d_L_d_out, learn_rate)

# Max Pooling layer
pool = MaxPool2()  # 26x26x8 -> 13x13x8
out = pool.forward(out)
d_L_d_out = np.random.randn(13, 13, 8)
d_L_d_input = pool.backprop(d_L_d_out)

# Softmax layer
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10
out = softmax.forward(out)
d_L_d_out = np.random.randn(10)
d_L_d_input = softmax.backprop(d_L_d_out, learn_rate)

print('Tests passed.')
