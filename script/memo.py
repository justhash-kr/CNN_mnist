# import numpy as np

# x = np.random.rand(10, 1, 28, 28)
# print(x.shape)

# print(x[1].shape)

# print(x[0,0].shape)

import torch
import torch.nn as nn

inputs = torch.Tensor(1,1,28,28)
print('{}'.format(inputs.shape))

conv1 = nn.Conv2d(1,32,3,padding=1)

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# print(conv2)

pool = nn.MaxPool2d(2)
# print(pool)

fc = nn.Linear(3136, 10)
# print(fc)

out = conv1(inputs)
print(out.shape)

out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

out = out.view(out.size(0),-1)
print(out.shape)

out = fc(out)
print(out.shape)