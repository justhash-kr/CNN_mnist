import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np
import cv2

from network import CNN

# print(torch.cuda.is_available())
# print(cv2.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN().to(device)


model_state = torch.load("/home/hash/python_test/model/practice/CNN_mnist/model/{}_epoch.pth".format(4))
model.load_state_dict(model_state)
with torch.no_grad():
    # X_test = mnist_test.data.view(len(mnist_test), 1, 28,28).float().to(device)
    # Y_test = mnist_test.targets.to(device)

    image = 255.0 - cv2.imread('/home/hash/Pictures/9.jpg', cv2.IMREAD_GRAYSCALE)
    image = image/255.0
    X_test = torch.Tensor([[image]]).to(device)
    # print(X_test)
    # print(X_test.shape)

    prediction = model(X_test)
    prediction = torch.argmax(prediction).item()
    print('Number is ', prediction)
    # accuracy = correc_prediction.float().mean()
    # print('Accuracy:',accuracy.item())