import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np
import cv2

from network import CNN

train = True
# print(torch.cuda.is_available())
# print(cv2.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device=='cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
traing_epochs = 20
batch_size = 100

mnist_train = dsets.MNIST(root='/home/hash/python_test/model/practice/CNN_mnist/data/MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='/home/hash/python_test/model/practice/CNN_mnist/data/MNIST_data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size=batch_size, # size of mini_batch
                                          shuffle=True,
                                          drop_last=True)

model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device) # softmax함수가 포함된 Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader) # mini_batch size가 100개 이므로, 600이 나온 거면 총 data가 600*100개 라는 의미.
# batch가 600개, batch size가 100. 하나의 배치에 100개의 data가 있다는 뜻.

if train:
    print('총 배치의 수 : {}'.format(total_batch))
    print('미니배치 사이즈 : {}'.format(batch_size))
    print('총 데이터 수 : {}'.format(total_batch*batch_size))
    for epoch in range(traing_epochs):
        avg_cost = 0

        for X,Y in data_loader: # X:minibatch, Y:label
            print(X[0,0])
            print(X.shape)
            exit()
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost/total_batch

        print('[Epoch: {:>4}] cost = {:>9}'.format(epoch+1, avg_cost))
        if (epoch+1)%5==0:
            torch.save(model.state_dict(),"/home/hash/python_test/model/practice/CNN_mnist/model/{}_epoch.pth".format((epoch+1)//5))
else:
    model_state = torch.load("/home/hash/python_test/model/practice/CNN_mnist/model/{}_epoch.pth".format(4))
    model.load_state_dict(model_state)
    with torch.no_grad():
        X_test = mnist_test.data.view(len(mnist_test), 1, 28,28).float().to(device)
        Y_test = mnist_test.targets.to(device)

        prediction = model(X_test)
        correc_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correc_prediction.float().mean()
        print('Accuracy:',accuracy.item())