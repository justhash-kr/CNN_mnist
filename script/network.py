import torch
import torch.nn as nn

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=3, stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2))
        
        self.fc = torch.nn.Linear(7*7*64,10, bias=True)

        self.softmax = torch.nn.Softmax()

        torch.nn.init.xavier_uniform_(self.fc.weight) # initialize weight of fc layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out