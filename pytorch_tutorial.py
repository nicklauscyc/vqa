import torch
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 


# compose together multiple different transformations
# transforms.ToTensor() converts a PIL Image/numpy.ndarray to tensor 
# transforms.Normalize normalizes a tensor image with mean and standard deviation (one for each channel)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# load data for the sample
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
                                        
# torch.utils.DataLoader works for map-style datasets & iterable-style datasets
# Need to figure out how to organize the data so that it can be read like this
# num_workers lets us to multi-process data loading 
# shuffle --> data is reshuffled at every epoch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

#create the neural net 
#Conv2d, first param is the number of in-channels (in our case 3 for RGB), second param is the number of out-channels, third is kernel size 
#Max Pool applies 2D max pooling over an input signal composes of several input planes...first param is kernel size, second is stride)
# nn.Linear applies a linear transformation to the incoming data, first param number of in channels, second is number of out channels

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # second number matches *
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # * first number here
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
