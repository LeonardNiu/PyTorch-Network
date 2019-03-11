import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from model import *
import utils
import os

bs = 100
lr = 0.001
epoch = 2

# data loading
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()
imagepath = "MNIST/."
traindata = torchvision.datasets.MNIST(
    root=imagepath, train=True, transform=train_transform, download=True
)
testdata = torchvision.datasets.MNIST(
    root=imagepath, train=False, transform=test_transform, download=True
)
trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=bs, shuffle=True, 
)
testloader = torch.utils.data.DataLoader(
    testdata, batch_size=bs, shuffle=True
)

net = Net()

print "-------------------------------------------------"
print net
print '-------------------------------------------------'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

def loadckpt():
    print('==> Resuming from checkpoint..')
    savepath = "./modelparam"
    assert os.path.isdir(savepath), 'Error: no checkpoint directory found!'
    assert os.path.isfile(os.path.join(savepath, 'LeNet_params.pkl')), 'Error: no checkpoint file found!'
    checkpoint = torch.load(os.path.join(savepath,'LeNet_params.pkl'))
    net.load_state_dict(checkpoint)


def train(epoch):
    for EPOCH in range(epoch):
        trainloss = 0.0
        correct = 0
        total = 0
        for i, (inputs, target) in enumerate(trainloader):
            inputs, target = Variable(inputs), Variable(target)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,target)
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).sum()

            utils.progress_bar(
                i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (trainloss/(i+1), 100.*correct/total, correct, total)
            )
    print "Training Process Is Over"

def saveparam():
    savepath = "./modelparam"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    torch.save(net.state_dict(), os.path.join(savepath, "LeNet_params.pkl"))

def test():
    net.eval()
    testloss = 0.0
    correct = 0
    total = 0
    for i, (inputs, target) in enumerate(testloader):
        inputs, target = Variable(inputs), Variable(target)
        outputs = net (inputs)
        loss = criterion(outputs,target)
        testloss += loss.item()
        _,predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum()
        utils.progress_bar(
                i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (testloss/(i+1), 100.*correct/total, correct, total)
            )
    print "Test Process Is Over"



if __name__ == '__main__':

    #loadckpt()
    train(epoch)
    test()
    #saveparam()
