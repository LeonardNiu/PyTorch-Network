#coding=utf-8

import torch
import torch.nn as nn
import argparse
import torchvision as tv
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import gzip, struct

from model import *

def _read(image, label):
    imagepath = './mnistdata/' + image
    labelpath = './mnistdata/' + label
    with gzip.open(labelpath) as fl:
        magic, num = struct.unpack(">II", fl.read(8))
        label = np.fromstring(fl.read(), dtype=np.int8)
    with gzip.open(imagepath, 'rb') as fi:
        magic, num, rows, cols = struct.unpack('>IIII', fi.read(16))
        image = np.fromstring(fi.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label

def getdata():
    trainimg, trainlbl = _read(
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'
    )
    testimg, testlbl = _read(
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    )
    return trainimg, trainlbl, testimg, testlbl

epoch = 10
bs = 25
lr = 0.001

transfrom = transforms.ToTensor()
trainimg, trainlbl, testimg, testlbl = getdata()
trainimg = torch.from_numpy(trainimg.reshape(-1, 1, 28, 28)).float() 
trainlbl = torch.from_numpy(trainlbl.astype(int))
testimg,testlbl = [torch.from_numpy(testimg.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(testimg.astype(int))]

traindataset = torch.utils.data.TensorDataset(trainimg, trainlbl)
test_dataset = torch.utils.data.TensorDataset(testimg, testlbl)
trainloader = torch.utils.data.DataLoader(
    traindataset, batch_size=bs, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=bs, shuffle=True, num_workers=2
)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)


def trainproc(EPOCH):
    total = 0.0
    correct = 0.0
    acc = 0.0
    for epoch in range(EPOCH):
        trainingloss = 0.0
        for i, (inputs, target) in enumerate(trainloader):
            inputs, target = Variable(inputs), Variable(target)

            optimizer.zero_grad()

            outputs = net(inputs)
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            trainingloss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            # total pics
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            acc = float(correct / total)

            if (i % 100 == 99):
                print ('Epoch is %d, iteration is %d, loss is %.03f'
                % (epoch + 1, i, trainingloss / 100)
                )
                print ('Acc is %.6f' % (acc))
                trainingloss = 0
       

def testproc():
    net.eval()
    testloss = 0.0
    correct = 0
    total = 0
    acc = 0.0
    for inputs, target in testloader:
        inputs, target = Variable(inputs), Variable(target)
        outputs = net(inputs)

        #_, predicted = torch.max(outputs.data, 1)
        predicted = outputs.data.max(1, keepdim=True)[1]
        total += target.size(0)
        # print predicted.size()
        # print target.size()
        correct += predicted.eq(target.data.view_as(predicted)).cpu().sum()
        #correct += predicted.eq(target.data).cpu().sum()
        acc = correct / total

    

if __name__ == '__main__':
    #trainproc(epoch)
    testproc()

    
