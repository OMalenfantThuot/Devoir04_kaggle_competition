# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop = nn.Dropout2d(p=0.05)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) #was 5
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv5 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv6 = nn.Conv2d(128, 256, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)        
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 31)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.drop(self.pool(self.bn2(F.relu(self.conv2(x)))))
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.drop(self.pool(self.bn3(F.relu(self.conv4(x)))))
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.drop(self.pool(self.bn4(F.relu(self.conv6(x)))))
        x = x.view(-1, 256 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class kaggle_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        dat = self.data[index]
        if self.transforms is not None:
            dat = self.transforms(dat)
        return (dat,self.labels[index])
   
    def __len__(self):
        return self.labels.shape[0]

def define_classes(labels):
    classes = {}
    label = []
    c = 0
    for i,j in labels:
        if j.decode('utf-8') not in classes:
            classes[j.decode('utf-8')] = c
            c += 1
        label.append(classes[j.decode('utf-8')])    
    return classes, torch.LongTensor(label)

def reshape_images(images):
    train_im = []
    #noise = torch.zeros(1,100,100)
    #c = 0
    #_,label = define_classes(train_labels)
    #for i in range(images.shape[0]):
    #    if label[i]==21:
    #        noise += torch.Tensor(images[i][1]).reshape((1,100,100)) / 255.
    #        c += 1.
    #noise /= c   
    for i in range(images.shape[0]):
        train_im.append(torch.Tensor(images[i][1].reshape((1,100,100))) / 255.)
    return train_im

def train(images, labels):
    
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = kaggle_dataset(
        reshape_images(images)[:9500],define_classes(labels)[1][:9500],
            transforms=transform)
    
    testset = kaggle_dataset(
        reshape_images(images)[9500:],define_classes(labels)[1][9500:],
            transforms=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

    classes = [str(i) for i in range(31)]
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.025, weight_decay=0.0001,
        momentum=0.9)
    
    loss_train = []
    loss_test = []
    err_train = []
    err_test = []
    for epoch in range(40):  # loop over the dataset multiple times
        if epoch>20:
            optimizer = optim.SGD(net.parameters(), lr=0.01,
                weight_decay=0.0001, momentum=0.9)
        if epoch>25: #was30
            optimizer = optim.SGD(net.parameters(), lr=0.001,
                weight_decay=0.0001, momentum=0.9)
        if epoch>30: #was40
            optimizer = optim.SGD(net.parameters(), lr=0.0005,
                weight_decay=0.0001, momentum=0.9)
        correct = 0.
        total = 0.
        running_loss_train = 0.0
        running_loss_test = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() / 9500
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        err_train.append(1 - correct / total)

        correct = 0.
        total = 0.
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                loss = criterion(outputs, labels.to(device))
                running_loss_test += loss.item() / 500
        err_test.append(1 - correct / total)
        
        loss_train.append(running_loss_train)
        loss_test.append(running_loss_test)
        print('Epoch: {}'.format(epoch))
        print('Train loss: {0:.4f} Train error: {1:.2f}'.format(
            loss_train[epoch], err_train[epoch]))
        print('Test loss: {0:.4f} Test error: {1:.2f}'.format(
            loss_test[epoch], err_test[epoch]))       
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print('Accuracy of the network on the 1000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(31))
    class_total = list(0. for i in range(31))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(device)).squeeze()
            for i in range(c.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(31):
        if class_total[i]!=0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i]/class_total[i]))
        else:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i]))
    return net, loss_train, loss_test, err_train, err_test

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)
    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)        

    images_train = np.load('train_images.npy',
        encoding='latin1')
    
    train_labels = np.genfromtxt('train_labels.csv',
        names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])
    
    t1 = time.time()    
    net, loss_train, loss_test, e_train, e_test = train(images_train, train_labels)    
    print(time.time()-t1)
    torch.cuda.empty_cache()

    plt.figure()
    plt.plot(range(40),loss_train, 'sk-', label='Train')
    plt.plot(range(40),loss_test, 'sr-', label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average cost')
    plt.legend()

    plt.figure()
    plt.plot(range(40),e_train, 'sk-', label='Train')
    plt.plot(range(40),e_test, 'sr-', label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()