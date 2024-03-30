import os
import shutil
import numpy as np


import torch
import pandas as pd
import whisper
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, random_split


import whisper
#model = whisper.load_model("medium")
device = 'cuda:0'
#model.to(device)
dataset = []

X = []
y = []
dataset_folder = '../FASA_data_whisperx/out_dataset'
tmp = os.listdir(dataset_folder)

valid_folder = ['4','5','6','7','8','9']
for folder in tmp:
    if folder[0] in valid_folder:
        files = os.listdir(os.path.join(dataset_folder,folder))
        for f in files:
            if f.endswith('.mp3'):
                dataset.append((os.path.join(dataset_folder,folder,f),int(folder[0])-4))

for (dir,c) in dataset:
    x_, sr = torchaudio.load(dir)
    X.append(x_)
    y.append(c)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        #self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        audio = whisper.pad_or_trim(x.flatten())
        x= whisper.log_mel_spectrogram(audio).to(device)

        y = self.targets[index]
        
        
        return x, y
    
    def __len__(self):
        return len(self.data)
dataset = MyDataset(X, y)

#dataset[0]
#dataloader = DataLoader(dataset, batch_size=5)

train_d, test_d = random_split(dataset,[0.8,0.2])

trainloader = DataLoader(train_d, batch_size=4,drop_last=True,shuffle=True)
testloader = DataLoader(test_d, batch_size=4,drop_last=True,shuffle=True)


EPOCH=30

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(80, 64, 16)
        self.conv2 = nn.Conv1d(64, 64, 16)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 32)
        self.conv4 = nn.Conv1d(128, 128, 32)
        
        self.conv5 = nn.Conv1d(128, 256, 32)
        '''
        self.conv6 = nn.Conv1d(256, 256, 16)
        self.conv7 = nn.Conv1d(256, 256, 16)
        
        self.conv8 = nn.Conv1d(256, 512, 32)
        self.conv9 = nn.Conv1d(512, 512, 32)
        self.conv10 = nn.Conv1d(512, 512, 32)
        '''
        #self.pool = nn.AvgPool1d(1024)
        self.fc1 = nn.Linear(256*680, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        '''
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool(x)
        
        #print(x.shape)
        '''
        #x = model.encoder(x)
        #print(x.shape)
        #x = self.pool(x)
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)


for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        

        inputs = inputs.to(device)

        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #outputs = net(model.encoder(inputs))
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'Running loss of the network on the train: {running_loss}')
    running_loss=0.0
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            images = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            #outputs = net(model.encoder(images))
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            

            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test: {100 * correct // total} %')

print('Finished Training')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        images = inputs.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        print(predicted)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test: {100 * correct // total} %')