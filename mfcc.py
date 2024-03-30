import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"]='./data'
import shutil
import numpy as np


import torch
import pandas as pd
import whisper
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, random_split

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer, WhisperConfig, WhisperProcessor
import deeplake
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
import hub
import librosa
normalizer = EnglishTextNormalizer()

#ds = deeplake.dataset("hub://activeloop/timit-train")
#ds = hub.load("hub://activeloop/timit-train",access_method='download')
ds = hub.load("./data/hub_activeloop_timit-train")
print(ds[0])
aaa
#ds_test = hub.load("hub://activeloop/timit-test",access_method='download')
ds_test = hub.load("./data/hub_activeloop_timit-test")
trainloader = ds.pytorch(batch_size=1, shuffle=False)

testloader = ds_test.pytorch(batch_size=1, shuffle=False)
device = 'cuda:1'
class MyDataset(Dataset):
    def __init__(self, dataloader, transform=None):
        self.data = []
        self.targets = []
        self.transcription = []
        for datapoint in dataloader:
            self.data.append(datapoint['audios'])
            self.targets.append(datapoint['dialects'])
            self.transcription.append(datapoint['texts'])
        #self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index].squeeze().float().numpy()

        mfcc = np.mean(librosa.feature.mfcc(y=x, sr=16000, n_mfcc=100).T, axis=0)

        #audio = whisper.pad_or_trim(x.flatten().float())
        #x = audio
        #x = whisper.log_mel_spectrogram(audio)

        d = self.targets[index].flatten()
        trans = self.transcription[index]
        
        return mfcc, d, trans
    def __len__(self):
        return len(self.data)

print("-----------------------START DATA PROCESSING-----------------------")
trainset = MyDataset(trainloader)
testset = MyDataset(testloader)
TRAIN = DataLoader(trainset, batch_size=1,drop_last=True,shuffle=True)
TEST = DataLoader(testset, batch_size=1,drop_last=True,shuffle=True)


EPOCH=30

print("----------------------MODEL START--------------------")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,25)
        self.fc3 = nn.Linear(25, 8)
    def forward(self, x):
        
        #x, _ = self.lstm(x)
        #x = x[:,-1,:]
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print(net)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(TRAIN, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data
        

        inputs = inputs.to(device)
        #print(inputs.shape)
        labels = labels.squeeze(axis=1).to(device)
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #outputs = net(model.encoder(inputs))
        outputs = net(inputs)
        #print(outputs.data.shape)
        #print(labels)
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
        for data in TEST:
            inputs, labels, _ = data
            
            inputs = inputs.to(device)
            labels = labels.squeeze(axis=1).to(device)
            # calculate outputs by running images through the network
            #outputs = net(model.encoder(images))
            outputs = net(inputs)
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
    for data in TEST:
        inputs, labels, _ = data
        
        inputs = inputs.to(device)
        labels = labels.squeeze(axis=0).to(device)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        print(predicted)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test: {100 * correct // total} %')