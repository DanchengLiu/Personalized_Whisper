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

normalizer = EnglishTextNormalizer()

#ds = deeplake.dataset("hub://activeloop/timit-train")
#ds = hub.load("hub://activeloop/timit-train",access_method='download')
ds = hub.load("./data/hub_activeloop_timit-train")


#ds_test = hub.load("hub://activeloop/timit-test",access_method='download')
ds_test = hub.load("./data/hub_activeloop_timit-test")
trainloader = ds.pytorch(batch_size=1, shuffle=False)

testloader = ds_test.pytorch(batch_size=1, shuffle=False)

device = 'cuda:0'
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
        x = self.data[index].squeeze(axis=0).float()
        
        #audio = whisper.pad_or_trim(x.flatten().float())
        #x = audio
        #x = whisper.log_mel_spectrogram(audio)

        d = self.targets[index].flatten()
        trans = self.transcription[index]
        
        return x, d, trans
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
        '''
        self.conv1 = nn.Conv1d(80, 64, 16)
        self.conv2 = nn.Conv1d(64, 64, 16)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 32)
        self.conv4 = nn.Conv1d(128, 128, 32)
        
        self.conv5 = nn.Conv1d(128, 256, 32)
        '''
        '''
        self.conv6 = nn.Conv1d(256, 256, 16)
        self.conv7 = nn.Conv1d(256, 256, 16)
        
        self.conv8 = nn.Conv1d(256, 512, 32)
        self.conv9 = nn.Conv1d(512, 512, 32)
        self.conv10 = nn.Conv1d(512, 512, 32)
        '''
        #self.pool = nn.AvgPool1d(1024)
        #self.fc1 = nn.Linear(256*680, 512)
        #self.fc2 = nn.Linear(512, 64)
        #self.fc3 = nn.Linear(64, 8)
        
        self.lstm = nn.LSTM(1,1024,3)
        self.fc1 = nn.Linear(1024,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64, 8)
        
    def forward(self, x):
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        '''
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
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
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

'''
MODEL = "openai/whisper-small.en"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#model = WhisperForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.to(device)
processor = WhisperProcessor.from_pretrained(MODEL)

def preprocess(data):
    processed_data = {}
    processed_data['input_features'] = processor(data["audio"]["array"], sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    processed_data['decoder_input_ids'] = processor.tokenizer('<|startoftranscript|>'+data['text'].lower(),return_tensors='pt').input_ids.to(device)
    processed_data['labels'] = processor.tokenizer(data['text'].lower()+'<|endoftext|>',return_tensors='pt').input_ids.to(device)
    return processed_data


wer_scores = []
for i in testloader:
    
    input_features = processor(i['audios'].squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    generated_ids = model.generate(input_features)
    generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    wer_ = wer(normalizer(generated_test_text), normalizer(i['texts'][0].lower()))
    wer_scores.append(wer_)
    print(str(i['dialects'])+'\t\t'+i['texts'][0]+'\t\t'+generated_test_text+'\t\t'+str(wer_))
print(sum(wer_scores)/len(wer_scores))

'''