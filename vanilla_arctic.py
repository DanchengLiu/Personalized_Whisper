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
#import deeplake
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
import librosa
#import hub

normalizer = EnglishTextNormalizer()

#ds = deeplake.dataset("hub://activeloop/timit-train")
#ds = hub.load("hub://activeloop/timit-train",access_method='download')


ds_root = './data/l2arctic_release_v5'

#Arabic: 0
#Chinese: 1
#Hindi: 2
#Korean: 3
#Spanish: 4
#Vietnamese: 5
accent_mapping = {
    'ABA':0,
    'SKA':0,
    'YBAA':0,
    'ZHAA':0,
    'BWC':1,
    'LXC':1,
    'NCC':1,
    'TXHC':1,
    'ASI':2,
    'RRBI':2,
    'SVBI':2,
    'TNI':2,
    'HJK':3,
    'HKK':3,
    'YDCK':3,
    'YKWK':3,
    'EBVS':4,
    'ERMS':4,
    'MBMPS':4,
    'NJS':4,
    'HQTV':5,
    'PNV':5,
    'THV':5,
    'TLV':5
}


print(accent_mapping)

all_data = []
for directory in os.listdir(ds_root):
    if directory in accent_mapping.keys():
        label = accent_mapping[directory]
        for item in os.listdir(os.path.join(ds_root,directory,'wav')):
            if item.endswith('.wav'):
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                all_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})
print(len(all_data))

device = 'cuda:0'
class MyDataset(Dataset):
    def __init__(self, raw_data, transform=None):
        self.data = []
        self.targets = []
        self.transcription = []
        for datapoint in raw_data:
            self.data.append(datapoint['audio'])
            self.targets.append(datapoint['accent'])
            self.transcription.append(datapoint['transcript'])
        #self.transform = transform
        
    def __getitem__(self, index):
        x,sr = librosa.load(self.data[index],sr=44100)
        x_s = librosa.resample(x,orig_sr = 44100,target_sr=16000)
        #x = librosa.util.fix_length(x,size=sr*3)
        #x = librosa.feature.melspectrogram(y=x, sr = sr, n_mels=80)
        # for 2d conv
        #x = np.expand_dims(x,axis=0)
        d = self.targets[index]
        t = self.transcription[index]
        #trans = self.transcription[index]
        
        return x_s, d, t
    def __len__(self):
        return len(self.data)
    
dataset = MyDataset(all_data)

train_d, test_d = random_split(dataset,[0.8,0.2])
TRAIN = DataLoader(train_d, batch_size=1,drop_last=True,shuffle=True)
TEST = DataLoader(test_d, batch_size=1,drop_last=True,shuffle=True)

print("-----------------------START DATA PROCESSING-----------------------")


'''
EPOCH=30

print("----------------------MODEL START--------------------")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3,padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3,padding=1)
        self.pool = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3,padding=1)
        
        self.conv5 = nn.Conv2d(32, 64, 3,padding=1)

        self.conv6 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv7 = nn.Conv2d(64, 128, 3,padding=1)
        
        self.conv8 = nn.Conv2d(128, 128, 3,padding=1)
        
        #self.avgpool = nn.AvgPool1d(128)
        self.fc1 = nn.Linear(10240, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 6)
        
        #self.lstm = nn.LSTM(1,1024,3)
        #self.fc1 = nn.Linear(1024,256)
        #self.fc2 = nn.Linear(256,64)
        #self.fc3 = nn.Linear(64, 8)
        
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        
        x = F.relu(self.conv8(x))

        x = self.pool(x)
        
        #print(x.shape)
        
        #x = model.encoder(x)
        #print(x.shape)
        #x = self.pool(x)
        #print(x.shape)
        #x, _ = self.lstm(x)
        #x = x[:,-1,:]
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        
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
optimizer = optim.Adam(net.parameters(), lr=1e-4)


for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(TRAIN, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        

        inputs = inputs.to(device)
        #print(inputs.shape)
        labels = labels.to(device)
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
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
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
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        print(predicted)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test: {100 * correct // total} %')
'''

MODEL = "openai/whisper-tiny.en"
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
for data in TEST:
    inputs, labels, text = data
    #print(text)
    #print(inputs.shape)
    #inputs = inputs.to(device)
    #labels = labels.to(device)    
    input_features = processor(inputs.squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    generated_ids = model.generate(input_features)
    generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    wer_ = wer(normalizer(generated_test_text), normalizer(text[0].lower()))
    wer_scores.append(wer_)
    print(str(labels)+'\t\t'+text[0]+'\t\t'+generated_test_text+'\t\t'+str(wer_))
    
print(sum(wer_scores)/len(wer_scores))

