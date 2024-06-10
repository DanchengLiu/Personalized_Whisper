import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"]='./data'
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import shutil
import numpy as np
from tqdm import tqdm

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

gender_mapping = { # M:0, F:1
 'ABA': 0,
 'SKA': 1,
 'YBAA': 0,
 'ZHAA': 1,
 'BWC': 0,
 'LXC': 1,
 'NCC': 1,
 'TXHC': 0,
 'ASI': 0,
 'RRBI': 0,
 'SVBI': 1,
 'TNI': 1,
 'HJK': 1,
 'HKK': 0,
 'YDCK': 1,
 'YKWK': 0,
 'EBVS': 0,
 'ERMS': 0,
 'MBMPS': 1,
 'NJS': 1,
 'HQTV': 0,
 'PNV': 1,
 'THV': 1,
 'TLV': 0
}

print(accent_mapping)

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

train_data = []
val_data=[]
test_data = []
for directory in os.listdir(ds_root):
    if directory in gender_mapping.keys():
        gender = gender_mapping[directory]
        accent = accent_mapping[directory]
        for item in os.listdir(os.path.join(ds_root,directory,'wav')):
            if item.endswith('.wav') and ('b01' in item or 'b00' in item):
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                val_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':accent,
                                 'gender': gender,
                                 'transcript': transcript})
     
            elif item.endswith('.wav') and 'b' in item:
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                test_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':accent,
                                 'gender': gender,
                                 'transcript': transcript})
            elif item.endswith('.wav'):
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                train_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':accent,
                                 'gender': gender,
                                 'transcript': transcript})              
      
print(len(train_data))
print(len(val_data))
print(len(test_data))

device = 'cuda:0'
class MyDataset(Dataset):
    def __init__(self, raw_data, transform=None, model_name = "openai/whisper-tiny.en"):
        self.data = []
        self.targets = []
        self.transcription = []
        self.gender = []
        self.MODEL = model_name
        self.processor = WhisperProcessor.from_pretrained(self.MODEL)
        for datapoint in raw_data:
            self.data.append(datapoint['audio'])
            self.targets.append(datapoint['accent'])
            self.transcription.append(datapoint['transcript'])
            self.gender.append(datapoint['gender'])
        #self.transform = transform
        
    def __getitem__(self, index):
        x,sr = librosa.load(self.data[index],sr=44100)
        x_s = torch.tensor(librosa.resample(x,orig_sr = 44100,target_sr=16000))
        x_s=whisper.pad_or_trim(x_s.flatten()).to(device)
        x_s = whisper.log_mel_spectrogram(x_s)
        #x = librosa.util.fix_length(x,size=sr*3)
        #x_s = librosa.feature.melspectrogram(y=x, sr = sr, n_mels=80)
        #x_s = np.expand_dims(x_s,0)
        #x_3s = torch.tensor(x_s[:,:,:300]).to(device)
        x_3s = torch.unsqueeze(x_s[:,:300],0)
        # for 2d conv
        #x = np.expand_dims(x,axis=0)
        
        d = self.targets[index]
        t = self.transcription[index]
        g = self.gender[index]
        #trans = self.transcription[index]
        
        return x_3s,x_s,d,t,g#x_3s, x_s.squeeze(), d, t
    def __len__(self):
        return len(self.data)
    
train_d = MyDataset(train_data)
val_d = MyDataset(val_data)
test_d = MyDataset(test_data)

#train_d, test_d = random_split(dataset,[0.8,0.2])
TRAIN = DataLoader(train_d, batch_size=256,drop_last=True,shuffle=True)
VAL = DataLoader(val_d, batch_size=256,drop_last=True,shuffle=True)
TEST = DataLoader(test_d, batch_size=256,drop_last=True,shuffle=True)

print("-----------------------START DATA PROCESSING-----------------------")


EPOCH=10


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
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3,padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, 3,padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3,padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3,padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        #self.conv11 = nn.Conv2d(256, 256, 3,padding=1)
        #self.conv12 = nn.Conv2d(256, 256, 3,padding=1)
        #self.batchnorm6 = nn.BatchNorm2d(512)
        #self.avgpool = nn.AvgPool1d(128)
        #self.fc1 = nn.Linear(4608, 512)
        #self.fc1 = nn.Linear(2304, 256)
        self.fc1_gender = nn.Linear(4608, 256)
        self.fc2_gender = nn.Linear(256, 64)
        self.fc3_gender = nn.Linear(64, 2)
        
        #self.lstm = nn.LSTM(1,1024,3)
        #self.fc1 = nn.Linear(1024,256)
        #self.fc2 = nn.Linear(256,64)
        #self.fc3 = nn.Linear(64, 8)
        
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.batchnorm1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.relu(self.batchnorm2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = F.relu(self.batchnorm3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        x = F.relu(self.batchnorm4(x))
        x = self.pool(x)

        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        x = F.relu(self.batchnorm5(x))
        x = self.pool(x)

        #x = F.relu(self.conv11(x))
        #x = self.conv12(x)
        #x = F.relu(self.batchnorm6(x))
        #x = self.pool(x)
                        
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
        #print(x.shape)
        x = F.relu(self.fc1_gender(x))
        x = F.relu(self.fc2_gender(x))
        x = self.fc3_gender(x)
        return x


net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
PATH_CLASSIFIER = './model/classifier_10layer.pt'
net.load_state_dict(torch.load(PATH_CLASSIFIER),strict=False)

print(net)
net.to(device)

for param in net.parameters():
    param.requires_grad = False
for param in net.fc1_gender.parameters():
    param.requires_grad = True    
for param in net.fc2_gender.parameters():
    param.requires_grad = True    
for param in net.fc3_gender.parameters():
    param.requires_grad = True    
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


train_loss_trace = []
val_acc_trace = []
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in tqdm(enumerate(TRAIN, 0)):
        # get the inputs; data is a list of [inputs, labels]
        #x_s,labels,trans = data
        #x_s = processor(x_s, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
        #inputs = x_s[:,:,:300]
        inputs, _, labels, trans,gender = data
        
        
        inputs = inputs
        #print(inputs.shape)
        labels = gender.to(device)
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
    train_loss_trace.append(running_loss)
    running_loss=0.0
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in VAL:
            inputs, _, labels, trans, gender = data
            #x_s,labels,trans = data
            #x_s = processor(x_s, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
            #inputs = x_s[:,:,:300]            
            inputs = inputs
            labels = gender.to(device)
            # calculate outputs by running images through the network
            #outputs = net(model.encoder(images))
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            

            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the val: {100 * correct // total} %')
    val_acc_trace.append(100 * correct // total)
print('Finished Training')
print('training loss curve is:')
print(train_loss_trace)
print('val acc curve is:')
print(val_acc_trace)
print('Saving weights')
PATH_CLASSIFIER_gender = './model/classifier_10layer_gender.pt'
torch.save(net.state_dict(), PATH_CLASSIFIER_gender)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in TEST:
        inputs, _, labels, trans, gender = data
        #x_s,labels,trans = data
        #x_s = processor(x_s, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
        #inputs = x_s[:,:,:300]        
        inputs = inputs
        labels = gender.to(device)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        #print(predicted)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test: {100 * correct // total} %')







