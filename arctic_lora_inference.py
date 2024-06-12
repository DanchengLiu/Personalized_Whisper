import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"]='./data'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

device = 'cuda'



class MyDataset(Dataset):
    def __init__(self, raw_data, transform=None, model_name = "openai/whisper-tiny.en"):
        self.data = []
        self.targets = []
        self.transcription = []
        self.MODEL = model_name
        self.processor = WhisperProcessor.from_pretrained(self.MODEL)
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
        
        x_s = self.processor(x_s, sampling_rate=16_000, return_tensors="pt").input_features
        x_3s = x_s[:,:,:300]

        # for 2d conv
        #x = np.expand_dims(x,axis=0)
        
        d = self.targets[index]
        t = self.transcription[index]
        #t = self.processor.tokenizer(t+'<|endoftext|>', return_tensors="pt").input_ids
        #t = self.processor.tokenizer.pad(t, return_tensors="pt")
        #print(t)
        #aaa
        #trans = self.transcription[index]
        
        return x_3s, x_s.squeeze(), d, t
        #return x_s, d,t
    def __len__(self):
        return len(self.data)
    
#dataset = MyDataset(all_data)
train_d = MyDataset(train_data)
test_d = MyDataset(test_data)
#train_d, test_d = random_split(dataset,[0.8,0.2])
TRAIN = DataLoader(train_d, batch_size=1,drop_last=True,shuffle=True)
TEST = DataLoader(test_d, batch_size=1,drop_last=True,shuffle=True)

print("-----------------------START DATA PROCESSING-----------------------")






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
        self.fc1 = nn.Linear(4608, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)
        
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def encoder(self,x):
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
        return x
    def classifier(self,x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x        

net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print(net)
net.to(device)


#model = WhisperForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)




import torch

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig

from transformers import AutoProcessor, Wav2Vec2ConformerForCTC, Wav2Vec2ForCTC
#processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
peft_model_id = "model/lora/base/checkpoint-1335" # Use the same model ID as before.
peft_model_id2 = "model/lora/base/checkpoint-890" # Use the same model ID as before.
peft_model_id3 = "model/lora/base/checkpoint-445" # Use the same model ID as before.
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=False, device_map="auto"
)
print(sum(p.numel() for p in model.parameters()))
#model = WhisperForConditionalGeneration.from_pretrained(
#    "openai/whisper-tiny.en", load_in_8bit=False, device_map="auto"
#)
from peft import prepare_model_for_kbit_training
model = PeftModel.from_pretrained(model, peft_model_id,adapter_name="1")
print(sum(p.numel() for p in model.parameters()))


some_num = 9
list_of_ad = ["1"]

for i in range(some_num):
    model.load_adapter(peft_model_id2,adapter_name=str(i+2))
    list_of_ad.append(str(i+2))

some_weight = [1/(some_num+1) for i in range(some_num+1)]
model.add_weighted_adapter(
                    adapters=list_of_ad,
                    weights=some_weight,
                    combination_type="cat",
                    adapter_name="cat"
                )
model.set_adapter("cat")
print(sum(p.numel() for p in model.parameters()))
#model = prepare_model_for_kbit_training(model)
MODEL = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(MODEL)


import time

total_time = 0.0
wer_scores = []
start_time = time.time()
for data in TEST:
    inputs_3s, input_features, labels, text = data
    input_features = input_features.to(device)
    #print(text)
    #print(inputs.shape)
    #inputs = inputs.to(device)
    #labels = labels.to(device)    
    #input_features = processor(inputs.squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    
    #inputs_3s = inputs_3s.to(device)
    #outputs = net.encoder(inputs_3s)
    #for i in range(some_num+1):
    #    outputs1 = net.classifier(outputs)
    #outputs2 = net.classifier(outputs)
    
    
    generated_ids = model.generate(input_features)
    generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    wer_ = wer(normalizer(text[0].lower()), normalizer(generated_test_text))
    wer_scores.append(wer_)
    #print(str(labels)+'\t\t'+text[0]+'\t\t'+generated_test_text+'\t\t'+str(wer_))
    
print(sum(wer_scores)/len(wer_scores))
end_time = time.time()
total_time+=end_time-start_time
print('-------------------------------------------------------------')
print(total_time)