import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"]='./data'
import shutil
import numpy as np
import time

from tqdm import tqdm
import torch
import pandas as pd
import whisper
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, random_split
from collections import defaultdict
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer, WhisperConfig, WhisperProcessor
#import deeplake
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
import librosa
#import hub
import pdb
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
normalizer = EnglishTextNormalizer()

ds_root = '/projects/jialing/DeepLearning/Personalized_Whisper/dataset'
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '7'

#Arabic: 0a
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

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

data_by_dialect = []
train_data = []
test_data = []
for directory in os.listdir(ds_root):
    if directory in accent_mapping.keys():
        label = accent_mapping[directory]
        for item in os.listdir(os.path.join(ds_root,directory,'wav')):
            if item.endswith('.wav') and 'b' in item:
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                test_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})
                data_by_dialect.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})
            elif item.endswith('.wav'):
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()
                train_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})
                data_by_dialect.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})
device = 'cuda:1'



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
        t1 = self.processor.tokenizer(t+'<|endoftext|>').input_ids
        #t = self.processor.tokenizer.pad(t, return_tensors="pt")
        #print(t)
        #aaa
        #trans = self.transcription[index]
        # pdb.set_trace()
        return x_3s, x_s.squeeze(), d, t
        # return {'3second':x_3s,
        #         'input_features':x_s.squeeze(), 
        #         'dialect':d, 
        #         'labels':t}
    def __len__(self):
        return len(self.data)


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
        self.conv3 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3,padding=1)
        
        self.conv5 = nn.Conv2d(32, 64, 3,padding=1)

        self.conv6 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv7 = nn.Conv2d(64, 128, 3,padding=1)
        
        self.conv8 = nn.Conv2d(128, 128, 3,padding=1)
        
        #self.avgpool = nn.AvgPool1d(128)
        self.fc1 = nn.Linear(11520, 512)
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


dialect_to_checkpoint = {
    '0': 'model/lora_5e4/dialect_0/checkpoint-114',
    '1': 'model/lora_5e4/dialect_1/checkpoint-114',
    '2': 'model/lora_5e4/dialect_2/checkpoint-114',
    '3': 'model/lora_5e4/dialect_3/checkpoint-114',
    '4': 'model/lora_5e4/dialect_4/checkpoint-114',
    '5': 'model/lora_5e4/dialect_5/checkpoint-114'   
}

net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print(net)
net.load_state_dict(torch.load('./classifier.pt'))
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()
MODEL = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(MODEL)




def preprocess(data):
    processed_data = {}
    processed_data['input_features'] = processor(data["audio"]["array"], sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    processed_data['decoder_input_ids'] = processor.tokenizer('<|startoftranscript|>'+data['text'].lower(),return_tensors='pt').input_ids.to(device)
    processed_data['labels'] = processor.tokenizer(data['text'].lower()+'<|endoftext|>',return_tensors='pt').input_ids.to(device)
    return processed_data

test_d = MyDataset(test_data)
TEST = DataLoader(test_d, batch_size=20, drop_last=True, shuffle=True)

# wer_scores = []
# peft_model_id = "/projects/jialing/DeepLearning/Personalized_Whisper/model/baseline/checkpoint-669"
# peft_config = PeftConfig.from_pretrained(peft_model_id)
# model = WhisperForConditionalGeneration.from_pretrained(
#     peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
# )
# model = PeftModel.from_pretrained(model, peft_model_id)


# Measure average inference time
# inference_times = []

# for data in tqdm(TEST, desc="Evaluating model"):
#     start_time = time.time()  # Start timing
    
#     inputs_3s, input_features, labels, text = data
#     classifier_output = net(inputs_3s.to(device))
#     max_indices = torch.argmax(classifier_output, dim=1)
    
#     input_features = input_features.half().to(device)
#     generated_ids = model.generate(input_features)
#     generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     wer_ = wer(normalizer(text[0].lower()), normalizer(generated_test_text))
#     wer_scores.append(wer_)

#     # Record and calculate inference time
#     end_time = time.time()
#     inference_time = end_time - start_time
#     inference_times.append(inference_time)
    
#     # print(f"{labels}\t{text[0]}\t{generated_test_text}\t{wer_}")

# # Calculate and print average inference time and WER
# average_inference_time = sum(inference_times) / len(inference_times)
# print(f"Average inference time per batch: {average_inference_time:.4f} seconds")
# print(f"Average WER: {sum(wer_scores) / len(wer_scores):.4f}")

# Dictionary to hold preloaded models
preloaded_models = {}

# Preload all necessary models
for dialect, checkpoint in dialect_to_checkpoint.items():
    peft_config = PeftConfig.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    preloaded_models[dialect] = PeftModel.from_pretrained(model, checkpoint)

print("All models loaded successfully.")


wer_scores = {}  # Dictionary to store cumulative WER for each label
count = {}       # Dictionary to count occurrences of each label
inference_times = []

for index, data in enumerate(tqdm(TEST, desc="Processing batches")):
    start_time = time.time()  
    inputs_3s, input_features, labels, text = data
    classifier_output = net(inputs_3s.to(device))
    max_indices = torch.argmax(classifier_output, dim=1)  # Get max index for each item in the batch

    for i, max_index in enumerate(max_indices):
        dialect_key = str(max_index.item())
        model = preloaded_models[dialect_key]

        # Handling input features per item if necessary
        current_input_features = input_features[i].half().to(device)
        generated_ids = model.generate(current_input_features.unsqueeze(0))
        generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Compute WER and update statistics
        wer_ = wer(normalizer(text[i].lower()), normalizer(generated_test_text))
        label_id = labels[i].item()  # Get label for current item

        if label_id not in wer_scores:
            wer_scores[label_id] = 0
            count[label_id] = 0
        wer_scores[label_id] += wer_
        count[label_id] += 1
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average inference time per batch: {average_inference_time:.4f} seconds")
# After loop, calculate averages
for label_id in wer_scores:
    average_wer = wer_scores[label_id] / count[label_id]
    print(f'Average WER for label {label_id}: {average_wer}')
