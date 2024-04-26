import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor
import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"]='./data'
import shutil
import numpy as np

import torch.nn.functional as F
import torch.optim as optim

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
from transformers import WhisperForConditionalGeneration, WhisperConfig
#import hub
from tqdm import tqdm
normalizer = EnglishTextNormalizer()
import pdb
#ds = deeplake.dataset("hub://activeloop/timit-train")
#ds = hub.load("hub://activeloop/timit-train",access_method='download')


ds_root = '/projects/jialing/DeepLearning/Personalized_Whisper/dataset'

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

device = 'cuda:7'
class MyDataset(Dataset):
    def __init__(self, raw_data, transform=None):
        self.data = []
        self.targets = []
        self.transcription = []
        for datapoint in raw_data:
            self.data.append(datapoint['audio'])
            self.targets.append(datapoint['accent'])
            self.transcription.append(datapoint['transcript'])

    def __getitem__(self, index):
        x,sr = librosa.load(self.data[index],sr=44100)
        x_s = librosa.resample(x,orig_sr = 44100,target_sr=16000)
        d = self.targets[index]
        t = self.transcription[index]
        return x_s, d, t
    def __len__(self):
        return len(self.data)
    
dataset = MyDataset(all_data)

train_d, test_d = random_split(dataset,[0.8,0.2])
TRAIN = DataLoader(train_d, batch_size=1,drop_last=True,shuffle=True)
TEST = DataLoader(test_d, batch_size=1,drop_last=True,shuffle=True)

import torch.nn as nn


from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
model = get_peft_model(model, lora_config)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model.print_trainable_parameters()

model.to(device)

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=True)

    for data in progress_bar:
        optimizer.zero_grad()
        inputs, labels, text = data
        input_features = processor(inputs.squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
        decoder_input_ids = processor.tokenizer('<|startoftranscript|>'+text[0].lower(),return_tensors='pt').input_ids.to(device)
        # text = processor.tokenizer(text[0].lower()+'<|endoftext|>',return_tensors='pt').input_ids.to(device)

        outputs = model(input_features, decoder_input_ids=decoder_input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
    avg_loss = total_loss / len(data_loader)
    print(f"Average training loss: {avg_loss:.6f}")


def test_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc='Testing', leave=True)
    with torch.no_grad():
        for data in TEST:
            inputs, labels, text = data
            input_features = processor(inputs.squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
            generated_ids = model.generate(input_features)
            generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            wer_ = wer(normalizer(generated_test_text), normalizer(text[0].lower()))
            wer_scores.append(wer_)
        print(sum(wer_scores)/len(wer_scores))
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss() 

# Train and test the model
train_model(model, TRAIN, optimizer, criterion, device)
test_model(model, TEST, device) 