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

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

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
            elif item.endswith('.wav'):
                f = open(os.path.join(ds_root,directory,'transcript',item.split('.wav')[0]+'.txt'))
                transcript = f.read()
                f.close()

                train_data.append({'audio':os.path.join(ds_root,directory,'wav',item),
                                 'accent':label,
                                 'transcript': transcript})                
print(len(train_data))
print(len(test_data))

device = 'cuda:0'



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
        t = self.processor.tokenizer(t+'<|endoftext|>').input_ids
        #t = self.processor.tokenizer.pad(t, return_tensors="pt")
        #print(t)
        #aaa
        #trans = self.transcription[index]
        
        return {#'3second':x_3s, 
                'input_features':x_s.squeeze(), 
                #'dialect':d, 
                'labels':t}
    def __len__(self):
        return len(self.data)
    
train_d = MyDataset(train_data)
test_d = MyDataset(test_data)
#_,test_d = random_split(test_d,[0.9,0.1])
#train_d, test_d = random_split(dataset,[0.8,0.2])
TRAIN = DataLoader(train_d, batch_size=32,drop_last=True,shuffle=True)
TEST = DataLoader(test_d, batch_size=32,drop_last=True,shuffle=True)

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


net = Net()
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print(net)
net.to(device)

# load classifier network


MODEL = "openai/whisper-tiny.en"

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#model = WhisperForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
#model.to(device)
processor = WhisperProcessor.from_pretrained(MODEL)

def preprocess(data):
    processed_data = {}
    processed_data['input_features'] = processor(data["audio"]["array"], sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    processed_data['decoder_input_ids'] = processor.tokenizer('<|startoftranscript|>'+data['text'].lower(),return_tensors='pt').input_ids.to(device)
    processed_data['labels'] = processor.tokenizer(data['text'].lower()+'<|endoftext|>',return_tensors='pt').input_ids.to(device)
    return processed_data

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./model/lora",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=100,
#    max_steps=100, # only for testing purposes, remove this from your final run :)
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_d,
    eval_dataset=test_d,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()



wer_scores = []
for data in TEST:
    inputs_3s, input_features, labels, text = data
    input_features = input_features.to(device)
    #print(text)
    #print(inputs.shape)
    #inputs = inputs.to(device)
    #labels = labels.to(device)    
    #input_features = processor(inputs.squeeze(), sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    generated_ids = model.generate(input_features)
    generated_test_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    wer_ = wer(normalizer(generated_test_text), normalizer(text[0].lower()))
    wer_scores.append(wer_)
    print(str(labels)+'\t\t'+text[0]+'\t\t'+generated_test_text+'\t\t'+str(wer_))
    
print(sum(wer_scores)/len(wer_scores))

