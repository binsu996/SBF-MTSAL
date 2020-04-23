import torch
from os.path import join,exists
from os import listdir
from librosa import load,stft
from utils import cmap
import numpy as np
from model import *
from loss import *
from progressbar import ProgressBar
import torcher

class Feeder(torch.utils.data.Dataset):
    def __init__(self,home_dir):
        self.filenames=listdir(join(home_dir,'mix'))
        self.len=len(self.filenames)
        self.home_dir=home_dir
        self.fft=lambda x:stft(x,n_fft=256,hop_length=128).T
        self.load=lambda x:load(x,sr=8000)[0]

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        if idx>=self.len:
            raise StopIteration
        name=self.filenames[idx]
        get_path=lambda x:join(self.home_dir,x,name)
        names=cmap(get_path,['s1','mix','aux'])
        wavs=cmap(self.load,names)
        specs=cmap(self.fft,wavs)
        s1,mix,aux=specs
        angle=np.cos(np.angle(mix)-np.angle(s1))
        s1,mix,aux=cmap(np.abs,[s1,mix,aux])
        return torch.Tensor(mix).cuda(),torch.Tensor(aux).cuda(),torch.Tensor(s1*angle).cuda()

model_name='modelx'
if exists(model_name):
    model=torch.load(model_name)
else:
    model=MaskEst().cuda()
loss=MTSALLoss
train_dataset=Feeder('../wsj0_2mix_extr/tr')
valid_dataset=Feeder('../wsj0_2mix_extr/cv')
opt=lambda x: torch.optim.Adam(x,lr=5e-4)

def cf(batch):
    Ys,As,Xs=list(map(torch.nn.utils.rnn.pad_sequence,zip(*batch)))
    return [Ys,As],Xs

train_dataloader=torch.utils.data.DataLoader(train_dataset,collate_fn=cf,batch_size=32)
valid_dataloader=torch.utils.data.DataLoader(valid_dataset,collate_fn=cf,batch_size=32)
trainer=torcher.Torcher(model,loss,opt)
trainer.fit(train_dataloader,valid_data=valid_dataloader,epochs=30,model_path=model_name)
    