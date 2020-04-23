import torch
from os.path import join,exists
from os import listdir
from librosa import load,stft
from utils import cmap
import numpy as np

class Feeder(torch.utils.data.Dataset):
    def __init__(self,home_dir,mode='train'):
        self.filenames=listdir(join(home_dir,'mix'))
        self.len=len(self.filenames)
        self.home_dir=home_dir
        self.fft=lambda x:stft(x,n_fft=256,hop_length=128).T
        self.load=lambda x:load(x,sr=8000)[0]
        self.mode=mode

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
        if self.mode=='train':
            return torch.Tensor(mix).cuda(),torch.Tensor(aux).cuda(),torch.Tensor(s1*angle).cuda()
        else:
            return torch.Tensor(mix).cuda(),torch.Tensor(aux).cuda(),mix,s1