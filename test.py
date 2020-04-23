import torch
from data import Feeder
from progressbar import ProgressBar
import numpy as np
import librosa
from os.path import join

test_dir='../wsj0_2mix_extr/tt'
model_name='modelx'
test_count=10
out_dir='output'

test_dataset=Feeder(test_dir,'test')
model=torch.load(model_name)


example_id=0
for mix_amp,aux_amp,mix,s1 in test_dataset:
    example_id+=1
    mix_amp=mix_amp.unsqueeze(1)
    aux_amp=aux_amp.unsqueeze(1)
    y_pred=model([mix_amp,aux_amp]).squeeze(1).cpu().detach().numpy()
    # print(y_pred.shape)
    mix_angle=np.angle(mix)
    y_real=y_pred*np.sin(mix_angle)
    y_imag=y_pred*np.cos(mix_angle)
    y_pred=y_real+1j*y_imag
    # print(mix.shape,y_pred.shape,s1.shape)
    wav_mix=librosa.istft(mix.T,hop_length=128)
    wav_pred=librosa.istft(y_pred.T,hop_length=128)
    wav_clean=librosa.istft(s1.T,hop_length=128)
    librosa.output.write_wav(join(out_dir,str(example_id)+'_mix.wav'),wav_mix,sr=8000,norm=True)
    librosa.output.write_wav(join(out_dir,str(example_id)+'_pred.wav'),wav_pred,sr=8000,norm=True)
    librosa.output.write_wav(join(out_dir,str(example_id)+'_clean.wav'),wav_clean,sr=8000,norm=True)
    if example_id+1>test_count:break

