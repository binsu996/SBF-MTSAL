import torch

class Auxiliary(torch.nn.Module):
    def __init__(self,input_shape):
        super(Auxiliary,self).__init__()
        self.blstm=torch.nn.LSTM(input_shape,256,bidirectional=True)
        self.net=torch.nn.Sequential(
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,30),
        )
    
    def forward(self,x):
        x=self.blstm(x)[0]
        return self.net(x).mean(dim=0,keepdim=True)


class MaskEst(torch.nn.Module):
    def __init__(self,input_shape=129):
        super(MaskEst,self).__init__()
        self.blstm1=torch.nn.LSTM(input_shape,512,bidirectional=True)
        self.blstm2=torch.nn.LSTM(512,512,bidirectional=True)
        self.net1=torch.nn.Sequential(
            torch.nn.Linear(1054,512),
            torch.nn.ReLU(),
        )
        self.net2=torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,input_shape)
        )
        self.aux=Auxiliary(input_shape)
    
    def forward(self,x):
        Y,A=x
        y=self.blstm1(Y)[0]
        a=self.aux(A)
        T,B,K=y.shape
        a=a.expand((T,-1,-1))
        y=torch.cat([y,a],-1)
        y=self.net1(y)
        y=self.blstm2(y)[0]
        return self.net2(y)*Y
        