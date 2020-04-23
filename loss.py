import torch
from utils import cmap


def fd(x):
    d1=x[:,:,2:]-x[:,:,:-2]
    d2=(x[:,:,3:]-x[:,:,:-3])*2
    d_sum=d1[:,:,1:]+d2
    n=2*(1+2**2)
    return d_sum/n
    
def MTSALLoss(y_pred,y_true):
    assert isinstance(y_pred,torch.Tensor)
    assert isinstance(y_true,torch.Tensor)
    loss1=torch.nn.MSELoss()(y_pred,y_true)
    y_pred1,y_true1=cmap(fd,[y_pred,y_true])
    loss2=torch.nn.MSELoss()(y_pred1,y_true1)
    y_pred2,y_true2=cmap(fd,[y_pred1,y_true1])
    loss3=torch.nn.MSELoss()(y_pred2,y_true2)
    return loss1+4.5*loss2+10*loss3