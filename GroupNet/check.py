import torch
from model.FocalSTR.net import FocalSTR

model = FocalSTR.load_from_checkpoint('/home/nrohit/IndianSTR/GrpNet_logs/FocalSTR/3915/checkpoints/epoch=5-val_loss=0.25-val_wrr2=0.72.ckpt')
t = torch.randn(32,3,128,128)
out = model(t)
print(out)