import torch
import pytorch_lightning as pl
from model import LightningModel

model = LightningModel.load_from_checkpoint('/workspace/MSSFinalProj/proj/tests3/lightning_logs/version_3/checkpoints/epoch=1-step=38.ckpt')
print(model)
