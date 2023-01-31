import torch
import datahandler
from model import LightningModel
from new_mus import newMus
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

hparams = {
        "mus_path": "musdb/",
        "num_bandwidths": 23,
        "bandwidths": "20,20,20,30,30,30,30,30,30,30,30,30,30,50,50,50,50,70,70,100,100,125",
        "bandwidth_freq_out_size": 128,
        "n_fft": 2048,
        "hop_length": 1024,
        "win_length": 2048,
        "conv_1_kernel_size":(1,7),
        "conv_1_stride":(1,3),
        "conv_2_kernel_size":(4,4),
        "conv_2_stride":(2,2),
        "conv_3_kernel_size":(1,7),
        "conv_3_stride":(1,3),
        "conv_3_ch_out_1":8,
        "time_steps": 431,
        "freq_bands": 1025,
        "n_mels": 32,
        "input_sampling_rate": 44100,
        "resampling_rate": 16000,
        "shortest_duration" : 5019648,
        "longest_duration" : 20000000,
        "segment_length" : 10,
        "sampling_rate" : 44100,
        "resampling_rate" : 16000,
        "discard_low_energy" : True,
        "drop_percentile" : 0.1,
        'chunks_below_percentile' : 0.5,
        'segment_overlap' : 0.5,
        'segment_chunks' : 10,
        'training_batch_size' : 16,
        'testing_batch_size' : 64,
        'filtered_training_indices' : '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,50,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85',
        'filtered_validation_indices' : '0,1,2,3,4,5,6,7,8,9,10,11,12,13'
    }

def collate(batch):
        return batch[0]

if __name__ == '__main__':
        torch.set_float32_matmul_precision('medium')
        musValidation = newMus('musdb/', 'valid', 'train', batch_size = 16,
                                filtered_indices = hparams['filtered_validation_indices'])
        valLoader = DataLoader(musValidation, batch_size = 1, collate_fn = collate, num_workers = 4)
        musTraining = newMus('musdb/', batch_size = 16,
                         filtered_indices = hparams['filtered_training_indices'])
        trainLoader = DataLoader(musTraining, batch_size = 1, collate_fn = collate, num_workers = 4)
        lightning = LightningModel(hparams)
        ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters = False)
        trainer = pl.Trainer(max_epochs=2, accelerator = 'gpu', devices = 2, strategy = ddp)
  #      trainer = pl.Trainer(max_epochs = 2, accelerator = 'cpu')
        trainer.fit(lightning)

