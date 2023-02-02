import torch
from datahandler import DataHandler
from model import LightningModel
from new_mus import newMus
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torchmodel import TorchModel
from loss import *

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

segment_samples = hparams['segment_length'] * hparams['sampling_rate']
segment_samples = int(segment_samples / hparams['hop_length']) * hparams['hop_length']
reserved = int(0.7 * hparams['batch_size'] * hparams['self.segment_samples'] + hparams['hop_length'])

def collate(batch):
        super_batch = []
        for stems in batch:
                x = trim_stems(stems, torch.randint(0, stems.shape[1], (1,)))
                super_batch.append(x)
        super_batch = torch.stack(super_batch, 0)
        return super_batch.view(len(batch) * batch[0].shape[0], reserved, 2)

def trim_stems(self, stems, start):
        # this function should trim the track to the shortest duration of the track from the start index, it allows looping back across the song.
        if start + self.reserved > stems.shape[1]:
            first_half = stems[:, start:, :]
            remaining = self.reserved - (stems.shape[1] - start)
            second_half = stems[:, :remaining, :]
            return torch.cat((first_half, second_half), axis=1)
        else:
            return stems[:, start:start+self.reserved, :]

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        loss: SSLoss,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss = loss

    def _run_batch(self, source):
        self.optimizer.zero_grad()
        data, labels = self.data_handler.batchize_training_item(source)
        stfts, chromas, mfccs = self.transforms(data)
        predicted_sources = self.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels[:,3])
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        iter = 0
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source in self.train_data:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Step: {iter}")
            source = source.to(self.gpu_id)
            self._run_batch(source)
            iter += 1

    def _save_checkpoint(self, epoch):
        ckp = {'model_state' : self.model.module.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'model' : self.model
                }
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                


def load_train_objs():
    train_set = musTraining = newMus('musdb/', batch_size = 16, filtered_indices = hparams['filtered_training_indices'])  # load your dataset
    model = TorchModel(hparams)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = SSLoss()
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        collate_fn = collate
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

def load_checkpoint(model: torch.nn.Module, optimizer: torch.nn.optim.Optimizer):
    PATH = "checkpoint.pt"
    ckp = torch.load(PATH)
    model.module.load_state_dict(ckp['model'])
    optimizer.load_state_dict(ckp['optimizer'])
    return model, optimizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
