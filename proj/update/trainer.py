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
        "batch_size": 1,
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
reserved = int(0.7 * hparams['training_batch_size'] * segment_samples + hparams['hop_length'])

def collate(batch):
        super_batch = []
        for stems in batch:
                x = trim_stems(stems, torch.randint(0, stems.shape[1], (1,)))
                super_batch.append(x)
        super_batch = torch.stack(super_batch, 0)
        return super_batch.view(len(batch) * batch[0].shape[0], reserved, 2)

def trim_stems(stems, start):
        # this function should trim the track to the shortest duration of the track from the start index, it allows looping back across the song.
        if start + reserved > stems.shape[1]:
            first_half = stems[:, start:, :]
            remaining = reserved - (stems.shape[1] - start)
            second_half = stems[:, :remaining, :]
            return torch.cat((first_half, second_half), axis=1)
        else:
            return stems[:, start:start+reserved, :]

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12974"
    init_process_group(backend='gloo', rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: TorchModel,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        loss: SSLoss,
        batch_size: int,
        data_handler: torch.nn.Module,
    ) -> None:
        self.gpu_id = gpu_id
        rank = gpu_id
        torch.cuda.set_device(gpu_id)
        print("transferring model to gpu ", rank)
        self.model = model.to(gpu_id)
        print("model transferred to gpu ", rank)
        self.data_handler = data_handler.to(gpu_id)
        print("transferred data handler to gpu" , rank)
        self.train_data = train_data
        print(rank)
        self.valid_data = valid_data
        self.optimizer = optimizer
        print(rank)
        self.save_every = save_every
        print(rank)
        self.model = DDP(model, device_ids=[gpu_id])
        print(rank)
        self.loss = loss
        self.loss = loss.to(gpu_id)
        print("Init completed ", rank)
        self.batch_size = batch_size

    def _run_training_batch(self, source):
        self.optimizer.zero_grad()
        data, labels = self.data_handler.batchize_training_item(source)
        stfts, chromas, mfccs = self.model.module.transforms(data)
        predicted_sources = self.model.module.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _run_validation_batch(self, source):
        data, labels = self.data_handler.batchize_training_item(source)
        stfts, chromas, mfccs = self.model.module.transforms(data)
        predicted_sources = self.model.module.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels)
        return loss
    
    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data) + len(self.valid_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        for i, source in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            loss = self._run_training_batch(source)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Training step: {i} | Loss: {loss}")
        self.model.eval()  
        for i, source in enumerate(self.valid_data):
            source = source.to(self.gpu_id)
            loss = self._run_validation_batch(source)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Validation step: {i} | Loss: {loss}")

    def _save_checkpoint(self, epoch):
        ckp = {'model_state' : self.model.module.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'model' : self.model
                }
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        rank = self.gpu_id
        for epoch in range(max_epochs):
            print(rank, " started epoch ", epoch)
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                


def load_train_objs(gpu_id):
    musTraining = newMus('musdb/', source = 'vocals', batch_size = 16, filtered_indices = hparams['filtered_training_indices'])
    musValidation = newMus('musdb/', subset = "train", split = "valid", source = 'vocals', batch_size = 16, filtered_indices = hparams['filtered_validation_indices']) # load your dataset
    model = TorchModel(hparams)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_handler = DataHandler(hparams['training_batch_size'],  hparams['shortest_duration'], hparams['sampling_rate'], hparams['resampling_rate'],
            hparams['hop_length'], hparams['longest_duration'], hparams['segment_overlap'], hparams['segment_chunks'], 
            hparams['segment_length'], hparams['chunks_below_percentile'], hparams['drop_percentile'])
    loss = SSLoss()
    return musTraining, musValidation, model, data_handler, optimizer, loss


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        collate_fn = collate,
        num_workers = 3
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    print("starting process: ", rank)
    ddp_setup(rank, world_size)
    print("ddp setup: ", rank)
    train_dataset, valid_dataset, model, data_handler, optimizer, loss = load_train_objs(rank)
    print("train objects loaded: ", rank)
    train_data = prepare_dataloader(train_dataset, batch_size)
    valid_data = prepare_dataloader(valid_dataset, batch_size)
    print("train data prepared ", rank)
    trainer = Trainer(model, train_data, valid_data, optimizer, rank, save_every, loss, batch_size, data_handler)
    print("Trainer initialized and starting training: ", rank)
    trainer.train(total_epochs)
    print("Destroing process group: ", rank)
    destroy_process_group()

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
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
    print("starting processes")
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, hparams['batch_size']), nprocs=world_size)
