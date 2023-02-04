import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from new_mus import newMus
from datahandler import DataHandler
from loss import *

class TorchModel(nn.Module): 
    
    def __init__(self, hparams):
        super().__init__()
        self.mus_path = hparams['mus_path']
        self.bandwidths = [int(bandwidth) for bandwidth in hparams['bandwidths'].split(',')]
        self.step = 0
        self.filtered_training_indices = hparams['filtered_training_indices']
        self.filtered_validation_indices = hparams['filtered_validation_indices']
        self.filtered_testing_indices = None # fix this
        self.n_mels = hparams['n_mels']
        self.N = hparams['bandwidth_freq_out_size']
        self.K = len(self.bandwidths)
        self.time_steps = hparams['time_steps']
        self.transforms = Transforms()
        self.kernel1 = hparams['conv_1_kernel_size']
        self.stride1 = hparams['conv_1_stride']
        self.kernel2 = hparams['conv_2_kernel_size']
        self.stride2 = hparams['conv_2_stride']
        self.sampling_rate = hparams['sampling_rate']
        self.resampling_rate = hparams['resampling_rate']
        self.batch_size = hparams['training_batch_size']
        self.hop_length = hparams['hop_length']
        self.segment_length = hparams['segment_length']
        self.resampling_ratio = self.sampling_rate / self.resampling_rate
        self.segment_samples = int(self.segment_length * self.sampling_rate)
        self.segment_samples = int(self.segment_samples / self.hop_length) * self.hop_length
        self.reserved = int(0.7 * self.batch_size * self.segment_samples + self.hop_length)
        self.loss = SSLoss()
        self.bandsplit = BandSplit(self.bandwidths, self.N)
        self.conv1 = ConvolutionLayer(self.K, self.K, self.kernel1, self.stride1)
        self.conv2 = ConvolutionLayer(self.K, self.K, self.kernel2, self.stride2)
        self.conv3 = ConvolutionLayer(1,8,kernel_size=(1,11),stride=(1,3))
        self.conv4 = ConvolutionLayer(8,22,kernel_size=(1,3),stride=(1,2))
        self.conv5 = ConvolutionLayer(1,8,kernel_size=(1,11),stride=(1,3))
        self.conv6 = ConvolutionLayer(8,22,kernel_size=(1,3),stride=(1,2))
        self.blstms1 = AlternatingBLSTMs(self.K, 24, 63, 64)
        self.blstms2 = AlternatingBLSTMs(self.K, 24, 96, 64)
        self.blstms3 = AlternatingBLSTMs(self.K, 24, 96, 63 )
        self.deconv1 = TransposeConvolutionLayer(self.K, self.K, self.kernel2, self.stride2)
        self.deconv2 = TransposeConvolutionLayer(self.K, self.K, self.kernel1, self.stride1)
        self.masks = MaskEstimation(self.bandwidths, 128,32)
        self.data_handler = DataHandler(hparams['training_batch_size'], hparams['shortest_duration'], hparams['sampling_rate'], hparams['resampling_rate'],
            hparams['hop_length'], hparams['longest_duration'], hparams['segment_overlap'], hparams['segment_chunks'], 
            hparams['segment_length'], hparams['chunks_below_percentile'], hparams['drop_percentile'])
    
    def forward(self, batch):
        data, labels = self.data_handler.batchize_training_item(batch)
        stfts, chromas, mfccs = self.transforms(data)
        return self.forward_pass(stfts, chromas, mfccs)
    
    def training_step(self, batch, batch_idx):
        if self.step % 200 == 0:
            torch.cuda.empty_cache()
        self.step += 1
        
        data, labels = self.data_handler.batchize_training_item(batch)
        stfts, chromas, mfccs = self.transforms(data)
        predicted_sources = self.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels[:,3])
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.step % 5 == 200:
            torch.cuda.empty_cache()
            
        self.step += 1
        data, labels = self.data_handler.batchize_training_item(batch)
        stfts, chromas, mfccs = self.transforms(data)
        predicted_sources = self.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels[:,3])
        self.log("valid_loss", loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.step % 200 == 0:
            torch.cuda.empty_cache()
        self.step += 1
        
        data, labels = self.data_handler.batchize_training_item(batch)
        stfts, chromas, mfccs = self.transforms(data)
        predicted_sources = self.forward_pass(stfts, chromas, mfccs)
        loss = self.loss(predicted_sources, labels[:,3])
        self.log("test_loss", loss, sync_dist=True)
        return loss
    
    def trim_stems(self, stems, start):
        # this function should trim the track to the shortest duration of the track from the start index, it allows looping back across the song.
        if start + self.reserved > stems.shape[1]:
            first_half = stems[:, start:, :]
            remaining = self.reserved - (stems.shape[1] - start)
            second_half = stems[:, :remaining, :]
            return torch.cat((first_half, second_half), axis=1)
        else:
            return stems[:, start:start+self.reserved, :]

    def collate(self, batch):
        super_batch = []
        for stems in batch:
            x = self.trim_stems(stems, torch.randint(0, stems.shape[1], (1,)))
            super_batch.append(x)
        super_batch = torch.stack(super_batch, 0)
        return super_batch.view(len(batch) * batch[0].shape[0], self.reserved, 2)

    def train_dataloader(self):
        musTraining = newMus('musdb/', batch_size = 16, filtered_indices = self.filtered_training_indices)
        trainLoader = DataLoader(musTraining, batch_size = 8, collate_fn = self.collate, num_workers = 8, shuffle = True, persistent_workers = True)
        return trainLoader

    def val_dataloader(self):
        musValidation = newMus('musdb/', subset = 'train', split= 'valid', batch_size = 16, filtered_indices = self.filtered_validation_indices)
        valLoader = DataLoader(musValidation, batch_size = 8, collate_fn = self.collate, num_workers = 4, shuffle = False, persistent_workers = True)
        return valLoader
    
    def test_dataloader(self):
        if self.testing_dataloader is None:
            musValidation = newMus('musdb/', subset = 'train', split= 'test', batch_size = 16, filtered_indices = self.filtered_testing_indices)
            valLoader = DataLoader(musValidation, batch_size = 1, collate_fn = self.collate, num_workers = 4)
            return valLoader
        else:
            return self.testing_dataloader
    
    def forward_pass(self, X, chromas, mfccs):
        X = X.permute(0,1,3,4,2)
        X1 = self.bandsplit(X)
        batch_size = X1.shape[0]
        #Shape: torch.Size([32, 22, 431, 128]) (batch_size, num_bands, time_steps, freq_N)
        X2 = self.conv1(X1)
        X3 = self.conv2(X2)
        mfccs = mfccs.reshape(batch_size,1,self.n_mels,-1)
        mfccs = self.conv3(mfccs)
        mfccs = self.conv4(mfccs)
        chromas = chromas.reshape(batch_size,1,self.n_mels,-1)
        chromas = self.conv5(chromas)
        chromas = self.conv6(chromas)
        X, _ = self.blstms1(X3)
        xmfccs = torch.cat((mfccs,X), 2)
        X, _ = self.blstms2(xmfccs)
        xchromas = torch.cat((chromas,X),2)
        X, _ = self.blstms3(xchromas)      
        X = self.deconv1(X + X3)
        X = self.deconv2(torch.cat((X, X[:,:,:,-1].unsqueeze(3)), 3) + X2)
        X = self.masks(X)
        return X
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)
        return optimizer
    

        
class Chroma(nn.Module):
    def __init__(self, n_fft, sampling_rate):
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        
    def forward(self, x):
        pass
    
        # x is a spectrogram with shape(... , T, F)
class Transforms(nn.Module):
    def __init__(self, input_freq = 44100, resample_freq = 16000, n_fft = 2048, hop_length = 1024, win_length=2048, n_mels = 32):
        super().__init__()
        self.stft = Spectrogram(n_fft = n_fft, hop_length = hop_length, win_length = win_length, power = None)
        self.mel = MelScale(sample_rate = resample_freq, n_mels = n_mels, n_stft = n_fft // 2 + 1)

    def forward(self, X):
        stft = self.stft(X)
        power_spectrogram = torch.abs(stft).pow(2)
        real = stft.real
        imag = stft.imag
        stft = torch.stack((real,imag), axis = 4)
        stft = stft.permute(0,1,4,2,3)
        mfccs= self.mel(power_spectrogram)
        
        chromas = self.mel(power_spectrogram)
        return stft, chromas, mfccs
    
def post_conv_dimensions(self,N,time_steps,in_channels):
        x = torch.randn(1,in_channels,N,time_steps).to(self.device)
        return self.conv2(self.conv1(x)).shape
        
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dtype='float'):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class TransposeConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dtype='float'):
        super(TransposeConvolutionLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

# This class defines a module that runs t
class AlternatingBLSTMs(nn.Module):
    def __init__(self, num_bands, time_steps, N, out_size, axis=1):
        super(AlternatingBLSTMs, self).__init__()
        self.band_blstm = BandBiLSTM(num_bands, time_steps, N)
        self.temporal_blstm = TemporalBiLSTM(num_bands, time_steps, N, out_size)
        self.num_bands = num_bands
        self.time_steps = time_steps
        self.N = N
        # hidden size = freq_steps_per_band * time_steps 

    def forward(self, x):
        # Input shape: (batch_size, num_bands, N, time_steps)
        # Prepare for Band BLSTM: shape = (batch_size, num_bands, N * time_steps)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_bands, -1)
        x = self.band_blstm(x)
        x = x.reshape(batch_size, self.time_steps, -1)
        x = self.temporal_blstm(x)
        #x += residual
        # Return the output of the module
        return x    
    
# This class defines a module that runs the input, with shape (num_bands, num_timesteps, N), through a normalization layer, then a temporal biLSTM, then a fully connected layer.
# Then, the output of that layer is of the same shape as the input to the module, which will be fed into a similar structure, but this time with a band biLSTM, following the same normalization, biLSTM, FC structure.
class BandBiLSTM(nn.Module):
    def __init__(self, num_bands, time_steps, N, axis=1):
        super(BandBiLSTM, self).__init__()
        self.norm = nn.GroupNorm(num_bands, num_bands)
        self.input_size = time_steps * N
        self.hidden_size = self.input_size // 2
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(N, N)
        self.axis = axis
        self.N = N
        self.num_bands = num_bands
        self.time_steps = time_steps
        # hidden size = freq_steps_per_band * time_steps 

    def forward(self, x):
        batch_size = x.shape[0]
        # (batch_size,time_steps, num_bands, N)
        x = self.norm(x)
        residual = x.clone().detach()
        x, lstm_vars = self.bilstm(x)
        # (batch_size, num_bands, 2 * hidden_size)
        x = x.reshape(batch_size, self.num_bands, self.time_steps, self.N)
        # (batch_size, num_bands, time_steps, N)
        x = self.fc(x)
        # (batch_size, num_bands, time_steps, N)
        #x += residual
        # Return the output of the module
        return x
    
class TemporalBiLSTM(nn.Module):
    def __init__(self, num_bands, time_steps, N, out, axis=1):
        super(TemporalBiLSTM, self).__init__()
        self.norm = nn.GroupNorm(time_steps, time_steps)
        self.input_size = num_bands * N
        self.hidden_size = num_bands * out // 2
        self.out = out
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(out, out)
        self.axis = axis
        self.N = N
        self.time_steps = time_steps
        self.num_bands = num_bands
        # hidden size = freq_steps_per_band * time_steps 

    def forward(self, x):
        batch_size = x.shape[0]
        # (batch_size,time_steps, num_bands, N)
        x = self.norm(x)
        residual = x.clone().detach()
        x, lstm_vars = self.bilstm(x)
        # (batch_size, num_bands, 2 * hidden_size)
        x = x.reshape(batch_size, self.num_bands, self.time_steps, self.out)
        # (batch_size, num_bands, time_steps, N)
        x = self.fc(x)
        x = x.permute(0,1,3,2)
        # (batch_size, num_bands, time_steps, N)
        #x += residual
        # Return the output of the module
        return x, lstm_vars

class BandSplit(torch.nn.Module):
    # Input shape: torch.Size([16, 2, 1025, 431, 2])
    def __init__(self, bandwidths, N):
        # bandwidth
        super(BandSplit, self).__init__()
        self.bandwidths = bandwidths
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2 * bandwidth) for bandwidth in self.bandwidths])
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(2 * bandwidth, N) for bandwidth in self.bandwidths])

    def forward(self, X):
        subband_spectrograms = []
        K = len(self.bandwidths)
        for i in range(K):
            start_index = sum(self.bandwidths[:i])
            end_index = start_index + self.bandwidths[i]
            subband_spectrogram = X[:, :,start_index:end_index, :]
            subband_spectrogram = subband_spectrogram.permute(0,1,4,2,3)
            subband_spectrogram = subband_spectrogram.reshape(2 * X.shape[0], X.shape[3], 2 * self.bandwidths[i])
            subband_spectrograms.append(subband_spectrogram)

        subband_features = []
        for i in range(K):
            norm_output = self.norm_layers[i](subband_spectrograms[i])
            fc_output = self.fc_layers[i](norm_output)
            subband_features.append(fc_output)

        Z = torch.stack(subband_features, dim=1)
        Z = Z.permute(0,1,3,2)
        return Z
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.MLP(x)
        return x
    
    
class MaskEstimation(nn.Module):
    def __init__(self, bandwidths, N, batch_size):
        super(MaskEstimation, self).__init__()
        self.num_bands = len(bandwidths)
        self.bandwidths = bandwidths
        self.batch_size = batch_size
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(N) for bandwidth in self.bandwidths])
        self.MLP_layers = torch.nn.ModuleList([MLP(N, bandwidth * 2, N * 2) for bandwidth in self.bandwidths])
    def forward(self, x):
        # Input shape: (batch_size, num_bands, N, T)
        time_steps = x.shape[3]
        x = x.permute(1, 0 , 3, 2)
        out = []
        # shape: (num_bands, batch_size, T, N)
        for i in range(self.num_bands):
            y = self.norm_layers[i](x[i])
            y = self.MLP_layers[i](y)
            out.append(y)
        out = torch.cat(out, 2)
        out = out.reshape(self.batch_size // 2, 2, sum(self.bandwidths), time_steps, 2)
        return out




  
