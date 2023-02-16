from convlstm import ConvLSTM, ConvLSTMCell
# This is a revision of my torchmodel.py file, this model will involve the same band split module, followed by deep layers of alternating BLSTMS, however,
# these layers will have multiple BLSTMs, one for each band and then one for each time step. The outputs of these BLSTMs will go through the a FC layer,
# and then get stacked into the same shape as the input. This will be repeated for each layer, and the last layer will go into the Mask Estimation Module.
# Importantly, the model will also use the chromagram and MFCCs as additional inputs to the model the get stacked with the input to the BLSTMs. However, the last layer
# forces the output to be of the same same as the band split module's output.
# step 1: alternative alternatingblstms module
# step 2: encoder decoder module for current alternatingblstms module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio
#TODO: add data augmentation, dropout, hyperparameter tuning, and an option for alternating blstms which assumes independence between time steps and bands


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.bandwidths = [int(bandwidth) for bandwidth in hparams['bandwidths'].split(',')]
        self.n_mels = hparams['n_mels']
        self.n_chroma = 12
        self.N = hparams['bandwidth_freq_out_size']
        self.K = len(self.bandwidths)
        self.time_steps = hparams['time_steps']
        self.sampling_rate = hparams['sampling_rate']
        self.resampling_rate = hparams['resampling_rate']
        self.batch_size = hparams['training_batch_size']
        self.hop_length = hparams['hop_length']
        self.segment_length = hparams['segment_length']
        self.resampling_ratio = self.sampling_rate / self.resampling_rate
        self.segment_samples = int(self.segment_length * self.sampling_rate)
        self.segment_samples = int(self.segment_samples / self.hop_length) * self.hop_length
        self.reserved = int(0.7 * self.batch_size * self.segment_samples + self.hop_length)
        self.transforms = Transforms(self.sampling_rate, self.resampling_rate, hparams['n_fft'], self.hop_length, hparams['hop_length'], self.n_mels)
        self.bandsplit = BandSplit(self.bandwidths[:], self.N, self.n_chroma, self.n_mels)
        self.encoder = ConvLSTMEncoder(hparams)
        self.decoder = ConvLSTMDecoder(hparams)
        self.masks = MaskEstimation(self.bandwidths, self.N, hparams['training_batch_size'] * 2)

    def forward(self, X):
        X, chromas, mfccs = self.transforms(X)
        X0 = X.permute(0,1,3,4,2)
        X = self.bandsplit(X0, chromas, mfccs)
        X, skips = self.encoder(X)
        X = self.decoder(X, skips)
        X = self.masks(X)
        X = X * X0
        return X

class Chroma(nn.Module):
    def __init__(self, n_fft, sampling_rate, n_chroma = 12):
        super().__init__()
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.n_chroma = n_chroma
        freq_bins = torch.linspace(0, sampling_rate / 2, n_fft // 2 + 1)
        mappings = self.semitone_mapping(freq_bins)
        self.semitone_bins = []
        for i in range(self.n_chroma):
            self.semitone_bins.append(torch.argwhere(mappings == torch.Tensor([i])))
    
    def semitone_mapping(self, freq):
        return self.n_chroma * torch.log2(freq / 440) % self.n_chroma

    def forward(self, X):
        chromas = []
        for i in range(12):
            chromas.append(torch.sum(X[:,:, self.semitone_bins[0],:], dim=2))
        chromas = torch.cat(chromas, axis = 2)
        return chromas

class Transforms(nn.Module):
    def __init__(self, input_freq = 44100, resample_freq = 16000, n_fft = 2048, hop_length = 1024, win_length=2048, n_mels = 32):
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_length, win_length = win_length, power = None)
        self.mel = torchaudio.transforms.MelScale(sample_rate = resample_freq, n_mels = n_mels, n_stft = n_fft // 2 + 1)
        self.chroma = Chroma(n_fft = n_fft, sampling_rate = resample_freq)

    def forward(self, X):
        stft = self.stft(X)
        power_spectrogram = torch.abs(stft).pow(2)
        real = stft.real
        imag = stft.imag
        stft = torch.stack((real,imag), axis = 4)
        stft = stft.permute(0,1,4,2,3)
        mfccs= self.mel(power_spectrogram)
        chromas = self.chroma(power_spectrogram)
        return stft, chromas, mfccs


class BandSplit(torch.nn.Module):
    # Input shape: torch.Size([16, 2, 1025, 431, 2])
    def __init__(self, bandwidths, N, n_chromas, n_mels):
        # bandwidth
        super(BandSplit, self).__init__()
        self.bandwidths = bandwidths
        self.K = len(bandwidths)
        self.bandwidths.append(n_chromas // 2)
        self.bandwidths.append(n_mels // 2)
        self.modified_K = self.K + 2
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2 * bandwidths[i]) for i in range(self.modified_K)])
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(2 * bandwidths[i], N) for i in range(self.modified_K)])



    def forward(self, X, chromas, mfccs):
        subband_spectrograms = []
        K = self.K
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

       # chromas = chromas.permute(0,1,3,2).reshape(32,-1,12)
       # mfccs = mfccs.permute(0,1,3,2).reshape(32,-1,32)

      #  chromas = self.norm_layers[K](chromas)
      #  mfccs = self.norm_layers[K+1](mfccs)
      #  chromas = self.fc_layers[K](chromas)
      #  mfccs = self.fc_layers[K+1](mfccs)
      #  subband_features.append(chromas)
      #  subband_features.append(mfccs)

        Z = torch.stack(subband_features, dim=1)
        return Z

class ConvLSTMEncoder(nn.Module):
    def __init__(self, hparams):
        super(ConvLSTMEncoder, self).__init__()
        self.num_layers = hparams['num_layers']
        self.activation = hparams['encoder_activation']
        self.out_conv_multiplier = hparams['out_conv_multiplier']
        self.in_time_dim = hparams['time_steps'] # change this to be dynamic
        self.K = hparams['K']
        self.in_channels = 1
        self.encoders = []
        for i in range(self.num_layers):
            convlstm = ConvLSTMCell(self.in_channels, self.in_channels * self.out_conv_multiplier, (3,3), True)
            norm = nn.GroupNorm(1, self.K)
            encoder = nn.Sequential(convlstm, norm)
            self.encoders.append(encoder)
            self.in_channels *= self.out_conv_multiplier
        self.out_channels = self.in_channels
    
    def forward(self, x):
        skips = []
        for i in range(self.num_layers):
            x = self.encoders[i](x,None)
            skips.append(x)
        return x, skips

class ConvLSTMDecoder(nn.Module):
    def __init__(self, hparams):
        super(ConvLSTMDecoder, self).__init__()
        self.num_layers = hparams['num_layers']
        self.activation = hparams['encoder_activation']
        self.out_conv_multiplier = hparams['out_conv_multiplier']
        self.in_time_dim = hparams['time_steps'] # change this to be dynamic
        self.K = hparams['K']
        self.in_channels = self.out_conv_multiplier ** self.num_layers
        self.decoders = []
        for i in range(self.num_layers):
            convlstm = ConvLSTMCell(self.in_channels, self.in_channels // self.out_conv_multiplier, (3,3), True)
            norm = nn.GroupNorm(1, self.K)
            decoder = nn.Sequential(convlstm, norm)
            self.decoders.append(decoder)
            self.in_channels = self.in_channels // self.out_conv_multiplier
        self.out_channels = self.in_channels
    
    def forward(self, x, skips):
        for i in range(self.num_layers):
            x = x + skips[-i-1]
            x = self.decoders[i](x,None)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
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
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        # Input shape: (batch_size, num_bands, N, T)
        time_steps = x.shape[2] 
        x = x.permute(1, 0 , 2, 3)
        out = []
        # shape: (num_bands, batch_size, T, N)
        for i in range(self.num_bands):
            y = self.norm_layers[i](x[i])
            y = self.MLP_layers[i](y)
            out.append(y)
        out = torch.cat(out, 2)
        out = self.sigmoid(out)
        out = out.reshape(self.batch_size // 2, 2, sum(self.bandwidths), time_steps, 2)
        return out
