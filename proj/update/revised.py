# This is a revision of my torchmodel.py file, this model will involve the same band split module, followed by deep layers of alternating BLSTMS, however,
# these layers will have multiple BLSTMs, one for each band and then one for each time step. The outputs of these BLSTMs will go through the a FC layer,
# and then get stacked into the same shape as the input. This will be repeated for each layer, and the last layer will go into the Mask Estimation Module.
# Importantly, the model will also use the chromagram and MFCCs as additional inputs to the model the get stacked with the input to the BLSTMs. However, the last layer
# forces the output to be of the same same as the band split module's output.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio

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
        self.alternating_lstms = AlternatingBLSTMs(6, self.N, self.time_steps, self.K)
        self.masks = MaskEstimation(self.bandwidths, self.N, hparams['training_batch_size'] * 2)

    def forward(self, X):
        X, chromas, mfccs = self.transforms(X)
        X0 = X.permute(0,1,3,4,2)
        X = self.bandsplit(X0, chromas, mfccs)
        X = self.alternating_lstms(X)
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

        chromas = chromas.permute(0,1,3,2).reshape(32,-1,12)
        mfccs = mfccs.permute(0,1,3,2).reshape(32,-1,32)

        chromas = self.norm_layers[K](chromas)
        mfccs = self.norm_layers[K+1](mfccs)
        chromas = self.fc_layers[K](chromas)
        mfccs = self.fc_layers[K+1](mfccs)
        subband_features.append(chromas)
        subband_features.append(mfccs)

        Z = torch.stack(subband_features, dim=1)
        return Z

class AlternatingBLSTMs(nn.Module):
    def __init__(self, num_layers, N, time_steps, num_bands):
        super().__init__()
        self.num_layers = num_layers
        self.N = N
        self.T = time_steps
        self.K = num_bands
        self.modified_K = num_bands + 2

        self.band_blstms = nn.ModuleList([BandBLSTModule(self.N, self.modified_K, self.T) for i in range(self.num_layers)])
        self.temporal_blstms = nn.ModuleList([TemporalBLSTMModule(self.N, self.modified_K, self.T) for i in range(self.num_layers)])

    def forward(self, X):
        # input shape (batch_size, num_features, num_bands, time_steps)
        for i in range(self.num_layers):
            X = self.temporal_blstms[i](X)
            X = self.band_blstms[i](X)
        return X[:, :self.K, :, :]


class BandBLSTModule(nn.Module):
    def __init__(self, N, K, T):
        super().__init__()
        self.N = N
        self.K = K
        self.T = T
        self.blstms = nn.ModuleList([nn.LSTM(input_size = self.N, hidden_size = self.N, num_layers = 1, batch_first = True, bidirectional = True) for i in range(self.T)])
        self.norm = nn.GroupNorm(1, self.N)
        self.fc = nn.ModuleList([nn.Linear(2 * self.N, self.N) for i in range(self.T)])

    def forward(self, X):
        skip = X
        X = X.permute(0,3,1,2)
        X = self.norm(X)
        X = X.permute(0,2,3,1)
        outputs = []
        for i in range(self.T):
            output, _ = self.blstms[i](X[:,:,i,:])
            output = self.fc[i](output)
            outputs.append(output)
        output = torch.stack(outputs, dim=2)
        output = output + skip
        return output
        

class TemporalBLSTMModule(nn.Module):
    def __init__(self, N, K, T):
        super().__init__()
        self.N = N
        self.K = K
        self.T = T
        self.blstms = nn.ModuleList([nn.LSTM(input_size = self.N, hidden_size = self.N, num_layers = 1, batch_first = True, bidirectional = True) for i in range(self.K)])
        self.norm = nn.GroupNorm(1, self.N)
        self.fc = nn.ModuleList([nn.Linear(2 * self.N, self.N) for i in range(self.K)])


    def forward(self, X):
        skip = X
        X = X.permute(0,3,1,2)
        X = self.norm(X)
        X = X.permute(0,2,3,1)
        outputs = []
        for i in range(self.K):
            output, _ = self.blstms[i](X[:,i,:,:])
            output = self.fc[i](output)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        output = output + skip
        return output


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
