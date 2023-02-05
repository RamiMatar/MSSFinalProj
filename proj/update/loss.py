import torch
import torch.nn as nn
import torchaudio
class SSLoss(nn.Module):
    def __init__(self, input_frequency = 44100, resampling_frequency = 16000, n_fft = 2048, hop_length = 1024, win_length = 2048):
        super().__init__()
        self.n_fft = n_fft
        self.input_frequency = input_frequency
        self.resampling_frequency = resampling_frequency
        self.resample = torchaudio.transforms.Resample(self.input_frequency, self.resampling_frequency).double()
        self.stft = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_length, win_length = win_length, power = None).double()
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft = n_fft, hop_length = hop_length, win_length = win_length).double()

    def forward(self, X_stft, Y_signal):
        batch_size = X_stft.shape[0]
        time_steps = X_stft.shape[3]
        Y_signal = Y_signal.reshape(Y_signal.shape[0]*2, -1)
        Y_stfts = self.stft(Y_signal)
        real = Y_stfts.real
        imag = Y_stfts.imag
        Y_stfts = torch.stack((real, imag), 3)
        Y_stfts = Y_stfts.reshape(X_stft.shape)
        X_signal = self.batch_time_signals(X_stft)
        print(X_signal.shape, Y_signal.shape)
        signal_loss = self.signal_mean_absolute_error(X_signal, Y_signal)
        stfts_loss = self.stft_mean_absolute_error(X_stft, Y_stfts)
        return signal_loss + stfts_loss, Y_stfts[0,0], X_stft[0,0], X_signal[0].unsqueeze(0), Y_signal[0].unsqueeze(0)

    def batch_time_signals(self, stfts):
        # This function should compute the time signals for each segment in the batch
        # input shape (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
        # output shape (batch_size, num_stems, segment_samples, num_channels)
        istft_list = []
        for i in range(stfts.shape[0]):
            for j in range(stfts.shape[1]):
                istft = self.istft(torch.complex(stfts[i,j,:,:,0], stfts[i,j,:,:,1]))
                istft_list.append(istft)
        istfts = torch.stack(istft_list, dim=0)
        return istfts
        
    def signal_mean_absolute_error(self, predicted_time_signal, real_time_signal):
        # This function will calculate the mean absolute error between the predicted and real time signals
        # The dimensions of the predicted and real time signals will be (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
        return torch.mean(torch.abs(predicted_time_signal - real_time_signal))

    def stft_mean_absolute_error(self, predicted_stfts, real_stfts):
        # This function will calculate the mean absolute error between the predicted and real STFTs
        # The dimensions of the predicted and real STFTs will be (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
        return torch.mean(torch.abs(predicted_stfts - real_stfts))

