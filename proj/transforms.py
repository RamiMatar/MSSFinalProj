import numpy as np
import scipy
import librosa
import torch

class Transforms():
    #def __init__(self, sample_rate=44100, n_fft=2048, hop_length=1024, win_length=2048, window='hann', center=True, pad_mode='reflect', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1, top_db=80.0, ref=1.0, amin=1e-10):
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=1024, win_length=2048, window='hann', center=True, pad_mode='reflect', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1, top_db=80.0, ref=1.0, amin=1e-10):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk
        self.norm = norm
        self.top_db = top_db
        self.ref = ref
        self.amin = amin

    def __call__(self, x): 
        #x = torch.from_numpy(x)
        #x = torch.Tensor(x)
        x = torch.tensor(x)
        x = x.permute(1, 0)
        x = self.stft(x)
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        return x

    def stft(self, x, n_fft=2048, hop_length=1024, win_length=2048, window=torch.hann_window(2048), center=True, pad_mode='reflect'):
        return torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)

    def istft(self, x, n_fft=2048, hop_length=1024, win_length=2048, window='hann', center=True):
        return torch.istft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center)

    def mel_spectrogram(self, x, sr=44100, n_fft=2048, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
        return librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm)

    def amplitude_to_db(self, x, top_db=80.0, ref=1.0, amin=1e-10):
        return librosa.amplitude_to_db(x, top_db=top_db, ref=ref, amin=amin)

    def RMS(self, audio_tensor):
        # this function is used to calculate the RMS of the audio signal
        # the input will be an audio tensor of shape (num_samples)
        # Note that this function is intended to use on smaller audio signals, and that
        # typically, longer signals get windows similar to the STFT in order to calculate more localized
        # RMS. The purpose of using this was for silent signal detection.
        squared_tensor = torch.pow(audio_tensor, 2)
        mean_power = torch.mean(squared_tensor)
        rms = torch.sqrt(mean_power)
        return rms
