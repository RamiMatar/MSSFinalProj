#This file will hold the loss functions for each loss I will test, including a composite loss function

import torch
import torch.nn as nn


def SSloss(predicted_stfts, real_stfts):
    # This function will calculate the loss for a given batch of predicted and real stems
    # The loss will be the sum of the losses for each stem
    # the dimensions of the predicted stems and real stems will be (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
    # The loss will be a scalar
    stft_loss = stft_mean_absolute_error(predicted_stfts, real_stfts)
    # We want to find the signal representation from the STFT we computed. We can do this by taking the inverse STFT
    # of each segment (input in the batch) separately and compare them. We don't need perfect reconstruction for our loss function,
    # but we will need it to evaluate the model.
    predicted_time_signals = batch_time_signals(predicted_stfts)
    real_time_signals = batch_time_signals(real_stfts)
    signal_loss = signal_mean_absolute_error(predicted_time_signals, real_time_signals)
    return signal_loss + stft_loss

def batch_time_signals(stfts):
    # This function should compute the time signals for each segment in the batch
    # input shape (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
    # output shape (batch_size, num_stems, segment_samples, num_channels)
    istft_list = []
    for i in range(stfts.shape[0]):
        for j in range(stfts.shape[1]):
            istft = torch.istft(stfts[i, j], n_fft=2048, hop_length=1024, win_length=2048, window=torch.hann_window(2048))
            istft_list.append(istft)
    istfts = torch.stack(istft_list, dim=0)
    return istfts

    
    
def signal_mean_absolute_error(predicted_time_signal, real_time_signal):
    # This function will calculate the mean absolute error between the predicted and real time signals
    # The dimensions of the predicted and real time signals will be (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
    return torch.mean(torch.abs(predicted_time_signal - real_time_signal))

def stft_mean_absolute_error(predicted_stfts, real_stfts):
    # This function will calculate the mean absolute error between the predicted and real STFTs
    # The dimensions of the predicted and real STFTs will be (batch_size, num_stems, stft_dim_F, stft_dim_T,num_channels)
    return torch.mean(torch.abs(predicted_stfts - real_stfts))

