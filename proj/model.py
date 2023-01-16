import torch.nn as nn
import torch
import torch.nn.functional as F

def test_module(module, input_size):
    """
    Test an nn.Module subclass with random input data of the given size.
    
    :param module: The nn.Module subclass to test.
    :param input_size: A tuple of integers specifying the size of the input data.
    """
    # Create random input data with the specified size
    input_data = torch.randn(input_size)
    print(f'Input shape: {input_data.shape}')

    # Run the input data through the module
    output = module(input_data)
    print(f'Output shape: {output.shape}')


class SingularValueRegularization(nn.Module):
    def __init__(self, num_singular_values, eps=1e-3):
        super().__init__()
        self.num_singular_values = num_singular_values # number of singular values to keep
        self.eps = eps 
    
    def forward(self, x):
        u, s, v = torch.svd(x) # perform singular value decomposition on x
        s[self.num_singular_values:] = self.eps # set the remaining singular values to epsilon
        x_hat = u @ torch.diag(s) @ v # construct x_hat using the first num_singular_values singular values
        return x_hat


class my_model(torch.nn.Module):
    def __init__(self, bandwidths, N):
        # bandwidth
        super(BandSplitModule, self).__init__()
        self.bandwidths = bandwidths
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2 * bandwidth) for bandwidth in self.bandwidths])
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(2 * bandwidth, N) for bandwidth in self.bandwidths])

    def forward(self, X):
        subband_spectrograms = []
        K = len(self.bandwidths)
        for i in range(K):
            start_index = sum(self.bandwidths[:i])
            end_index = start_index + self.bandwidths[i]
            subband_spectrogram = X[:, start_index:end_index, :]
            subband_spectrograms.append(subband_spectrogram)

        subband_features = []
        for i in range(K):
            norm_output = self.norm_layers[i](subband_spectrograms[i])
            fc_output = self.fc_layers[i](norm_output)
            subband_features.append(fc_output)

        Z = torch.stack(subband_features, dim=1)

        return Z


# This class defines a module that runs the input, with shape (num_bands, num_timesteps, N), through a normalization layer, then a temporal biLSTM, then a fully connected layer.
# Then, the output of that layer is of the same shape as the input to the module, which will be fed into a similar structure, but this time with a band biLSTM, following the same normalization, biLSTM, FC structure.
class GeneralBiLSTMUnit(nn.Module):
    def __init__(self, N, hidden_size, axis=1):
        super(GeneralBiLSTMUnit, self).__init__()
        self.norm = nn.GroupNorm(1, N)
        self.bilstm = nn.LSTM(N, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, N)
        self.axis = axis

    def forward(self, x):
        # Normalize the input
        x = self.norm(x)

        # Apply the temporal biLSTM layer
        x, _ = self.bilstm(x.transpose(0,1))

        # Apply the fully connected layer
        x = self.fc(x)
        x = x.transpose(0,1)
        
        x += input 
        # Return the output of the module
        return x

class InterleavedBiLSTMs(nn.Module):
    def __init__(self, N, hidden_size):
        super(InterleavedBiLSTMs, self).__init__()
        self.temporal_bilstm = GeneralBiLSTMUnit(N, hidden_size)
        self.band_bilstm = GeneralBiLSTMUnit(N, hidden_size)

    def forward(self, x):
        x = self.temporal_bilstm(x)
        x = self.band_bilstm(x)
        return x


