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
        self.blstm = nn.LSTM(input_size = self.N, hidden_size = self.N, num_layers = 1, batch_first = True, bidirectional = True)
        self.norm = nn.GroupNorm(1, self.N)
        self.fc = nn.Linear(2 * self.N, self.N)

    def forward(self, X):
        skip = X
        print(skip.shape)
        X = X.permute(0,3,1,2)
        X = self.norm(X)
        X = X.permute(0,2,3,1)
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])
        X, _ = self.blstm(X)
        X = self.fc(X)
        X = X + skip
        return X
        

class TemporalBLSTMModule(nn.Module):
    def __init__(self, N, K, T):
        super().__init__()
        self.N = N
        self.K = K
        self.T = T
        self.blstm = nn.LSTM(input_size = self.N, hidden_size = self.N, num_layers = 1, batch_first = True, bidirectional = True)
        self.norm = nn.GroupNorm(1, self.N)
        self.fc = nn.Linear(2 * self.N, self.N)

    def forward(self, X):
        skip = X
        print(skip.shape)
        X = X.permute(0,3,1,2)
        X = self.norm(X)
        X = X.permute(0,2,1,3)
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])
        X, _ = self.blstm(X)
        X = self.fc(X)
        X = X + skip
        return X


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
        print(x.shape)
        print(self.num_bands)
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