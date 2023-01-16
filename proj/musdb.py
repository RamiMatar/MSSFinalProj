"""
This is the file to retreive the data from MusDB, it allows us to choose between the HQ dataset with .wav files
or the compressed dataset with .stem.mp4 files. It also allows us to choose between the train and test subsets of the dataset.
The class MusdbDataset is a subclass of torch.utils.data.Dataset, which allows us to use the PyTorch DataLoader class
to load the data in batches. The __init__ method initializes the dataset, and the __getitem__ method returns the data
for a given index. The __len__ method returns the length of the dataset.

NOTES TO SELF: Fix threshold to set only off mix source
                OTHER functions than RMS
                Make self.batch_size
"""

import musdb
import torch
import torch.nn as nn
# Padding, constant batch size


class MusdbDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, musdb_root, split='train', subset='train', is_wav=False, sample_rate=44100, segment_length = 10, segment_chunks = 10, discard_low_energy = True, segment_overlap = 0.5, drop_percentile =  0.1, chunks_below_percentile = 0.5):
        # Check if a GPU is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        assert(subset == 'train' or subset == 'test')
        self.mode = subset
        self.mus = musdb.DB(musdb_root, subsets=subset, is_wav=is_wav)
        self.durations = dict()
        self.init_durations()
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.shortest_duration = self.shortest_duration_in_samples(self.mus)
        self.split = split
        self.discard_low_energy = discard_low_energy
        self.segment_length = segment_length
        self.segment_chunks = segment_chunks
        self.chunks_below_percentile = chunks_below_percentile
        self.segment_overlap = segment_overlap
        self.drop_percentile = drop_percentile
        self.segment_samples = int(self.segment_length * self.sample_rate)
        self.stft_size = self.STFT_dimensions(self.segment_samples)
        self.chunk_samples = int(self.segment_samples / self.segment_chunks)
        self.num_stems = self.mus.tracks[0].stems.shape[0]
        self.num_channels = self.mus.tracks[0].stems.shape[2]


    def __len__(self):
        return len(self.mus.tracks)


    def __getitem__(self, idx):
        # this function should return a batch of segment STFTs from the song as well as their stem STFTs.
        track = self.mus.tracks[idx]
        # stems is a list of the stems of the track, in the order of the stems in the track
        stems = torch.as_tensor(track.stems, device = self.device)
        if self.mode == 'train':
            segments = self._batchize_training_item(stems)
        else:
            segments = self._batchize_testing_item(stems)
        # we apply a STFT to each of those segments, and that is how we achieve our constant batch and input sizes.
        stfts = self.stft_segments(segments)
        # we then return the segments as a torch tensor
        data = stfts[0]
        labels = stfts[1:]
        return data, labels

    def _batchize_training_item(self,stems):
        # we need to trim the stems to the shortest duration track, starting from a random location
        stems = self.trim_stems(stems, self.random_start(stems.duration))
        # we split the stems from shape (num_stems, num_samples, num_channels) into a tensor with shape (num_stems, num_segments, num_samples_per_segment, num_channels)
        segments = self.split_track(stems)
        # now we need to drop out the low energy segments
        segments = self.high_energy_segments(segments)
        # then we choose a random and continuous yet constant number of segments from the track minus the dropped segments
        return segments

    def _batchize_testing_item(self,stems):
        # first we make sure the length of the song will produce a whole number of segments by padding with zeros to the end of the next segment.
        stems = self.add_zero_padding(stems)
        # now we extend the song with zeros to make sure we have an equal batch size for every output.
        to_pad = self.longest_duration_in_samples() - stems.shape[1]
        stems = self.add_N_zeros(stems, to_pad)
        # now we split the stems into equal size segments
        segments = self.split_track(stems)
        # we are now ready to operate on the song since we don't want to drop or modify our data as we conserve it to reconstruct our signal.
        return segments

    def collate(batch):
        # we define a custom function that overrides the PyTorch built in collate in order to make sure
        # our input to the model matches the dimensionality we want.
        # our input to the function is a list of (input, label) tuples. The list will have size = batch_size as defined in the DataLoader
        # We will use a batch_size of 1 to receive a list of tuples (in this case a list with 1 tuple), each tuple has two elements:
        # First, the input, a tensor with shape (batch_size, stft_dim_F, stft_dim_T, 2)
        # Second, the label, a tensor with shape (num_stems - 1, batch_size, stft_dim_F, stft_dim_T, 2)
        input, labels = batch[0]
        return input, labels

    def init_durations(self):
        for idx, track in enumerate(self.mus.tracks):
            self.durations[idx] = track.stems.shape[1]

    def stft_segments(self, segments):
        # this function should take in a list of segments and apply a STFT to each of them
        # it should return a tensor of STFTs, of shape (num_stems, batch_size, STFT_F, STFT_T, num_channels)
        # where batch_size is the number of STFTs we can fit into a batch
        # STFT_F is the number of frequency bins in the STFT
        # STFT_T is the number of time bins in the STFT
        # num_stems is the number of stems in the track
        # num_channels is the number of channels in the track
        # the STFTs should be applied to each segment, and then the segments should be concatenated along the batch_size axis
        stfts = [[] for i in range(self.num_stems)]
        # we first need to find the batch size
        batch_size = self.batch_size()
        # we then keep only the first batch_size segments, and apply a STFT to each of them. 
        for i in range(self.num_stems):
            segment = segments[:batch_size, i, :, :].view(batch_size, self.num_channels, self.segment_samples)
            for j in range(batch_size):
                # we apply a STFT to each segment
                batch_stfts = [[] for z in range(self.num_channels)]
                for k in range(self.num_channels):
                    stft = self.transforms.stft(segment[j,k])
                    batch_stfts[k].append(stft)
                batch_stfts = torch.stack(batch_stfts)
                stfts[i].append(batch_stfts)
                # we then add the STFT to the list of STFTs
        stfts = torch.stack(stfts)
        complex_number_dim = 2
        return stfts.view(batch_size, self.num_stems, self.stft_size[0], self.stft_size[1], complex_number_dim)
        
    def trim_stems(self, stems, start):
        print(self.shortest_duration)
        # this function should trim the track to the shortest duration of the track from the start index, it allows looping back across the song.
        if start + self.shortest_duration > stems.shape[1]:
            first_half = stems[:, start:, :]
            remaining = self.shortest_duration - (stems.shape[1] - start)
            second_half = stems[:, :remaining, :]
            return torch.cat((first_half, second_half), axis=1)
        else:
            return stems[:, start:start+self.shortest_duration, :]

    def pad_stems(self, stems):
        length_in_samples = stems.shape[1]
        to_pad = self.longest_duration_in_samples() - length_in_samples

    def batch_size(self):
        # this function should tell us how many STFTs we can fit into a batch based on finding the floor power of 2 of the number of STFTs we fit over the duration of the song
        # each STFT will represent a STFT over a fixed segment length.
        num_segments = self.num_segments_in_track(self.shortest_duration)
        # we anticipate a drop of up to twice the drop percentile (impossible, just to be safe) of the segments.
        num_segments = int(num_segments * (1 - 2 * self.drop_percentile))
        # We return the closest power of two to that anticipated number of segments. Of course we use a floor because we want to fill every batch.
        return 2 ** int(torch.floor(torch.log2(num_segments)))

    def num_segments_in_track(self, duration_in_samples):
        # this function should return the number of segments in the track and consider the overlap factor, self.segment_overlap
        return int(torch.ceil(duration_in_samples / (self.segment_samples * (1 - self.segment_overlap))))

    def random_start(self, duration_in_samples):
        # this function should return a random start index for the track
        return torch.randint(0, duration_in_samples)


    def STFT_dimensions(self, duration_in_samples):
        # this function should return the dimensions of the STFT of the shortest track in the dataset
        F = int(self.transforms.n_fft / 2 + 1)
        T = int(torch.ceil(duration_in_samples / self.transforms.hop_length))
        return (F,T)
        

    def shortest_duration_in_samples(self, mus):
        # this function should return the shortest duration of the stems in the track
        return int(min([track.stems.shape[1] for track in mus.tracks]))

    def longest_duration_in_samples(self, mus):
        # this function should return the longest duration of the stems in the track
        return max(min([track.stems.shape[1] for track in mus.tracks]))

    def is_high_energy_segment(self, segment, threshold):
        # this function decides based on the provided threshold whether a sufficient number of chunks in the segment have an energy above the threshold
        mix_chunk_energies = self.segment_chunk_energies(segment)[0,:]
        return len(torch.argwhere(mix_chunk_energies > threshold)) > (self.chunks_below_percentile * self.segment_chunks)

    def high_energy_segments(self, segments):
        # this function should take in a full track's stems, it will then split the track into segments. 
        # The segments should have an overlap factor of self.segment_overlap.
        # Then, it will split each segment into chunks, and it will calculate the energy of each segment, and store the energy of each segment in a list.
        # With this list, it will calculate the percentile of the energy of the chunks, and it will discard the segments where 25% of the chunks have an energy below the percentile.
        # It will then return the list of segments that have a high enough energy.
        high_energy_indices = []
        threshold = self.segment_energy_threshold(segments)
        for idx, segment in enumerate(segments):
            if self.is_high_energy_segment(segment, threshold):
                high_energy_indices.append(idx)
        return segments[high_energy_indices]

    def segment_energy_threshold(self, segments):
        # this function should split every segment into self.segment_chunks chunks, and it will calculate the energy of each chunk using the RMS energy function.
        # It will save the energy of each chunk in a list, and it will return the value self.percentile_dropped percentile of the list.
        chunk_energies = []
        for segment in segments:
            chunk_energies.extend(self.segment_chunk_energies(segment))
        chunk_energies = torch.stack(chunk_energies)
        percentile = torch.quantile(chunk_energies, self.drop_percentile, interpolation='midpoint')
        return percentile
            
    def segment_chunk_energies(self, segment):
        # this function should split the segment into self.segment_chunks chunks, and it will calculate the energy of each chunk using the RMS energy function.
        # It will save the energy of each chunk in a list, and it will return the list.
        chunk_energies = [[] for i in range(self.num_stems)]
        segment_samples = segment.shape[1]
        chunk_samples = int(segment_samples / self.segment_chunks)
        for i in range(0, segment_samples, chunk_samples):
            chunk = segment[:, i:i+chunk_samples]
            mix_track = chunk[0,:,:]
            rms = self.transforms.rms(mix_track)
            chunk_energies.append(rms)
        return torch.stack(chunk_energies).view(self.segment_chunks)

    def split_track(self, stems):
        # this function should take in a full track, and it will split the track into segments. 
        # The segments should have an overlap factor of self.segment_overlap.
        # We add zero padding to the track to make sure that the track is divisible by the segment length.
        # Then, it will split each segment into chunks, and it will return the list of chunks.
        # The input is a tensor with shape (num_stems, num_samples, num_channels)
        # The output is a tensor array with shape (num_stems, num_segments, num_samples_per_segment, num_channels)
        stems = self.add_zero_padding(stems)
        segments = []
        num_samples = stems.shape[1]
        step_in_samples = int(self.segment_samples * (1 - self.segment_overlap))
        for i in range(0, num_samples - step_in_samples, step_in_samples):
            segment = stems[:, i:i+self.segment_samples]
            segments.append(segment)
        return segments

    def add_zero_padding(self, stems):
        # this function should add zero padding to the track to make sure that the track is divisible by the segment length and the residue from the overlap.
        # the length of the array has to be segment_length + k * samples_in_steps for some nonnegative integer k.
        num_samples = stems.shape[1]
        step_in_samples = int(self.segment_samples * (1 - self.segment_overlap))
        samples_in_last_segment = num_samples % step_in_samples
        if samples_in_last_segment != 0:
            padding = torch.zeros((stems.shape[0], step_in_samples - samples_in_last_segment, stems.shape[2]), device = self.device)
            return torch.cat((stems, padding), axis=1)
        else:
            return stems

    def add_N_zeros(self, stems, N):
        zeros = torch.zeros((stems.shape[0], N, stems.shape[2]))
        return torch.cat((stems, zeros), axis = 1)

