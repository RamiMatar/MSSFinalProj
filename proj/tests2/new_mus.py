import musdb
import torch
class newMus(torch.utils.data.Dataset):
    def __init__(self, musdb_root, split='train', subset='train', filtered_indices = None, batch_size = None, is_wav=False, sample_rate=44100, segment_length = 10, segment_chunks = 10, discard_low_energy = True, segment_overlap = 0.5, drop_percentile =  0.1, chunks_below_percentile = 0.5):
        assert(subset == 'train' or subset == 'test')
        self.mode = subset
        self.split = split 
        self.mus = musdb.DB(musdb_root, subsets=subset, split=split, is_wav=is_wav)
        self.sample_rate = sample_rate
        self.discard_low_energy = discard_low_energy
        self.segment_length = segment_length
        self.segment_chunks = segment_chunks
        self.chunks_below_percentile = chunks_below_percentile
        self.segment_overlap = segment_overlap
        self.drop_percentile = drop_percentile
        self.segment_samples = int(self.segment_length * self.sample_rate)
        self.chunk_samples = int(self.segment_samples / self.segment_chunks)
        if filtered_indices is None or batch_size is None:
            self.durations = dict()
            self.filtered_indices = dict()
            self.len = self.init_durations()
            self.shortest_duration = self.shortest_duration_in_samples(self.mus)
            self.batch_size = self.find_batch_size()
        else:
            self.filtered_indices = [int(x) for x in filtered_indices.split(',')]
            self.batch_size = batch_size
            self.len = len(self.filtered_indices)

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        # this function should return a batch of segment STFTs from the song as well as their stem STFTs.
        track = self.mus.tracks[self.filtered_indices[idx]]
        # stems is a list of the stems of the track, in the order of the stems in the track
        return torch.tensor(track.stems,dtype=torch.double)


    def init_durations(self):
        pos = 0
        for idx, track in enumerate(self.mus.tracks):
            print(track.name)
            self.durations[idx] = track.stems.shape[1]
            if self.durations[idx] >= self.segment_samples * 9:
                self.filtered_indices[pos] = idx
                pos += 1
        return pos
        
    def find_batch_size(self):
        # this function should tell us how many STFTs we can fit into a batch based on finding the floor power of 2 of the number of STFTs we fit over the duration of the song
        # each STFT will represent a STFT over a fixed segment length.
        num_segments = self.num_segments_in_track(self.shortest_duration)
        # we anticipate a drop of up to twice the drop percentile (impossible, just to be safe) of the segments.
        num_segments = torch.Tensor([int(num_segments * (1 - 2 * self.drop_percentile))])
        # We return the closest power of two to that anticipated number of segments. Of course we use a floor because we want to fill every batch.
        return 2 ** int(torch.floor(torch.log2(num_segments)))

    def num_segments_in_track(self, duration_in_samples):
        # this function should return the number of segments in the track and consider the overlap factor, self.segment_overlap
        return int(torch.ceil(torch.Tensor([duration_in_samples / (self.segment_samples * (1 - self.segment_overlap))])))

    def shortest_duration_in_samples(self, mus):
        # this function should return the shortest duration of the stems in the track
        min = 100000000
        for i, dur in self.durations.items():
            if i in self.filtered_indices.values():
                if dur < min:
                    min = dur
        return min

    def longest_duration_in_samples(self, mus):
        # this function should return the longest duration of the stems in the track
        return max([duration for duration in self.durations.values()])
