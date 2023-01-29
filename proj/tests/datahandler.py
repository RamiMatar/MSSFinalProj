import torch.nn as nn
import torch

class DataHandler():
    def __init__(self,batch_size,shortest_duration, sampling_rate, longest_duration, segment_overlap, segment_chunks, segment_length, chunks_below_percentile, drop_percentile):
        self.sample_rate = sampling_rate
        self.shortest_duration = shortest_duration
        self.longest_duration = longest_duration
        self.segment_length = segment_length
        self.segment_chunks = segment_chunks
        self.chunks_below_percentile = chunks_below_percentile
        self.segment_overlap = segment_overlap
        self.drop_percentile = drop_percentile
        self.segment_samples = int(self.segment_length * self.sample_rate)
        self.chunk_samples = int(self.segment_samples / self.segment_chunks)
        self.batch_size = batch_size
        
    def batchize_training_item(self,stems):
        # we need to trim the stems to the shortest duration track, starting from a random location
        stems = self.trim_stems(stems, self.random_start(stems.shape[1]))
        # we split the stems from shape (num_stems, num_samples, num_channels) into a tensor with shape (num_stems, num_segments, num_samples_per_segment, num_channels)
        print(stems.shape)
        segments = self.split_track(stems)
        print(segments.shape)
        # now we need to drop out the low energy segments
        segments = self.high_energy_segments(segments)
        print(segments.shape)
        # then we choose a random and continuous yet constant number of segments from the track minus the dropped segments
        return segments[:self.batch_size]

    def batchize_testing_item(self,stems):
        # first we make sure the length of the song will produce a whole number of segments by padding with zeros to the end of the next segment.
        stems = self.add_zero_padding(stems)
        # now we extend the song with zeros to make sure we have an equal batch size for every output.
        to_pad = self.longest_duration_in_samples() - stems.shape[1]
        stems = self.add_N_zeros(stems, to_pad)
        # now we split the stems into equal size segments
        segments = self.split_track(stems)
        # we are now ready to operate on the song since we don't want to drop or modify our data as we conserve it to reconstruct our signal.
        return segments[:self.batch_size]
        
        
    def trim_stems(self, stems, start):
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

    def num_segments_in_track(self, duration_in_samples):
        # this function should return the number of segments in the track and consider the overlap factor, self.segment_overlap
        return int(torch.ceil(torch.Tensor(duration_in_samples / (self.segment_samples * (1 - self.segment_overlap)))))

    def random_start(self, duration_in_samples):
        # this function should return a random start index for the track
        return torch.randint(0, duration_in_samples, (1,))


    def longest_duration_in_samples(self, mus):
        # this function should return the longest duration of the stems in the track
        return max(min([track.stems.shape[1] for track in mus.tracks]))

    def is_high_energy_segment(self, segment, threshold):
        # this function decides based on the provided threshold whether a sufficient number of chunks in the segment have an energy above the threshold
        mix_chunk_energies = self.segment_chunk_energies(segment)[:]
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
        high_energy_indices = torch.tensor(high_energy_indices, dtype= torch.int)
        return torch.index_select(segments, 0, high_energy_indices)

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
        chunk_energies = []
        segment_samples = segment.shape[1]
        chunk_samples = int(segment_samples / self.segment_chunks)
        for i in range(0, segment_samples, chunk_samples):
            chunk = segment[:, i:i+chunk_samples]
            mix_track = chunk[0,:,:]
            squared_tensor = torch.pow(chunk, 2)
            mean_power = torch.mean(squared_tensor)
            rms = torch.sqrt(mean_power)
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
        segments = torch.stack(segments)
        return segments

    def add_zero_padding(self, stems):
        # this function should add zero padding to the track to make sure that the track is divisible by the segment length and the residue from the overlap.
        # the length of the array has to be segment_length + k * samples_in_steps for some nonnegative integer k.
        num_samples = stems.shape[1]
        step_in_samples = int(self.segment_samples * (1 - self.segment_overlap))
        samples_in_last_segment = num_samples % step_in_samples
        if samples_in_last_segment != 0:
            padding = torch.zeros((stems.shape[0], step_in_samples - samples_in_last_segment, stems.shape[2]))
            return torch.cat((stems, padding), axis=1)
        else:
            return stems

    def add_N_zeros(self, stems, N):
        zeros = torch.zeros((stems.shape[0], N, stems.shape[2]))
        return torch.cat((stems, zeros), axis = 1)
    