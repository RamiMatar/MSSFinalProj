# I will use the museval package to calculate the SDR, SIR, and SAR scores as well as
# the time invariant SDR, SIR SAR variants.
# Importantly, this class will allow us to test the model on full songs. For training, we make some sacrifices to opitimize 
# computations and introduce randomization and we end up processing the song in parts over training. This is also because our reconstruction
# task is to recreate the segments of the song that were tested and to compare only with those features.
import torch
import museval 
from loss import batch_time_signals
from musdb import collate
from model import my_model

def load_model(model_path):
    model = my_model()
    model.load_state_dict(model_path)
    model.eval()
    return model

class Evaluate():
    def __init__(self, test_dataset, model_path):
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, collate_fn = collate)
        self.model = load_model(model_path)
        self.mus = test_dataset.mus

    def reconstruct_all_and_evaluate(self):
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                mixture_stft, _ = data
                predicted_stems_stft = self.model(mixture_stft)
                self.reconstruct_one_and_evaluate(predicted_stems_stft, i)

    def reconstruct_one_and_evaluate(self, predicted_stems_stfts, i): 
        stem_estimates_time_signals = self.reconstruct_one_song(predicted_stems_stfts)
        self.evaluate_one_song(stem_estimates_time_signals, i)

    def evaluate_one_song(self, stems_estimates_time_signals, i):
        # this function will evaluate the song using the museval package.
        # it will receive the time signals of the stems and the song number in the dataset.
        # it will both save the results in the results and print the scores after evaluating each song.
        scores = museval.eval_mus_track(self.mus[i], stems_estimates_time_signals, output_dir='results')   
        print(scores)

    def reconstruct_one_song(self, predicted_stems_stft):
        # this function will reconstruct the song from the predicted stems stft and return the time signals of the stems.
        # We will use the overlap and add method to reconstruct the song after we recover the time signal for each segment for each stem.
        time_signals = batch_time_signals(predicted_stems_stft)
        stem_estimates_time_signals = self.overlap_and_add(time_signals)
        return stem_estimates_time_signals

    def overlap_and_add(self, time_signals):
        # receives all the reconstructed time signals and overlaps them into one signal to reconstruct the original song.
        # this will happen based on the parameters in the dataset for segment length and overlap.
        step_in_samples = time_signals.shape[2] * (1 - self.test_dataset.overlap)
        segment_length = time_signals.shape[2]
        num_segments = time_signals.shape[1]
        full_signal_length = segment_length + step_in_samples * (num_segments - 1)
        new_signal = torch.zeros((time_signals.shape[0], full_signal_length))
        for i in range(num_segments):
            new_signal[:, i * step_in_samples : i * step_in_samples + segment_length] += time_signals[:, i, :]
        return new_signal


