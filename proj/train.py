# This class should define the training loop for the model. We will define a training loop function
# as well as a train_one_epoch function. This is based on the tutorial on the official PyTorch website: 
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
# we will also implement our training via train/validate/test splits, and save our best models parameters.
# I will use SummaryWriter, a useful class for reporting our data from PyTorch.
import torch
from torch.utils.tensorboard import SummaryWriter
from musdb import collate
import datetime

class Train():
    def __init__(self, train_dataset, validation_dataset, summary_writer, optimizer, model, learning_rate, loss, epochs = 1000, report_loss_frequency=15):
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collate)
        self.validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, collate_fn=collate)
        self.writer = SummaryWriter('Runs/{}_{}'.format())
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.report_loss_frequency = report_loss_frequency
        self.run_model()

    def train_one_epoch(self, epoch_number):
        epoch_loss = 0.0
        last_loss = 0.0
        for i, data in enumerate(self.train_dataloader):
            # We will use the PyTorch Data Loader class to easily iterate through the data and collect our batches.
            # See musdb.py for a deeper explanation of how we collect batches. Simply speaking, we take a an arbitrary
            # fixed-length part of the song and we split it into fixed length segments, taking the STFT of each. 
            mixture_stft, real_stems_stft = data
            # PyTorch accumulates gradients by default, so we zero out the gradients before each batch update in order to 
            # ensure that the optimizer only uses the gradients from the current batch during the update step.
            # This also should help with performance and memory costs.
            self.optimizer.zero_grad()
            # We call our model to make a prediction
            predicted_stems_stft = self.model(mixture_stft)
            # We compute the loss from our loss function defined in loss.py
            loss = self.loss(predicted_stems_stft, real_stems_stft)
            # We calculate the gradient
            loss.backward()
            # Then update the weights based on the gradient
            self.optimizer.step()
            # In order to report our loss, we print the loss every 15 batches
            accumulating_loss += loss.item()
            if i % self.report_loss_frequency == self.report_loss_frequency - 1:
                last_loss = accumulating_loss
                print("Batch {}. Average loss over last {} batches is: {}".format(i+1, last_loss))
                writer_index = epoch_number * len(self.train_dataloader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, writer_index)
                accumulating_loss = 0
            
            return last_loss

    def run_model(self, epochs):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        #set arbitrarily large initial validation loss
        best_vloss = 2.0 ** 30
        
        for epoch in range(epochs):
            print("STARTING EPOCH {}".format(epoch_number + 1))
            # We want to learn in our learning loops, so we make sure our gradients are updating during each epoch.
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, self.writer)
            # We don't need or want the gradient opertaion on any of the 
            self.model.train(False)

            accumulating_vloss = 0.0

            for i, vdata in enumerate(self.validation_dataloader):
                vmixture_stft, vreal_stems_stft = vdata
                vpredicted_stems_stft = self.model(vmixture_stft)
                vloss = self.loss(vpredicted_stems_stft, vreal_stems_stft)
                accumulating_vloss += vloss
            
            avg_vloss = accumulating_vloss / (i + 1)
            print('LOSS training set: {} validation set: {}'.format(avg_loss, avg_vloss))
            
            # log all of this 
            self.writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)
            
            epoch_number += 1


                


    
    


