"""main.py"""

import time
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import numpy as np
from lstmmodel import LSTMModel
from tacdataset import TACDataset
from emode import EMode

class Main:
    """Main class."""

    def __init__(self, root_dir, input_size, output_size, hidden_size,
                 num_lstm_layers, batch_size, sequence_size):
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.sequence_size = sequence_size

    def train(self, number_epochs, learning_rate, participant_number):
        # Default output path
        pnum = str(participant_number)

        """Run model."""
        # LSTM model (2 LSTM layers + 2 fully-connected layers)
        model = LSTMModel(self.input_size, self.output_size, self.hidden_size,
                          self.num_lstm_layers, self.batch_size, self.sequence_size)

        # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Huber loss function
        criterion = nn.MSELoss()

        # dataloader parameters
        params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'drop_last': True}

        # creating dataset object
        training_set = TACDataset(self.root_dir, pnum, EMode.TRAIN, self.sequence_size, 'E')
        training_generator = data.DataLoader(training_set, **params)

        validation_set = TACDataset(self.root_dir, pnum, EMode.VALIDATION, self.sequence_size, 'E')
        validation_generator = data.DataLoader(validation_set, **params)

        #train_loss_it_file = open(self.root_dir + 'losses/P' + pnum + '/train/per_iteration.csv', "a+")
        #train_loss_ep_file = open(self.root_dir + 'losses/P' + pnum + '/train/per_epoch.csv', "a+")
        #val_loss_file = open(self.root_dir + 'losses/P' + pnum + '/val/per_epoch.csv', "a+")

        lr_counter = 0
        for i in range(1, number_epochs):
            print("Epoch " + str(i))
            losses = []
            for inputs, targets in training_generator:
                # zeroing gradient
                model.zero_grad()

                # feeding input into model
                outputs = model(inputs)

                # computing and backpropagation loss
                loss = criterion(outputs, targets.view(-1, self.output_size))
                
                # storing loss
                losses.append(loss.data.item())
                #print(losses[-1])

                # zeroing out optimizer gradient
                optimizer.zero_grad()

                # zeroing the gradient
                loss.backward()

                # updating parameters
                optimizer.step()

                # printing to iteration file
                #train_loss_it_file.write(str(loss.data.item()) + '\n')

            # printing to validation file
            #train_loss_ep_file.write(str(np.average(losses)) + '\n')

            # evaluating validation set
            if i % 100 == 1:
                # saving model state
                #torch.save(model.state_dict(), self.root_dir + 'states/P' + pnum + '/model_' + str(i) + '.pt')

                model.eval()
                loss_validation = []

                for inputs, targets in validation_generator:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.view(-1, self.output_size))
                    loss_validation.append(loss.data.item())

                #val_loss_file.write(str(np.average(loss_validation)) + '\n')
                model.train()

            #if i % 150 == 0 and lr_counter < 8:
            #    for param_group in optimizer.param_groups:
            #        print('halving learning rate to: ' + str(param_group['lr'] / 2))
            #        param_group['lr'] = param_group['lr'] / 2
            #        lr_counter += 1

        #train_loss_it_file.close()
        #train_loss_ep_file.close()
        #val_loss_file.close()

        return model, None#, training_set.scaler

    def test(self, model, scaler, valid_participant_number, test_samples):
        """Tests the trained network with data from valid subject an imposters"""
        pnum = str(valid_participant_number)

        # setting model to evaluation mode
        model.eval()

        # initializing valid and imposter loss
        valid_loss = []
        imposter_loss = []

        # dataloader parameters
        params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'drop_last': True}

        valid_set = TACDataset(self.root_dir, pnum, EMode.TEST, self.sequence_size, 'E', scaler)
        valid_generator = data.DataLoader(valid_set, **params)

        # defining criterion
        criterion = nn.MSELoss()

        # 23 participants - valid
        num_valid = (test_samples / self.batch_size) * 22

        # testing valid participant
        valid_counter = 0
        with torch.no_grad():
            while valid_counter < num_valid:
                for inputs, targets in valid_generator:
                    if valid_counter == num_valid:
                        break

                    outputs = model(inputs)
                    loss = criterion(outputs, targets.view(-1, self.output_size))
                    valid_loss.append(loss.data.item())
                    valid_counter += 1

        # num tests for imposters
        num_imposter = (test_samples / self.batch_size)

        for part in range(2, 27):
            if part not in (valid_participant_number, 20, 22):
                # testing imposter
                imposter_counter = 0

                imposter_set = TACDataset(self.root_dir, str(part),
                                          EMode.TEST, self.sequence_size, 'E', scaler)
                imposter_generator = data.DataLoader(imposter_set, **params)

                with torch.no_grad():
                    while imposter_counter < num_imposter:
                        for inputs, targets in imposter_generator:
                            if imposter_counter == num_imposter:
                                break

                            outputs = model(inputs)
                            loss = criterion(outputs, targets.view(-1, self.output_size))
                            imposter_loss.append(loss.data.item())
                            imposter_counter += 1

        np.savetxt(self.root_dir + 'losses/P' + pnum + '/test/valid_loss_new_architecture.csv', valid_loss, fmt='%2.5f')
        np.savetxt(self.root_dir + 'losses/P' + pnum + '/test/imposter_loss_new_architecture.csv', imposter_loss, fmt='%2.5f')

        return valid_loss, imposter_loss

    def test_stressed(self, model, scaler, valid_participant_number, test_samples):
        """Tests the trained network with data from valid subject an imposters"""
        pnum = str(valid_participant_number)

        # setting model to evaluation mode
        model.eval()

        # initializing relaxed and stressed loss
        relaxed_loss = []
        stressed_loss = []

        # dataloader parameters
        params = {'batch_size': self.batch_size,
                'shuffle': True,
                'drop_last': True}

        relaxed_set = TACDataset(self.root_dir, pnum, EMode.TEST, self.sequence_size, 'E', scaler)
        relaxed_generator = data.DataLoader(relaxed_set, **params)

        # defining criterion
        criterion = nn.MSELoss()

        # 23 participants - valid
        test_size = (test_samples / self.batch_size) * 22

        # testing valid participant
        relaxed_counter = 0
        with torch.no_grad():
            while relaxed_counter < test_size:
                for inputs, targets in relaxed_generator:
                    if relaxed_counter == test_size:
                        break

                    outputs = model(inputs)
                    loss = criterion(outputs, targets.view(-1, self.output_size))
                    relaxed_loss.append(loss.data.item())
                    relaxed_counter += 1

        stressed_set = TACDataset(self.root_dir, pnum, EMode.TEST, self.sequence_size, 'H', scaler)
        stressed_generator = data.DataLoader(stressed_set, **params)

        stressed_counter = 0
        with torch.no_grad():
            while stressed_counter < test_size:
                for inputs, targets in stressed_generator:
                    if stressed_counter == test_size:
                        break

                    outputs = model(inputs)
                    loss = criterion(outputs, targets.view(-1, self.output_size))
                    stressed_loss.append(loss.data.item())
                    stressed_counter += 1


        np.savetxt(self.root_dir + 'losses/P' + pnum + '/test/relaxed_loss_new_architecture.csv', relaxed_loss, fmt='%2.5f')
        np.savetxt(self.root_dir + 'losses/P' + pnum + '/test/stressed_loss_new_architecture.csv', stressed_loss, fmt='%2.5f')

        return relaxed_loss, stressed_loss
