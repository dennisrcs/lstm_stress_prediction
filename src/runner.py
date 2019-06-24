"""Script for training LSTMs using the TAC data"""
from main import Main
import numpy as np
import torch
from lstmmodel import LSTMModel

# model parameters
ROOT_DIR = 'C:/Users/silva.dennis/Documents/workplace/'
INPUT_SIZE = 52
OUTPUT_SIZE = 4
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 16
SEQUENCE_SIZE = 5

# run parameters
LEARNING_RATE = 0.01
EPOCHS_NUMBER = 5000

# test parameters
TEST_SAMPLES = 160

for p in range(5, 27):
    if p is not 20 and p is not 22:
        PARTICIPANT_NUMBER = p
        print('Evaluating participant ' + str(p) + '...')

        # run
        main = Main(ROOT_DIR, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LSTM_LAYERS, BATCH_SIZE, SEQUENCE_SIZE)
        model, scaler = main.train(EPOCHS_NUMBER, LEARNING_RATE, PARTICIPANT_NUMBER)
        valid_loss, imposter_loss = main.test_stressed(model, scaler, PARTICIPANT_NUMBER, TEST_SAMPLES)
