"""TAC dataset utilities."""
import torch
from torch.autograd import Variable
from torch.utils import data
import pandas as pd
import numpy as np
from emode import EMode
from sklearn import preprocessing

class TACDataset(data.Dataset):
    """TAC dataset with auxiliary methods."""

    def __init__(self, root_dir, participant_number, mode, sequence_size, session_type, scaler=None):
        self.root_dir = root_dir
        self.sequence_size = sequence_size
        self.scaler = scaler

        # loading training data
        if mode == EMode.TRAIN:
            train_x_1, train_y_1 = self.loadData(participant_number, session_type, '1')
            train_x_2, train_y_2 = self.loadData(participant_number, session_type, '2')
            self.train_x = np.concatenate((train_x_1, train_x_2))
            self.train_y = np.concatenate((train_y_1, train_y_2))
        elif mode == EMode.VALIDATION:
            self.train_x, self.train_y = self.loadData(participant_number, session_type, '3')
        else:
            self.train_x, self.train_y = self.loadData(participant_number, session_type, '4')

        ## if scaler != None, apply transform
        #if self.scaler is None:
        #    self.scaler = preprocessing.StandardScaler().fit(self.dataframe_y)

        # applying transform (normalizing data)
        #self.train_y = self.scaler.transform(self.dataframe_y)

    def loadData(self, participant_num, session_type, session_num):
        filename_x = self.root_dir + 'dataset/P' + participant_num + '/X_S' + session_num + session_type + '.csv'
        dataframe_x = pd.read_csv(filename_x, names=['X'])

        # loading labels
        filename_y = self.root_dir + 'dataset/P' + participant_num + '/Y_S' + session_num + session_type + '.csv'
        dataframe_y = pd.read_csv(filename_y, names=['DD', 'UD', 'UU', 'DU'])

        # splitting training data
        inputs = dataframe_x.X
        outputs = np.column_stack((dataframe_y['DD'],
                                   dataframe_y['UD'],
                                   dataframe_y['UU'],
                                   dataframe_y['DU']))

        return inputs, outputs

    def __len__(self):
        return len(self.train_x) - self.sequence_size

    def __getitem__(self, index):

        # initializing inputs and targets to be returned
        inputs = []
        targets = []

        # returning batch of size batch x sequence x dimension
        for i in range(0, self.sequence_size):
            single_input = self.from_string_to_two_hot(self.train_x[index + i])
            single_target = self.train_y[index + i]

            inputs.append(single_input)
            targets.append(single_target)

        # converting to tensor
        inputs_tensor = Variable(torch.Tensor(inputs))
        targets_tensor = Variable(torch.Tensor(targets))

        return inputs_tensor, targets_tensor

    def from_string_to_two_hot(self, two_hot_string):
        """Converts a two-hot string to an array"""
        two_hot_array = []
        for item in two_hot_string:
            two_hot_array.append(int(item))
        return two_hot_array
