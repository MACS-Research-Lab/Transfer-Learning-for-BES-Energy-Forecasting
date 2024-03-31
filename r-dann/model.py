import torch.nn as nn
from functions import ReverseLayerF
import torch


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLayer, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.Linear = nn.Linear(hidden_size, 50)

    def forward(self, input):
        x = self.LSTM(input)
        out = self.Linear(x[0])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()

        # LSTM + Dense model for feature extraction - Create custom LSTM and modify forward func
        self.feature = LSTMLayer(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

        # Class classifier
        self.class_classifier = nn.Sequential()

        self.class_classifier.add_module("c_fc1", nn.Linear(50, 100))
        self.class_classifier.add_module("c_relu1", nn.ReLU(True))
        self.class_classifier.add_module("c_drop1", nn.Dropout(dropout_rate))
        self.class_classifier.add_module("c_fc2", nn.Linear(100, 100))
        self.class_classifier.add_module("c_relu2", nn.ReLU(True))
        self.class_classifier.add_module(
            "c_fc3", nn.Linear(100, 1)
        )  # 1 for regression output

        # Domain classifier
        self.domain_classifier = nn.Sequential()

        self.domain_classifier.add_module("d_fc1", nn.Linear(50, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))
        self.domain_classifier.add_module("d_fc2", nn.Linear(100, 2))
        self.domain_classifier.add_module(
            "d_softmax", nn.LogSoftmax(dim=0)
        )  # fixme: change to 0

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)  # Assuming batch_first=True
        feature = feature[:, -1, :]  # Get the output of the last time step

        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
