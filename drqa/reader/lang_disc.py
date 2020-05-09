import torch
from torch import autograd, nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
    Ref : https://github.com/ccsasuke/adan/blob/master/code/models.py
"""
class LanguageDetector(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        hidden_size_2 = hidden_size // 1000
        h1, h2 = 0, 0
        for i in range(num_layers):
            h1 = hidden_size
            h2 = hidden_size_2
            if i > 0:
                h1 = hidden_size_2
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(h1, h2))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(h2))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(h2, 1))

    def forward(self, input):
        # mask = mask.unsqueeze(1)
        # input = torch.matmul (mask, input)
        # input = input.squeeze (1)
        return self.net(input)

class LanguageDetectorRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, batch_norm=False):
        super(LanguageDetectorRNN, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # input size = (B, T, F) -> output size = ()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        
    def forward(self, input):
        # input : (B, T, F)
        lengths = input.shape[1] * (torch.ones(input.shape[0]).int())
        packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        with torch.backends.cudnn.flags(enabled=False):
            output, (h, c) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[:, -1,:]

        out = self.linear(output)
        return out
