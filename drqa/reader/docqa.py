import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class DocQA (nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, doc_hidden_size, question_hidden_size, normalize=True):
        super(DocQA, self).__init__()

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, doc_hiddens, question_hidden, x1_mask):
        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
