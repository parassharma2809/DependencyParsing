import torch
from torch import nn


class ParserModel(nn.Module):
    """
    Pytorch Model definition class for the parser.
    """
    def __init__(self, input_dim, embedding_dim, vocab_size, hidden_dim, output_dim, dropout_p=0.1):
        """
        Init for the model
        :param input_dim: [int] : input dimensions
        :param embedding_dim: [int] : output dimensions
        :param vocab_size: [int] : vocab size for embeddings
        :param hidden_dim: [int] : hidden dimension size
        :param output_dim: [int] : number of output classes
        :param dropout_p: [float] : dropout probability
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(input_dim * embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Performs forward pass on input
        :param x: [tensor] : input
        :return: [tensor] : output
        """
        x = self.embeddings(x).view((x.size(0), -1))
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
