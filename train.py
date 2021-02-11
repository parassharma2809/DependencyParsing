import argparse
import pickle

import torch
import torch.optim as optim
from torch import nn

from parser_model import ParserModel


def get_features_from_converted_data(feature_file, data_parser):
    """
    Gets the features from the .converted file
    :param feature_file: [string] .converted file to generate features
    :param data_parser: [object] parser object
    :return: feature set , labels
    """
    file = open(feature_file, 'r')
    feature_set = []
    labels = []
    for line in file.readlines():
        line_split = line.strip().split('\t')
        label = data_parser.action2idx[line_split[1]]
        feature_list = line_split[0].split(' ')
        features = []
        for feature in feature_list:
            features.append(data_parser.token2idx[feature])
        feature_set.append(features)
        labels.append(label)
    return feature_set, labels


def train_parse(args):
    """
    Train the parser model.
    :param args: [object] command line args
    """
    # Load arguments and file
    with open(args.v, 'rb') as file:
        data_parser = pickle.load(file)
    data, labels = get_features_from_converted_data(args.i, data_parser)
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    hidden_dim = args.u
    dropout = args.d
    epochs = args.e
    batch_size = args.b
    embedding_dim = args.E

    # model definition
    model = ParserModel(data_parser.num_feats, embedding_dim, len(data_parser.token2idx), hidden_dim,
                        data_parser.num_actions,
                        dropout)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.l)
    model.train()

    if batch_size > len(data):
        batch_size = len(data)

    # Training loop
    for epoch in range(epochs):
        print('Epoch ', epoch + 1)
        losses = 0.0
        for i in range(0, len(data), batch_size):
            # Taking configurations in batches
            if i + batch_size > len(data):
                x = data[i:len(data)]
                y = labels[i:len(data)]
            else:
                x = data[i:i + batch_size]
                y = labels[i:i + batch_size]

            optimizer.zero_grad()
            predictions = model(x)  # Forward pass
            loss = loss_criterion(predictions, y)  # loss calculation
            losses += loss.item()
            loss.backward()  # backward pass
            optimizer.step()
        print('Loss', losses / (len(data) // batch_size))
    with open(args.o, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dependency Parser training arguments')

    parser.add_argument('-u', type=int, default=200, help='number of hidden units')
    parser.add_argument('-l', type=float, default=0.001, help='learning rate')
    parser.add_argument('-E', type=int, default=50, help='embedding dimension length')
    parser.add_argument('-b', type=int, default=64, help='mini-batch size')
    parser.add_argument('-e', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('-d', type=float, default=0.5, help='dropout value')
    parser.add_argument('-i', type=str, help='feature file', required=True)
    parser.add_argument('-v', type=str, default='train.vocab', help='vocab file for data parser', required=True)
    parser.add_argument('-o', type=str, help='model file to be written', required=True)

    args = parser.parse_args()
    train_parse(args)
