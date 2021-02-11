import argparse
import pickle

import torch

from parse_utils import ParserUtils
from preprocess import Preprocess


def parse(args):
    """
    Parses the given test file.
    :param args: [object] command line args
    """
    dev_file = args.i
    pp = Preprocess()
    dev_dataset = pp.process(dev_file, False)
    # Load model and vocabulary
    with open(args.m, 'rb') as file:
        model = pickle.load(file)
    with open(args.v, 'rb') as file2:
        dep_parser = pickle.load(file2)
    model.eval()
    vec_data = dep_parser.get_tokenized_index_vectors(dev_dataset)
    parent_list = []
    label_list = []
    for k in range(len(vec_data)):  # for each training sentence
        stack = [0]
        num_words = len(vec_data[k]['word']) - 1
        buf = [i + 1 for i in range(num_words)]
        arcs = []
        parents = dict()
        labels = dict()
        for p in range(num_words * 2):
            feats = dep_parser.process_features(stack, buf, arcs, vec_data[k])  # extract features for this configuration
            data = torch.tensor(feats).view(1, -1)
            with torch.no_grad():
                predictions = model(data)
                pred_softmax = torch.log_softmax(predictions, dim=1)
                sorted_pred_softmax, softmax_indices = torch.sort(pred_softmax, dim=1, descending=True)
            legal_label_ind = ParserUtils.get_legal_label_index(dep_parser.idx2action, softmax_indices.view(-1),
                                                                len(stack),
                                                                len(buf))  # get legal labels
            ParserUtils.perform_action(k, dep_parser.idx2action, dep_parser.token2idx, legal_label_ind, stack, buf, arcs, parents, labels,
                                       vec_data)  # perform the transition according to legal label
        parent_list.append(parents)
        label_list.append(labels)
    ParserUtils.write_output(parent_list, label_list, dev_file, args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser arguments')

    parser.add_argument('-i', type=str, help='test file')
    parser.add_argument('-m', type=str, help='model file')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-v', type=str, default='train.vocab', help='vocab file')

    args = parser.parse_args()
    parse(args)
