import argparse
from preprocess import Preprocess
from data_parser import Parser
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data with features and labels')

    parser.add_argument('-i', type=str, help='data file to prepare')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-v', type=str, default='train.vocab', help='.vocab file for data parser')
    args = parser.parse_args()

    pp = Preprocess()
    dataset = pp.process(args.i, True)  # preprocess the data file.
    parser = Parser(dataset)
    feature_label_list = parser.process_and_get_features(dataset)  # extract tokenized features
    file = open(args.o, 'w')
    for feature, label in feature_label_list:
        str_to_write = ' '.join([parser.idx2token[f] for f in feature])
        str_to_write += '\t' + parser.idx2action[label]
        file.write(str_to_write + '\n')
    with open(args.v, 'wb') as file2:
        pickle.dump(parser, file2)
    file.close()
    file2.close()
