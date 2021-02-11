# These tags will be used as prefix in the vocab dictionary to distinguish between words, pos and actions.
LAB = '<l>'  # tag for label
POS = '<p>'  # tag for pos
NULL = '<null>'  # tag for null words
UNK = '<unk>'  # tag for out of vocab words
ROOT = '<root>'  # tag for root label


class Parser:
    """
    This class prepares the data from the file with the features and associated label and action.
    """

    def __init__(self, data):
        self.root = 'root'
        # Labels
        labels = [self.root] + list(
            set([label for example in data for label in example['label'] if label != self.root]))
        token2idx = {LAB + label: k for k, label in enumerate(labels)}  # Add labels to dictionary
        self.labels = labels  # store all the possible labels
        self.num_labels = len(labels)
        self.L_NULL = len(token2idx)
        token2idx[LAB + NULL] = self.L_NULL  # Add unknown label.

        # Actions
        actions = ['L>' + label for label in labels]  # append all left arcs with L> to actions
        actions += ['R>' + label for label in labels]  # append all right arcs with R> to actions
        actions += ['S']  # shift action
        self.num_actions = len(actions)
        self.action2idx = {a: i for i, a in enumerate(actions)}  # store all possible actions
        self.idx2action = {i: a for i, a in enumerate(actions)}

        # POS tags
        pos = set([p for example in data for p in example['c_pos']])  # get all unique pos tags
        pos_dict = {POS + p: i + len(token2idx) for i, p in enumerate(pos)}  # append all pos tags with POS tag
        token2idx.update(pos_dict)  # store all possible pos tags
        token2idx[POS + UNK] = self.P_UNK = len(token2idx)  # add pos for unknown tag
        token2idx[POS + NULL] = self.P_NULL = len(token2idx)  # add pos for null tag
        token2idx[POS + ROOT] = self.P_ROOT = len(token2idx)  # add pos for root tag

        # Word Lemmas
        words = set([w for example in data for w in example['lemma']])  # get all unique lemmas
        word_dict = {w: i + len(token2idx) for i, w in enumerate(words)}
        token2idx.update(word_dict)  # store all possible words
        token2idx[UNK] = self.UNK = len(token2idx)  # add unknown word
        token2idx[NULL] = self.NULL = len(token2idx)  # add null word
        token2idx[ROOT] = self.ROOT = len(token2idx)  # add root word

        self.token2idx = token2idx  # store this token dict
        self.idx2token = {v: k for k, v in self.token2idx.items()}

        self.num_feats = 48  # selected number of features is 48
        self.num_tokens = len(self.token2idx)

    def get_tokenized_index_vectors(self, data):
        """
        converts the data to corresponding index vectors
        :param data: list[dict], tokenized data
        :return: list[dict], tokenized and indexed data
        """
        vec_data = []
        for example in data:
            word = [self.ROOT] + [self.token2idx[w] if w in self.token2idx.keys() else self.UNK for w in
                                  example['lemma']]
            pos = [self.P_ROOT] + [self.token2idx[POS + p] if POS + p in self.token2idx.keys() else self.P_UNK for p in
                                   example['c_pos']]
            head = [-1] + example['head']
            label = [-1] + [self.token2idx[LAB + ll] if LAB + ll in self.token2idx.keys() else -1 for ll in
                            example['label']]
            entry = dict()
            entry['word'] = word
            entry['pos'] = pos
            entry['head'] = head
            entry['label'] = label
            vec_data.append(entry)
        return vec_data

    def process_example_vectors(self, vec_data):
        """
        Generates features from the vectorized data.
        :param vec_data: [list[dict]] : vectorized data.
        :return: [list[list]] : processed features
        """
        processed_data = []
        for k in range(len(vec_data)):
            example = vec_data[k]
            stack = [0]
            num_words = len(example['word']) - 1
            buf = [i + 1 for i in range(num_words)]
            arcs = []
            data = []
            for i in range(num_words * 2):
                label = self.get_labels(stack, buf, example)  # get label from oracle
                data.append((self.process_features(stack, buf, arcs, example), label))  # append the generate features
                if label == self.num_actions - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                elif label < self.num_labels:
                    arcs.append((stack[-1], stack[-2], label))
                    stack = stack[:-2] + [stack[-1]]
                else:
                    arcs.append((stack[-2], stack[-1], label - self.num_labels))
                    stack = stack[:-1]
            else:
                processed_data += data
        return processed_data

    def add_left_child(self, features, data, null_data, left_child, num):
        if len(left_child) > num:
            features.append(data[left_child[num]])
        else:
            features.append(null_data)

    def add_right_child(self, features, data, null_data, right_child, num):
        if len(right_child) > num:
            features.append(data[right_child[num]])
        else:
            features.append(null_data)

    def add_left_grand_child(self, features, data, null_data, left_grand_child, num):
        if len(left_grand_child) > num:
            features.append(data[left_grand_child[num]])
        else:
            features.append(null_data)

    def add_right_grand_child(self, features, data, null_data, right_grand_child, num):
        if len(right_grand_child) > num:
            features.append(data[right_grand_child[num]])
        else:
            features.append(null_data)

    def add_features(self, features, feat_type, example, left_child, right_child, left_grand_child, right_grand_child):
        data = None
        null_data = None
        if feat_type == 'word':
            data = example['word']
            null_data = self.NULL
        elif feat_type == 'pos':
            data = example['pos']
            null_data = self.P_NULL
        elif feat_type == 'label':
            data = example['label']
            null_data = self.L_NULL
        self.add_left_child(features, data, null_data, left_child, 0)
        self.add_right_child(features, data, null_data, right_child, 0)
        self.add_left_child(features, data, null_data, left_child, 1)
        self.add_right_child(features, data, null_data, right_child, 1)
        self.add_left_grand_child(features, data, null_data, left_grand_child, 0)
        self.add_right_grand_child(features, data, null_data, right_grand_child, 0)

    def get_left_child(self, k, arcs):
        return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

    def get_right_child(self, k, arcs):
        return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                      reverse=True)

    def process_features(self, stack, buf, arcs, ex):
        """
        Generates features using the stack and buffer values.
        :param stack: stack
        :param buf: buffer
        :param arcs: arc-labels
        :param ex: vectorized input sentence
        :return: generated features
        """
        tmp, features = [], []
        for x in stack[-3:]:
            tmp.append(ex['word'][x])
        features += ([self.NULL] * (3 - len(stack)) + tmp)
        tmp2 = []
        for x in buf[:3]:
            tmp2.append(ex['word'][x])
        features += ([self.NULL] * (3 - len(buf)) + tmp2)

        tmp3, p_features = [], []
        for x in stack[-3:]:
            tmp3.append(ex['pos'][x])
        p_features += ([self.P_NULL] * (3 - len(stack)) + tmp3)
        tmp4 = []
        for x in buf[:3]:
            tmp4.append(ex['pos'][x])
        p_features += ([self.P_NULL] * (3 - len(buf)) + tmp4)
        l_features = []

        for i in range(2):
            if i < len(stack):
                k = stack[-i - 1]
                left_child = self.get_left_child(k, arcs)
                right_child = self.get_right_child(k, arcs)
                left_grand_child = []
                right_grand_child = []
                if len(left_child) > 0:
                    left_grand_child = self.get_left_child(left_child[0], arcs)
                if len(right_child) > 0:
                    right_grand_child = self.get_right_child(right_child[0], arcs)
                self.add_features(features, 'word', ex, left_child, right_child, left_grand_child, right_grand_child)
                self.add_features(p_features, 'pos', ex, left_child, right_child, left_grand_child, right_grand_child)
                self.add_features(l_features, 'label', ex, left_child, right_child, left_grand_child, right_grand_child)
            else:
                features += [self.NULL] * 6
                p_features += [self.P_NULL] * 6
                l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.num_feats
        return features

    def get_labels(self, stack, buf, example):
        """
        Works as oracle for the parser.
        :param stack: stack
        :param buf: buffer
        :param example: vectorized input sentence
        :return: arc-label
        """
        if len(stack) < 2:
            return self.num_actions - 1
        cand1 = stack[-1]
        cand2 = stack[-2]
        par1 = example['head'][cand1]
        par2 = example['head'][cand2]
        lab1 = example['label'][cand1]
        lab2 = example['label'][cand2]

        # check for left arc
        if cand2 > 0 and par2 == cand1:
            return lab2
        # check for right arc
        elif cand2 >= 0 and cand2 == par1 and not any([x for x in buf if example['head'][x] == cand1]):
            return lab1 + self.num_labels
        else:
            return self.num_actions - 1

    def process_and_get_features(self, data):
        """
        Generates the tokenized features for the input data
        :param data: input data
        :return: tokenized features
        """
        vector_data = self.get_tokenized_index_vectors(data)
        features = self.process_example_vectors(vector_data)
        return features
