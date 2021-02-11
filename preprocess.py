class Preprocess:
    def __init__(self):
        pass

    def check_if_projective(self, head_list, is_training=True):
        """
        Checks if the tree for the given sentence is projective.
        :param head_list: list, list of heads for all the words in the sentence.
        :param is_training: boolean, if it's training or testing
        :return: boolean, whether sentence is projective or not
        """
        if not is_training:  # In testing we won't have head list, so need to check for projective.
            return True
        arcs = []
        for i, h in enumerate(head_list):
            arcs.append((i + 1, h))  # prepare list of arcs from the given head list

        for arc in arcs:  # select an arc
            min_p = min(arc[0], arc[1])
            max_p = max(arc[0], arc[1])
            for arc1 in arcs:  # for all the other arcs
                a1 = min(arc1[0], arc1[1])
                a2 = max(arc1[0], arc1[1])
                if min_p == a1 and max_p == a2:
                    continue
                # check if any arcs are crossing, if yes then tree is non-projective
                if (max_p > a1 > min_p and not (max_p > a2 > min_p)) \
                        or (max_p > a2 > min_p and not (max_p > a1 > min_p)):
                    if min_p == a1 or max_p == a1 or max_p == a2 or min_p == a2:
                        continue
                    return False
        return True

    def process(self, data_file, is_training=True):
        """
        Process the given data file.
        :param data_file: string, path to file
        :param is_training: boolean, if it's training or testing
        :return: list[dict], processed data
        """
        data = []
        f = open(data_file)
        word, lemma, c_pos, f_pos, head, label, other_dep = [], [], [], [], [], [], []
        for line in f.readlines():
            lin = line.strip().split('\t')
            if len(lin) == 10:
                word.append(lin[1].lower())
                lemma.append(lin[2].lower())
                c_pos.append(lin[3])
                f_pos.append(lin[4])
                if is_training:
                    head.append(int(lin[6]))
                else:
                    head.append(lin[6])
                label.append(lin[7])
                other_dep.append(lin[8])
            elif len(word) > 0:
                if self.check_if_projective(head, is_training):  # Add to data only if projective
                    data.append(
                        {'word': word, 'lemma': lemma, 'c_pos': c_pos, 'f_pos': f_pos, 'head': head, 'label': label,
                         'other_dep': other_dep})
                word, lemma, c_pos, f_pos, head, label, other_dep = [], [], [], [], [], [], []
        if len(word) > 0:
            if self.check_if_projective(head, is_training):
                data.append({'word': word, 'lemma': lemma, 'c_pos': c_pos, 'f_pos': f_pos, 'head': head, 'label': label,
                             'other_dep': other_dep})
        f.close()
        return data
