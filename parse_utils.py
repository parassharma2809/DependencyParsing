class ParserUtils:
    """
    Utility class.
    """
    @staticmethod
    def write_output(parent_list, label_list, dev_file, out_file):
        """
        Writes the output of the parser to a file.
        :param parent_list: [list] : list of parents
        :param label_list: [list] : list of corresponding arc labels
        :param dev_file: [string] : path to test file
        :param out_file: [string] : path to output file
        """
        f = open(dev_file, 'r')
        f2 = open(out_file, 'w')
        n = 0
        for line in f.readlines():
            parent = parent_list[n]
            label = label_list[n]
            lin = line.strip().split('\t')
            if len(lin) == 10:
                if int(lin[0]) in parent.keys():
                    lin[6] = parent[int(lin[0])]
                else:
                    lin[6] = '0'
                if int(lin[0]) in label.keys():
                    lin[7] = label[int(lin[0])]
                else:
                    lin[7] = '_'
                f2.write('\t'.join(map(str, lin)))
                f2.write('\n')
            else:
                n += 1
                f2.write('\n')
        f2.close()
        f.close()

    @staticmethod
    def get_legal_label_index(idx2action, softmax_indices, stack_len, buf_len):
        """
        Returns the legal label index.
        :param idx2action: [dict] : index to actions dictionary
        :param softmax_indices: [list] : list of values returned by model.
        :param stack_len: [int] : length of stack
        :param buf_len: [int] : length of buffer
        :return: [int] : legal label index
        """
        for n in range(len(softmax_indices)):
            action = idx2action[softmax_indices[n].item()]
            if (action == 'S' and buf_len > 0) or ('L>' in action and stack_len > 2) or (
                    'R>' in action and stack_len >= 2):
                return softmax_indices[n].item()
            else:
                continue
        return None

    @staticmethod
    def perform_action(k, idx2action, token2idx, legal_label_ind, stack, buf, arcs, parents, labels, vec_data):
        """
        performs the action as predicted by the model.
        :param k: [int] : iteration number
        :param idx2action: [dict] : index to action dictionary
        :param token2idx: [dict] token to index dictionary
        :param legal_label_ind: [int] : index of the legal arc-label
        :param stack: [list] : stack
        :param buf: [list] : buffer
        :param arcs: [list] : current arc list
        :param parents: [list] : parents for words
        :param labels: [list] : label list
        :param vec_data: [list] : vector data
        :return:
        """
        if idx2action[legal_label_ind] == 'S':
            stack.append(buf[0])
            buf.pop(0)
        elif 'L>' in idx2action[legal_label_ind]:
            arcs.append((stack[-1], stack[-2]))
            parents[stack[-2]] = stack[-1]
            vec_data[k]['head'][stack[-2]] = stack[-1]
            labels[stack[-2]] = idx2action[legal_label_ind].split('>')[1]
            vec_data[k]['label'][stack[-2]] = token2idx['<l>' + labels[stack[-2]]]
            stack.pop(-2)
        elif 'R>' in idx2action[legal_label_ind]:
            arcs.append((stack[-2], stack[-1]))
            parents[stack[-1]] = stack[-2]
            vec_data[k]['head'][stack[-1]] = stack[-2]
            labels[stack[-1]] = idx2action[legal_label_ind].split('>')[1]
            vec_data[k]['label'][stack[-1]] = token2idx['<l>' + labels[stack[-1]]]
            stack.pop(-1)
