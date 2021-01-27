'''
Dung Doan
'''

import torch

class InputExample(object):
    def __init__(self, guid, text_a):
        self.guid = guid
        self.text_a = text_a

def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    for line in f:
        if '# sent_id' in line or '# text' in line or '# newdoc' in line or '# source' in line or '# orig' in line:
            continue

        if len(line) == 0 or line[0] == '\n':
            if len(sentence) > 0:
                data.append((sentence))
                sentence = []
            continue

        splits = line.split('\t')
        sentence.append(splits[1])

    if len(sentence) > 0:
        data.append((sentence))
        sentence = []

    return data

class DataProcessor(object):

    def _get_train_examples(self, data_dir):
        raise NotImplementedError()

    def _get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def _get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        return readfile(input_file)

class DPProcessor(DataProcessor):

    def get_train_examples(self, train_data):
        return self._create_examples(self._read_tsv(train_data), "train")

    def get_dev_examples(self, dev_data):
        return self._create_examples(self._read_tsv(dev_data), "dev")

    def get_test_examples(self, test_data):
        return self._create_examples(self._read_tsv(test_data), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, sentence in enumerate(lines):
            # if i == 100:
            #     break
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=sentence))
        return examples





















