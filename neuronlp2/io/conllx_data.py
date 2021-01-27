__author__ = 'max'

import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch

from neuronlp2.io.reader import CoNLLXReader, CoNLLXReaderTransform
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID, NUM_CHAR_PAD
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 140]


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True):

    def expand_vocab():
        vocab_set = set(vocab_list)
        max_sent_length = 0
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                sent_length = 1
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        for val in _buckets:
                            if val > sent_length:
                                sent_length = val
                                break
                        if sent_length > max_sent_length:
                            max_sent_length = sent_length
                        sent_length = 1
                        continue

                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    # word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    word = tokens[1]

                    pos = tokens[4]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)
        return max_sent_length

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    max_sent_length = 0
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = defaultdict(int)
        sent_length = 1
        with open(train_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    for val in _buckets:
                        if val > sent_length:
                            sent_length = val
                            break
                    if sent_length > max_sent_length:
                        max_sent_length = sent_length
                    sent_length = 1
                    continue

                tokens = line.split('\t')
                for char in tokens[1]:
                    char_alphabet.add(char)

                sent_length += 1
                # word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                word = tokens[1]
                # word = word.lower()

                # vocab[word] += 1
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

                pos = tokens[4]
                pos_alphabet.add(pos)

                type = tokens[7]
                type_alphabet.add(type)

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            # assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            # expand_vocab()
            sent_length = expand_vocab()
            if sent_length > max_sent_length:
                max_sent_length = sent_length

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length


def read_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
              max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_test=False):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):
        wids, cid_seqs, pids, hids, tids = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, :len(cids)] = cids
            cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[i, :inst_size] = tids
        tid_inputs[i, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
        # masks
        masks[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks = torch.from_numpy(masks)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                   'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
    return data_tensor, data_size


def read_bucketed_data(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet,
                       max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

def read_data_transform(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet, feature_bert_path, elmo_path,
              max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReaderTransform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        if use_elmo:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids])
        elif use_bert:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.bert_ids])
        elif use_elmo_bert:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids,
                 inst.bert_ids])
        else:
            data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])

        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    if use_elmo:
        elmo_dim = 1024
        scale = np.sqrt(3.0 / elmo_dim)
        elmo_inputs = np.random.uniform(-scale, scale, [data_size, max_length, 1024]).astype(np.float32)

    if use_bert:
        bert_dim = 768
        scale = np.sqrt(3.0 / bert_dim)
        bert_inputs = np.random.uniform(-scale, scale, [data_size, max_length, 768]).astype(np.float32)

    if use_elmo_bert:
        elmo_dim = 1024
        scale = np.sqrt(3.0 / elmo_dim)
        elmo_inputs = np.random.uniform(-scale, scale, [data_size, max_length, 1024]).astype(np.float32)

        bert_dim = 768
        scale = np.sqrt(3.0 / bert_dim)
        bert_inputs = np.random.uniform(-scale, scale, [data_size, max_length, 768]).astype(np.float32)

    masks = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):
        if use_elmo:
            wids, cid_seqs, pids, hids, tids, elids = inst
        elif use_bert:
            wids, cid_seqs, pids, hids, tids, berids = inst
        elif use_elmo_bert:
            wids, cid_seqs, pids, hids, tids, elids, berids = inst
        else:
            wids, cid_seqs, pids, hids, tids = inst

        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, :len(cids)] = cids
            cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[i, :inst_size] = tids
        tid_inputs[i, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
        # masks
        masks[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        if use_elmo:
            elmo_inputs[i, :inst_size] = elids[0]

        if use_bert:
            bert_inputs[i, :inst_size] = berids[0:inst_size]

        if use_elmo_bert:
            elmo_inputs[i, :inst_size] = elids[0]
            bert_inputs[i, :inst_size] = berids[0:inst_size]

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks = torch.from_numpy(masks)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    if use_elmo:
        elmos = torch.from_numpy(elmo_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'ELMO': elmos}

    elif use_bert:
        berts = torch.from_numpy(bert_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'BERT': berts}

    elif use_elmo_bert:
        elmos = torch.from_numpy(elmo_inputs)
        berts = torch.from_numpy(bert_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'ELMO': elmos, 'BERT': berts}

    else:
        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                       'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
    return data_tensor, data_size

def read_bucketed_data_transform(source_path: str, word_alphabet: Alphabet, char_alphabet: Alphabet, pos_alphabet: Alphabet, type_alphabet: Alphabet, feature_bert_path, elmo_path,
                       max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReaderTransform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                if use_elmo:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids])
                elif use_bert:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.bert_ids])
                elif use_elmo_bert:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids,
                         inst.bert_ids])
                else:
                    data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        if use_elmo:
            elmo_dim = 1024
            scale = np.sqrt(3.0 / elmo_dim)
            elmo_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 1024]).astype(np.float32)

        if use_bert:
            bert_dim = 768
            scale = np.sqrt(3.0 / bert_dim)
            bert_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 768]).astype(np.float32)

        if use_elmo_bert:
            elmo_dim = 1024
            scale = np.sqrt(3.0 / elmo_dim)
            elmo_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 1024]).astype(np.float32)

            bert_dim = 768
            scale = np.sqrt(3.0 / bert_dim)
            bert_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 768]).astype(np.float32)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            if use_elmo:
                wids, cid_seqs, pids, hids, tids, elids = inst
            elif use_bert:
                wids, cid_seqs, pids, hids, tids, berids = inst
            elif use_elmo_bert:
                wids, cid_seqs, pids, hids, tids, elids, berids = inst
            else:
                wids, cid_seqs, pids, hids, tids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            if use_elmo:
                elmo_inputs[i, :inst_size] = elids[0]

            if use_bert:
                bert_inputs[i, :inst_size] = berids[0:inst_size]

            if use_elmo_bert:
                elmo_inputs[i, :inst_size] = elids[0]
                bert_inputs[i, :inst_size] = berids[0:inst_size]

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks = torch.from_numpy(masks)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        if use_elmo:
            elmos = torch.from_numpy(elmo_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                           'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'ELMO': elmos}
            data_tensors.append(data_tensor)

        elif use_bert:
            berts = torch.from_numpy(bert_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                           'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'BERT': berts}
            data_tensors.append(data_tensor)

        elif use_elmo_bert:
            elmos = torch.from_numpy(elmo_inputs)
            berts = torch.from_numpy(bert_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                           'MASK': masks, 'SINGLE': single, 'LENGTH': lengths, 'ELMO': elmos, 'BERT': berts}
            data_tensors.append(data_tensor)

        else:
            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types,
                           'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
            data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes

def read_data_(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False, len_thresh=None, use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    counter_added = 0
    reader = CoNLLXReaderTransform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        if inst_size <= len_thresh:
            sent = inst.sentence
            for bucket_id, bucket_size in enumerate(_buckets):
                if inst_size < bucket_size:
                    if use_elmo:
                        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids])
                    elif use_bert:
                        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.bert_ids])
                    elif use_elmo_bert:
                        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.elmo_ids, inst.bert_ids])
                    else:
                        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                    max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                    if max_char_length[bucket_id] < max_len:
                        max_char_length[bucket_id] = max_len
                    break
            counter_added += 1

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    reader.close()
    print("Total number of data: %d, used: %d" % (counter, counter_added))
    return data, max_char_length

def read_data_to_tensor(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, max_size=None,
                          normalize_digits=True, symbolic_root=False, symbolic_end=False,
                          device=torch.device('cpu'), volatile=False, len_thresh=100000, use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data, max_char_length = read_data_(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path,
                                      max_size=max_size, normalize_digits=normalize_digits, symbolic_root=symbolic_root,
                                      symbolic_end=symbolic_end, len_thresh=len_thresh, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        if use_elmo:
            elmo_dim = 1024
            scale = np.sqrt(3.0 / elmo_dim)
            elmo_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 1024]).astype(np.float32)

        if use_bert:
            bert_dim = 768
            scale = np.sqrt(3.0 / bert_dim)
            bert_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 768]).astype(np.float32)

        if use_elmo_bert:
            elmo_dim = 1024
            scale = np.sqrt(3.0 / elmo_dim)
            elmo_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 1024]).astype(np.float32)

            bert_dim = 768
            scale = np.sqrt(3.0 / bert_dim)
            bert_inputs = np.random.uniform(-scale, scale, [bucket_size, bucket_length, 768]).astype(np.float32)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            if use_elmo:
                wids, cid_seqs, pids, hids, tids, elids = inst
            elif use_bert:
                wids, cid_seqs, pids, hids, tids, berids = inst
            elif use_elmo_bert:
                wids, cid_seqs, pids, hids, tids, elids, berids = inst
            else:
                wids, cid_seqs, pids, hids, tids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            if use_elmo:
                elmo_inputs[i, :inst_size] = elids[0]

            if use_bert:
                bert_inputs[i, :inst_size] = berids[0:inst_size]

            if use_elmo_bert:
                elmo_inputs[i, :inst_size] = elids[0]
                bert_inputs[i, :inst_size] = berids[0:inst_size]

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        single = torch.from_numpy(single).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        if use_elmo:
            elmos = torch.from_numpy(elmo_inputs).to(device)
            data_variable.append((words, chars, pos, heads, types, masks, single, lengths, elmos))
        elif use_bert:
            berts = torch.from_numpy(bert_inputs).to(device)
            data_variable.append((words, chars, pos, heads, types, masks, single, lengths, berts))
        elif use_elmo_bert:
            elmos = torch.from_numpy(elmo_inputs).to(device)
            berts = torch.from_numpy(bert_inputs).to(device)
            data_variable.append((words, chars, pos, heads, types, masks, single, lengths, elmos, berts))
        else:
            data_variable.append((words, chars, pos, heads, types, masks, single, lengths))

    return data_variable, bucket_sizes


def get_batch_tensor(data, batch_size, unk_replace=0., use_elmo=False, use_bert=False, use_elmo_bert=False):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    if use_elmo:
        words, chars, pos, heads, types, masks, single, lengths, elmos = data_variable[bucket_id]
    elif use_bert:
        words, chars, pos, heads, types, masks, single, lengths, berts = data_variable[bucket_id]
    elif use_elmo_bert:
        words, chars, pos, heads, types, masks, single, lengths, elmos, berts = data_variable[bucket_id]
    else:
        words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]

    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    index = index.to(words.device)

    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, bucket_length)
        noise = masks.new_empty(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    if use_elmo:
        return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index], elmos[index]
    elif use_bert:
        return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index], berts[index]
    elif use_elmo_bert:
        return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index], elmos[index], berts[index]
    else:
        return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]

def iterate_batch_tensor(data, batch_size, unk_replace=0., shuffle=False, use_elmo=False, use_bert=False, use_elmo_bert=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        if use_elmo:
            words, chars, pos, heads, types, masks, single, lengths, elmos = data_variable[bucket_id]
        elif use_bert:
            words, chars, pos, heads, types, masks, single, lengths, berts = data_variable[bucket_id]
        elif use_elmo_bert:
            words, chars, pos, heads, types, masks, single, lengths, elmos, berts = data_variable[bucket_id]
        else:
            words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]

        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = masks.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            if use_elmo:
                yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                      masks[excerpt], lengths[excerpt], elmos[excerpt]
            elif use_bert:
                yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                      masks[excerpt], lengths[excerpt], berts[excerpt]
            elif use_elmo_bert:
                yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                      masks[excerpt], lengths[excerpt], elmos[excerpt], berts[excerpt]
            else:
                yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                      masks[excerpt], lengths[excerpt]
