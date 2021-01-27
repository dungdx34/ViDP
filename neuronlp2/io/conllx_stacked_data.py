__author__ = 'max'

import numpy as np
import torch
from neuronlp2.io.reader import CoNLLXReader, CoNLLXReaderTransform
from neuronlp2.io.conllx_data import _buckets, NUM_SYMBOLIC_TAGS, create_alphabets
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE


def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


def _generate_stack_inputs(heads, types, prior_order):
    if prior_order == 'deep_first':
        child_ids = _obtain_child_index_for_depth(heads, True)
    elif prior_order == 'shallow_first':
        child_ids = _obtain_child_index_for_depth(heads, False)
    elif prior_order == 'left2right':
        child_ids = _obtain_child_index_for_left2right(heads)
    elif prior_order == 'inside_out':
        child_ids = _obtain_child_index_for_inside_out(heads)
    else:
        raise ValueError('Unknown prior order: %s' % prior_order)

    stacked_heads = []
    children = []
    siblings = []
    stacked_types = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    stack = [0]
    position = 1
    while len(stack) > 0:
        head = stack[-1]
        stacked_heads.append(head)
        siblings.append(sibs[head])
        child_id = child_ids[head]
        skip_connect.append(prev[head])
        prev[head] = position
        if len(child_id) == 0:
            children.append(head)
            sibs[head] = 0
            stacked_types.append(PAD_ID_TAG)
            stack.pop()
        else:
            child = child_id.pop(0)
            children.append(child)
            sibs[head] = child
            stack.append(child)
            stacked_types.append(types[child])
        position += 1

    return stacked_heads, children, siblings, stacked_types, skip_connect


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
              max_size=None, normalize_digits=True, prior_order='inside_out', use_test=False):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
        data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

    masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)

    for i, inst in enumerate(data):
        wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
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
        # masks_e
        masks_e[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        inst_size_decoder = 2 * inst_size - 1
        # stacked heads
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[i, :inst_size_decoder] = ssids
        ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[i, :inst_size_decoder] = skip_ids
        skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[i, :inst_size_decoder] = 1.0

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks_e = torch.from_numpy(masks_e)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    stacked_heads = torch.from_numpy(stack_hid_inputs)
    children = torch.from_numpy(chid_inputs)
    siblings = torch.from_numpy(ssid_inputs)
    stacked_types = torch.from_numpy(stack_tid_inputs)
    skip_connect = torch.from_numpy(skip_connect_inputs)
    masks_d = torch.from_numpy(masks_d)

    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
    return data_tensor, data_size


def read_bucketed_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                       max_size=None, normalize_digits=True, prior_order='inside_out'):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
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

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
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
            # masks_e
            masks_e[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            inst_size_decoder = 2 * inst_size - 1
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = torch.from_numpy(masks_d)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
        data_tensors.append(data_tensor)

    return data_tensors, bucket_sizes


def read_data_transform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path,
              max_size=None, normalize_digits=True, prior_order='inside_out', use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReaderTransform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
        if use_elmo:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children,
                 siblings, stacked_types, skip_connect, inst.elmo_ids])
        elif use_bert:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children,
                 siblings, stacked_types, skip_connect, inst.bert_ids])
        elif use_elmo_bert:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children,
                 siblings, stacked_types, skip_connect, inst.elmo_ids,
                 inst.bert_ids])
        else:
            data.append(
                [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children,
                 siblings, stacked_types, skip_connect])

        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
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

    masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

    masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)

    for i, inst in enumerate(data):
        if use_elmo:
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, elids = inst
        elif use_bert:
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, berids = inst
        elif use_elmo_bert:
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, elids, berids = inst
        else:
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst

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
        # masks_e
        masks_e[i, :inst_size] = 1.0
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

        inst_size_decoder = 2 * inst_size - 1
        # stacked heads
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[i, :inst_size_decoder] = ssids
        ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[i, :inst_size_decoder] = skip_ids
        skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[i, :inst_size_decoder] = 1.0

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks_e = torch.from_numpy(masks_e)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    stacked_heads = torch.from_numpy(stack_hid_inputs)
    children = torch.from_numpy(chid_inputs)
    siblings = torch.from_numpy(ssid_inputs)
    stacked_types = torch.from_numpy(stack_tid_inputs)
    skip_connect = torch.from_numpy(skip_connect_inputs)
    masks_d = torch.from_numpy(masks_d)

    if use_elmo:
        elmos = torch.from_numpy(elmo_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'ELMO': elmos}

    elif use_bert:
        berts = torch.from_numpy(bert_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'BERT': berts}

    elif use_elmo_bert:
        elmos = torch.from_numpy(elmo_inputs)
        berts = torch.from_numpy(bert_inputs)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'ELMO': elmos, 'BERT': berts}

    else:
        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}

    return data_tensor, data_size


def read_bucketed_data_transform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path,
                       max_size=None, normalize_digits=True, prior_order='inside_out', use_elmo=False, use_bert=False, use_elmo_bert=False, use_test=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReaderTransform(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, feature_bert_path, elmo_path, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)

                if use_elmo:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads,
                         children, siblings, stacked_types, skip_connect, inst.elmo_ids])
                elif use_bert:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads,
                         children, siblings, stacked_types, skip_connect, inst.bert_ids])
                elif use_elmo_bert:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads,
                         children, siblings, stacked_types, skip_connect, inst.elmo_ids,
                         inst.bert_ids])
                else:
                    data[bucket_id].append(
                        [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads,
                         children, siblings, stacked_types, skip_connect])

                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False, use_elmo=use_elmo, use_bert=use_bert, use_elmo_bert=use_elmo_bert, use_test=use_test)
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

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)

        for i, inst in enumerate(data[bucket_id]):
            if use_elmo:
                wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, elids = inst
            elif use_bert:
                wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, berids = inst
            elif use_elmo_bert:
                wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, elids, berids = inst
            else:
                wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst

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
            # masks_e
            masks_e[i, :inst_size] = 1.0
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

            inst_size_decoder = 2 * inst_size - 1
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = torch.from_numpy(masks_d)

        if use_elmo:
            elmos = torch.from_numpy(elmo_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'ELMO': elmos}
            data_tensors.append(data_tensor)

        elif use_bert:
            berts = torch.from_numpy(bert_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'BERT': berts}
            data_tensors.append(data_tensor)

        elif use_elmo_bert:
            elmos = torch.from_numpy(elmo_inputs)
            berts = torch.from_numpy(bert_inputs)

            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d, 'ELMO': elmos, 'BERT': berts}
            data_tensors.append(data_tensor)

        else:
            data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
            data_tensors.append(data_tensor)

    return data_tensors, bucket_sizes