from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys

sys.path.append(".")
sys.path.append("..")

import json
import torch
from neuronlp2.io import get_logger
from neuronlp2.io import conllx_data
from neuronlp2.io import CoNLLXWriter
from neuronlp2.models import DeepBiAffineTransform
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from transformers import AutoTokenizer
from transformers import AutoModel
import os, re
from vncorenlp import VnCoreNLP
from flask import Flask
from flask import request
import logging
from werkzeug.exceptions import BadRequest
import api.http_response_message as http_response_message
import api.http_sender as http_sender
from api.configs import *
from api.utils import DPProcessor

annotator = VnCoreNLP(vncore_path, annotators="wseg,pos", max_heap_size='-Xmx2g')
print('vncore loaded.')

def word_segmentation(text):
    try:
        text = re.sub("\s+", " ", text)
        words = []
        postags = []
        sentences = annotator.annotate(text)['sentences']
        for sentence in sentences:
            words += [word['form'] for word in sentence]
            postags += [word['posTag'] for word in sentence]
        return words, postags
    except Exception as err:
        print(err)
    return [], []

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, token_type_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    unk_token = tokenizer.unk_token
    cls_token_segment_id = 0
    sequence_a_segment_id = 0
    pad_token_id = tokenizer.pad_token_id
    pad_token_segment_id = 0
    mask_padding_with_zero = True
    start_token = 0
    end_token = 2

    for (ex_index, example) in enumerate(examples):
        words = example.text_a
        tokens = []

        for i, word in enumerate(words):
            word = word.replace(" ", "_")

            word_tokens = tokenizer.encode(word)
            word_tokens = word_tokens[1:-1]
            if len(word_tokens) > 1:
                word_tokens = [word_tokens[0]]

            if not word_tokens:
                word_tokens = [unk_token]

            tokens.extend(word_tokens)

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        tokens += [end_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [start_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokens
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(input_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(len(input_mask), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_length)
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids))

    return features

app = Flask(__name__)

logger = get_logger("DP_API")

use_gpu = True
use_elmo_bert = True

@app.route('/nlp/parser', methods=['POST'])
def parser():
    try:
        params = request.json
        if params is None:
            return http_sender.send_error_response(http_response_message.ResponseCode.JSON_SYNTAX_ERROR)
        if 'sentence' in params:
            query = params['sentence'].strip()

            logging.info("Input: " + query)

            # check query null
            if query.strip() == "":
                return http_sender.send_error_response(http_response_message.ResponseCode.EMPTY_REQUEST)

            result_segment = ''
            words, postags = word_segmentation(query)
            for index, (word, pos) in enumerate(zip(words, postags)):
                word = word.replace("_", " ")
                if pos == 'CH':
                    pos = 'PUNCT'
                elif pos == 'L':
                    pos = 'DET'
                elif pos == 'A':
                    pos = 'ADJ'
                elif pos == 'R':
                    pos = 'ADV'
                elif pos == 'Np':
                    pos = 'NNP'
                elif pos == 'M':
                    pos = 'NUM'
                elif pos == 'E':
                    pos = 'PRE'
                elif pos == 'P':
                    pos = 'PRO'
                elif pos == 'Cc':
                    pos = 'CC'
                elif pos == 'T':
                    pos = 'PART'
                elif pos == 'Y':
                    pos = 'NNP'
                elif pos == 'Cb':
                    pos = 'CC'
                elif pos == 'Eb':
                    pos = 'FW'
                elif pos == 'Ni':
                    pos = 'Ny'
                elif pos == 'B':
                    pos = 'NNP'
                elif pos == 'L':
                    pos = 'DET'
                elif pos == 'Aux':
                    pos = 'AUX'
                elif pos == 'NN':
                    pos = 'N'

                result_segment += str(index + 1) + '\t' + word + '\t' + word.lower() + '\t' + pos + '\t' + pos + '\t' \
                           + '_' + '\t' + '_' + '\t' + '_' + '\t' + '_' + '\t' + '_' + '\n'

            result_segment = result_segment.strip()

            # split data for test
            test_folder = 'tmp'
            if not os.path.exists(test_folder):
                os.mkdir(test_folder)
            else:
                for file in os.listdir(test_folder):
                    os.remove(test_folder + '/' + file)

            output_path = test_folder + '/test.txt'
            fout = open(output_path, 'w')
            fout.write(result_segment + '\n')
            fout.close()

            alphabet_path = os.path.join(model_path, 'alphabets/')
            model_name = os.path.join(model_path, 'network.pt')
            word_alphabet, char_alphabet, pos_alphabet, \
            type_alphabet, max_sent_length = conllx_data.create_alphabets(alphabet_path, None,
                                                                          data_paths=[None, None],
                                                                          max_vocabulary_size=50000,
                                                                          embedd_dict=None)

            num_words = word_alphabet.size()
            num_chars = char_alphabet.size()
            num_pos = pos_alphabet.size()
            num_types = type_alphabet.size()

            logger.info("Word Alphabet Size: %d" % num_words)
            logger.info("Character Alphabet Size: %d" % num_chars)
            logger.info("POS Alphabet Size: %d" % num_pos)
            logger.info("Type Alphabet Size: %d" % num_types)

            tokenizer = AutoTokenizer.from_pretrained(phobert_path)
            model_bert = AutoModel.from_pretrained(phobert_path)

            processor = DPProcessor()

            test_path = 'tmp/test.txt'
            feature_bert_path = 'tmp/phobert_features.pth'
            train_examples = processor.get_train_examples(test_path)
            all_lengths = []
            for t in train_examples:
                all_lengths.append(len(t.text_a))
            max_seq_len = max(all_lengths) + 1

            if max_seq_len > 512:
                max_seq_len = 512
                logger.info("Max sequence length reset to 512")

            device = torch.device("cuda")
            model_bert.to(device)

            train_features = convert_examples_to_features(train_examples, max_seq_len, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

            model_bert.eval()
            to_save = {}

            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids = batch

                with torch.no_grad():
                    all_encoder_layers = model_bert(input_ids, attention_mask=input_mask,
                                                    token_type_ids=token_type_ids)

                output_ = all_encoder_layers[0]

                for j in range(len(input_ids)):
                    sent_id = j + step * 32
                    layer_output = output_[j, :input_mask[j].to('cpu').sum()]
                    to_save[sent_id] = layer_output.detach().cpu().numpy()

            torch.save(to_save, feature_bert_path)

            data_test = conllx_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet,
                                                        type_alphabet,
                                                        feature_bert_path, elmo_path, symbolic_root=True,
                                                        device=device, use_elmo=False, use_bert=False, use_elmo_bert=True,
                                                        use_test=True)

            pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

            logger.info('model: %s' % model_name)

            def load_model_arguments_from_json():
                arguments = json.load(open(arg_path, 'r'))
                return arguments['args'], arguments['kwargs']

            arg_path = model_name[0:-1] + '.arg.json'
            if not os.path.isfile(arg_path):
                arg_path = model_name + '.arg.json'

            args_, kwargs = load_model_arguments_from_json()
            network = DeepBiAffineTransform(*args_, **kwargs, use_elmo=False, use_bert=False,
                                            use_elmo_bert=True)

            network.load_state_dict(torch.load(model_name))

            if use_gpu:
                network.cuda()
            else:
                network.cpu()

            network.eval()

            decode = network.decode_mst

            out_filename = 'tmp/test'
            pred_writer.start(out_filename + '_pred.conll')

            for batch in conllx_data.iterate_batch_tensor(data_test, 1, use_elmo=False, use_bert=False, use_elmo_bert=True):
                sys.stdout.flush()

                word, char, pos, heads, types, masks, lengths, elmos, berts = batch

                heads_pred, types_pred = decode(word, char, pos, elmos, berts, mask=masks, length=lengths,
                                                leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                word = word.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()

                pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)

            pred_writer.close()

            sents_gold = result_segment.split('\n')
            result = ''
            test_path = 'tmp/test_pred.conll'
            lines = open(test_path, 'r').readlines()
            for i, line in enumerate(lines):
                if line.strip() != '':
                    sent = sents_gold[i]
                    words_gold = sent.split('\t')
                    word = words_gold[1]

                    line = line.strip()
                    words = line.split('\t')
                    line = words[0] + '\t' + word + '\t' + word.lower() + '\t' + words[4] + '\t' + words[4] + '\t_\t' + words[6] + '\t' + words[7] + '\t_\t_' + '\n'
                    if line != '':
                        result += line + '\n'
            result = result.strip()

            logging.info("Result: " + str(result))

            return http_sender.send_http_result(result)
        else:
            return http_sender.send_error_response(http_response_message.ResponseCode.INPUT_FORMAT_ERROR)
    except BadRequest:
        return http_sender.send_error_response(http_response_message.ResponseCode.JSON_SYNTAX_ERROR)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port_dp)







