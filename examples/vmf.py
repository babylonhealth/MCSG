# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import argparse
import simplejson as json
import io

from senteval.utils import dotdict
from similarity import get_similarity_by_name

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'


def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='FUzzy Set Similarity')
parser.add_argument('--output_path', type=str, default='bayes.json',
                    help='Output path for results')

args = parser.parse_args()


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id, norm=False):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                np_vector = np.fromstring(vec, sep=' ')
                if norm:
                    np_vector = np_vector / np.linalg.norm(np_vector)
                word_vec[word] = np_vector

    # logging.info('Found {0} words with word vectors, out of \
    #     {1} words'.format(len(word_vec), len(word2id)))
    # import ipdb; ipdb.set_trace()
    return word_vec


WORD_VEC_MAP = {
    'glove': 'glove.840B.300d.txt',
    'word2vec_GN': 'GoogleNews-vectors-negative300.txt',
    'fasttext': 'fasttext-crawl-300d-2M.txt'
}


def get_word_vec_path_by_name(word_vec_name):
    base_path = '../data/word_vectors/'
    return base_path + WORD_VEC_MAP[word_vec_name]


def prepare(params, samples):
    sim_params = params.sim_params
    params.similarity = get_similarity_by_name(sim_params.similarity)
    word_vec_path = get_word_vec_path_by_name(sim_params.word_vec)
    norm = sim_params.norm
    params.wvec_dim = 300

    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(word_vec_path,
                                       params.word2id,
                                       norm=norm)


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        sentfreq = []
        for word in sent:
            if word in params.word_vec:
                wv = params.word_vec[word]
                if isinstance(wv, tuple):
                    iv, wv = wv
                    sentfreq.append(iv)
                sentvec.append(wv)
        sentvec.append(params.word_vec[params.padding])
        if not sentvec:
            sentvec = np.zeros((1, params.wvec_dim))

        sentvec = np.array(sentvec)
        if not sentfreq:
            embeddings.append(sentvec)
        else:
            embeddings.append((sentvec, np.array(sentfreq).reshape(-1, 1) ))
    return embeddings


if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = []
    for word_vec in ['fasttext', 'glove', 'word2vec_GN']:
        print( 'Word vectors: {0}'.format(word_vec))
        for word_count_path in [None]:
            for norm in [True]:
                for sim_name in ['von_mises_correction_tic', 'von_mises_correction_aic']:
                    print('Similarity: {0}'.format(sim_name))
                    params_senteval = {'task_path': PATH_TO_DATA,
                                       'usepytorch': True,
                                       'kfold': 10,
                                       'padding': '.'}

                    # Word2Vec Google News does not have a . embedding
                    if word_vec == 'word2vec_GN':
                        params_senteval['padding'] = 'the'

                    sim_params = dotdict({'similarity': sim_name,
                                          'word_vec': word_vec,
                                          'word_count_path': word_count_path,
                                          'norm': norm})
                    params_senteval['sim_params'] = sim_params

                    se = senteval.engine.SE(params_senteval, batcher, prepare)
                    result = se.eval(transfer_tasks)
                    result_dict = {
                        'param': dict(sim_params),
                        'eval': result
                    }
                    results.append(result_dict)
                    with open(args.output_path, 'w') as f:
                        json.dump(results, f)
