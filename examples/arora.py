# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import itertools

import collections

import senteval

CountsFile = collections.namedtuple('CountsFile', 'name path')
SIF_CNT = CountsFile(name='sif_counts', path='../data/misc/enwiki_vocab_min200.txt')


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'

sys.path.insert(0, PATH_TO_SENTEVAL)

import io
import numpy as np
import logging
from sklearn.decomposition import TruncatedSVD


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


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id,
                norm=False,
                path_to_counts=None,
                remove_stopwords=False):
    word_vec = {}
    word_freq_map = None
    if path_to_counts:
        word_freq_map = _get_word_freq_map(path_to_counts)

    with io.open(path_to_vec, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if remove_stopwords and word in stopwords:
                continue
            if word in word2id:
                np_vector = np.fromstring(vec, sep=' ')
                if norm:
                    np_vector = np_vector / np.linalg.norm(np_vector)
                if word_freq_map:
                    np_vector = _get_word_weight(word, word_freq_map) * np_vector
                word_vec[word] = np_vector

    # logging.info('Found {0} words with word vectors, out of \
    #     {1} words'.format(len(word_vec), len(word2id)))
    # import ipdb; ipdb.set_trace()
    return word_vec


def _get_word_freq_map(path_to_counts):
    word_count_list = []

    total_count = 0.0
    with io.open(path_to_counts, 'r') as f:
        for line in f:
            word_count = line.split(' ')
            word = word_count[0]
            count = float(word_count[1])
            total_count += count
            word_count_list.append((word, count))

    word_freq_map = {}
    for word_count in word_count_list:
        word_freq_map[word_count[0]] = word_count[1] / total_count

    return word_freq_map


def _get_word_weight(word, word_freq_map, a=1e-3):
    word_freq = word_freq_map.get(word, 0.0)
    return a / (a + word_freq)


WORD_VEC_MAP = {
    'glove': 'glove.840B.300d.txt',
    'word2vec_GN': 'GoogleNews-vectors-negative300.txt',
    'fasttext': 'fasttext-crawl-300d-2M.txt',
}


def get_word_vec_path_by_name(word_vec_name):
    base_path = '../data/word_vectors/'
    return base_path + WORD_VEC_MAP[word_vec_name]


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    word_vec_path = get_word_vec_path_by_name(params.word_vec_name)
    word_count_path = params.word_count_path
    norm = params.norm
    params.wvec_dim = 300

    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(word_vec_path,
                                        params.word2id,
                                        norm=norm,
                                        path_to_counts=word_count_path,
                                        remove_stopwords=False)
    # Comment the 3 lines below to get SIF results
    params.pc = None
    X = batcher(params, samples)
    params.pc = compute_pc(X, npc=1)
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, axis=0)
        if params.pc is not None:
            pc = params.pc
            sentvec = sentvec - sentvec.dot(pc.T) * pc

        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings



if __name__ == "__main__":


    word_vectors = ['glove', 'word2vec_GN', 'fasttext']

    word_counts = [SIF_CNT]

    similarities = ['sif']

    experiments = list(itertools.product(word_vectors,
                                         similarities,
                                         word_counts))

    logging.info('Running {0} experiments. Good luck! :)\n\n\n'.format(len(experiments)))

    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = []

    for idx, experiment in enumerate(experiments):
        word_vec_name = experiment[0]
        sim_name = experiment[1]
        word_counts = experiment[2]

        logging.info('Word vectors: {0}'.format(word_vec_name))
        logging.info('Word Counts : {0}'.format(word_counts.name))
        logging.info('Similarity: {0}'.format(sim_name))
        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'word_vec_name': word_vec_name,
            'word_count_name': word_counts.name,
            'word_count_path': word_counts.path,
            'similarity_name': sim_name
        }
        params_senteval.update(params_experiment)

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        se.eval(transfer_tasks)
