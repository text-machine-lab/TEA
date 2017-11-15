import sys
import os
import numpy as np
import cPickle as pickle
import nltk


def load_word2vec_binary(fname='/data1/nlp-data/GoogleNews-vectors-negative300.bin', verbose=1, dev=False):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    if verbose:
        print 'loading word2vec'
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            # short circuit (when we just care about pipeline, not actually using this for tests)
            if dev:
                if line >= 500:
                    break
            # display how long it takes?
            if verbose:
                if line % (vocab_size/40) == 0:
                    print '%6.2f %%' % (100*float(line)/vocab_size)
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    return word_vecs


def load_word2vec_dep(fname):
    word_vecs = pickle.load(open(fname))
    return word_vecs


def load_glove(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

# The folliwing functions are adapted from https://github.com/facebookresearch/InferSent/blob/master/data.py

def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent):  # original version uses split()
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec