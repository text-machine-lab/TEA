import sys
import os
import numpy as np


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
