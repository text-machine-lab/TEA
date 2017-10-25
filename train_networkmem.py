'''
Training interface for Neural network model to detect and classify TLINKS between entities.
'''
from __future__ import print_function
import sys
import os
from code.config import env_paths
import numpy
numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle
import json
import threading
import Queue
import time

from code.learning.network_mem import NetworkMem
from code.notes.TimeNote import TimeNote
from code.learning.word2vec import load_word2vec_binary, load_glove

from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

from code.learning.ntm_models import LABELS, DENSE_LABELS, EMBEDDING_DIM, MAX_LEN


def main():
    '''
    Process command line arguments and then generate trained models (One for detection of links, one for classification)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        help="Directory containing training annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--val_dir",
                        default=None,
                        help="Directory containing validation annotations")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--no_val",
                        action='store_true',
                        default=False,
                        help="No validation. Use all training data to train.")

    parser.add_argument("--pair_type",
                        default='both',
                        help="specify the entity type to train: intra, cross or both")

    parser.add_argument("--nolink",
                        default=None,
                        type=float,
                        help="no link downsampling ratio. e.g. 0.5 means # of nolinks are 50% of # positive tlinks")

    parser.add_argument("--no_ntm",
                        action='store_true',
                        default=False,
                        help="specify whether to use neural turing machine. default is to use ntm (no_ntm=false).")

    args = parser.parse_args()

    assert args.pair_type in ('intra', 'cross', 'dct', 'all')

    # validate file paths
    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.train_dir) is False:
        sys.exit("invalid path to directory containing training data")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    print("arguments:\n", args)

    # get files in directory
    files = glob.glob(os.path.join(args.train_dir, '*'))
    gold_files = []
    tml_files = []

    for f in files:
        if "E3input" in f:
            tml_files.append(f)
        elif f.endswith('.tml'):
            gold_files.append(f)

    gold_files.sort()
    tml_files.sort()

    if args.val_dir is None:
        val_files = None
    else:
        val_files = glob.glob(os.path.join(args.val_dir, '*'))
        val_files.sort()

    # one-to-one pairing of annotated file and un-annotated
    # assert len(gold_files) == len(tml_files)

    model_destination = os.path.join(args.model_destination, args.pair_type) + '/'
    if not os.path.exists(model_destination):
        os.makedirs(model_destination)

    if args.no_val:
        earlystopping = EarlyStopping(monitor='loss', patience=15, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='loss', save_best_only=True)
    else:
        earlystopping = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='val_loss', save_best_only=True)

    # create a sinlge model, then save architecture and weights
    if args.load_model:
        try:
            model = load_model(model_destination + 'model.h5')
        except:
            model = model_from_json(open(model_destination + '.arch.json').read())
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            model.load_weights(model_destination + '.weights.h5')
    else:
        model = None

    N_CLASSES = len(LABELS)
    notes = get_notes(gold_files, args.newsreader_annotations)
    n = len(notes)
    splits = 20  # the number of chunks we divide a batch/document into
    rounds = 10  # number of epochs to use all training data, good for fast check

    # the steps_per_epoch is useful if a single document is divided into chunks
    # if we use a whole document as a patch, it will be just the number of documents
    if args.pair_type == 'cross':
        batch_size = 300 / splits
        steps_per_epoch = 43000/batch_size + n/2  # 42758 entries, 169 notes
    elif args.pair_type == 'intra':
        batch_size = 200 / splits
        steps_per_epoch = 24000/batch_size + n/2  # 24312 entries, 169 notes
    elif args.pair_type == 'all':
        batch_size = 500 / splits
        steps_per_epoch = 68000/batch_size + n/2
    else:
        batch_size = 50
        steps_per_epoch = n
    steps_per_epoch /= rounds

    if not args.no_val:
        val_notes = get_notes(val_files, args.newsreader_annotations)
        m = len(val_notes)
        if args.pair_type == 'cross':
            validation_steps = 2400/batch_size + m/2  # 2368 entries, 9 notes, 2000/batch_size is reasonable
        elif args.pair_type == 'intra':
            validation_steps = 1400/batch_size + m/2  # 1352 entries, 9 notes,
        elif args.pair_type == 'all':
            validation_steps = 4000/batch_size + m/2
        else:
            validation_steps = m
    else:
        validation_steps = None

    print("use batch size", batch_size)
    print("steps_per_epoch", steps_per_epoch)
    print("validation_steps", validation_steps)
    print("loading word vectors...")
    word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)
    # network.word_vectors = load_glove(os.environ["TEA_PATH"] + '/glove.6B.200d.txt')
    # network.word_vectors = load_glove(os.environ["TEA_PATH"] + '/glove.6B.300d.txt')
    network = NetworkMem()
    network.word_vectors = word_vectors


    training_data = network.generate_training_input(notes, args.pair_type, max_len=MAX_LEN, nolink_ratio=args.nolink, no_ntm=args.no_ntm)
    if args.no_ntm:
        training_data_gen = training_data  # no sclicing
    else:
        training_data_gen = network.slice_data(training_data, batch_size)

    if not args.no_val and val_notes is not None:
        val_data = network.generate_training_input(val_notes, args.pair_type, max_len=MAX_LEN, no_ntm=args.no_ntm)
        if args.no_ntm:
            val_data_gen = val_data
        else:
            val_data_gen = network.slice_data(val_data, batch_size)

    else:
        val_data_gen = None

    print("model to load", model)
    model, history = network.train_model(model=model, no_ntm=args.no_ntm, epochs=100, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                         input_generator=training_data_gen, val_generator=val_data_gen,
                                         weight_classes=True, encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.6,
                                         LSTM_size=128, dense_size=128, max_len=MAX_LEN, nb_classes=N_CLASSES, callbacks=[checkpoint, earlystopping],
                                         batch_size=batch_size)

    # evaluation
    test_data = network.generate_test_input(val_notes, args.pair_type, max_len=MAX_LEN, no_ntm=args.no_ntm)
    if args.no_ntm:
        test_data_gen = test_data
    else:
        test_data_gen = network.slice_data(test_data, batch_size)

    network.predict(model, test_data_gen, batch_size=batch_size, evaluation=True, smart=True, no_ntm=args.no_ntm)

    architecture = model.to_json()
    open(model_destination + '.arch.json', "wb").write(architecture)
    model.save_weights(model_destination + '.weights.h5')
    model.save(model_destination + 'final_model.h5')
    json.dump(history, open(model_destination + 'training_history.json', 'w'))

    return model, history

def basename(name):
    name = os.path.basename(name)
    name = name.replace('.TE3input', '')
    name = name.replace('.tml', '')
    return name


def get_notes(files, newsreader_dir):

    if not files:
        return None

    notes = []
    if DENSE_LABELS:
        # denselabels = cPickle.load(open(newsreader_dir+'dense-labels.pkl'))
        denselabels = cPickle.load(open(newsreader_dir + 'dense-labels-single.pkl'))
    else:
        denselabels = None

    for i, tml in enumerate(files):
        if i % 10 == 0:
            print('processing file {}/{} {}'.format(i + 1, len(files), tml))
        if os.path.isfile(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle")):
            tmp_note = cPickle.load(open(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle"), "rb"))
        else:
            tmp_note = TimeNote(tml, tml, denselabels=denselabels)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))

        if DENSE_LABELS and tmp_note.denselabels is None: # handle old note files without dense labels
            tmp_note.denselabels = denselabels
            tmp_note.get_id_to_denselabels()
        notes.append(tmp_note)
    return notes


if __name__ == "__main__":
    main()
