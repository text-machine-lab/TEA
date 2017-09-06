'''
Training interface for Neural network model to detect and classify TLINKS between entities.
'''

import sys
import os
from code.config import env_paths
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle
import json

from code.learning.network import Network
from code.notes.TimeNote import TimeNote
from code.learning.word2vec import load_word2vec_binary

from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

N_CLASSES = 13


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

    parser.add_argument("--pair_ordered",
                        action='store_true',
                        default=False,
                        help="Only consider pairs in their narrative order (order in text)")

    parser.add_argument("--nolink_ratio",
                        default=1.0,
                        type=float,
                        help="no link downsampling ratio. e.g. 0.5 means # of nolinks are 50% of # positive tlinks")

    args = parser.parse_args()

    assert args.pair_type in ('intra', 'cross', 'both', 'dct')

    # validate file paths
    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.train_dir) is False:
        sys.exit("invalid path to directory containing training data")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    # get files in directory
    files = glob.glob(os.path.join(args.train_dir, '*'))
    gold_files = []
    tml_files  = []

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

    model_destination = os.path.join(args.model_destination, args.pair_type) + '/'
    if not os.path.exists(model_destination):
        os.makedirs(model_destination)

    if args.no_val:
        earlystopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='loss', save_best_only=True)
    else:
        earlystopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='loss', save_best_only=True)

    if args.load_model:
        try:
            NNet = load_model(model_destination + 'model.h5')
        except:
            NNet = model_from_json(open(model_destination + '.arch.json').read())
            opt = SGD(lr=0.003, momentum=0.9, decay=0.0, nesterov=False)
            NNet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            NNet.load_weights(model_destination + '.weights.h5')
    else:
        NNet = None

    NN, history = trainNetwork(gold_files, val_files, args.newsreader_annotations, args.pair_type, ordered=args.pair_ordered,
                               no_val=args.no_val, nolink_ratio=args.nolink_ratio, callbacks=[checkpoint, earlystopping], train_dir=args.train_dir)
    architecture = NN.to_json()
    open(model_destination + '.arch.json', "wb").write(architecture)
    NN.save_weights(model_destination + '.weights.h5')
    json.dump(history, open(model_destination + 'training_history.json', 'w'))


def basename(name):
    name = os.path.basename(name)
    name = name.replace('.TE3input', '')
    name = name.replace('.tml', '')
    return name


def get_notes(files, newsreader_dir):

    if not files:
        return None

    notes = []

    for i, tml in enumerate(files):
        if i % 10 == 0:
            print 'processing file {}/{} {}'.format(i + 1, len(files), tml)
        if os.path.isfile(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle")):
            tmp_note = cPickle.load(open(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle"), "rb"))
        else:
            tmp_note = TimeNote(tml, tml)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))

        notes.append(tmp_note)
    return notes


def trainNetwork(gold_files, val_files, newsreader_dir, pair_type, ordered=False, no_val=False, nolink_ratio=1.0, callbacks=[], train_dir='./'):
    '''
    Train a neural network for classification of temporal realtions.
    '''

    print "Called trainNetwork"

    global N_CLASSES

    if not os.path.isfile(train_dir+'training_data.pkl'):
        notes = get_notes(gold_files, newsreader_dir)
    if not no_val:
        val_notes = get_notes(val_files, newsreader_dir)

    network = Network()
    print "loading word vectors..."
    network.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)

    if os.path.isfile(train_dir+'training_data.pkl'):
        print "loading pkl file... this may take over 10 minutes"
        training_data = cPickle.load(open(train_dir+'training_data.pkl'))
        print "training data size:", training_data[0].shape, training_data[1].shape, len(training_data[2])
    else:
        # nolink_ration = # no tlink cases / # tlink cases
        training_data = network._get_training_input(notes, pair_type=pair_type, nolink_ratio=nolink_ratio, shuffle=True, ordered=ordered)
        print "training data size:", training_data[0].shape, training_data[1].shape, len(training_data[2])

        if not no_val and val_notes is not None:
            val_data = network._get_test_input(val_notes, pair_type=pair_type, ordered=ordered)
            print "validation data size:", val_data[0].shape, val_data[1].shape, len(val_data[2])
        else:
            val_data = None

        del network.word_vectors
        NNet, history = network.train_model(None, epochs=200, training_input=training_data, val_input=val_data, no_val=no_val, weight_classes=False, batch_size=100,
        encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.6, reg_W=0, reg_B=0, reg_act=0, LSTM_size=256,
        dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=N_CLASSES, callbacks=callbacks, ordered=ordered)

        return NNet, history

if __name__ == "__main__":
  main()
