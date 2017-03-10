'''
Training interface for Neural network model for SemEval 2010 task 8, only to classify "other" and "non-other"
'''

import sys
import os
from code.config import env_paths
import time
import json
import numpy
numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

from code.learning import network_binary
from code.notes.EntNote import EntNote

ignore_order = False

def main():
    '''
    Process command line arguments and then generate trained models (One for detection of links, one for classification)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("training_dir",
                        type=str,
                        help="Directory of training file")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    # not really used, just for consistency
    parser.add_argument("--single_pass",
                        action='store_true',
                        default=True,
                        help="Train a single pass model that performs both detection and classification")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--test_dir",
                        type=str,
                        default='',
                        help="Use test data for validation. All training data will be used for training.")

    args = parser.parse_args()

    newsreader_dir = args.newsreader_annotations

    print "training dir:", args.training_dir
    # validate file paths
    if os.path.isfile(args.training_dir) is False:
        gold_files = glob.glob(args.training_dir.rstrip('/')+'/*')
        gold_files.sort()
        if not gold_files:
            sys.exit("training file for semeval 10 task 8 not found")
    else:
        gold_files = [args.training_dir]

    if args.test_dir and os.path.isfile(args.test_dir) is False:
        test_files = glob.glob(args.test_dir.rstrip('/')+'/*')
        test_files.sort()
    elif args.test_dir:
        test_files = [args.test_dir]
    else:
        print "No test data provided. Will use 1/5 training data for validation."
        test_files = None

    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    start = time.time()

    checkpoint = ModelCheckpoint(args.model_destination+'binary_model.h5', monitor='val_acc', save_best_only=True)
    earlystopping = EarlyStopping(monitor='loss', patience=50, verbose=0, mode='auto')

    if args.load_model:
        try:
            NNet = load_model(args.model_destination + 'binary_model.h5')
        except:
            NNet = model_from_json(open(args.model_destination + '.arch.json').read())
            #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            #           decay=0.0)  # learning rate 0.001 is the default value
            opt = SGD(lr=0.1, momentum=0.99, decay=0.0, nesterov=False)
            NNet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            NNet.load_weights(args.model_destination + '.binary_weights.h5')
    else:
        NNet = None

    NN, history = trainNetwork(gold_files, newsreader_dir, test_files=test_files, model=NNet, callbacks=[checkpoint, earlystopping])
    architecture = NN.to_json()
    open(args.model_destination + '.binary_arch.json', "wb").write(architecture)
    NN.save_weights(args.model_destination + '.binary_weights.h5')
    NN.save(args.model_destination + 'final_binary_model.h5')
    json.dump(history, open(args.model_destination + 'binary_training_history.json', 'w'))

    print "training finished. used %.2f sec" %(time.time()-start)


def get_notes(files, newsreader_dir):

    # filenames without directory and extension
    basenames = [os.path.splitext(file)[0].split('/')[-1] for file in files]
    note_files = sorted([os.path.join(newsreader_dir, basename + ".parsed.pickle") for basename in basenames])

    # Read in notes
    notes = []
    for i, note_file in enumerate(note_files):
        if os.path.isfile(note_file):
            ent_note = cPickle.load(open(note_file, "rb"))
        else:
            ent_note = EntNote(files[i], overwrite=False)
            cPickle.dump(ent_note, open(note_file, "wb"))

        notes.append(ent_note)
    return notes


def trainNetwork(gold_files, newsreader_dir, test_files=None, model=None, callbacks=[]):
    '''
    train::trainNetwork()

    Purpose: Train a neural network for classification of realtions. Assumes events and timexes
        will be provided at prediction time

    @param gold_file: training file containing sentences and relations
    '''

    print "Called trainNetwork"

    notes = get_notes(gold_files, newsreader_dir)
    if test_files:
        test_notes = get_notes(test_files, newsreader_dir)
    else:
        test_notes = None


    nb_class = 2

    # if set shuffle to false, we can have the same dev set each time
    data = network_binary._get_training_input(notes, shuffle=True)
    if test_notes:
        test_data = network_binary._get_training_input(test_notes, shuffle=False)
    else:
        test_data = None

    NNet, history = network_binary.train_model(None, model=model, epochs=200, training_input=data, test_input=test_data, weight_classes=False, batch_size=100,
    encoder_dropout=0, decoder_dropout=0.3, input_dropout=0.3, reg_W=0.00001, reg_B=0, reg_act=0, LSTM_size=64, dense_size=10, maxpooling=True,
    data_dim=300, max_len='auto', nb_classes=nb_class, callbacks=callbacks)

    return NNet, history



if __name__ == "__main__":
  main()
