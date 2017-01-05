'''
Training interface for Neural network model for SemEval 2010 task 8
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

from code.learning import network_sem10
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

    parser.add_argument("--single_pass",
                        action='store_true',
                        default=False,
                        help="Train a single pass model that performs both detection and classification")

    parser.add_argument("--ignore_order",
                        action='store_true',
                        default=False,
                        help="Ignore the order of two entities when assigning labels")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--test_dir",
                        type=str,
                        default='',
                        help="Use test data for validation. All training data will be used for training.")

    args = parser.parse_args()

    global ignore_order
    ignore_order = args.ignore_order
    network_sem10.set_ignore_order(ignore_order)

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

    checkpoint = ModelCheckpoint(args.model_destination+'model.h5', monitor='val_acc', save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')

    # create a sinlge model, then save architecture and weights
    if args.single_pass:
        if args.load_model:
            try:
                NNet = load_model(args.model_destination + 'model.h5')
            except:
                NNet = model_from_json(open(args.model_destination + '.arch.json').read())
                #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                #           decay=0.0)  # learning rate 0.001 is the default value
                opt = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False) # default lr=0.01
                NNet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                NNet.load_weights(args.model_destination + '.weights.h5')
        else:
            NNet = None

        NN, history = trainNetwork(gold_files, newsreader_dir, test_files=test_files, model=NNet, two_pass=False, callbacks=[checkpoint, earlystopping])
        architecture = NN.to_json()
        open(args.model_destination + '.arch.json', "wb").write(architecture)
        NN.save_weights(args.model_destination + '.weights.h5')
        NN.save(args.model_destination + 'final_model.h5')
        json.dump(history, open(args.model_destination + 'training_history.json', 'w'))

    # create a pair of models, one for detection, one for classification. Then save architecture and weights
    else:
        if args.load_model:
            # load both passes
            classifier = model_from_json(open(args.model_destination + '.class.arch.json').read())
            classifier.load_weights(args.model_destination + '.class.weights.h5')

            detector = model_from_json(open(args.model_destination + '.detect.arch.json').read())
            detector.load_weights(args.model_destination + '.detect.weights.h5')
            NNet = (detector, classifier)
        else:
            NNet = (None, None)

        # train models
        NN, history = trainNetwork(gold_files, newsreader_dir, model=NNet)
        detector, classifier = NN
        # save models
        detect_arch = detector.to_json()
        class_arch = classifier.to_json()
        open(args.model_destination + '.detect.arch.json', "wb").write(detect_arch)
        open(args.model_destination + '.class.arch.json', "wb").write(class_arch)
        detector.save_weights(args.model_destination + '.detect.weights.h5')
        classifier.save_weights(args.model_destination + '.class.weights.h5')

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

def trainNetwork(gold_files, newsreader_dir, test_files=None, model=None, two_pass=True, callbacks=[]):
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

    global ignore_order
    if ignore_order:
        nb_class = 10
    else:
        nb_class = 20 # in fact only 19 are used

    if two_pass:

        detect_data = network_sem10._get_training_input(notes, presence=True, no_none=False)
        classify_data = network_sem10._get_training_input(notes, presence=False, no_none=True)

        detector = network_sem10.train_model(None, model=model[0], epochs=150, training_input=detect_data, weight_classes=False, batch_size=256,
        encoder_dropout=0, decoder_dropout=0, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=300, dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=2)

        # use max input length from detector
        max_len = detector.input_shape[0][2]

        classifier = network_sem10.train_model(None, model=model[1], epochs=500, training_input=classify_data, weight_classes=False, batch_size=256,
        encoder_dropout=0., decoder_dropout=0., input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=300, dense_size=100, maxpooling=True, data_dim=300, max_len=max_len, nb_classes=nb_class)

        return detector, classifier

    else:

        # if set shuffle to false, we can have the same dev set each time
        data = network_sem10._get_training_input(notes, shuffle=False)
        if test_notes:
            test_data = network_sem10._get_training_input(test_notes, shuffle=False)
        else:
            test_data = None

        NNet, history = network_sem10.train_model(None, model=model, epochs=500, training_input=data, test_input=test_data, weight_classes=False, batch_size=100,
        encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.4, reg_W=0.00001, reg_B=0, reg_act=0, LSTM_size=256, dense_size=100, maxpooling=True,
        data_dim=300, max_len='auto', nb_classes=nb_class, callbacks=callbacks)

        return NNet, history



if __name__ == "__main__":
  main()
