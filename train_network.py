'''
Training interface for Neural network model to detect and classify TLINKS between entities.
'''

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

from code.learning.network import Network

from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

timenote_imported = False
N_CLASSES = 13

def main():
    '''
    Process command line arguments and then generate trained models (One for detection of links, one for classification)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        type=str,
                        nargs=1,
                        help="Directory containing training input and gold annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    # parser.add_argument("--no_detector",
    #                     action='store_true',
    #                     default=False,
    #                     help="Do not train a detection model")

    # parser.add_argument("--no_classifier",
    #                     action='store_true',
    #                     default=False
    #                     help="Do not train a classification model")

    parser.add_argument("--single_pass",
                        action='store_true',
                        default=False,
                        help="Train a single pass model that performs both detection and classification")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--no_val",
                        action='store_true',
                        default=False,
                        help="No validation. Use all training data to train.")

    args = parser.parse_args()

    # validate file paths
    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.train_dir[0]) is False:
        sys.exit("invalid path to directory containing training data")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    newsreader_dir = args.newsreader_annotations

    train_dir = None

    if '/*' != args.train_dir[0][-2:]:
        train_dir = args.train_dir[0] + '/*'
    else:
        train_dir = args.train_dir[0]

    # get files in directory
    files = glob.glob(train_dir)

    gold_files = []
    tml_files  = []

    for f in files:
        if f.endswith('.pkl'):
            continue
        if "E3input" in f:
            tml_files.append(f)
        else:
            gold_files.append(f)

    gold_files.sort()
    tml_files.sort()

    # one-to-one pairing of annotated file and un-annotated
    assert len(gold_files) == len(tml_files)

    checkpoint = ModelCheckpoint(args.model_destination+'model.h5', monitor='val_acc', save_best_only=True)
    if args.no_val:
        earlystopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')
    else:
        earlystopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    # create a sinlge model, then save architecture and weights
    if args.single_pass:
        if args.load_model:
            try:
                NNet = load_model(args.model_destination + 'model.h5')
            except:
                NNet = model_from_json(open(args.model_destination + '.arch.json').read())
                #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                #           decay=0.0)  # learning rate 0.001 is the default value
                opt = SGD(lr=0.003, momentum=0.9, decay=0.0, nesterov=False)
                NNet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                NNet.load_weights(args.model_destination + '.weights.h5')
        else:
            NNet = None

        NN, history = trainNetwork(tml_files, gold_files, newsreader_dir, no_val=args.no_val, two_pass=False, callbacks=[checkpoint, earlystopping], train_dir=args.train_dir[0].strip('*'))
        architecture = NN.to_json()
        open(args.model_destination + '.arch.json', "wb").write(architecture)
        NN.save_weights(args.model_destination + '.weights.h5')
        NN.save(args.model_destination + 'final_model.h5')
        json.dump(history, open(args.model_destination + 'training_history.json', 'w'))

    # create a pair of models, one for detection, one for classification. Then save architecture and weights
    else:
        # train models
        detector, classifier = trainNetwork(tml_files, gold_files, newsreader_dir)

        # save models
        detect_arch = detector.to_json()
        class_arch = classifier.to_json()
        open(args.model_destination + '.detect.arch.json', "wb").write(detect_arch)
        open(args.model_destination + '.class.arch.json', "wb").write(class_arch)
        detector.save_weights(args.model_destination + '.detect.weights.h5')
        classifier.save_weights(args.model_destination + '.class.weights.h5')


def trainNetwork(tml_files, gold_files, newsreader_dir, no_val=False, two_pass=True, callbacks=[], train_dir='./'):
    '''
    train::trainNetwork()

    Purpose: Train a neural network for classification of temporal realtions. Assumes events and timexes
        will be provided at prediction time

    @param tml_files: List of unlabled (no timex, etc) timeML documents
    @param gold_files: Fully labeled gold standard timeML documents
    '''

    print "Called trainNetwork"

    global timenote_imported
    global N_CLASSES

    if not os.path.isfile(train_dir+'training_data.pkl'):
        # Read in notes
        notes = []

        basename = lambda x: os.path.basename(x[0:x.index(".tml")])

        pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

        tmp_note = None

        for i, example in enumerate(zip(tml_files, gold_files)):
            tml, gold = example

            assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)


            print '\n\nprocessing file {}/{} {}'.format(i + 1,
                                                        len(zip(tml_files, gold_files)),
                                                        tml)
            if basename(tml) + ".parsed.pickle" in pickled_timeml_notes:
                tmp_note = cPickle.load(open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "rb"))
            else:
                if timenote_imported is False:
                    from code.notes.TimeNote import TimeNote
                    timenote_imported = True
                tmp_note = TimeNote(tml, gold)
                cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))

            notes.append(tmp_note)

    network = Network()
    if two_pass:
        return

        detect_data = network._get_training_input(notes, presence=True, no_none=False)
        classify_data = network._get_training_input(notes, presence=False, no_none=True)

        detector = network.train_model(None, epochs=150, training_input=detect_data, weight_classes=False, batch_size=256,
        encoder_dropout=0, decoder_dropout=0, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=64, dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=2)

        # use max input length from detector
        max_len = detector.input_shape[2]

        classifier = network.train_model(None, epochs=500, training_input=classify_data, weight_classes=False, batch_size=256,
        encoder_dropout=0., decoder_dropout=0., input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=64, dense_size=100, maxpooling=True, data_dim=300, max_len=max_len, nb_classes=6)

        return detector, classifier

    else:

        if os.path.isfile(train_dir+'training_data.pkl'):
            print "loading pkl file... this may take over 10 minutes"
            data = cPickle.load(open(train_dir+'training_data.pkl'))
        else:
            data = network._get_training_input(notes, nolink_ratio=0, shuffle=True)
            #cPickle.dump(data, open(train_dir+'training_data.pkl', 'w'))

        NNet, history = network.train_model(None, epochs=150, training_input=data, no_val=no_val, weight_classes=False, batch_size=100,
        encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.6, reg_W=0, reg_B=0, reg_act=0, LSTM_size=256,
        dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=N_CLASSES, callbacks=callbacks)

        return NNet, history

if __name__ == "__main__":
  main()
