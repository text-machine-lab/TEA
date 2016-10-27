"""
Interface to perform predicting of TIMEX, EVENT and TLINK annotations,
using a neural network model for TLINK annotations.
"""

import sys
from code.config import env_paths
import numpy
numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import cPickle
import glob
import os

from keras.models import model_from_json
from keras.models import load_model

from code.notes.EntNote import EntNote
from code.learning import network_sem10

ignore_order = False


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("predict_dir",
                        help="Directory containing test input")

    parser.add_argument("model_destination",
                        help="Where trained model is located")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--single_pass",
                        action='store_true',
                        default=False,
                        help="Predict using a single pass model")

    parser.add_argument("--evaluate",
                        action='store_true',
                        default=False,
                        help="Use gold data from the given files to produce evaluation metrics")

    parser.add_argument("--ignore_order",
                        action='store_true',
                        default=False,
                        help="Ignore the order of two entities when assigning labels")

    args = parser.parse_args()
    global ignore_order
    ignore_order = args.ignore_order
    network_sem10.set_ignore_order(ignore_order)

    newsreader_dir = args.newsreader_annotations

    if os.path.isdir(newsreader_dir) is False:
        sys.exit("invalid path for time note dir")

    if os.path.isfile(args.predict_dir) is False:
        test_files = glob.glob(args.predict_dir.rstrip('/')+'/*')
        test_files.sort()
        if not test_files:
            sys.exit("training file for semeval 10 task 8 not found")
    else:
        test_files = [args.predict_dir]

    model_path = args.model_destination


    basenames = [os.path.splitext(test_file)[0].split('/')[-1] for test_file in test_files]
    note_files = sorted([os.path.join(newsreader_dir, basename + ".parsed.pickle") for basename in basenames])


    # Read in notes
    notes = []
    for i, note_file in enumerate(note_files):
        if os.path.isfile(note_file):
            ent_note = cPickle.load(open(note_file, "rb"))
        else:
            ent_note = EntNote(test_files[i], overwrite=False)
            cPickle.dump(ent_note, open(note_file, "wb"))

        notes.append(ent_note)

    if ignore_order:
        nb_class = 10
    else:
        nb_class = 20 # in fact only 19 are used

    if args.single_pass:
        # load model
        #NNet = model_from_json(open(model_path + '.arch.json').read())
        #NNet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #NNet.load_weights(model_path + '.weights.h5')
        NNet = load_model(model_path + '.model.h5')      

        # run prediction cycle
        network_sem10.single_predict(notes, NNet, nb_class, evalu=args.evaluate)

    else:
        # load both passes
        classifier = model_from_json(open(model_path + '.class.arch.json').read())
        classifier.load_weights(model_path + '.class.weights.h5')

        detector = model_from_json(open(model_path + '.detect.arch.json').read())
        detector.load_weights(model_path + '.detect.weights.h5')

        # run prediction cycle
        labels, filter_lists = network_sem10.predict(notes, detector, classifier, evalu=args.evaluate)

if __name__ == '__main__':
    main()
