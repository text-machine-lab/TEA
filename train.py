import sys
import os
import cPickle
import argparse
import re
import glob

from code.config import env_paths

from code.notes.TimeNote import TimeNote
from code import model
from code import network

if "TEA_PATH" not in os.environ:
    sys.exit("TEA_PATH environment variable not specified, it is the directory containg train.py")

if "PY4J_DIR_PATH" is os.environ:
    sys.exit("PY4J_DIR_PATH environment variable not specified")


def main():

    """ Processes command line arguments and then generates a trained model on files provided.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        type=str,
                        nargs=1,
                        help="Directory containing training input and gold annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")
    parser.add_argument("--neural_network", '-n',
                        action='store_true',
                        help="set flag to use a neural network model rather than SVM for tlink identification")

    args = parser.parse_args()

    if os.path.isdir(args.train_dir[0]) is False:
        exit("invalid path to directory containing training data")

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
        if "E3input" in f:
            tml_files.append(f)
        else:
            gold_files.append(f)

    gold_files.sort()
    tml_files.sort()

    # one-to-one pairing of annotated file and un-annotated
    assert len(gold_files) == len(tml_files)

    # create the model
    if args.neural_network == True:
        model = trainNetwork(tml_files, gold_files)
    else:
        model = trainModel(tml_files, gold_files, False)

    # store model as pickle object.
    with open(args.model_destination, "wb") as modFile:
        cPickle.dump(model, modFile)


def trainModel( tml_files, gold_files, grid ):
    """
    train::trainModel()

    Purpose: Train a model for classification of events, timexes, and temporal relations based
       on given training data

    @param training_list: List of strings containing file paths for .tml training documents
    """

    print "Called train"

    # Read in notes
    notes = []

    basename = lambda x: os.path.basename(x[0:x.index(".tml")])

    for i, example in enumerate(zip(tml_files, gold_files)):

        tml, gold = example

        assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)

        print '\n\nprocessing file {}/{} {}'.format(i + 1,
                                                    len(zip(tml_files, gold_files)),
                                                    tml)

        tmp_note = TimeNote(tml, gold)
        notes.append(tmp_note)

    mod = model.Model(grid=grid)
    mod.train(notes)

    return mod

def trainNetwork(tml_files, gold_files):
    '''
    train::trainNetwork()

    Purpose: Train a neural network for classification of temporal realtions. Assumes events and timexes
        will be provided at prediction time

    @param tml_files: List of unlabled (no timex, etc) timeML documents
    @param gold_files: Fully labeled gold standard timeML documents
    '''
    print "Called trainNetwork"

    # Read in notes
    notes = []

    basename = lambda x: os.path.basename(x[0:x.index(".tml")])

    for i, example in enumerate(zip(tml_files, gold_files)):

        tml, gold = example

        assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)

        print '\n\nprocessing file {}/{} {}'.format(i + 1,
                                                    len(zip(tml_files, gold_files)),
                                                    tml)

        tmp_note = TimeNote(tml, gold)
        notes.append(tmp_note)

    mod = network.NNModel()
    mod.train(notes)

    return mod


if __name__ == "__main__":
  main()
