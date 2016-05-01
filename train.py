import sys
from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import os
import cPickle
import argparse
import re
import glob

from code.notes.TimeNote import TimeNote
from code.learning import model

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

if __name__ == "__main__":
  main()
