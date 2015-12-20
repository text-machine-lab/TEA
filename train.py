import os
import cPickle
import argparse
import re
import glob

from code.notes.TimeNote import TimeNote
from code import model

if "TEA_PATH" not in os.environ:
    exit("TEA_PATH environment variable not specified, it is the directory containg train.py")

if "PY4J_DIR_PATH" not in os.environ:
    exit("PY4J_DIR_PATH environment variable not specified")

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

def main():


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

    assert len(gold_files) == len(tml_files)

    model = trainModel(tml_files, gold_files, False)

    with open(args.model_destination, "wb") as modFile:
        cPickle.dump(model, modFile)


def trainModel( tml_files, gold_files, grid ):
    '''
    train::trainModel()

    Purpose: Train a model for classification of events, timexes, and temporal relations based
       on given training data

    @param training_list: List of strings containing file paths for .tml training documents
    '''

    print "Called train"

    # Read in notes
    notes = []

    basename = lambda x: os.path.basename(x[0:x.index(".tml")])

    i = 1

    for tml, gold in zip(tml_files, gold_files):

        assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)

        print '\n\nprocessing file {}/{} {}'.format(i, len(zip(tml_files, gold_files)), tml)

        tmp_note = TimeNote(tml, gold)
        notes.append(tmp_note)

        i += 1

    mod = model.Model(grid=grid)
    mod.train(notes)

    return mod

if __name__ == "__main__":
  main()
