import os
import cPickle
import argparse
import re
import glob

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

from code.notes.TimeNote import TimeNote

from code import model

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        nargs=1,
                        help="Directory containing training input and gold annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("--grid",
                        dest="grid",
                        action="store_true",
                        help="Enable or disable grid search")

    args = parser.parse_args()

    train_dir = None

    if '/*' != args.train_dir[0][-2:]:
        train_dir = args.train_dir[0] + '/*'

    else:
        train_dir = args.train_dir[0]


    print "\ntraining dir: {}\n".format(train_dir)

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

    model = trainModel(tml_files, gold_files, args.grid)

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

    for tml, gold in zip(tml_files, gold_files):

        assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)

        tmp_note = TimeNote(tml, gold)
        notes.append(tmp_note)

    mod = model.Model(grid=grid)
    mod.train(notes)

    return mod

if __name__ == "__main__":
  main()
