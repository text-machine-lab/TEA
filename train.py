"""Interface to perform training of models for temporal entity and relation extraction.
"""

import sys
import os
from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle

from code.learning import model

timenote_imported = False

def main():
    """ Process command line arguments and then generate trained models (4, one for each pass) on files provided.
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

    parser.add_argument("newsreader_annotations",
                        #type=str,
                        #nargs=1,
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--no_event",
                        action='store_true',
                        default=False)

    parser.add_argument("--no_timex",
                        action='store_true',
                        default=False)

    parser.add_argument("--no_tlink",
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    train_event = not(args.no_event)
    train_timex = not(args.no_timex)
    train_tlink = not(args.no_tlink)

    print "\n\tTRAINING:\n"
    print "\t\tTIMEX {}".format(train_timex)
    print "\t\tEVENT {}".format(train_event)
    print "\t\tTLINK {}".format(train_tlink)
    print "\n"

    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.train_dir[0]) is False:
        sys.exit("invalid path to directory containing training data")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    newsreader_dir = args.newsreader_annotations
    #print "NEWSREADER"
    #print newsreader_dir
    #sys.exit("done")
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

    # create the model, then save architecture and weights
    if args.neural_network == True:
        from code.learning import network
        NN = trainNetwork(tml_files, gold_files, newsreader_dir)
        architecture = NN.classifier.to_json()
        open(args.model_destination + '.arch.json', "w").write(architecture)
        NN.classifier.save_weights(args.model_destination + '.weights.h5')

    else:
        models, vectorizers = trainModel(tml_files, gold_files, False, train_timex, train_event, train_tlink, newsreader_dir)

        # store model as pickle object.
        model.dump_models(models, vectorizers, args.model_destination)


def trainModel( tml_files, gold_files, grid, train_timex, train_event, train_tlink, newsreader_dir):
    """
    train::trainModel()

    Purpose: Train a model for classification of events, timexes, and temporal relations based
       on given training data

    @param training_list: List of strings containing file paths for .tml training documents
    """

    global timenote_imported

    print "Called train"

    # Read in notes
    notes = []

    basename = lambda x: os.path.basename(x[0:x.index(".tml")])

    pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

    print pickled_timeml_notes

    tmp_note = None

    for i, example in enumerate(zip(tml_files, gold_files)):

        tml, gold = example

        assert basename(tml) == basename(gold), "mismatch\n\ttml: {}\n\tgold:{}".format(tml, gold)

        print '\n\nprocessing file {}/{} {}'.format(i + 1,
                                                    len(zip(tml_files, gold_files)),
                                                    tml)

        stashed_name = os.path.basename(tml)
        stashed_name = stashed_name.split('.')
        stashed_name = stashed_name[0:stashed_name.index('tml')]
        stashed_name = '.'.join(stashed_name)

        if stashed_name + ".parsed.pickle" in pickled_timeml_notes:
            tmp_note = cPickle.load(open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "rb"))
        else:
            if timenote_imported is False:
                from code.notes.TimeNote import TimeNote
                timenote_imported = True
            tmp_note = TimeNote(tml, gold)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))
        notes.append(tmp_note)

    print "training..."
    return model.train(notes, train_timex, train_event, train_tlink)


def trainNetwork(tml_files, gold_files, newsreader_dir):
    '''
    train::trainNetwork()

    Purpose: Train a neural network for classification of temporal realtions. Assumes events and timexes
        will be provided at prediction time

    @param tml_files: List of unlabled (no timex, etc) timeML documents
    @param gold_files: Fully labeled gold standard timeML documents
    '''

    from code.learning import network

    print "Called trainNetwork"

    global timenote_imported

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

    mod = network.NNModel()
    mod.train(notes, epochs=35)

    return mod


if __name__ == "__main__":
  main()
