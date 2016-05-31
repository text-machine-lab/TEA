"""
Interface to perform predicting of TIMEX, EVENT and TLINK annotations,
using a neural network model for TLINK annotations.
"""

import sys
from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import cPickle
import glob
import os

from keras.models import model_from_json

from code.learning import network

timenote_imported = False

def main():

    global timenote_imported

    parser = argparse.ArgumentParser()

    parser.add_argument("predict_dir",
                        nargs=1,
                        help="Directory containing test input")

    parser.add_argument("model_destination",
                        help="Where trained model is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--use_gold",
                        action='store_true',
                        default=False,
                        help="Use gold taggings for EVENT and TIMEX")

    parser.add_argument("--single_pass",
                        action='store_true',
                        default=False,
                        help="Predict using a single pass model")

    args = parser.parse_args()

    annotation_destination = args.annotation_destination
    newsreader_dir = args.newsreader_annotations

    if os.path.isdir(annotation_destination) is False:
        sys.exit("\n\noutput destination does not exist")
    if os.path.isdir(newsreader_dir) is False:
        sys.exit("invalid path for time note dir")

    predict_dir = args.predict_dir[0]

    if os.path.isdir(predict_dir) is False:
        sys.exit("\n\nno output directory exists at set path")

    model_path = args.model_destination

    # bad form, but it is annoying for this to inputted just to be told args are invalid.
    from code.notes.TimeNote import TimeNote
    from code.learning import model

    notes = []

    pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

    if args.use_gold:
        if '/*' != args.predict_dir[0][-2:]:
            predict_dir = predict_dir + '/*'

        # get files in directory
        files = glob.glob(predict_dir)

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

        basename = lambda x: os.path.basename(x[0:x.index(".tml")])

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


    if args.single_pass:
        # load model
        NNet = model_from_json(open(model_path + '.arch.json').read())
        NNet.load_weights(model_path + '.weights.h5')

        # run prediction cycle
        labels, filter_lists = network.evaluate(notes, NNet)

    else:
        # load both passes
        classifier = model_from_json(open(model_path + '.class.arch.json').read())
        classifier.load_weights(model_path + '.class.weights.h5')

        detector = model_from_json(open(model_path + '.detect.arch.json').read())
        detector.load_weights(model_path + '.detect.weights.h5')

        # run prediction cycle
        labels, filter_lists = network.predict(notes, detector, classifier)

    # labels are returned as a 1 dimensional numpy array, with labels for all objects.
    # we track the current index to find labels for given notes
    label_index = 0

    # filter unused pairs and write each pair to the TimeML file
    for note, del_list in zip(notes, filter_lists):
        # get entity pairs, offsets, tokens, and event/timex entities
        entities = note.get_tlink_id_pairs()
        offsets = note.get_token_char_offsets()

        # flatten list of tokens
        tokenized_text = note.get_tokenized_text()
        tokens = []
        for line in tokenized_text:
            tokens += tokenized_text[line]

        event_timex_labels = []
        # flatten list of labels
        for label_list in note.get_labels():
            event_timex_labels += label_list

        # sort del_list to be in ascending order, and remove duplicates
        del_list = list(set(del_list))
        del_list.sort()
        del_list.reverse()
        # loop through indicies starting at the back to preserve earlier indexes
        for index in del_list:
            del entities[index]

        note_labels = labels[label_index:label_index+len(entities)]

        note.write(event_timex_labels, note_labels, entities, offsets, tokens, annotation_destination)

        label_index += len(entities)

if __name__ == '__main__':
    main()
