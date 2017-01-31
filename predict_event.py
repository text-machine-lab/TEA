"""Interface to perform predicting of TIMEX, EVENT and TLINK annotations.
"""

import sys
from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import os
import cPickle
import argparse
import glob

from code.learning import model
from code.learning.model_event import EventWriter
from code.learning.model_event import tag_timex
from keras.models import load_model

from code.learning.word2vec import load_word2vec_binary

timenote_imported = False

def main():

    global timenote_imported

    parser = argparse.ArgumentParser()

    parser.add_argument("timex_dir",
                        nargs=1,
                        help="Directory containing files with timex tags")

    parser.add_argument("model_destination",
                        help="Where trained event model is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")


    args = parser.parse_args()

    annotation_destination = args.annotation_destination

    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(annotation_destination) is False:
        sys.exit("\n\noutput destination does not exist")

    newsreader_dir = args.newsreader_annotations

    predict_dir = args.timex_dir[0]

    if os.path.isdir(predict_dir) is False:
        sys.exit("\n\nno output directory exists at set path")

    model_path = args.model_destination

    files_to_annotate = glob.glob(predict_dir + "/*")

    pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

    # event model
    NNet = load_model(os.path.join(model_path, 'model.h5'))
    word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)

    #read in files as notes
    for i, tml in enumerate(files_to_annotate):

        print '\nannotating file: {}/{} {}\n'.format(i+1, len(files_to_annotate), tml)

        stashed_name = os.path.basename(tml)
        stashed_name = stashed_name.split('.')
        if 'tml' in stashed_name:
            stashed_name = stashed_name[0:stashed_name.index('tml')]
        stashed_name = '.'.join(stashed_name)

        if stashed_name + ".parsed.predict.pickle" in pickled_timeml_notes:
            print "loading stashed"
            note = cPickle.load(open(newsreader_dir + "/" + stashed_name + ".parsed.predict.pickle", "rb"))
        else:
            if timenote_imported is False:
                from code.notes.TimeNote import TimeNote
                timenote_imported = True
            note = TimeNote(tml, tml) # need the second argument to get timex tags
            cPickle.dump(note, open(newsreader_dir + "/" + stashed_name + ".parsed.predict.pickle", "wb"))

        entityLabels = [label for line in note.iob_labels for label in line]
        tokens = [token for num in note.pre_processed_text for token in note.pre_processed_text[num]]

        event_writer = EventWriter(note, word_vectors=word_vectors, NNet=NNet)
        tml_root = event_writer.tag_text(entityLabels, tokens, note)
        note_path = os.path.join(annotation_destination, stashed_name + ".tml")
        EventWriter.write_tags(tml_root, note_path)


if __name__ == '__main__':
	main()
