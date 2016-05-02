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

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("predict_dir",
                        nargs=1,
                        help="Directory containing test input")

    parser.add_argument("model_destination",
                        help="Where trained model is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")

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

    predict_event = not(args.no_event)
    predict_timex = not(args.no_timex)
    predict_tlink = not(args.no_tlink)

    annotation_destination = args.annotation_destination

    if os.path.isdir(annotation_destination) is False:
        sys.exit("\n\noutput destination does not exist")

    predict_dir = args.predict_dir[0]

    if os.path.isdir(predict_dir) is False:
        sys.exit("\n\nno output directory exists at set path")

    model_path = args.model_destination

    keys = ["TIMEX", "EVENT", "EVENT_CLASS", "TLINK"]
    flags = [predict_timex, predict_event, predict_event, predict_tlink]

    # make sure appropriate models exist for what needs to be done.
    for key, flag in zip(keys, flags):
        if flag is True:
            m_path = model_path + "_{}_MODEL".format(key)
            v_path = model_path + "_{}_VECT".format(key)
            if os.path.isfile(m_path) is False:
                sys.exit("\n\nmissing model: {}".format(m_path))
            if os.path.isfile(v_path) is False:
                sys.exit("\n\nmissing vectorizer: {}".format(v_path))

    # bad form, but it is annoying for this to inputted just to be told args are invalid.
    from code.notes.TimeNote import TimeNote
    from code.learning import model

    files_to_annotate = glob.glob(predict_dir + "/*")

    #load data from files
    notes = []

    model.load_models(model_path, predict_timex, predict_event, predict_tlink)

    #read in files as notes
    for i, tml in enumerate(files_to_annotate):

        print '\nannotating file: {}/{} {}\n'.format(i+1, len(files_to_annotate), tml)

        note = TimeNote(tml)

        entityLabels, OriginalOffsets, tlinkLabels, tokens = model.predict(note,
                                                                           predict_timex,
                                                                           predict_event,
                                                                           predict_tlink)

        tlinkIdPairs = note.get_tlink_id_pairs()
        offsets = note.get_token_char_offsets()

        assert len(OriginalOffsets) == len(offsets)

        note.write(entityLabels, tlinkLabels, tlinkIdPairs, offsets, tokens, annotation_destination)


if __name__ == '__main__':
	main()
