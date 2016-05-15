"""Debugging TimeNote
"""

# temporary until NN interface is updated
import cPickle


import sys
import os

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] +['..']))

sys.path.insert(0, TEA_HOME_DIR)

from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle

timenote_imported = False


def main():

    global timenote_imported

    """ Process command line arguments and then generate trained models (4, one for each pass) on files provided.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("notes_dir",
                        help="Directory containing training input and gold annotations")
    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    args = parser.parse_args()

    if os.path.isdir(args.notes_dir) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid stashed notes dir")

    notes_dir = None
    newsreader_dir = args.newsreader_annotations

    print args.notes_dir

    notes_dir = args.notes_dir + '/*'

    # get files in directory
    files = glob.glob(notes_dir)

    print files

    pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

    print pickled_timeml_notes

    for tml in files:

        stashed_name = os.path.basename(tml)
        stashed_name = stashed_name.split('.')
        stashed_name = stashed_name[0:stashed_name.index('tml')]
        stashed_name = '.'.join(stashed_name)

        if stashed_name + ".parsed.pickle" in pickled_timeml_notes:
            tmp_note = cPickle.load(open(newsreader_dir + "/" + os.path.basename(tml) + ".parsed.pickle", "rb"))
        else:
            if timenote_imported is False:
                from code.notes.TimeNote import TimeNote
                timenote_imported = True
            tmp_note = TimeNote(tml)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + os.path.basename(tml) + ".parsed.pickle", "wb"))




if __name__ == "__main__":
    main()
