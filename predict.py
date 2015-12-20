import os
import cPickle
import argparse
import glob

if "TEA_PATH" not in os.environ:
    exit("TEA_PATH environment variable not specified, it is the directory containg predict.py")

if "PY4J_DIR_PATH" not in os.environ:
    exit("PY4J_DIR_PATH environment variable not specified")


os.environ["TEA_PATH"] = os.getcwd()

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument("predict_dir",
                        nargs=1,
                        help="Directory containing test input")

    parser.add_argument("model_destination",
                        help="Where trained model is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")


    args = parser.parse_args()

    annotation_destination = args.annotation_destination

    if os.path.isdir(annotation_destination) is False:

        exit("\n\noutput destination does not exist")

    predict_dir = None
    model = None

    predict_dir = args.predict_dir[0]

    if os.path.isdir(predict_dir) is False:

        exit("\n\nno output directory exists at set path")

    if os.path.isfile(args.model_destination) is False:

        exit("\n\nno model exists at set path")

    # bad form, but it is annoying for this to inputted just to be told args are invalid.
    from code.notes.TimeNote import TimeNote
    from code import model

    modfile = args.model_destination

    files_to_annotate = glob.glob(predict_dir + "/*")

    #load data from files
    notes = []

    with open(modfile) as modelfile:
        model = cPickle.load(modelfile)

    #read in files as notes
    for i, tml in enumerate(files_to_annotate):

        print '\nannotating file: {}/{} {}\n'.format(i+1, len(files_to_annotate), tml)

        note = TimeNote(tml)

        entityLabels, OriginalOffsets, tlinkLabels, tokens = model.predict(note)

        tlinkIdPairs = note.get_tlink_id_pairs()

        offsets = note.get_token_char_offsets()

        assert len(OriginalOffsets) == len(offsets)

        note.write(entityLabels, tlinkLabels, tlinkIdPairs, offsets, tokens, annotation_destination)


if __name__ == '__main__':
	main()
