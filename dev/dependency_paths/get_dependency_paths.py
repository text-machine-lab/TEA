"""Interface to extract dependency paths from timeml files.
"""

# temporary until NN interface is updated
import sys
import os

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] +['..','..']))
sys.path.insert(0, TEA_HOME_DIR)

import argparse
import glob
import cPickle

def process_input_files(input_files):
    """Process input file list and then assign pairings.
       Also verifies that proper pairings exist.
    """

    timeml_files = {}

    for f in input_files:

        # try to get the unique file id
        basename = os.path.basename(f)
        is_annotated = False
        end = basename.find(".tml.TE3input")

        # this must be an annotated file
        if end == -1:
            end = basename.find(".tml")
            is_annotated = True

        # unique file id
        file_id = basename[0:end]

        # store pairings in a dictionary.
        if file_id not in timeml_files:
            timeml_files[file_id] = {}
        timeml_files[file_id]["annotation" if is_annotated else "timeml"] = f

    # verify pairings are kosher.
    problem_files = []
    for file_id in timeml_files:
        if len(timeml_files[file_id].keys()) != 2:
            problem_files.append(file_id)

    # print out problem pairings.
    if problem_files:
        print '\n\tERROR: File pairings'
        for file_id in problem_files:
            print "\n\t\tPairing issue for file: {}".format(file_id)
        print
        sys.exit()

    return timeml_files

def main():
    """ Process command line arguments and then pre-process timeml files.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("input_files",
                        nargs='*',
                        help="A list of files file paths. Annotated and unannotated are expected to be present within the list of file paths: <file_id>.tml.E3input and a corresponding <file_id>.tml, indicates the annotated gold file.")

    parser.add_argument("output_dir",
                        type=str,
                        help="Where dependency paths for a timeml file are written.")

    parser.add_argument("newsreader_dir",
                        type=str,
                        help="Where timeml Note objects are stored. This is needed because it takes a long time for timeml notes to be processed by newsreader.")

    args = parser.parse_args()

    timeml_files = process_input_files(args.input_files)

    if os.path.isdir(args.output_dir) is False:
        sys.exit("\n\tERROR: Invalid dir to store dependency paths: {}\n".format(args.output_dir))
    if os.path.isdir(args.newsreader_dir) is False:
        sys.exit("\n\tERROR: Invalid dir to store Note objects: {}\n".format(args.newsreader_dir))

    print "\n\t# of files to process: {}".format(len(timeml_files))
    print "\tdepenendecy output path: {}".format(args.output_dir)
    print "\tNote pickle dump path: {}".format(args.newsreader_dir)
    print

    # get the files that are already stashed.
    pickled_timeml_notes = [os.path.basename(f) for f in glob.glob(os.path.join(args.newsreader_dir,'*'))]
    notes = {}

    num_files = len(timeml_files)
    timenote_imported = False

    for i, file_id in enumerate(timeml_files):

        print '\n\tProcessing file {}/{}: {}'.format(i + 1,num_files,file_id)

        if file_id + ".parsed.pickle" in pickled_timeml_notes:
            print "\t\tLoading stashed model"
            tmp_note = cPickle.load(open(os.path.join(args.newsreader_dir,file_id + ".parsed.pickle"), "rb"))
        else:
            print "\t\tCreating Note object."
            if timenote_imported is False:
                print "\t\tImporting TimeNote module, will take a minute..."
                from code.notes.TimeNote import TimeNote
                timenote_imported = True
            tmp_note = TimeNote(timeml_files[file_id]["timeml"], timeml_files[file_id]["annotation"])
            print "\t\tStashing Note object."
            cPickle.dump(tmp_note, open(os.path.join(args.newsreader_dir,file_id + ".parsed.pickle"), "wb"))
        notes[file_id]= tmp_note

    for note_id in notes:

        dependency_output_file = open(os.path.join(args.output_dir,note_id + ".dependencies"),"wb")

        print "\n\tNote: {}".format(note_id)
        l_paths, r_paths, pairs = _get_token_id_subpaths(notes[note_id])

        for l_path, r_path, pair in zip(l_paths,r_paths,pairs):
            dependency_output_file.write(' '*10 + "Relation type: {}\n".format(pair["rel_type"]))
            dependency_output_file.write(' '*10 + "source token: {} ({})\n".format(pair["source_token"],pair["src_id"]))
            dependency_output_file.write(' '*10 + "target token: {} ({})\n".format(pair["target_token"],pair["target_id"]))
            dependency_output_file.write(' '*10 + "Left path: {}\n".format(notes[note_id].get_tokens_from_ids(l_path)))
            dependency_output_file.write(' '*10 + "Right path: {}\n".format(notes[note_id].get_tokens_from_ids(r_path)))
            dependency_output_file.write('\n\n')


def _get_token_id_subpaths(note):
    """Extract ids for the tokens in each half of the shortest dependency path between each token in each relation
       This is copy pasted from connor's branch to see what the paths he is obtaining are.
    """
    # TODO: for now we only look at the first token in a given entity. Eventually, we should get all tokens in the entity

    iob_labels = note.get_labels()
    pairs = note.get_tlinked_entities()

    left_paths = []
    right_paths = []
    id_pairs = []

    for pair in pairs:
        target_id = ''
        source_id = ''

        # extract and reformat the ids in the pair to be of form t# instead of w#
        # events may be linked to document creation time, which will not have an id
        if 'id' in pair["target_entity"][0]:
            target_id = pair["target_entity"][0]["id"]
            target_id = 't' + target_id[1:]
        if 'id' in pair["src_entity"][0]:
            source_id = pair["src_entity"][0]["id"]
            source_id = 't' + source_id[1:]

        left_path, right_path = note.dependency_paths.get_left_right_subpaths(target_id, source_id)
        left_paths.append(left_path)
        right_paths.append(right_path)

        target_token = pair["target_entity"][0]["token"] if "token" in pair["target_entity"][0] else "DOC TIME"
        src_token    = pair["src_entity"][0]["token"] if "token" in pair["src_entity"][0] else "DOC TIME"

        target_entity_id = "t0"
        src_entity_id = "t0"

        if target_token != "DOC TIME":
            target_entity_id = iob_labels[pair["target_entity"][0]["sentence_num"]-1][pair["target_entity"][0]["token_offset"]]["entity_id"]
        if src_token != "DOC TIME":
            src_entity_id = iob_labels[pair["src_entity"][0]["sentence_num"]-1][pair["src_entity"][0]["token_offset"]]["entity_id"]
#        print pair["target_entity"]

        id_pairs.append({"target_token":target_token,
                         "target_id":target_entity_id,
                         "source_token":src_token,
                         "src_id":src_entity_id,
                         "rel_type":pair["rel_type"]})

    return left_paths, right_paths, id_pairs

if __name__ == "__main__":
  main()

