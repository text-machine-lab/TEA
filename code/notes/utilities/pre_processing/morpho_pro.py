import subprocess
import os
import glob
import sys

from code.config import env_paths

_ENV_PATHS = env_paths()

if _ENV_PATHS["MORPHO_DIR_PATH"] == None:
    sys.exit("ERROR: path for morphopro not set")

_TEA_HOME = os.path.join(*([os.path.dirname(os.path.abspath(__file__))]+[".."]*4))

_dict_entry = lambda morpho_line, sentence_num, sentence_index: {"token_morpho": morpho_line[0],
                                                                 "pos_morpho": morpho_line[1],
                                                                 "morphology_morpho": morpho_line[2],
                                                                 "chunk_morpho": morpho_line[3],
                                                                 "sentence_num_morpho":sentence_num,
                                                                 "sentence_index_morpho":sentence_index}

_new_sentence = lambda l: l[0:1] == ''
_comment      = lambda l: l[0:1] == '#'

def process(text, base_filename=None, overwrite=False, verbose=False):
    """Perform morphological analysis on a text doc.

       text: body of text document
       base_filename: base name of time ml docuemnt, ex: APW19980219.0476 from APW19980219.0476.tml.TE3input
       overwrite: overwrite existing base_filename. Set to false to load existing annotation, since morphopro takes a long time to load.
    """

    if verbose:
        print "len(text): {}".format(len(text))
        print "base_filename: {}".format(base_filename)

    # TODO: make direct api calls and load morphopro into memory.

    _DEFAULT_OUTPUT_DIR = os.path.join(_TEA_HOME, "morpho_output")

    dest_path = None
    morpho_output = None

    if base_filename is None:
        dest_path = os.path.join(_DEFAULT_OUTPUT_DIR, "tmp.morpho")
    else:
        dest_path = os.path.join(_DEFAULT_OUTPUT_DIR, base_filename + ".morpho")

    processed_files = [os.path.basename(f_path) for f_path in glob.glob(os.path.join(_DEFAULT_OUTPUT_DIR,"*"))]

    if overwrite is False and base_filename is not None and base_filename + ".morpho" in processed_files:
        if verbose:
            print "stashed morpho processed file found"
        # return contents of existing file
        morpho_output = open(dest_path,"rb").read()
    else:
        if verbose:
            print "morphopro processing file: ", base_filename
        morpho = subprocess.Popen(["bash",
                                   os.path.join(_ENV_PATHS["MORPHO_DIR_PATH"], "textpro.sh"), # script we're running.
                                   "-l", # select language.
                                   "eng",
                                   "-c", # select what we want to do to text.
                                   "token+pos+full_morpho+chunk",
                                   "-d", # disable tokenization and sentence splitting to ensure 1-1 corresondence with newsreader,
                                   "tokenization+sentence"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)

        morpho_output, _ = morpho.communicate(text)

        # save parsed output to file
        with open(dest_path,"wb") as f:
            f.write(morpho_output)

    # print morpho_output

    output = []

    # TODO: goofy. needs to be fixed. doing this for consistency even though we are GOOFY.
    # TODO: alos kind of redundent. but I kept it in.
    sentence_num = 1
    token_index  = 0

    sentence = []
    output.append(sentence)

    for line in morpho_output.strip('\n').split('\n')[4:]:
        if _new_sentence(line):
            sentence_num += 1
            token_index = 0
            sentence = []
            output.append(sentence)
        else:
            # take advantage of python referencing
            sentence.append(_dict_entry(line.split('\t'), sentence_num, token_index))
            token_index += 1

    sentences = output

    # chunk the morphologies together because the news reader constituency tree stinks!
    for sentence in sentences:
        if verbose:
            print
            print sentence
            print

        for i, token in enumerate(sentence):
            if verbose:
                print
                print i
                print token["chunk_morpho"]

            token["chunked_morphologies_morpho"] = []

            if "B-" in token["chunk_morpho"]:
                if verbose:
                    print "start of chunk"
                # go right until we hit anything that is not "I-"

                token["chunked_morphologies_morpho"].append(token["morphology_morpho"])

                for j, _token in enumerate(sentence[i+1:]):
                    if "I-" not in _token["chunk_morpho"]:
                        break
                    if verbose:
                        print
                        print "\t", j
                        print "\t", _token["chunk_morpho"]
                        print

                    # TODO: append to a list!
                    token["chunked_morphologies_morpho"].append(_token["morphology_morpho"])

            elif "I-" in token["chunk_morpho"]:
                if verbose:
                    print "in chunk"

                # go to left until we hit B-.*
                index = i
                while index >= 0:
                    if verbose:
                        print
                        print "\t", index
                        print "\t", sentence[index]["chunk_morpho"]
                        print
                    if "B-" in sentence[index]["chunk_morpho"]:
                        token["chunked_morphologies_morpho"].insert(0,sentence[index]["morphology_morpho"])
                        break
                    token["chunked_morphologies_morpho"].insert(0,sentence[index]["morphology_morpho"])
                    index -= 1

                # go right until we don't hit an I-, will be either O or B-
                for j, _token in enumerate(sentence[i+1:]):
                    if "I-" not in _token["chunk_morpho"]:
                        break
                    if verbose:
                        print
                        print "\t", j
                        print "\t", _token["chunk_morpho"]
                        print

                    # TODO: append to a list!
                    token["chunked_morphologies_morpho"].append(_token["morphology_morpho"])

            else:
#                if verbose:
#                    print "not important"
                pass

            if verbose:
                print
                print "CHUNKED MORPHO: ", token["chunked_morphologies_morpho"]

    return output

