"""Evaluate F-score, Accuracy and Percision for each individual class and overall.
"""

import os
import argparse
import glob
import sys
import copy
import re

from string import whitespace

from code.notes.utilities.timeml_utilities import get_tagged_entities
from code.notes.utilities.timeml_utilities import get_text
from code.notes.utilities.timeml_utilities import get_text_with_taggings

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("predicted",
        help = "The directory that contains predicted timeml files",
    )

    parser.add_argument("gold",
        help = "The directory that contains gold standard timeml files",
    )

    # Parse command line arguments
    args = parser.parse_args()

    if os.path.isdir(args.predicted) is False:
        sys.exit("ERROR: invalid predicted directory")
    if os.path.isdir(args.gold) is False:
        sys.exit("ERROR: invalid gold directory")

    print "\n"
    print "\tpredicted DIR: {}".format(args.predicted)
    print "\tgold DIR: {}".format(args.gold)
    print "\n"

    wildcard = "*"

    # List of gold data
    gold_files = glob.glob( os.path.join(args.gold, wildcard) )

    gold_files_map = map_files(gold_files)

    # List of predictions
    pred_files = glob.glob( os.path.join(args.predicted, wildcard) )
    pred_files_map = map_files(pred_files)

    # Grouping of text, predictions, gold
    files = []

    for gold_basename in gold_files_map:
        if gold_basename in pred_files_map:
            files.append((pred_files_map[gold_basename], gold_files_map[gold_basename]))
        else:
            sys.exit("missing predicted file: {}".format(gold_basename))

    for predicted, gold in files:

        predicted_entities = extract_labeled_entities(predicted)
        gold_entities = extract_labeled_entities(gold)

        incorrect = []
        correct   = []

        # get mismatching xml and print them.
        for offset in gold_entities:
            gold_class_type = gold_entities[offset]["xml_element"].attrib["class"] if "class" in gold_entities[offset]["xml_element"].attrib else gold_entities[offset]["xml_element"].attrib["type"]
            if offset in predicted_entities:

                predicted_class_type = predicted_entities[offset]["xml_element"].attrib["class"] if "class" in predicted_entities[offset]["xml_element"].attrib else predicted_entities[offset]["xml_element"].attrib["type"]

                if gold_class_type != predicted_class_type:
                    incorrect.append((gold_entities[offset]["text"],
                                      gold_class_type + " ({})".format(gold_entities[offset]["xml_element"].tag),
                                      predicted_class_type + " ({})".format(predicted_entities[offset]["xml_element"].tag)))
                else:
                    correct.append((gold_entities[offset]["text"],
                                    gold_class_type + " ({})".format(gold_entities[offset]["xml_element"].tag),
                                    predicted_class_type + " ({})".format(predicted_entities[offset]["xml_element"].tag)))
            # mismatch
            else:
                incorrect.append((gold_entities[offset]["text"],
                                  gold_class_type + " ({})".format(gold_entities[offset]["xml_element"].tag),
                                  "O"))

        # mismatches
        for offset in predicted_entities:
            predicted_class_type = predicted_entities[offset]["xml_element"].attrib["class"] if "class" in predicted_entities[offset]["xml_element"].attrib else predicted_entities[offset]["xml_element"].attrib["type"]
            if offset not in gold_entities:
                incorrect.append((predicted_entities[offset]["text"],
                                  "O",
                                  predicted_class_type + " ({})".format(predicted_entities[offset]["xml_element"].tag)))

        col_width = max(len(entry) for line in incorrect + correct for entry in line) + 2

        repeat = 0
        print "\t\tincorrect taggings"
        for line in incorrect:
            if repeat % 15 == 0 or repeat == 0:
                print "\n\t\t\t{}{}{}\n".format("token".ljust(col_width),
                                          "gold label".ljust(col_width),
                                          "predicted label".ljust(col_width))

            print "\t\t\t" + "".join(entry.ljust(col_width) for entry in line)

            repeat += 1

        repeat = 0

        print "\n\t\tcorrect taggings"
        for line in correct:
            if repeat % 15 == 0 or repeat == 0:
                print "\n\t\t\t{}{}{}\n".format("token".ljust(col_width),
                                          "gold label".ljust(col_width),
                                          "predicted label".ljust(col_width))

            print "\t\t\t" + "".join(entry.ljust(col_width) for entry in line)

            repeat += 1


    return


def map_files(files):
    """ assign to each file the basename (no extension) """
    output = {}
    for f in files:
        basename = os.path.basename(f).split('.')[0]
        output[basename] = f
    return output


def extract_labeled_entities(annotated_timeml):
    """raw_text: unannotated timeml file
       labeled_text: annotated timeml file

       extracts the labeled entities from labeled documents with char offsets included.
    """

    tagged_entities = get_tagged_entities(annotated_timeml)
    _tagged_entities = copy.deepcopy(tagged_entities)

    raw_text = get_text(annotated_timeml)
    labeled_text = get_text_with_taggings(annotated_timeml)

    # lots of checks!
    for char in ['\n'] + list(whitespace):
        raw_text     = raw_text.strip(char)
        labeled_text = labeled_text.strip(char)

    raw_text     = re.sub(r"``", r"''", raw_text)
    labeled_text = re.sub(r'"', r"'", labeled_text)

    raw_text = re.sub("<TEXT>\n+", "", raw_text)
    raw_text = re.sub("\n+</TEXT>", "", raw_text)

    labeled_text = re.sub("<TEXT>\n+", "", labeled_text)
    labeled_text = re.sub("\n+</TEXT>", "", labeled_text)

    # incase there is no new line!
    labeled_text = re.sub("<TEXT>", "", labeled_text)
    labeled_text = re.sub("</TEXT>", "", labeled_text)

    raw_index = 0
    labeled_index = 0

    raw_char_offset = 0
    labeled_char_offset = 0

    # should we count?
    count_raw = True
    count_labeled = True

    # used to compare and make sure texts match up after parsing raw and labeled texts.
    text1 = ""
    text2 = ""

    start_count = 0
    end_count = 0

    offsets = {}

    tagged_element = None

    # need to get char based offset for each tagging within annotated timeml doc.
    while raw_index < len(raw_text) or labeled_index < len(labeled_text):

        if raw_index < len(raw_text):
            if count_raw is True:
                raw_char_offset += 1
                text1 += raw_text[raw_index]
            raw_index += 1

        if labeled_index < len(labeled_text):

            # TODO: change this to be an re match.
            if labeled_text[labeled_index:labeled_index+1] == '<' and labeled_text[labeled_index:labeled_index+2] != '</':

                tagged_element = tagged_entities.pop(0)

                count_labeled = False
                start_count += 1

            elif labeled_text[labeled_index:labeled_index+2] == '</':
                count_labeled = False
                start_count += 1

            if labeled_text[labeled_index:labeled_index+1] == ">":

                if tagged_element != None:

                    start = labeled_char_offset
                    end   = labeled_char_offset+len(tagged_element.text) - 1

                    # spans should be unique?
                    offsets[(start, end)] = {"xml_element":tagged_element, "text":tagged_element.text}

                    # ensure the text at the offset is correct
                    assert raw_text[start:end + 1] == tagged_element.text, "\'{}\' != \'{}\'".format( raw_text[start:end + 1], tagged_element.text)
                    tagged_element = None

                end_count += 1
                count_labeled = True

                labeled_index += 1
                continue

            if count_labeled is True:
                labeled_char_offset += 1
                text2 += labeled_text[labeled_index]

            labeled_index += 1


    # verify correctness
    assert text1 == text2, "{} != {}".format(text1, text2)
    assert start_count == end_count, "{} != {}".format(start_count, end_count)
    assert raw_index == len(raw_text) and labeled_index == len(labeled_text)
    assert raw_char_offset == labeled_char_offset
    assert len(tagged_entities) == 0
    assert tagged_element is None
    assert len(offsets) == len(_tagged_entities)

    return offsets

if __name__ == "__main__":
    main()


