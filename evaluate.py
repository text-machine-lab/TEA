"""Evaluate F-score, Accuracy and Percision for each individual class and overall.
"""

import os
import argparse
import glob
import sys
import copy
import re

from string import whitespace

from code.notes.utilities.timeml_utilities import get_tlinks
from code.notes.utilities.timeml_utilities import get_tagged_entities
from code.notes.utilities.timeml_utilities import get_text
from code.notes.utilities.timeml_utilities import get_text_with_taggings
from code.notes.utilities.timeml_utilities import get_make_instances

# the mapped values are index values in a 2-D list.
_TIMEX_LABELS = {
                 'DATE':0,
                 'TIME':1,
                 'DURATION':2,
                 'SET':3,
                 'O':4,
                }

_EVENT_LABELS = {
                 'REPORTING':0,
                 'PERCEPTION':1,
                 'ASPECTUAL':2,
                 'I_ACTION':3,
                 'I_STATE':4,
                 'STATE':5,
                 'OCCURRENCE':6,
                 'O':7
                }


_POS_LABELS = {
                'ADJECTIVE':0,
                'NOUN':1,
                'VERB':2,
                'PREP':3,
              }

_POL_LABELS = {
                'NEG':0,
                'POS':1,
              }

_TENSE_LABELS = {
                 'PAST':0,
                 'PRESENT':1,
                 'FUTURE':2,
                 'NONE':3,
                 'INFINITIVE':4,
                 'PRESPART':5,
                 'PASTPART':6,
                }

_ASPECT_LABELS = {
                    'PROGRESSIVE':0,
                    'PERFECTIVE':1,
                    'PERFECTIVE_PROGRESSIVE':2,
                    'NONE':3
                 }

_TLINK_LABELS = {
                    'BEFORE':0,
                    'AFTER':1,
                    'INCLUDES':2,
                    'IS_INCLUDED':3,
                    'DURING':4,
                    'DURING_INV':5,
                    'SIMULTANEOUS':6,
                    'IAFTER':7,
                    'IBEFORE':8,
                    'IDENTITY':9,
                    'BEGINS':10,
                    'ENDS':11,
                    'BEGUN_BY':12,
                    'ENDED_BY':13,
                    'NONE_TLINK':14, # not a TIMEML labeling. our own to indicate no tlink
                }



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

    _display_timex_confusion_matrix(files)
    _display_event_confusion_matrix(files)
    _display_pos_confusion_matrix(files)
    _display_pol_confusion_matrix(files)
    _display_TENSE_confusion_matrix(files)
    _display_ASPECT_confusion_matrix(files)
    _display_TLINK_confusion_matrix(files)

    return


def _display_event_confusion_matrix(files):

    confusion = [[0] * len(_EVENT_LABELS) for l in _EVENT_LABELS]

    # keep track of how many times this class occurs in gold or predict
    label_counts = {label:0 for label in _EVENT_LABELS}

    predicted_entities_in_file = []
    gold_entities_in_file = []

    # get annotated elements per file
    for predicted_file, gold_file in files:

        predicted_entities, _ = extract_labeled_entities(predicted_file)
        gold_entities, _ = extract_labeled_entities(gold_file)

        # get mismatching xml and print them.
        for offset in gold_entities:

            gold_class_type = 'O'
            predicted_class_type = 'O'

            if gold_entities[offset]["xml_element"].tag == "EVENT":
                gold_class_type = gold_entities[offset]["xml_element"].attrib["class"]

            label_counts[gold_class_type] += 1

            if offset in predicted_entities:
                if predicted_entities[offset]["xml_element"].tag == "EVENT":
                    predicted_class_type = predicted_entities[offset]["xml_element"].attrib["class"]

                    predicted_entities.pop(offset)

            assert gold_class_type in _EVENT_LABELS
            assert predicted_class_type in _EVENT_LABELS

            label_counts[predicted_class_type] += 1
            confusion[_EVENT_LABELS[gold_class_type]][_EVENT_LABELS[predicted_class_type]] += 1

        for offset in predicted_entities:
            predicted_class_type = 'O'

            if predicted_entities[offset]["xml_element"].tag == "EVENT":
                predicted_class_type = predicted_entities[offset]["xml_element"].attrib["class"]

            assert predicted_class_type in _EVENT_LABELS
            confusion[_EVENT_LABELS['O']][_EVENT_LABELS[predicted_class_type]] += 1
            label_counts[predicted_class_type] += 1

    name = "EVENT"

    display_confusion(name, confusion, _EVENT_LABELS, label_counts)

def _display_TLINK_confusion_matrix(files):
    """Print results of tlink tagging
    """

    confusion = [[0] * len(_TLINK_LABELS) for label in _TLINK_LABELS]

    # keep track of how many times this class occurs in gold or predict
    label_counts = {label:0 for label in _TLINK_LABELS}

    predicted_entities_in_file = []
    gold_entities_in_file = []

    # get annotated elements per file
    for predicted_file, gold_file in files:
        predicted_tlinks = extract_tlinks(predicted_file)
        gold_tlinks = extract_tlinks(gold_file)

        for gold_pair in gold_tlinks:
            if gold_pair in predicted_tlinks:
                confusion[_TLINK_LABELS[gold_tlinks[gold_pair]]][_TLINK_LABELS[predicted_tlinks[gold_pair]]] += 1
                predicted_tlinks.pop(gold_pair)
            else:
                confusion[gold_tlinks[gold_pair]]["NONE_TLINK"] += 1
            label_counts[gold_tlinks[gold_pair]] += 1
        for predicted_pair in predicted_tlinks:
            confusion[_TLINK_LABELS["NONE_TLINK"]][_TLINK_LABELS[predicted_tlinks[predicted_pair]]] += 1

            label_counts[predicted_tlinks[predicted_pauir]] += 1

    display_confusion("TLINK", confusion, _TLINK_LABELS, label_counts, padding=0)


def _display_timex_confusion_matrix(files):
    """Print results of timex tagging

       files: [(predicted_file_path, gold_file_path),...]
    """

    confusion = [[0] * len(_TIMEX_LABELS) for timex_label in _TIMEX_LABELS]

    # keep track of how many times this class occurs in gold or predict
    label_counts = {label:0 for label in _TIMEX_LABELS}

    predicted_entities_in_file = []
    gold_entities_in_file = []

    # get annotated elements per file
    for predicted_file, gold_file in files:

        predicted_entities, _ = extract_labeled_entities(predicted_file)
        gold_entities, _ = extract_labeled_entities(gold_file)

        # get mismatching xml and print them.
        for offset in gold_entities:

            gold_class_type = 'O'
            predicted_class_type = 'O'

            if gold_entities[offset]["xml_element"].tag == "TIMEX3":
                gold_class_type = gold_entities[offset]["xml_element"].attrib["type"]

            label_counts[gold_class_type] += 1

            if offset in predicted_entities:
                if predicted_entities[offset]["xml_element"].tag == "TIMEX3":
                    predicted_class_type = predicted_entities[offset]["xml_element"].attrib["type"]

                    predicted_entities.pop(offset)

            assert gold_class_type in _TIMEX_LABELS
            assert predicted_class_type in _TIMEX_LABELS

            label_counts[predicted_class_type] += 1
            confusion[_TIMEX_LABELS[gold_class_type]][_TIMEX_LABELS[predicted_class_type]] += 1

        for offset in predicted_entities:
            predicted_class_type = 'O'

            if predicted_entities[offset]["xml_element"].tag == "TIMEX3":
                predicted_class_type = predicted_entities[offset]["xml_element"].attrib["type"]

            confusion[_TIMEX_LABELS['O']][_TIMEX_LABELS[predicted_class_type]] += 1
            label_counts[predicted_class_type] += 1

    name = "TIMEX"

    display_confusion(name, confusion, _TIMEX_LABELS, label_counts)

def display_confusion(name, confusion, labels, label_counts, padding=5):
    """Display a confuson matrix for some given labels.

       name: a string for displaying type of confusion matrix
       confusion: entries to my matrix
       labels: the types of labels
       label_counts: total counts of occurrences of a label in gold or predict
    """

    # Display the confusion matrix
    col_names = labels.keys()
    row_entries = []

    for act, act_v in labels.items():
        line = [act]
        line += [str(confusion[act_v][pre_v]) for pre, pre_v in labels.items()]

        row_entries.append(line)

    col_width = max(len(entry) for line in [col_names] + row_entries for entry in line) + padding

    print "\n\t{} CONFUSION MATRIX\n".format(name)
    print "\t\t{}{}".format(' '*col_width, ''.join([col_name.ljust(col_width) for col_name in col_names]))
    for line in row_entries:
        print "\t\t","".join([entry.ljust(col_width) for entry in line])
    print


    # Compute the analysis stuff
    precision = []
    recall = []
    specificity = []
    f1 = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    print "\n\t{} Analysis\n".format(name)
    print '\t\t{}{}'.format(' '*col_width, ''.join([metric.ljust(col_width) for metric in ["Precision", "Recall", "F1"]]))
    print

    for lab, lab_v in labels.items():

        tp = confusion[lab_v][lab_v]
        fp = sum(confusion[v][lab_v] for k, v in labels.items() if v != lab_v)
        fn = sum(confusion[lab_v][v] for k, v in labels.items() if v != lab_v)

        p_num = tp
        p_den = (tp + fp) + 1e-10

        p = float(p_num) / p_den

        r_num = tp
        r_den = (tp + fn) + 1e-10

        r = float(r_num) / r_den

        f = 2 * ((p * r) / ((p + r) + 1e-10))

#        print
#        print "lab: ", lab
#        print "tp: ",tp
#        print "fp: ", fp
#        print "fn: ", fn

        print "\t\t{}{}".format(lab.ljust(col_width), ''.join(["{:.5f}".format(entry).ljust(col_width) for entry in [p, r, f]]))

        # just ignore this then. we didn't predict and it didn't occur in gold.
        # we will ways do really well on O so don't include in average.
        if lab == 'O' or label_counts[lab] == 0: continue
        # going to negatively impact the overall average of individual scores.
        precision += [p]
        recall += [r]
        f1 += [f]


    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f1 = sum(f1) / len(f1)

    print "\n\t{} SUMMARY\n".format(name)
    print '\t\t{}{}'.format(' '*col_width, ''.join([metric.ljust(col_width) for metric in ["Precision", "Recall", "F1"]]))
    print  "\t\t{}{}".format("Average: ".ljust(col_width), ''.join(["{:.5f}".format(metric).ljust(col_width) for metric in [precision, recall, f1]]))
    print

def _compare_file(predicted, gold):
    """Look at where predictions differ from gold and where they are the same
    """

    print "\t\ttgold file: ", gold
    print "\t\tpredicted file: ", predicted
    print

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
    print "\t\t\tincorrect taggings"
    for line in incorrect:
        if repeat % 15 == 0 or repeat == 0:
            print "\n\t\t\t\t{}{}{}\n".format("token".ljust(col_width),
                                      "gold label".ljust(col_width),
                                      "predicted label".ljust(col_width))

        print "\t\t\t\t" + "".join(entry.ljust(col_width) for entry in line)

        repeat += 1

    repeat = 0

    print "\n\t\t\tcorrect taggings"
    for line in correct:
        if repeat % 15 == 0 or repeat == 0:
            print "\n\t\t\t\t{}{}{}\n".format("token".ljust(col_width),
                                      "gold label".ljust(col_width),
                                      "predicted label".ljust(col_width))

        print "\t\t\t\t" + "".join(entry.ljust(col_width) for entry in line)

        repeat += 1



def map_files(files):
    """ assign to each file the basename (no extension) """
    output = {}
    for f in files:
        basesplit = os.path.basename(f).split('.')
        basename = ''
        # add all extension markers until the TimeML extension is reached
        for string in basesplit:
            if string != 'tml' and string != 'TE3input':
                basename += string + '.'

        # slice off trailing .
        basename = basename[:-1]
        output[basename] = f
    return output

def _display_pos_confusion_matrix(files):
    """For all predicted EVENTINSTANCES that match gold EVENTINSTANCES
       display confusion matrix along with F-measure, precision and recall for
       each part of speech class.

       These results don't consider FP or FN event instances. Only look at matches.
       We are only concerned with evaluating how well for the EVENTs we do label correctly
       do we get its attributes right.
    """


    confusion = [[0] * len(_POS_LABELS) for l in _POS_LABELS]
    label_counts = {label:0 for label in _POS_LABELS}

    for predicted, gold in files:

        # predicted id's will never match up completely to gold. so convert them to offset
        predicted_make_instance_offsets = extract_make_instance_offsets(predicted)
        gold_make_instance_offsets = extract_make_instance_offsets(gold)

        for gold_offset in gold_make_instance_offsets:

            if gold_offset in predicted_make_instance_offsets:

                gold_pos_type = gold_make_instance_offsets[gold_offset]["pos"]
                predicted_pos_type = predicted_make_instance_offsets[gold_offset]["pos"]

                confusion[_POS_LABELS[gold_pos_type]][_POS_LABELS[predicted_pos_type]] += 1

                label_counts[gold_pos_type] += 1

    display_confusion("MAKEINSTANCE POS", confusion, _POS_LABELS, label_counts)


def _display_pol_confusion_matrix(files):
    """For all predicted EVENTINSTANCES that match gold EVENTINSTANCES
       display confusion matrix along with F-measure, precision and recall for
       each polarity class.

       These results don't consider FP or FN event instances. Only look at matches.
       We are only concerned with evaluating how well for the EVENTs we do label correctly
       do we get its attributes right.
    """


    confusion = [[0] * len(_POL_LABELS) for l in _POL_LABELS]
    label_counts = {label:0 for label in _POL_LABELS}

    for predicted, gold in files:

        # predicted id's will never match up completely to gold. so convert them to offset
        predicted_make_instance_offsets = extract_make_instance_offsets(predicted)
        gold_make_instance_offsets = extract_make_instance_offsets(gold)

        for gold_offset in gold_make_instance_offsets:

            if gold_offset in predicted_make_instance_offsets:

                gold_pol_type = gold_make_instance_offsets[gold_offset]["polarity"]
                predicted_pol_type = predicted_make_instance_offsets[gold_offset]["polarity"]

                confusion[_POL_LABELS[gold_pol_type]][_POL_LABELS[predicted_pol_type]] += 1

                label_counts[gold_pol_type] += 1

    display_confusion("MAKEINSTANCE POL", confusion, _POL_LABELS, label_counts)

def _display_TENSE_confusion_matrix(files):
    """For all predicted EVENTINSTANCES that match gold EVENTINSTANCES
       display confusion matrix along with F-measure, precision and recall for
       each tense class.

       These results don't consider FP or FN event instances. Only look at matches.
       We are only concerned with evaluating how well for the EVENTs we do label correctly
       do we get its attributes right.
    """


    confusion = [[0] * len(_TENSE_LABELS) for l in _TENSE_LABELS]
    label_counts = {label:0 for label in _TENSE_LABELS}

    for predicted, gold in files:

        # predicted id's will never match up completely to gold. so convert them to offset
        predicted_make_instance_offsets = extract_make_instance_offsets(predicted)
        gold_make_instance_offsets = extract_make_instance_offsets(gold)

        for gold_offset in gold_make_instance_offsets:

            if gold_offset in predicted_make_instance_offsets:

                gold_TENSE_type = gold_make_instance_offsets[gold_offset]["tense"]
                predicted_TENSE_type = predicted_make_instance_offsets[gold_offset]["tense"]

                confusion[_TENSE_LABELS[gold_TENSE_type]][_TENSE_LABELS[predicted_TENSE_type]] += 1

                label_counts[gold_TENSE_type] += 1

    display_confusion("MAKEINSTANCE TENSE", confusion, _TENSE_LABELS, label_counts)

def _display_ASPECT_confusion_matrix(files):
    """For all predicted EVENTINSTANCES that match gold EVENTINSTANCES
       display confusion matrix along with F-measure, precision and recall for
       each ASPECT class.

       These results don't consider FP or FN event instances. Only look at matches.
       We are only concerned with evaluating how well for the EVENTs we do label correctly
       do we get its attributes right.
    """


    confusion = [[0] * len(_ASPECT_LABELS) for l in _ASPECT_LABELS]
    label_counts = {label:0 for label in _ASPECT_LABELS}

    for predicted, gold in files:

        # predicted id's will never match up completely to gold. so convert them to offset
        predicted_make_instance_offsets = extract_make_instance_offsets(predicted)
        gold_make_instance_offsets = extract_make_instance_offsets(gold)

        for gold_offset in gold_make_instance_offsets:

            if gold_offset in predicted_make_instance_offsets:

                gold_ASPECT_type = gold_make_instance_offsets[gold_offset]["aspect"]
                predicted_ASPECT_type = predicted_make_instance_offsets[gold_offset]["aspect"]

                confusion[_ASPECT_LABELS[gold_ASPECT_type]][_ASPECT_LABELS[predicted_ASPECT_type]] += 1

                label_counts[gold_ASPECT_type] += 1

    display_confusion("MAKEINSTANCE ASPECT", confusion, _ASPECT_LABELS, label_counts)

def extract_make_instance_offsets(annotated_file):
    """Map MAKEINSTANCE entities to their respective offsets within text
    """

    _, id_to_offset = extract_labeled_entities(annotated_file)
    make_instances = get_make_instances(annotated_file)

    make_instance_offsets = {}

    for make_instance in make_instances:
        offset = id_to_offset[make_instance.attrib["eventID"]]
        make_instance_offsets[offset] = {"tense":make_instance.attrib["tense"],
                                         "aspect":make_instance.attrib["aspect"],
                                         "polarity":make_instance.attrib["polarity"],
                                         "pos":make_instance.attrib["pos"]}

    return make_instance_offsets


def extract_tlinks(annotated_timeml):
    """Return offset pairs and the relations between them.
    """


    # offset pairs to rel types
    tlinks = {}

    _, id_to_offset = extract_labeled_entities(annotated_timeml)
    make_instances = get_make_instances(annotated_timeml)

    event_instance_id_to_event_id = {make_instance.attrib["eiid"]:make_instance.attrib["eventID"] for make_instance in make_instances}

    # add in an offset for DOCTIME
    id_to_offset['t0'] = (-1,-1)

    for tlink in get_tlinks(annotated_timeml):
        attribs = tlink.attrib

        """
        print id_to_offset

        print
        print attribs["eventInstanceID"]
        print event_instance_id_to_event_id[attribs["eventInstanceID"]]
        print id_to_offset[event_instance_id_to_event_id[attribs["eventInstanceID"]]]
        print
        """

        event_offset = id_to_offset[event_instance_id_to_event_id[attribs["eventInstanceID"]]]
        target_offset = id_to_offset[event_instance_id_to_event_id[attribs["relatedToEventInstance"]] if "relatedToEventInstance" in attribs else attribs["relatedToTime"]]
        if (event_offset, target_offset) in tlinks:
            sys.exit("ERROR: duplicate TLINKs")

        tlinks[(event_offset, target_offset)] = attribs["relType"]

    return tlinks


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
    id_to_offset = {}

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

                    attrib = tagged_element.attrib
                    if "eid" in attrib or "tid" in attrib:
                        ent_id = attrib["eid"] if "eid" in attrib else attrib["tid"]

                        id_to_offset[ent_id] = (start, end)

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

    return offsets, id_to_offset

if __name__ == "__main__":
    main()


