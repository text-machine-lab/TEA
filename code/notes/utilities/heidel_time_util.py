import os
import itertools
import sys
import re
import copy

from string import whitespace

from timeml_utilities import annotate_root
from timeml_utilities import annotate_text_element
from timeml_utilities import get_doctime_timex
from timeml_utilities import get_make_instances
from timeml_utilities import get_stripped_root
from timeml_utilities import get_tagged_entities
from timeml_utilities import get_text
from timeml_utilities import get_text_element
from timeml_utilities import get_text_element_from_root
from timeml_utilities import get_tagged_entities_from_heidel
from timeml_utilities import get_text_with_taggings
from timeml_utilities import get_tlinks
from timeml_utilities import set_text_element

from xml_utilities import get_raw_text
from xml_utilities import get_root
from xml_utilities import write_root_to_file

def get_iobs_heidel(note):

    print "CALLED get_iobs_heidel"

    # don't want to modify original
    pre_processed_text = copy.deepcopy(note.pre_processed_text)

    # need to create a list of tokens
    iob_labels = []

    tagged_entities = []

    _tmp_tagged_entities = get_tagged_entities_from_heidel(note._heideltime_annotations)

    heidel_text = copy.deepcopy(note._heideltime_annotations)

    """
    print "heidel text:"
    print heidel_text
    print
    print "-----"*5
    """

    # get timexes. timexes can be nested in TIMEX3INTERVAL taggings.
    for entity in _tmp_tagged_entities:
        if entity.tag == "TIMEX3INTERVAL":
            timex_entities = list(entity)
            #print "# OF TIMEX ENTITIES: ", len(timex_entities)
            for t in timex_entities:
                tagged_entities.append(t)
        else:
            tagged_entities.append(entity)

    _tagged_entities = copy.deepcopy(tagged_entities)

    raw_text = get_text(note.note_path)

    # lots of checks!
    for char in ['\n'] + list(whitespace):
        raw_text     = raw_text.strip(char)
        heidel_text = heidel_text.strip(char)

    raw_text     = re.sub(r"``", r"''", raw_text)
    raw_text = re.sub(r'"', r"'", raw_text)

    raw_text = re.sub("<TEXT>\n+", "", raw_text)
    raw_text = re.sub("\n+</TEXT>", "", raw_text)

    heidel_text = heidel_text.split('\n')[3:-1]

    heidel_text = [re.sub(r"``", r"''", chunk) for chunk in heidel_text]
    heidel_text = [re.sub(r'"', r"'", chunk) for chunk in heidel_text]

    # just in case.
    heidel_text[0] = re.sub("<TimeML>", "", heidel_text[0])
    heidel_text[-1] = re.sub("</TimeML>.*", "", heidel_text[-1])
    heidel_text = "\n".join(heidel_text)
    heidel_text = heidel_text.strip("\n")

    raw_index = 0
    heidel_index = 0

    raw_char_offset = 0
    heidel_char_offset = 0

    # should we count?
    count_raw = True
    count_heidel = True

    text1 = ""
    text2 = ""

    start_count = 0
    end_count = 0

    offsets = {}

    tagged_element = None

    in_interval = False

    # need to get char based offset for each tagging within annotated timeml doc.
    while raw_index < len(raw_text) or heidel_index < len(heidel_text):

        if raw_index < len(raw_text):
            if count_raw is True:
                raw_char_offset += 1
                text1 += raw_text[raw_index]
            raw_index += 1

        if heidel_index < len(heidel_text):

            # TODO: change this to be an re match.
            if heidel_text[heidel_index:heidel_index+7] == '<TIMEX3' and heidel_text[heidel_index:heidel_index+15] != '<TIMEX3INTERVAL':
                tagged_element = tagged_entities.pop(0)
                count_heidel = False

                # TODO: check behind for TIMEX3>or TIMEX3INTERVAL AS WELL. BACK TO BACK DON'T DOUBLE STARTOCUNT INCR
                #if in_interval is False and heidel_text[max(0,heidel_index-8):heidel_index] != '</TIMEX3' and heidel_text[max(0,heidel_index-9):heidel_index] != 'INTERVAL':
                start_count += 1
            elif heidel_text[heidel_index:heidel_index+2] == "</" or heidel_text[heidel_index:heidel_index+1] == '<':
                count_heidel = False
                start_count += 1
            else:
                pass

            if heidel_text[heidel_index:heidel_index+1] == ">":

                if tagged_element != None:

                    start = heidel_char_offset
                    end   = heidel_char_offset+len(tagged_element.text) - 1

                    # spans should be unique?
                    offsets[(start, end)] = {"tagged_xml_element":tagged_element, "text":tagged_element.text, "norm_val":tagged_element.attrib["value"]}

#                    print
#                    print "raw token: ", raw_text[start:end + 1]
#                    print "tagged_element.text: ", tagged_element.text
#                    print

                    # ensure the text at the offset is correct
                    assert raw_text[start:end + 1] == tagged_element.text, "\'{}\' != \'{}\'".format( raw_text[start:end + 1], tagged_element.text)
                    tagged_element = None

                end_count += 1

                if heidel_text[heidel_index:heidel_index+8] != "><TIMEX3" and heidel_text[heidel_index:heidel_index+9] != "></TIMEX3":
                    count_heidel = True

                heidel_index += 1

                continue

            if count_heidel is True:
                heidel_char_offset += 1
                text2 += heidel_text[heidel_index]

            heidel_index += 1

    """
    print "TEXT1: "
    print text1
    print "TEXT2: "
    print text2
    print "equal: ", text1 == text2
    print "len text1: ", len(text1)
    print "len text2: ", len(text2)
    print "len equal: ", len(text1) == len(text2)

    for i,c in enumerate(text1):
        print "char text1: ", c
        print "char text2: ", text2[i]
        if c != text2[i]:
            print "CHAR MISMATCH: {} != {}".format(c, text2[i])

    print "last char text1: \'{}\'".format(text1[-1])
    print "last char text2: \'{}\'".format(text2[-1])
    """

#    exit()
    assert text1 == text2, "{} != {}".format(text1, text2)
    assert start_count == end_count, "{} != {}".format(start_count, end_count)
    assert raw_index == len(raw_text) and heidel_index == len(heidel_text)
    assert raw_char_offset == heidel_char_offset
    assert len(tagged_entities) == 0
    assert tagged_element is None
    assert len(offsets) == len(_tagged_entities)

    for sentence_num in sorted(pre_processed_text.keys()):

        # list of dicts
        sentence = pre_processed_text[sentence_num]

        # iobs in a sentence
        iobs_sentence = []

        # need to assign the iob labels by token index
        for token in sentence:

            # set proper iob label to token
            iob_label, entity_type, entity_id, norm_value = get_label(token, offsets)

            if iob_label is not 'O':
                assert entity_id is not None
                assert entity_type in ['TIMEX3']
            else:
                assert entity_id is None
                assert entity_type is None

            #if token["token"] == "expects":
            #    print "Found expects"
            #    print "iob_label: ", iob_label
            #    print "entity_type: ", entity_type
            #    print "entity_id: ", entity_id
            #    print
            #    sys.exit("done")

            iobs_sentence.append({'entity_label':iob_label,
                                  'entity_type':entity_type,
                                  'entity_id':entity_id,
                                  'norm_val':norm_value})

        iob_labels.append(iobs_sentence)

    print iob_labels

    return iob_labels


def get_label(token, offsets):

    # NOTE: never call this directly. input is tested within _read
    tok_span = (token["char_start_offset"], token["char_end_offset"])

    label = 'O'
    entity_id = None
    entity_type = None
    norm_value = None

    for span in offsets:

        if offsets[span]["tagged_xml_element"].tag not in ["EVENT", "TIMEX3"]:
            continue

        if same_start_offset(span, tok_span):

            labeled_entity = offsets[span]["tagged_xml_element"]

            if 'class' in labeled_entity.attrib:
                label = 'B_' + labeled_entity.attrib["class"]
            elif 'type' in labeled_entity.attrib:
                label = 'B_' + labeled_entity.attrib["type"]

            if 'eid' in labeled_entity.attrib:
                entity_id = labeled_entity.attrib["eid"]
            else:
                entity_id = labeled_entity.attrib["tid"]

            if "norm_val" in offsets[span]:
                norm_value = offsets[span]["norm_val"]

            entity_type = labeled_entity.tag

            break

        elif subsumes(span, tok_span):

            labeled_entity = offsets[span]["tagged_xml_element"]

            if 'class' in labeled_entity.attrib:
                label = 'I_' + labeled_entity.attrib["class"]
            else:
                label = 'I_' + labeled_entity.attrib["type"]

            if 'eid' in labeled_entity.attrib:
                entity_id = labeled_entity.attrib["eid"]
            else:
                entity_id = labeled_entity.attrib["tid"]

            if "norm_val" in offsets[span]:
                norm_value = offsets[span]["norm_val"]

            entity_type = labeled_entity.tag

            break

#        if token["token"] == "expects":

#            print
#            print "Token span: ", tok_span
#            print "Label found: ", label
#            print

#            sys.exit("found it")

    if entity_type == "EVENT":
        # don't need iob tagging just what the type is.
        # multi token events are very rare.
        label = label[2:]

    return label, entity_type, entity_id, norm_value

def same_start_offset(span1, span2):
    """
    doees span1 share the same start offset?
    """
    return span1[0] == span2[0]

def subsumes(span1, span2):
    """
    does span1 subsume span2?
    """
    return span1[0] < span2[0] and span2[1] <= span1[1]

if __name__ == "__main__":
    pass


