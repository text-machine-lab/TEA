
import xml.etree.ElementTree as ET
from .note_utils import valid_path

from . import xml_utilities

from string import whitespace

import re

import glob

def get_text_element(timeml_doc):

    root = xml_utilities.get_root(timeml_doc)

    text_element = None

    for e in root:
        if e.tag == "TEXT":

            text_element = e
            break

    return text_element


def get_text_element_from_root(timeml_root):

    # exit("called get text element from root")

    text_element = None

    for e in timeml_root:
        if e.tag == "TEXT":

            text_element = e
            break

    return text_element


def set_text_element(timeml_root, text_element):

    for e in timeml_root:
        if e.tag == "TEXT":

            e = text_element
            break

    return timeml_root

def annotate_text_element(timeml_root, tag, start, end, attributes = {}):
    '''
    returns modified version of the passed timeml_doc root with the annotations
    added in the correct positions
    '''

    text_element = get_text_element_from_root(timeml_root)

    element = ET.Element(tag, attributes)

    text = text_element.text

    start = start + 1
    end = end + 2

    newText = text[:start]
    eleText = text[start:end]
    tail = text[end:]

    text_element.text = newText
    element.text = eleText
    element.tail = tail

    text_element.insert(0, element)

    return text_element


def annotate_root(timeml_root, tag, attributes = {}):
    ''' adds a sub element to root'''

    element = ET.Element(tag, attributes)
    element.tail = "\n"

    timeml_root.append(element)

    return timeml_root

def get_text_with_taggings(timeml_doc):

    text_e = get_text_element(timeml_doc)

    string = ET.tostring(text_e)

    for char in ['\n'] + list(whitespace):

        string = string.strip(char)

    string = xml_utilities.strip_quotes(string)

    return string

def get_text(timeml_doc):
    """ gets raw text of document, xml tags removed """

    text_e = get_text_element(timeml_doc)

    string =  ET.tostring(text_e)

    string = ET.tostring(text_e, encoding='utf8', method='text')

    for char in ['\n'] + list(whitespace):

        string = string.strip(char)

    string = xml_utilities.strip_quotes(string)

    return string

def get_tagged_entities(timeml_doc):
    """ gets tagged entities within timeml text """

    text_element = get_text_element(timeml_doc)

    return list(text_element)

def display_taggings_in_doc(timeml_docs):
    """ get thes unique taggings witin timeml text """

    # used for debugging

    types = {}

    for doc in timeml_docs:

        tagged_entities = get_tagged_entities(doc)

        for entity in tagged_entities:

            entity_type, sub_type = get_entity_type(entity)

            if entity_type not in types:

                types[entity_type] = []

            if sub_type not in types[entity_type]:

                types[entity_type].append(sub_type)

    # print  them all in a nice formatting

    for entity_type in types:

        print(entity_type)

        for sub_type in types[entity_type]:

            print("\t\t" + sub_type)

def get_make_instances(timeml_doc):
    """ gets the event instances in a timeml doc """
    root = xml_utilities.get_root(timeml_doc)

    make_instances = []

    for e in root:

        if e.tag == "MAKEINSTANCE":
            make_instances.append(e)

    return make_instances


def get_tlinks(timeml_doc):

    """ get tlinks from annotated document """

    root = xml_utilities.get_root(timeml_doc)

    tlinks = []

    for e in root:

        if e.tag == "TLINK":

            tlinks.append(e)

    return tlinks

def display_tlink_types(timeml_docs):

    types = set()

    for doc in timeml_docs:

        tlinks = get_tlinks(doc)

        for tlink in tlinks:

            types.add(tlink.attrib["relType"])

    print("TLINK TYPES:")

    for t in types:

        print("\t\t" + t)


def get_entity_type(tagged_entity):

    entity_type = tagged_entity.tag

    sub_type = None

    if 'class' in tagged_entity.attrib:
        sub_type = tagged_entity.attrib["class"]
    elif 'type' in tagged_entity.attrib:
        sub_type = tagged_entity.attrib["type"]
    elif entity_type == "SIGNAL":
        sub_type = ""
    else:
        print(entity_type)
        exit("unknowned type")

    return (entity_type, sub_type)


def get_doctime_timex(timeml_doc):

    """ get the document creation time timex """

    root = xml_utilities.get_root(timeml_doc)

    doctime = None

    for e in root:

        if e.tag == "DCT":
            doctime = e[0]
            break

    return doctime

if __name__ == "__main__":

    display_taggings_in_doc(glob.glob("/data2/kwacome/Temporal-Entity-Annotator-TEA-/annotated_data/*"))

    display_tlink_types(glob.glob("/data2/kwacome/Temporal-Entity-Annotator-TEA-/annotated_data/*"))

    print("nothing to do here")

# EOF

