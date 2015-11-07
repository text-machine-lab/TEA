
import xml.etree.ElementTree as ET
from note_utils import valid_path

import xml_utilities

def get_text_element(timeml_doc):

    root = xml_utilities.get_root(timeml_doc)

    text_element = None

    for e in root:
        if e.tag == "TEXT":

            text_element = e
            break

    return text_element

def get_text(timeml_doc):
    """ gets raw text of document, xml tags removed """

    text_e = get_text_element(timeml_doc)

    text_e = text_e

    string = list(ET.tostring(text_e, encoding='utf8', method='text'))

    # retains a newline after <TEXT> tag...
    if string[0] == '\n':
        string.pop(0)

    if string[-1] == '\n':
        string.pop(len(string) - 1)

    return "".join(string)


def get_tagged_entities(timeml_doc):
    """ gets tagged entities within timeml text """

    text_element = get_text_element(timeml_doc)

    elements = []

    for element in text_element:

        elements.append(element)

    return elements

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


if __name__ == "__main__":

    print get_make_instances("wsj_1025.tml")


    print get_tlinks("wsj_1025.tml")

    print "nothing to do here"

# EOF

