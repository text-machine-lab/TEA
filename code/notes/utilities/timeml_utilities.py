
import xml.etree.ElementTree as ET
from note_utils import valid_path

def get_text_element(timeml_doc):

    root = get_root(timeml_doc)

    text_element = None

    for e in root:
        if e.tag == "TEXT":
            text_element = e
            break

    return text_element

def get_root(timeml_doc):

    valid_path(timeml_doc)

    tree = ET.parse(timeml_doc)
    root = tree.getroot()

    return root

def get_raw_text(timeml_doc):
    """ gets raw text of document, xml tags removed """

    text_e = get_text_element(timeml_doc)

    return ET.tostring(text_e, encoding='utf8', method='text')

def get_tagged_entities(timeml_doc):
    """ gets tagged entities within timeml text """

    text_element = get_text_element(timeml_doc)

    elements = {}

    for element in text_element:

        if element.tag in elements:
            elements[element.tag].append(element)
        else:
            elements[element.tag] = []

    return elements

if __name__ == "__main__":
    print "nothing to do here"

# EOF

