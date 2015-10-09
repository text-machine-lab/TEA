
import xml.etree.ElementTree as ET
"""
from lxml import etree

tree = etree.parse('somefile.xml')
notags = etree.tostring(tree, encoding='utf8', method='text')
"""

def get_text_element(tree):

    root = tree.getroot()

    text_element = None

    for e in root:
        if e.tag == "TEXT":
            text_element = e
            break

    return text_element

def get_raw_text(timeml_doc):

    #TODO: test valid path

    tree = ET.parse(timeml_doc)

    text_e = get_text_element(tree)

    return ET.tostring(text_e, encoding='utf8', method='text')

print get_raw_text('ABC19980108.1830.0711.tml')

