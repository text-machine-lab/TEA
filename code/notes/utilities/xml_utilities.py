
import xml.etree.ElementTree as ET
from .note_utils import valid_path
from string import whitespace

import re

def strip_quotes(text):
    """ the pipeline we use does really weird stuff to quotes. just going to remove them for now or forever """

    text     = re.sub(r"``", r"''", text)
    text     = re.sub(r'"', r"'", text)

    return text

def get_root(xml_doc_path):

    valid_path(xml_doc_path)

    tree = ET.parse(xml_doc_path)
    root = tree.getroot()

    return root

def get_root_from_str(xml_doc_contents):

    root = ET.fromstring(xml_doc_contents)

    return root

def get_raw_text(xml_tree_element):
    """ get raw text with xml encodings a string """

    text =  ET.tostring(xml_tree_element)

    return text

def write_root_to_file(xml_root, file_path):

	tree = ET.ElementTree(xml_root)

	print(file_path)
	tree.write(file_path, xml_declaration=True, encoding="us-ascii")

if __name__ == "__main__":
    print("nothing to do here")

# EOF

