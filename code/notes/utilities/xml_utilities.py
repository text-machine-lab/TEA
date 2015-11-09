
import xml.etree.ElementTree as ET
from note_utils import valid_path

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

    return ET.tostring(xml_tree_element)

def write_root_to_file(xml_root):

	tree = ET.ElementTree(xml_root)

	tree.write("/home/connor/Workspaces/TEA/Temporal-Entity-Annotator-TEA-/text")


if __name__ == "__main__":
    print "nothing to do here"

# EOF

