
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

if __name__ == "__main__":
    print "nothing to do here"

# EOF

