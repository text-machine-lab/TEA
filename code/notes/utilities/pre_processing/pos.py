
import subprocess
import os
import re
import sys

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities
import news_reader

def _get_terms_element(ixa_tok_output):

   xml_root = xml_utilities.get_root_from_str(ixa_tok_output)

   terms_element = None

   for e in xml_root:
       if e.tag == "terms":
           terms_element = e
           break

   return terms_element


def _get_naf_pos_terms(ixa_tok_output):

    terms_element = _get_terms_element(ixa_tok_output)

    naf_pos_terms = []

    for e in terms_element:
        if e.tag == "term":
            naf_pos_terms.append(e)

    return naf_pos_terms


def get_pos_tags(ixa_tok_output):

    """
    look here for interpreting pos taggings:

        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """

    pos_tags = []

    for naf_term in _get_naf_pos_terms(ixa_tok_output):

        pos_tag      = naf_term.attrib["morphofeat"]

        # I think I can just change t1 to w1, seems like one to one mapping.
        id_str       = naf_term.attrib["id"].replace('t','w')

        pos_tags.append({"pos_tag":pos_tag, "id":id_str})

    return pos_tags


if __name__ == "__main__":
    print get_pos_tags(news_reader.pre_process(open("test.txt", "rb").read()))
    pass
# EOF

