
import os
import sys

naf_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, naf_utilities_path)

import naf_utilities
import news_reader

def get_pos_tags(ixa_tok_output):

    """
    look here for interpreting pos taggings:

        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """

    pos_tags = []

    for naf_term in naf_utilities._get_naf_terms(ixa_tok_output):

        pos_tag      = naf_term.attrib["morphofeat"]

        # I think I can just change t1 to w1, seems like one to one mapping.
        id_str       = naf_term.attrib["id"].replace('t','w')

        pos_tags.append({"pos_tag":pos_tag, "id":id_str})

    return pos_tags


if __name__ == "__main__":
    print get_pos_tags(news_reader.pre_process("this is a test sentence!"))
    pass
# EOF

