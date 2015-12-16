
import sys
import os

naf_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, naf_utilities_path)

import naf_utilities

def get_main_verbs(naf_tagged_doc):

    tokens = naf_utilities._get_predicate_tokens(naf_tagged_doc)

    return tokens

