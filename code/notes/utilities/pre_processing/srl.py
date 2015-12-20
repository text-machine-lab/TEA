
import sys
import os

naf_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, naf_utilities_path)

import naf_utilities

def get_main_verbs(naf_tagged_doc):

    tokens = naf_utilities._get_predicate_tokens(naf_tagged_doc)

    return tokens

def get_predicate_info(naf_tagged_doc):

    return naf_utilities._get_token_id_to_participant_map(naf_tagged_doc)

