
import sys
import os

naf_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, naf_utilities_path)

import naf_utilities

def get_taggings(naf_tagged_doc):

    taggings = {}

    # a list of dicts of form [ {"ner_tag":<tag>, "entity":<list of targets>}, ...]
    for mapping in naf_utilities._get_ner_labels(naf_tagged_doc):

        ne_id =  mapping["ne_id"]
        ner_tag   =  mapping["ner_tag"]
        target_ids = mapping["target_ids"]

        for target_id in target_ids:

            assert target_id not in taggings

            taggings[target_id] = {"ne_id":ne_id,
                                    "ner_tag":ner_tag}

    return taggings

if __name__ == "__main__":

    pass

