
import news_reader
import tokenize
import pos
import parse
import ner
import srl

import os
import sys

utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, utilities_path)

import lemmas
import timeml_utilities

from news_reader import NewsReader

pre_processor = NewsReader()

def pre_process(text):

    naf_tagged_doc = pre_processor.pre_process(text)

    tokens, tokens_to_offset = tokenize.get_tokens(naf_tagged_doc)
    pos_tags = pos.get_pos_tags(naf_tagged_doc)
    token_lemmas   = lemmas.get_lemmas(naf_tagged_doc)
    ner_tags       = ner.get_taggings(naf_tagged_doc)
    main_verbs     = srl.get_main_verbs(naf_tagged_doc)

    constituency_trees = parse.get_constituency_trees(naf_tagged_doc)

    sentences = {}

    token_offset = 0

    assert len(tokens) == len(pos_tags)
    assert len(tokens) == len(token_lemmas)

    id_to_tok = {}

    for tok, pos_tag, lemma in zip(tokens, pos_tags, token_lemmas):

        tmp = []

        char_start = tok["char_start_offset"]
        char_end   = tok["char_end_offset"]

        assert text[char_start:char_end + 1] == tok["token"], "{} != {}".format(text[char_start:char_end+1], tok["token"])
        assert tok["id"] == pos_tag["id"]
        assert tok["id"] == lemma["id"]

        tok.update(pos_tag)
        tok.update(lemma)

        """
        # get the categories a token falls under.
        # {0:.., 1:,...} the lower the number the more specific.
        grammar_categories = constituency_trees[tok["sentence_num"]].get_phrase_memberships(tok["id"])

        tok.update(pos_tag)

        for category in grammar_categories:
            tok.update({"grammar_category{}".format(category):grammar_categories[category]})
        """

        if tok["sentence_num"] in sentences:
            sentences[tok["sentence_num"]].append(tok)
            tok["token_offset"] = token_offset
        else:
            sentences[tok["sentence_num"]] = [tok]
            token_offset = 0
            tok["token_offset"] = token_offset

        assert tok["id"] not in id_to_tok

        id_to_tok[tok["id"]] = tok

        token_offset += 1

    sentence_features = {}

    # sentence based features
    for key in sentences:

        features_for_current_sentence = {}

        parse_tree = constituency_trees[key].get_parenthetical_tree(sentences[key])
        features_for_current_sentence['constituency_tree'] = parse_tree

        sentence_features[key] = features_for_current_sentence

    for target_id in ner_tags:

        assert target_id in id_to_tok, "{} not in id_to_tok".format(target_id)

        id_to_tok[target_id].update(ner_tags[target_id])

    for main_verb in main_verbs:

        assert main_verb in id_to_tok

        id_to_tok[main_verb].update({"is_main_verb":True})

    # make sure all the other tokens have is_main_verb

    for tok in tokens:

        if "is_main_verb" not in tok:
            tok.update({"is_main_verb":False})


    # one tree per sentence
    # TODO: doesn't actually assert the sentences match to their corresponding tree
    assert( len(sentences) == len(constituency_trees))

    return sentences, tokens_to_offset, sentence_features

if __name__ == "__main__":
    pre_process(timeml_utilities.get_text("APW19980820.1428.tml.TE3input"))

#    print pre_process(timeml_utilities.get_text("APW19980820.1428.tml"))
    pass

