"""Processes contents of text document.
"""

import os
import sys
import naf_parse

from news_reader import NewsReader
import morpho_pro

pre_processor = None

def pre_process(text, filename):
    """ pre-process contents of a document

    Example:
        text = open("some_doc.txt","rb").read()
        pre_process(text)
    """

    global pre_processor

    if pre_processor is None:
        pre_processor = NewsReader()

    naf_tagged_doc = pre_processor.pre_process(text)

    tokens,     tokens_to_offset,\
    pos_tags,   token_lemmas,\
    ner_tags,   constituency_trees,\
    main_verbs, tok_id_to_predicate_info = naf_parse.parse(naf_tagged_doc)
    #coreferent_lists = naf_parse.parse(naf_tagged_doc)

    base_filename = os.path.basename(filename)

    """
    print "\ntokens:\n"
    print tokens
    print "\n\n"

    print "\ntokens_to_offset:\n"
    print tokens_to_offset
    print "\n\n"

    print "\ntoken_lemmas:\n"
    print token_lemmas
    print "\n\n"

    print "\nner_tags:\n"
    print ner_tags
    print "\n\n"

    print "\nmain_verbs:\n"
    print main_verbs
    print "\n\n"

    print "\ntok_id_to_predicate_info:\n"
    print tok_id_to_predicate_info
    print "\n\n"

    print "\nconstituency_trees:\n"
    print constituency_trees
    print "\n\n"
    """

    sentences = {}

    token_offset = 0

    assert len(tokens) == len(pos_tags)
    assert len(tokens) == len(token_lemmas)

    id_to_tok = {}

    morpho_pro_input = []

    for tok, pos_tag, lemma in zip(tokens, pos_tags, token_lemmas):

        tmp = []

        char_start = tok["char_start_offset"]
        char_end   = tok["char_end_offset"]

        assert text[char_start:char_end + 1] == tok["token"], "{} != {}".format(text[char_start:char_end+1], tok["token"])
        assert tok["id"] == pos_tag["id"]
        assert tok["id"] == lemma["id"]

        tok.update(pos_tag)
        tok.update(lemma)

        grammar_categories = []

        # get the categories a token falls under.
        # {0:.., 1:,...} the lower the number the more specific.
        if tok["sentence_num"] in constituency_trees:
            grammar_categories = constituency_trees[tok["sentence_num"]].get_phrase_memberships(tok["id"])

        tok.update(pos_tag)

        # get verb tense:
        # TODO: make this better..
        if tok["pos_tag"] in ["VBD", "VBP"]:
            tok.update({"tense":"PAST"})
        else:
            tok.update({"tense":"PRESENT"})

        tok.update({"grammar_categories":grammar_categories})

        if tok["sentence_num"] in sentences:
            sentences[tok["sentence_num"]].append(tok)
            tok["token_offset"] = token_offset

            morpho_pro_input.append(tok["token"])

        else:
            sentences[tok["sentence_num"]] = [tok]
            token_offset = 0
            tok["token_offset"] = token_offset

            if morpho_pro_input != []:
                morpho_pro_input.append("")
            morpho_pro_input.append(tok["token"])

        assert tok["id"] not in id_to_tok

        id_to_tok[tok["id"]] = tok

        token_offset += 1

    sentence_features = {}

    # sentence based features
    for key in sentences:
        features_for_current_sentence = {}
        parse_tree = None

        if key in constituency_trees:
            parse_tree = constituency_trees[key].get_parenthetical_tree(sentences[key])
        else:
            parse_tree = []

        features_for_current_sentence['constituency_tree'] = parse_tree
        sentence_features[key] = features_for_current_sentence


    for target_id in ner_tags:
        assert target_id in id_to_tok, "{} not in id_to_tok".format(target_id)

        id_to_tok[target_id].update(ner_tags[target_id])
        ne_chunk = ""

        for _id in ner_tags[target_id]["ne_chunk_ids"]:
            ne_chunk += id_to_tok[_id]["token"]

        id_to_tok[target_id].update({"ne_chunk":ne_chunk})

    for main_verb in main_verbs:

        assert main_verb in id_to_tok

        id_to_tok[main_verb].update({"is_main_verb":True})

    """
    for coref_id in coreferent_lists:
        for span in coreferent_lists[coref_id]:
            for tok_id in span:

                tok_id = 'w' + tok_id[1:]
                assert tok_id in id_to_tok

                id_to_tok[tok_id].update({"coref_chain":coref_id})
    """

    # print morpho_pro_input
    morpho_pro_input = "\n".join(morpho_pro_input)

    # print morpho_pro_input

    morpho_output = morpho_pro.process(morpho_pro_input, base_filename)

    # print morpho_output

    # make sure all the other tokens have is_main_verb
    for tok in tokens:
        if "is_main_verb" not in tok:
            tok.update({"is_main_verb":False})

        if "ne_id" not in tok:
            tok.update({"ne_id":tok["char_start_offset"]})
            tok.update({"ner_tag":'NONE'})
            tok.update({"ne_chunk":"NULL"})

        """
        if "coref_chain" not in tok:
            tok.update({"coref_chain":"None"})
        """

        if tok["id"] in tok_id_to_predicate_info:
            semantic_roles = tok_id_to_predicate_info[tok["id"]]["semantic_role"]
            tok_predicate_info = tok_id_to_predicate_info[tok["id"]]
            preposition_ids = tok_predicate_info.pop("toks_preposition")
            preposition_tokens = []

            for tok_id in preposition_ids:
                preposition_tokens.append(id_to_tok[tok_id]["token"])

            tok_predicate_info["preposition_tokens"] = preposition_tokens
            tok.update({"semantic_roles":semantic_roles})
            tok.update(tok_predicate_info)

        # add constituency phrase membership
        tok.update({"constituency_phrase":constituency_trees[tok["sentence_num"]].get_phrase_membership(tok["id"])})

        # verify that there is a 1-1 correspondence between morphopro tokenization and newsreader.
        if tok["token_offset"] >= len(morpho_output[tok["sentence_num"]-1]):
            sys.exit("missing token from morphology processing")
        elif tok["token"] != morpho_output[tok["sentence_num"]-1][tok["token_offset"]]["token_morpho"]:
            # print "morpho token: ", morpho_output[tok["sentence_num"]-1][tok["token_offset"]]["token_morpho"]
            # print "newsreader token: ", tok["token"]
            # print "newsreader: ", [t for t in tokens if t["sentence_num"] == tok["sentence_num"]]
            # print "morpho sentence: ", morpho_output[tok["sentence_num"]-1]
            sys.exit("token mismatch between newsreader tokenization and morphorpo")
        else:
            # good to go
            tok.update(morpho_output[tok["sentence_num"]-1][tok["token_offset"]])

    dependency_paths = naf_parse.DependencyPath(naf_tagged_doc)

    # one tree per sentencei
    # TODO: doesn't actually assert the sentences match to their corresponding tree

    if len(constituency_trees) > 0:
        assert( len(sentences) == len(constituency_trees))

    return sentences, tokens_to_offset, sentence_features, dependency_paths, id_to_tok

if __name__ == "__main__":
    pass

