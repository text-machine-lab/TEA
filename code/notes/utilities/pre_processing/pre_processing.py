"""Processes contents of text document.
"""

import os
import sys
import naf_parse

from news_reader import NewsReader
import morpho_pro

pre_processor = None

def pre_process(text, filename, overwrite=False):
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
    predicate_ids, tok_id_to_predicate_info, coreferent_lists = naf_parse.parse(naf_tagged_doc)

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

    #print "tokens: ", tokens
    #print "pos's: ", pos_tags

    for tok, pos_tag, lemma in zip(tokens, pos_tags, token_lemmas):

        tmp = []

        char_start = tok["char_start_offset"]
        char_end   = tok["char_end_offset"]

        try:
            assert text[char_start:char_end + 1] == tok["token"], "{} != {}".format(text[char_start:char_end+1], tok["token"])
        except AssertionError:
            print text[char_start:char_end + 1], tok["token"]
            print "Unexpected error:", sys.exc_info()[0]
            # sys.exit()
        assert tok["id"] == pos_tag["id"]
        assert tok["id"] == lemma["id"]

        tok.update(pos_tag)
        tok.update(lemma)

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

    # print "tokens: ", tokens
    # print "id_to_tok: ", id_to_tok

    for tok_id in predicate_ids:
        id_to_tok[tok_id]["is_predicate"] = True

    for tok_id in tok_id_to_predicate_info:
        id_to_tok[tok_id]["predicate_tokens"] = [id_to_tok[i]["token"] for i in tok_id_to_predicate_info[tok_id]["predicate_ids"]]

    for sentence in constituency_trees:
        #print
        #print "sentence num: ", sentence
        #print

        verb_phrase_tokens = constituency_trees[sentence].get_main_verb_phrase_tokens()

    #    print "verb phrase tokens: ", verb_phrase_tokens

        verbs = [token for token in verb_phrase_tokens if id_to_tok[token["id"]]["pos"] == "V"]
        if len(verbs) > 0:
            min_depth = min([v["depth"] for v in verbs])
            main_verbs = [verb for verb in verbs if verb["depth"] == min_depth]

            for main_verb in main_verbs:
                id_to_tok[main_verb["id"]]["is_main_verb"] = True
        #print
            #print "main_verbs: ", main_verbs
        #print

       #     print
     #       print "token id: ",token
      #      print "token: ", token
         #   print "token pos: ", id_to_tok[token["id"]]["pos"]
        #    print


    for target_id in ner_tags:
        assert target_id in id_to_tok, "{} not in id_to_tok".format(target_id)

        id_to_tok[target_id].update(ner_tags[target_id])
        ne_chunk = ""

        for _id in ner_tags[target_id]["ne_chunk_ids"]:
            ne_chunk += id_to_tok[_id]["token"]

        id_to_tok[target_id].update({"ne_chunk":ne_chunk})

    for coref_id in coreferent_lists:
        for span in coreferent_lists[coref_id]:
            for tok_id in span:

                tok_id = 'w' + tok_id[1:]
                assert tok_id in id_to_tok

                id_to_tok[tok_id].update({"coref_chain":coref_id})

    # # print morpho_pro_input
    # morpho_pro_input = "\n".join(morpho_pro_input)
    #
    # # print morpho_pro_input
    #
    # morpho_output = morpho_pro.process(morpho_pro_input, base_filename, overwrite=overwrite)

    # print morpho_output

    # # make sure all the other tokens have is_main_verb
    # for tok in tokens:
    #     if "is_predicate" not in tok:
    #         tok["is_predicate"] = False
    #
    #     if "is_main_verb" not in tok:
    #         tok.update({"is_main_verb":False})
    #
    #     if "ne_id" not in tok:
    #         tok.update({"ne_id":tok["char_start_offset"]})
    #         tok.update({"ner_tag":'NONE'})
    #         tok.update({"ne_chunk":"NULL"})
    #
    #     if "coref_chain" not in tok:
    #         tok.update({"coref_chain":"None"})
    #
    #     if tok["id"] in tok_id_to_predicate_info:
    #         semantic_roles = tok_id_to_predicate_info[tok["id"]]["semantic_role"]
    #         tok.update({"semantic_roles":semantic_roles})
    #     else:
    #         tok.update({"semantic_roles":[]})
    #
    #     # verify that there is a 1-1 correspondence between morphopro tokenization and newsreader.
    #     if tok["token_offset"] >= len(morpho_output[tok["sentence_num"]-1]):
    #         sys.exit("missing token from morphology processing")
    #     elif tok["token"] != morpho_output[tok["sentence_num"]-1][tok["token_offset"]]["token_morpho"]:
    #         if tok["token"] in ('"', ""''"", "``"):
    #             tok.update(morpho_output[tok["sentence_num"] - 1][tok["token_offset"]])
    #         else:
    #             print "morpho token: ", morpho_output[tok["sentence_num"]-1][tok["token_offset"]]["token_morpho"]
    #             print "newsreader token: ", tok["token"]
    #         # print "newsreader: ", [t for t in tokens if t["sentence_num"] == tok["sentence_num"]]
    #         # print "morpho sentence: ", morpho_output[tok["sentence_num"]-1]
    #             sys.exit("token mismatch between newsreader tokenization and morphorpo")
    #     else:
    #         # good to go
    #         tok.update(morpho_output[tok["sentence_num"]-1][tok["token_offset"]])

    dependency_paths = naf_parse.DependencyPath(naf_tagged_doc)

    # one tree per sentencei
    # TODO: doesn't actually assert the sentences match to their corresponding tree

    if len(constituency_trees) > 0:
        assert( len(sentences) == len(constituency_trees))

    return sentences, tokens_to_offset, sentence_features, dependency_paths, id_to_tok

if __name__ == "__main__":
    pass

