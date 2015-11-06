
import news_reader
import tokenize
import pos
import parse

def pre_process(text):

    naf_tagged_doc = news_reader.pre_process(text)

    tokens = tokenize.get_tokens(naf_tagged_doc)
    pos_tags = pos.get_pos_tags(naf_tagged_doc)
    constituency_trees = parse.get_constituency_trees(naf_tagged_doc)

    sentences = {}

    for tok, pos_tag in zip(tokens, pos_tags):

        tmp = []

        assert tok["id"] == pos_tag["id"]

        # get the categories a token falls under.
        # {0:.., 1:,...} the lower the number the more specific.
        grammar_categories = constituency_trees[tok["sentence_num"]].get_phrase_memberships(tok["id"])

        tok.update(pos_tag)
        tok.update({"grammar_categories":grammar_categories})

        print tok

        if tok["sentence_num"] in sentences:
            sentences[tok["sentence_num"]].append(tok)
        else:
            sentences[tok["sentence_num"]] = [tok]

    # one tree per sentence
    # TODO: doesn't actually assert the sentences match to their corresponding tree
    assert( len(sentences) == len(constituency_trees))

    return sentences

if __name__ == "__main__":
#    print pre_process(""" One reason people lie is to achieve personal power. Achieving personal power is helpful for someone who pretends to be more confident than he really is. """)
    pre_process("I wrote a complete sentence. I got a good grade on it")

