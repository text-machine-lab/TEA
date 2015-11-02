
import news_reader
import tokenize
import pos

def pre_process(text):

    naf_tagged_doc = news_reader.pre_process(text)

    tokens = tokenize.get_tokens(naf_tagged_doc)
    pos_tags = pos.get_pos_tags(naf_tagged_doc)

    sentences = {}

    for tok, pos_tag, in zip(tokens, pos_tags):

        tmp = []

        assert tok["id"] == pos_tag["id"]

        tok.update(pos_tag)

        if tok["sentence_num"] in sentences:
            sentences[tok["sentence_num"]].append(tok)
        else:
            sentences[tok["sentence_num"]] = [tok]

    return sentences

if __name__ == "__main__":
#    print pre_process(""" One reason people lie is to achieve personal power. Achieving personal power is helpful for someone who pretends to be more confident than he really is. """)

    print pre_process("I wrote a complete sentence. I got a good grade on it")

