import subprocess
import os
import sys

TEA_HOME = os.path.join(*([os.path.dirname(os.path.abspath(__file__))]+[".."]*3))


def get_temporal_discourse_connectives(sentence_constituency):

    connectives_text = _add_discourse(sentence_constituency)

    connectives_tokens = parse_discourse(connectives_text)

    return connectives_tokens

def _add_discourse(sentence_constituency):
    '''takes a constituency parse for a sentence and adds discourse tags'''

    discourse = subprocess.Popen(["perl",
                                TEA_HOME + "/dependencies/AddDiscourse/addDiscourse.pl",
                                "--parses",
                                "/dev/stdin"],
                                cwd=TEA_HOME + "/dependencies/AddDiscourse",
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)

    output, _ = discourse.communicate(sentence_constituency)

    return output

def parse_discourse(discourse_connective_annotations):
    ''' extracts Temporal connective tokens from a discourse connective parse '''


    temporal_offsets = []

    starting_offset = 1

    token_offsets = get_token_offsets(discourse_connective_annotations)

    #get starting offsets of the temporal tags
    while 0 < starting_offset:
        offset = discourse_connective_annotations.find('#Temporal', starting_offset)
        #used to identify discourse connectives that span multipul characters
        if offset is not -1:
            discourse_id = offset - 1
            token_end_offset = offset - 2
            temporal_offsets.append((token_end_offset, discourse_id))
        starting_offset = offset + 1

    temporal_connectives = []


    for temporal_offset in temporal_offsets:

        # search token list for a token with the identified end offset, start offset, connective id, and token index
        for i, token_offset in enumerate(token_offsets):
            if temporal_offset[0] == token_offset[1]:
                temporal_connectives.append({"token_offset":i, "discourse_id":discourse_connective_annotations[temporal_offset[1]], "token":discourse_connective_annotations[token_offset[0]: token_offset[1]]})
                break

    return temporal_connectives

def get_token_offsets(sentence_constituency):
    ''' identify tokens in a discource connective '''

    token_offsets = []

    for i, char in enumerate(sentence_constituency):
        # first closing parenthisis in a set will always follow a token
        if char is ')' and sentence_constituency[i - 1] is not ')':

            #find start of the token
            offset_start = i
            while sentence_constituency[offset_start - 1] is not ' ':
                offset_start -= 1

            #find end of the token
            offset_end = offset_start
            while offset_end is not len(sentence_constituency):

                if sentence_constituency[offset_end + 1] is '#':
                    offset_end += 1
                    break

                if sentence_constituency[offset_end + 1] is ')':
                    offset_end += 1
                    break

                offset_end += 1

            token_offsets.append((offset_start, offset_end))

    return token_offsets


if __name__ == "__main__":

    inputText = "(S (NP (NN Selling)) (VP (VBD picked) (PRT (RP up)) (SBAR (IN as) (S (S (NP (JJ previous) (NNS buyers)) (VP (VBD bailed) (PRT (RP out)) (PP (IN of) (NP (PRP$ their) (NNS positions))))) (CC and) (S (NP (JJ aggressive) (JJ short) (NN sellers--anticipating)) (VP (ADVP (RB further)) (VBD declines--moved) (PRT (RP in))))))) (. .))(S (NP (PRP$ My) (JJ favorite) (NNS colors)) (VP (AUX are) (ADJP (JJ red) (CC and) (JJ green))) (. .))(S (S (NP (NP (DT The) (NNS asbetsos) (NN fiber)) (, ,) (NP (NN crocidolite)) (, ,)) (VP (AUX is) (ADJP (RB unusually) (JJ resilient)) (SBAR (IN once) (S (NP (PRP it)) (VP (VBZ enters) (NP (NP (DT the) (NNS lungs)) (, ,) (PP (IN with) (NP (NP (RB even) (JJ brief) (NNS exposures)) (PP (TO to) (NP (PRP it))) (VP (VBG causing) (NP (NP (NNS symptoms)) (SBAR (WHNP (WDT that)) (S (VP (VBP show) (PRT (RP up)) (ADVP (NP (NNS decades)) (RB later))))))))))))))) (, ,) (NP (NNS researchers)) (VP (VBD said)) (. .))(S (S (NP (NP (DT A) (NN form)) (PP (IN of) (NP (NN asbestos)))) (ADVP (RB once)) (VP (VBD used) (S (VP (TO to) (VP (VB make) (S (NP (NNP Kent) (NN cigarette) (NNS filters)) (VP (AUX has) (VP (VBN caused) (NP (NP (DT a) (JJ high) (NN percentage)) (PP (IN of) (NP (NN cancer) (NNS deaths)))) (PP (IN among) (NP (NP (DT a) (NN group)) (PP (IN of) (NP (NNS workers))))) (VP (VBN exposed) (PP (TO to) (NP (PRP it))) (ADVP (NP (QP (JJR more) (IN than) (CD 30)) (NNS years)) (RB ago))))))))))) (, ,) (NP (NNS researchers)) (VP (VBD reported)) (. .))(S (PP (IN In) (NP (NN addition))) (, ,) (NP (DT the) (NNP Cray-3)) (VP (MD will) (VP (VB contain) (NP (NP (CD 16) (NN processors--twice)) (CONJP (RB as) (RB many) (IN as)) (NP (DT the) (JJS largest) (JJ current) (NN supercomputer))))) (. .))"

    get_temporal_discourse_connectives(inputText)
