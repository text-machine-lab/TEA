import subprocess
import os
import sys

def get_temporal_discourse_connectives(sentence_constituency):

	connectives_text = _add_discourse(sentence_constituency)
	print connectives_text
	# connectives_tags = parse_discourse(connectives_text)


def _add_discourse(sentence_constituency):
	'''takes a constituency parse for a sentence and adds discourse tags'''

	discourse = subprocess.Popen(["perl",
								os.environ["TEA_PATH"] + "/code/notes/AddDiscourse/addDiscourse.pl",
								"--parses",
								"/dev/stdin"],
								cwd=os.environ["TEA_PATH"] + "/code/notes/AddDiscourse",
								stdin=subprocess.PIPE,
								stdout=subprocess.PIPE)

	output, _ = discourse.communicate(sentence_constituency)

	return output

# def parse


if __name__ == "__main__":

	inputText = "(S (NP (NN Selling)) (VP (VBD picked) (PRT (RP up)) (SBAR (IN as) (S (S (NP (JJ previous) (NNS buyers)) (VP (VBD bailed) (PRT (RP out)) (PP (IN of) (NP (PRP$ their) (NNS positions))))) (CC and) (S (NP (JJ aggressive) (JJ short) (NN sellers--anticipating)) (VP (ADVP (RB further)) (VBD declines--moved) (PRT (RP in))))))) (. .))(S (NP (PRP$ My) (JJ favorite) (NNS colors)) (VP (AUX are) (ADJP (JJ red) (CC and) (JJ green))) (. .))(S (S (NP (NP (DT The) (NNS asbetsos) (NN fiber)) (, ,) (NP (NN crocidolite)) (, ,)) (VP (AUX is) (ADJP (RB unusually) (JJ resilient)) (SBAR (IN once) (S (NP (PRP it)) (VP (VBZ enters) (NP (NP (DT the) (NNS lungs)) (, ,) (PP (IN with) (NP (NP (RB even) (JJ brief) (NNS exposures)) (PP (TO to) (NP (PRP it))) (VP (VBG causing) (NP (NP (NNS symptoms)) (SBAR (WHNP (WDT that)) (S (VP (VBP show) (PRT (RP up)) (ADVP (NP (NNS decades)) (RB later))))))))))))))) (, ,) (NP (NNS researchers)) (VP (VBD said)) (. .))(S (S (NP (NP (DT A) (NN form)) (PP (IN of) (NP (NN asbestos)))) (ADVP (RB once)) (VP (VBD used) (S (VP (TO to) (VP (VB make) (S (NP (NNP Kent) (NN cigarette) (NNS filters)) (VP (AUX has) (VP (VBN caused) (NP (NP (DT a) (JJ high) (NN percentage)) (PP (IN of) (NP (NN cancer) (NNS deaths)))) (PP (IN among) (NP (NP (DT a) (NN group)) (PP (IN of) (NP (NNS workers))))) (VP (VBN exposed) (PP (TO to) (NP (PRP it))) (ADVP (NP (QP (JJR more) (IN than) (CD 30)) (NNS years)) (RB ago))))))))))) (, ,) (NP (NNS researchers)) (VP (VBD reported)) (. .))(S (PP (IN In) (NP (NN addition))) (, ,) (NP (DT the) (NNP Cray-3)) (VP (MD will) (VP (VB contain) (NP (NP (CD 16) (NN processors--twice)) (CONJP (RB as) (RB many) (IN as)) (NP (DT the) (JJS largest) (JJ current) (NN supercomputer))))) (. .))"

	get_temporal_discourse_connectives(inputText)