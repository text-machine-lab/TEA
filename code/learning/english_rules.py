"""The rules within english_rules.txt come from the original authors of this paper.
"""
import os
import re
import sys

from code.notes.utilities.pre_processing import morpho_pro


TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + [".."]*2))

# be/indic/pres + _v_/gerund/pres = tense=PRESENT, aspect=PROGRESSIVE
_ACTIVE_VOICE_PRESENT_PROGRESSIVE_RE = "^be\+.*\+indic\+pres||^.+\+v\+.*gerund\+pres"
# have/indic/pres + be/part/past + _v_/gerund/pres = tense=PRESENT, aspect=PERFECTIVE_PROGRESSIVE
_ACTIVE_VOICE_PRESENT_PERFECTIVE_PROGRESSIVE_RE = "^have\+.*\+indic\+pres||^be\+.*\+part\+past||^.+\+v\+.*gerund\+pres"
 # have/indic/pres + _v_/part/past = tense=PRESENT, aspect=PERFECTIVE
_ACTIVE_VOICE_PRESENT_PERFECTIVE_RE = "^have\+.*\+indic\+pres||.+\+v\+.*part\+past"

# be/indic/past + _v_/gerund/pres = tense=PAST, aspect=PROGRESSIVE
_ACTIVE_VOICE_PAST_PROGRESSIVE_RE = "^be\+.*\+indic\+past||^.+\+v\+.*gerund\+pres"
# have/indic/past + _v_/part/past = tense=PAST, aspect=PERFECTIVE
_ACTIVE_VOICE_PAST_PERFECTIVE_RE =  "^have\+.*\+indic\+past||^.+\+v\+.*part\+past"
# have/indic/past + be/part/past + _v_/gerund/pres = tense=PAST, aspect=PERFECTIVE_PROGRESSIVE
_ACTIVE_VOICE_PAST_PERFECTIVE_PROGRESSIVE_RE = "^have\+.*\+indic\+past||^be\+.*\+part\+past||^.+\+v\+.*gerund\+pres"

# will/indic/pres + _v_/infin/pres = tense=FUTURE, aspect=NONE
_ACTIVE_VOICE_FUTURE_NONE_A_RE = "^will\+.*\+indic\+pres||^.+\+v\+.*infin\+pres"
# will/indic/pres + _v_/indic/pres = tense=FUTURE, aspect=NONE
_ACTIVE_VOICE_FUTURE_NONE_B_RE = "^will\+.*\+indic\+pres||^.+\+v\+.*indic\+pres"
# be/indic/pres + go/gerund/pres + to + _v_/infin/pres = tense=FUTURE, aspect=NONE
_ACTIVE_VOICE_FUTURE_NONE_C_RE = "^be\+.*\+indic\+pres||^go\+.*\+gerund\+pres||^to\+.*||^.+\+v\+.*infin\+pres"
# will/indic/pres + be/infin/pres + _v_/gerund/pres = tense=FUTURE, aspect=PROGRESSIVE
_ACTIVE_VOICE_FUTURE_PROGRESSIVE_A_RE = "^will\+.*\+indic\+pres||^be\+.*\+infin\+pres||^.+\+v\+.*gerund\+pres"
# be/indic/pres + go/gerund/pres + to + be/infin/pred + _v_/infin/pres = tense=FUTURE, aspect=PROGRESSIVE
_ACTIVE_VOICE_FUTURE_PROGRESSIVE_B_RE = "^be\+.*\+indic\+pres||^go\+.*\+gerund\+pres||^to\+.*||^be\+.*infin\+pres||^.+\+v\+.*gerund\+pres"
# will/indic/pres + have/infin/pres + _v_/part/past = tense=FUTURE, aspect=PERFECTIVE
_ACTIVE_VOICE_FUTURE_PERFECTIVE_RE = "^will\+.*\+indic\+pres||^have\+.*\+infin\+pres||^.+\+v\+.*part\+past"
# will/indic/pres + have/infin/pres + be/part/past +  _v_/gerund/pres = tense=FUTURE, aspect=PERFECTIVE_PROGRESSIVE
_ACTIVE_VOICE_FUTURE_PERFECTIVE_PROGRESSIVE_RE = "^will\+.*\+indic\+pres||^have\+.*\+infin\+pres||^be\+.*part\+past||^.+\+v\+.*gerund\+pres"


_RULE_NAMES = {

                _ACTIVE_VOICE_PRESENT_PROGRESSIVE_RE: "ACTIVE VOICE: PRESENT PROGRESSIVE",
                _ACTIVE_VOICE_PRESENT_PERFECTIVE_PROGRESSIVE_RE: "ACTIVE VOICE: PRESENT PERFECTIVE-PROGRESSIVE",
                _ACTIVE_VOICE_PRESENT_PERFECTIVE_RE: "ACTIVE VOICE: PRESENT PERFECTIVE",

                _ACTIVE_VOICE_PAST_PROGRESSIVE_RE: "ACTIVE VOICE: PAST PROGRESSIVE",
                _ACTIVE_VOICE_PAST_PERFECTIVE_RE: "ACTIVE VOICE: PAST PERFECTIVE",
                _ACTIVE_VOICE_PAST_PERFECTIVE_PROGRESSIVE_RE: "ACTIVE VOICE: PERFECTIVE PROGRESSIVE",

                _ACTIVE_VOICE_FUTURE_NONE_A_RE: "ACTIVE VOICE: FUTURE NONE",
                _ACTIVE_VOICE_FUTURE_NONE_B_RE: "ACTIVE VOICE: FUTURE NONE",
                _ACTIVE_VOICE_FUTURE_NONE_C_RE: "ACTIVE VOICE: FUTURE NONE",
                _ACTIVE_VOICE_FUTURE_PROGRESSIVE_A_RE: "ACTIVE VOICE: FUTURE PROGRESSIVE",
                _ACTIVE_VOICE_FUTURE_PROGRESSIVE_B_RE: "ACTIVE VOICE: FUTURE PROGRESSIVE",
                _ACTIVE_VOICE_FUTURE_PERFECTIVE_RE: "ACTIVE VOICE: FUTURE PERFECTIVE",
                _ACTIVE_VOICE_FUTURE_PERFECTIVE_PROGRESSIVE_RE: "ACTIVE VOICE: PERFECTIVE PROGRESSIVE",

              }

# should match
_POSITIVE_CASES = {

                    _ACTIVE_VOICE_PRESENT_PROGRESSIVE_RE: "is teaching",
                    _ACTIVE_VOICE_PRESENT_PERFECTIVE_PROGRESSIVE_RE: "has been teaching",
                    _ACTIVE_VOICE_PRESENT_PERFECTIVE_RE: "has taught",

                    _ACTIVE_VOICE_PAST_PROGRESSIVE_RE: "was teaching",
                    _ACTIVE_VOICE_PAST_PERFECTIVE_RE: "had taught",
                    _ACTIVE_VOICE_PAST_PERFECTIVE_PROGRESSIVE_RE: "had been teaching",

                    _ACTIVE_VOICE_FUTURE_NONE_A_RE: "will teach",
                    _ACTIVE_VOICE_FUTURE_NONE_B_RE: "will teach",
                    _ACTIVE_VOICE_FUTURE_NONE_C_RE: "is going to teach",
                    _ACTIVE_VOICE_FUTURE_PROGRESSIVE_A_RE: "will be teaching",
                    _ACTIVE_VOICE_FUTURE_PROGRESSIVE_B_RE: "is going to be teaching",
                    _ACTIVE_VOICE_FUTURE_PERFECTIVE_RE: "will have taught",
                    _ACTIVE_VOICE_FUTURE_PERFECTIVE_PROGRESSIVE_RE: "will have been teaching",

                 }

# should never match
_NEGATIVE_CASES = {

                    _ACTIVE_VOICE_PRESENT_PROGRESSIVE_RE: "was teaching",
                    _ACTIVE_VOICE_PRESENT_PERFECTIVE_PROGRESSIVE_RE: "had been teaching",
                    _ACTIVE_VOICE_PRESENT_PERFECTIVE_RE: "had taught",

                    _ACTIVE_VOICE_PAST_PROGRESSIVE_RE: "is teaching",
                    _ACTIVE_VOICE_PAST_PERFECTIVE_RE: "has taught",
                    _ACTIVE_VOICE_PAST_PERFECTIVE_PROGRESSIVE_RE: "has been teaching",

                    _ACTIVE_VOICE_FUTURE_NONE_A_RE: "would teach",
                    _ACTIVE_VOICE_FUTURE_NONE_B_RE: "would teach",
                    _ACTIVE_VOICE_FUTURE_NONE_C_RE: "was going to teach",
                    _ACTIVE_VOICE_FUTURE_PROGRESSIVE_A_RE: "would be teaching",
                    _ACTIVE_VOICE_FUTURE_PROGRESSIVE_B_RE: "was going to be teaching",
                    _ACTIVE_VOICE_FUTURE_PERFECTIVE_RE: "would have taught",
                    _ACTIVE_VOICE_FUTURE_PERFECTIVE_PROGRESSIVE_RE: "would have been teaching",

                  }


def _test_generate_morpho_input(CASES):

    morphopro_input = []

    # generate input for morphopro and process them all at once to save time.
    for i, rule in enumerate(CASES):
        morphopro_input += CASES[rule].split(' ')
        # morphopro can take in already tokenized text, one line per token in sentence.
        # blank lines indicate start of new sentence.
        # don't add a trailing blank line.
        if i+1 < len(CASES):
            morphopro_input.append("")

    return morphopro_input

def _test_cases(CASES, EXAMPLES, verbose=False):

    results = {}

    # one rule for each line
    for rule, example in zip(CASES.keys(), EXAMPLES):
        if verbose is True:
            print
            print "\t\tRULE: ", _RULE_NAMES[rule]
            print "\t\tEXAMPLE: \'{}\'".format(CASES[rule])

        # one condition for each token
        conditions = rule.split('||')

        rule_holds = True

        for condition, token in zip(conditions, example):
            # for some reason I store the morphology in a list. which is not necessasry.
            morphology = token["morphology_morpho"][0]
            if True not in [re.search(condition, m) != None for m in morphology.split(' ')]:
                rule_holds = False

            if verbose:
                print
                print "\t\t\tCONDITION: ", condition
                print "\t\t\tMORPHOLOGY: ", morphology

        results[_RULE_NAMES[rule]] = rule_holds

    return results


def _test_english_rules(verbose=False):
    """A suite of tests for each morphology rule.
    """

    print
    print "Testing rules:"

    if _POSITIVE_CASES.keys() != _NEGATIVE_CASES.keys():
        sys.exit("ERROR english_rules.py: either missing positive or negative cases example to test")

    positive_morpho_input = _test_generate_morpho_input(_POSITIVE_CASES)
    negative_morpho_input = _test_generate_morpho_input(_NEGATIVE_CASES)

    morphopro_input = "\n".join(positive_morpho_input + [""] + negative_morpho_input)

    morphopro_output = morpho_pro.process(morphopro_input, base_filename="english_rules_text", overwrite=True)

    if len(morphopro_output) != len(_POSITIVE_CASES.keys() + _NEGATIVE_CASES.keys()):
        sys.exit("ERROR _test_english_rules(): morphopro output did not processing input correctly")

    if verbose:
        print "\tPOSITIVE: "

    pos_cases = len(_POSITIVE_CASES.keys())

    positive_test_results = _test_cases(_POSITIVE_CASES, morphopro_output[0:pos_cases], verbose=verbose)

    if verbose:
        print "\tNEGATIVE: ",

    negative_test_results = _test_cases(_NEGATIVE_CASES, morphopro_output[pos_cases:], verbose=verbose)

    # summarize and output. indicate if everything passed!
    print
    print "SUMMARY: "

    matched = [rule for rule in positive_test_results if positive_test_results[rule] is True]
    did_not_match = [rule for rule in negative_test_results if negative_test_results[rule] is False]

    print "\tPOSITIVE: Passed {}/ Total {}".format(len(matched), len(positive_test_results))
    print "\tNEGATIVE: Passed {}/ Total {}".format(len(did_not_match), len(negative_test_results))




def _tests_passive_voice():

    from code.notes.utilities.pre_processing import morpho_pro

    # be/indic/pres + _v_/part/past = tense=PRESENT, aspect=NONE
    active_voice = "\n".join(["is", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice present none test passed"
    else:
        sys.exit("passive voice present none test failed")

    # be/indic/pres + be/gerund/pres + _v_/part/past = tense=PRESENT, aspect=PROGRESSIVE
    active_voice = "\n".join(["is", "being", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice present progressive test passed"
    else:
        sys.exit("passive voice present progressive test failed")

    # have/indic/pres + be/part/past + _v_/part/past = tense=PRESENT, aspect=PERFECTIVE
    active_voice = "\n".join(["have", "been", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice present perfective test passed"
    else:
        sys.exit("passive voice present perfective test failed")

    # be/indic/past + _v_/part/past = tense=PAST, aspect=NONE
    active_voice = "\n".join(["was", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice past none test passed"
    else:
        sys.exit("passive voice past none test failed")

    # be/indic/past + be/gerund/pres + _v_/part/past = tense=PAST, aspect=PROGRESSIVE
    active_voice = "\n".join(["was", "being", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice past progressive test passed"
    else:
        sys.exit("passive voice past progressive test failed")


    # have/indic/past + be/part/past + _v_/part/past = tense=PAST, aspect=PERFECTIVE
    active_voice = "\n".join(["had", "been", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice past perfective test passed"
    else:
        sys.exit("passive voice past perfective test failed")

    # will/indic/pres + be/infin/pres + _v_/part/past = tense=FUTURE, aspect=NONE
    active_voice = "\n".join(["will", "be", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future none test passed"
    else:
        sys.exit("passive voice future none test failed")

    # be/indic/pres + go/gerund/pres + to + be/infin/pres + _v_/part/past = tense=FUTURE, aspect=NONE
    active_voice = "\n".join(["is", "going", "to", "be", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]
    active_voice_morphology_tok5 = active_voice_output[0][4]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^go\+.*\+gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok5.split(' ')]:
        print "passive voice future none test passed"
    else:
        sys.exit("passive voice future none test failed")

    # will/indic/pres + have/infin/pres + be/part/past + _v_/part/past = tense=FUTURE, aspect=PERFECTIVE
    active_voice = "\n".join(["will", "have", "been", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    return

def _tests_auxiliar():

    from code.notes.utilities.pre_processing import morpho_pro

    # have/indic/pres + to + _v_/infin/pres = tense=PRESENT, aspect=NONE
    active_voice = "\n".join(["has", "to", "teach"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future none test passed"
    else:
        sys.exit("passive voice future none test failed")

    # have/indic/pres + to + be/infin/pres + _v_/gerund/pres = tense=PRESENT, aspect=PROGRESSIVE
    active_voice = "\n".join(["has", "to", "be", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # have/indic/pres + to + have/infin/pres + _v_/part/past = tense=PRESENT, aspect=PERFECTIVE
    active_voice = "\n".join(["has", "to", "have", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # have/indic/pres + to + have/infin/pres + be/part/past + _v_/gerund/pres = tense=PRESENT, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["has", "to", "have", "been", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]
    active_voice_morphology_tok5 = active_voice_output[0][4]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^be\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok4.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok5.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # have/indic/past + to + _v_/infin/pres = tense=PAST, aspect=NONE
    active_voice = "\n".join(["had", "to", "teach"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # have/indic/past + to + be/infin/pres + _v_/gerund/pres = tense=PAST, aspect=PROGRESSIVE
    active_voice = "\n".join(["had", "to", "be", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # will/indic/pres + have/infin/pres + to + _v_/infin/pres = tense=FUTURE, aspect=NONE
    active_voice = "\n".join(["will", "have", "to", "teach"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # will/indic/pres + have/infin/pres + to + be/infin/pres + _v_/gerund/pres = tense=FUTURE, aspect=PROGRESSIVE
    active_voice = "\n".join(["will", "have", "to", "be", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]
    active_voice_morphology_tok5 = active_voice_output[0][4]["morphology_morpho"][0]

    if True in [re.search("^will\+.*indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^to.*",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok5.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # (must|should|may|might|can|could|would) + _v_/infin/pres = tense=NONE, aspect=NONE
    active_voice = "\n".join(["could", "teach"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^(must|should|may|might|can|could|would)\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # (must|should|may|might|can|could|would) + be/infin/pres + _v_/gerund/pres = tense=NONE, aspect=PROGRESSIVE
    active_voice = "\n".join(["could", "be", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^(must|should|may|might|can|could|would)\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # (must|should|may|might|can|could|would) + have/infin/pres +_v_/part/past = tense=NONE, aspect=PERFECTIVE
    active_voice = "\n".join(["could", "have", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^(must|should|may|might|can|could|would)\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # (must|should|may|might|can|could|would) + have/infin/pres + been/part/past +_v_/gerund/pres = tense=NONE, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["could", "have", "been", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^(must|should|may|might|can|could|would)\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # (must|should|may|might|can|could|would) + be/infin/pres + _v_/part/past = tense=PAST, aspect=NONE
    active_voice = "\n".join(["could", "be", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^(must|should|may|might|can|could|would)\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


def _do_did():

    from code.notes.utilities.pre_processing import morpho_pro

    # do/indic/past + _v_/infin/pres = tense=PAST, aspect=NONE
    active_voice = "\n".join(["did", "care"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^do\+.*indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # do/indic/pres + _v_/infin/pres = tense=PRESENT, aspect=NONE
    active_voice = "\n".join(["do", "care"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^do\+.*indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


def _infinitive():
    from code.notes.utilities.pre_processing import morpho_pro

    # to + _v_/infin/pres = tense=INFINITIVE, aspect=NONE
    active_voice = "\n".join(["to", "release"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    print active_voice_morphology_tok1
    print active_voice_morphology_tok2

    if True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # to + _v_/indic/pres = tense=INFINITIVE, aspect=NONE
    active_voice = "\n".join(["to", "release"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*indic\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # to + be/infin/pres + _v_/gerund/pres = tense=INFINITIVE, aspect=PROGRESSIVE
    active_voice = "\n".join(["to", "be", "releasing"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # to + have/infin/pres + _v_/part/past = tense=INFINITIVE, aspect=PERFECTIVE
    active_voice = "\n".join(["to", "have", "released"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # to + have/infin/pres + be/part/past + _v_/gerund/pres = tense=INFINITIVE, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["to", "have", "been", "releasing"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    return


def _single_word_VP_test():

    from code.notes.utilities.pre_processing import morpho_pro

    # _v_/gerund/pres = tense=PRESPART, aspect=NONE
    phrase = "\n".join(["releasing"])
    output = morpho_pro.process(phrase, base_filename="test", overwrite=True)

    tok = output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in tok.split(' ')]:
        print "active voice present progressive test passed"
    else:
        sys.exit("active voice present progressive test failed")

    # _v_/indic/pres = tense=PRESENT, aspect=NONE
    phrase = "\n".join(["release"])
    output = morpho_pro.process(phrase, base_filename="test", overwrite=True)

    tok = output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*indic\+pres",m) != None for m in tok.split(' ')]:
        print "active voice present progressive test passed"
    else:
        sys.exit("active voice present progressive test failed")

    # _v_/indic/past = tense=PAST, aspect=NONE
    phrase = "\n".join(["released"])
    output = morpho_pro.process(phrase, base_filename="test", overwrite=True)

    tok = output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*indic\+past",m) != None for m in tok.split(' ')]:
        print "active voice present progressive test passed"
    else:
        sys.exit("active voice present progressive test failed")

    # _v_/part/past = tense=PASTPART, aspect=NONE
    phrase = "\n".join(["released"])
    output = morpho_pro.process(phrase, base_filename="test", overwrite=True)

    tok = output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*part\+past",m) != None for m in tok.split(' ')]:
        print "active voice present progressive test passed"
    else:
        sys.exit("active voice present progressive test failed")


def _two_piece_VP():

    from code.notes.utilities.pre_processing import morpho_pro

    # be/part/past + _v_/gerund/pres = tense=NONE, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["been", "running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # be/part/past + _v_/part/past = tense=NONE, aspect=PERFECTIVE
    active_voice = "\n".join(["been", "tried"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    pass


def _prepart():


    from code.notes.utilities.pre_processing import morpho_pro

    # prespart
    # _v_/gerund/pres = tense=PRESPART, aspect=NONE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    pass


def _adjectives():

    from code.notes.utilities.pre_processing import morpho_pro

    """
    # adjective
    # be/indic/pres + _a_ = tense=PRESENT, aspect=NONE
    active_voice = "\n".join(["is", "blue"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    print active_voice_morphology_tok1
    print active_voice_morphology_tok2

    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.*\+adj\+.*",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    # be/indic/pres + be/gerund/pres + _a_ = tense=PRESENT, aspect=PROGRESSIVE
    active_voice = "\n".join(["is", "being", "scared"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.*\+adj\+.*",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # have/indic/pres + be/part/past + _a_ = tense=PRESENT, aspect=PERFECTIVE
    active_voice = "\n".join(["has", "been", "scared"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.*\+adj\+.*",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")

    """

    exit()

    # be/indic/past + _a_ = tense=PAST, aspect=NONE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # be/indic/past + be/gerund/pres + _a_ = tense=PAST, aspect=PROGRESSIVE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # have/indic/past + be/part/past + _a_ = tense=PAST, aspect=PERFECTIVE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # will/indic/pres + be/infin/pres + _a_ = tense=FUTURE, aspect=NONE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")


    # will/indic/pres + have/infin/pres + be/part/past + _a_ = tense=FUTURE, aspect=PERFECTIVE
    active_voice = "\n".join(["running"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]

    if True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')]:
        print "passive voice future perfective test passed"
    else:
        sys.exit("passive voice future perfective test failed")



    pass

def _nominal_nouns():

    # noun
    # be/indic/pres + _n_ = tense=PRESENT, aspect=NONE
    # be/indic/pres + be/gerund/pres + _n_ = tense=PRESENT, aspect=PROGRESSIVE
    # have/indic/pres + be/part/past + _n_ = tense=PRESENT, aspect=PERFECTIVE

    # be/indic/past + _n_ = tense=PAST, aspect=NONE
    # be/indic/past + be/gerund/pres + _n_ = tense=PAST, aspect=PROGRESSIVE
    # have/indic/past + be/part/past + _n_ = tense=PAST, aspect=PERFECTIVE

    # will/indic/pres + be/infin/pres + _n_ = tense=FUTURE, aspect=NONE
    # will/indic/pres + have/infin/pres + be/part/past + _n_ = tense=FUTURE, aspect=PERFECTIVE
    pass

def _preposition():

    # preposition
    # be/indic/pres + _p_ = tense=PRESENT, aspect=NONE
    # be/indic/pres + be/gerund/pres + _p_ = tense=PRESENT, aspect=PROGRESSIVE
    # have/indic/pres + be/part/past + _p_ = tense=PRESENT, aspect=PERFECTIVE

    # be/indic/past + _p_ = tense=PAST, aspect=NONE
    # be/indic/past + be/gerund/pres + _p_ = tense=PAST, aspect=PROGRESSIVE
    # have/indic/past + be/part/past + _p_ = tense=PAST, aspect=PERFECTIVE

    # will/indic/pres + be/infin/pres + _p_ = tense=FUTURE, aspect=NONE
    # will/indic/pres + have/infin/pres + be/part/past + _p_ = tense=FUTURE, aspect=PERFECTIVE

    pass



def get_tense_aspect(token, id_to_tok):

    num_tokens = None
    tokens_morphology = None

    if token["constituency_phrase"]["phrase"] == "NONE":
        num_tokens = 1
        tokens_morphology = [token["morphology_morpho"]]
    else:
        tokens_morphology = [id_to_tok[t_id]["morphology_morpho"] for t_id in token["constituency_phrase"]["ordered_phrase_members"]]
        num_tokens = len(tokens_morphology)

    print num_tokens
    print tokens_morphology

    tense  = "NONE"
    aspect = "NONE"

    if num_tokens == 1:
        # TODO: add the rules
        return "NONE", "NONE"
    elif num_tokens == 2:
        # TODO: add the rules
        tok1_match = False
        tok1_morphology = tokens_morphology[0]

        tok2_match = False
        tok2_morphology = tokens_morphology[1]

        # be/indic/pres + _v_/gerund/pres = tense=PRESENT, aspect=PROGRESSIVE
        if True in [re.search("^be\+.*\+indic\+pres",m) for m in tok1_morphology.split(' ')] and\
           True in [re.search("^.+\+v\+.+\+part\+part",m) for m in tok2_morphology.split(' ')]:
            print "YES"
        # have/indic/pres + _v_/part/past = tense=PRESENT, aspect=PERFECTIVE
        # TODO: ....

        exit("hit 2 case")
        return "NONE", "NONE"
    elif num_tokens == 3:
        # TODO: add the rules
        # have/indic/pres + be/part/past + _v_/gerund/pres = tense=PRESENT, aspect=PERFECTIVE_PROGRESSIVE
        return "NONE", "NONE"
    else:
        # TODO: get better way of handling no match cases?
        return "NONE", "NONE"

if __name__ == "__main__":
    print _get_candidate_rules(2)
    pass

