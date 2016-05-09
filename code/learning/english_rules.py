"""The rules within english_rules.txt come from the original authors of this paper.
"""
import os
import re
import sys

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + [".."]*2))

#_INVALID_START_LINE_CHARS=tuple(['','#','\n'])

#def _read_rules():
#    """Parse the english_rules.txt file.
#    """

#    rules = {}

#    for l in open(os.path.join(*[TEA_HOME_DIR,"code","learning","english_rules.txt"]), "rb"):
#        if l[0:1] not in _INVALID_START_LINE_CHARS:
#            token_count = len(l.split('+'))

#            l.strip('\n')

#            if token_count in rules:
#                rules[token_count].append(l.strip('\n'))
#            else:
#                rules[token_count] = [l.strip('\n')]

#    return rules

# This will be instantiated on import and on executed
#_RULES = _read_rules()

#def get_tense_polarity_aspect(token):
#    print token

#    return

#def _get_candidate_rules(num_tokens):
#    """Get rule
#    """

#    if num_tokens in _RULES:
#        return _RULES[num_tokens]
#    else:
#        return []

def _tests_active_voice():
    """Run morphopro on text and verify patterns work properly.

       Needs to be executed by a *.py file within TEA_HOME

       TODO: make more thorough
    """

    print "running tense aspect rule unit tests"

    from code.notes.utilities.pre_processing import morpho_pro

    active_voice = "\n".join(["is", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # be/indic/pres + _v_/gerund/pres = tense=PRESENT, aspect=PROGRESSIVE
    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice present progressive test passed"
    else:
        sys.exit("active voice present progressive test failed")

    active_voice = "\n".join(["has", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # have/indic/pres + _v_/part/past = tense=PRESENT, aspect=PERFECTIVE
    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice present perfective test passed"
    else:
        sys.exit("active voice present perfective test failed")

    active_voice = "\n".join(["has", "been", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    # have/indic/pres + be/part/past + _v_/gerund/pres = tense=PRESENT, aspect=PERFECTIVE_PROGRESSIVE
    if True in [re.search("^have\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "active voice present perfective_progressive test passed"
    else:
        sys.exit("active voice present perfective_progressive test failed")

    active_voice = "\n".join(["was", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # be/indic/past + _v_/gerund/pres = tense=PAST, aspect=PROGRESSIVE
    if True in [re.search("^be\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice past progressive test passed"
    else:
        sys.exit("active voice past progressive test failed")

    active_voice = "\n".join(["had", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # have/indic/past + _v_/part/past = tense=PAST, aspect=PERFECTIVE
    if True in [re.search("^have\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice past perfective test passed"
    else:
        sys.exit("active voice past perfective test failed")

    # have/indic/past + be/part/past + _v_/gerund/pres = tense=PAST, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["had", "been", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^have\+.*\+indic\+past",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+part\+past",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "active voice past perfective_progressive test passed"
    else:
        sys.exit("active voice past perfective_progressive test failed")

    active_voice = "\n".join(["will", "teach"])

    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # will/indic/pres + _v_/infin/pres = tense=FUTURE, aspect=NONE
    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice future none  passed"
    else:
        sys.exit("active voice future none test failed")

    active_voice = "\n".join(["will", "teach"])

    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]

    # will/indic/pres + _v_/indic/pres = tense=FUTURE, aspect=NONE
    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^.+\+v\+.*indic\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')]:
        print "active voice future none  passed"
    else:
        sys.exit("active voice future none test failed")

    active_voice = "\n".join(["is", "going", "to", "teach"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    # be/indic/pres + go/gerund/pres + to + _v_/infin/pres = tense=FUTURE, aspect=NONE
    if True in [re.search("^be\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^go\+.*\+gerund\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^to\+.*",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*infin\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "active voice future none test passed"
    else:
        sys.exit("active voice future none test failed")


    # will/indic/pres + be/infin/pres + _v_/gerund/pres = tense=FUTURE, aspect=PROGRESSIVE
    active_voice = "\n".join(["will", "be", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^be\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "active voice future progressive test passed"
    else:
        sys.exit("active voice future progressive test failed")

    # be/indic/pres + go/gerund/pres + to + be/infin/pred + _v_/infin/pres = tense=FUTURE, aspect=PROGRESSIVE
    active_voice = "\n".join(["is", "going", "to", "be", "teaching"])
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
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok5.split(' ')]:
        print "active voice future progressive test passed"
    else:
        sys.exit("active voice future progressive test failed")

    # will/indic/pres + have/infin/pres + _v_/part/past = tense=FUTURE, aspect=PERFECTIVE
    active_voice = "\n".join(["will", "have", "taught"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^.+\+v\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')]:
        print "active voice future perfective test passed"
    else:
        sys.exit("active voice future perfective test failed")


    # will/indic/pres + have/infin/pres + be/part/past +  _v_/gerund/pres = tense=FUTURE, aspect=PERFECTIVE_PROGRESSIVE
    active_voice = "\n".join(["will", "have", "been", "teaching"])
    active_voice_output = morpho_pro.process(active_voice, base_filename="active_voice_test", overwrite=True)

    active_voice_morphology_tok1 = active_voice_output[0][0]["morphology_morpho"][0]
    active_voice_morphology_tok2 = active_voice_output[0][1]["morphology_morpho"][0]
    active_voice_morphology_tok3 = active_voice_output[0][2]["morphology_morpho"][0]
    active_voice_morphology_tok4 = active_voice_output[0][3]["morphology_morpho"][0]

    if True in [re.search("^will\+.*\+indic\+pres",m) != None for m in active_voice_morphology_tok1.split(' ')] and\
       True in [re.search("^have\+.*\+infin\+pres",m) != None for m in active_voice_morphology_tok2.split(' ')] and\
       True in [re.search("^be\+.*part\+past",m) != None for m in active_voice_morphology_tok3.split(' ')] and\
       True in [re.search("^.+\+v\+.*gerund\+pres",m) != None for m in active_voice_morphology_tok4.split(' ')]:
        print "active voice future perfective_progressive test passed"
    else:
        sys.exit("active voice future perfective_progressive test failed")

    return


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

