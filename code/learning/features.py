""" Timex patterns and pattern matching code taken from https://bitbucket.org/qwaider/textpro-en/src/fd203d54b184f2b1a5d43c7dee5aad131dce9280/FBK-timepro/resources/english-rules-temporal-markers?at=master
"""

import copy
from code.notes.utilities.add_discourse import get_temporal_discourse_connectives
import re
import re_timex_patterns
import nominalization

# these were extracted from the TimeBank corpus, and have been hardcoded here for convenience
temporal_signals = [['in'],      ['on'],                  ['after'],       ['since'],
                    ['until'],   ['in', 'advance', 'of'], ['before'],      ['to'],
                    ['at'],      ['during'],              ['ahead', 'of'], ['of'], ['by'],
                    ['between'], ['as', 'of'],            ['from'],        ['as', 'early', 'as'],
                    ['for'],     ['around'],              ['over'],        ['prior', 'to'],
                    ['when'],    ['should'],              ['within'],      ['while']]


def get_window_features(index, features_in_sentence):

    # TODO: figure out how I am going to get this to be efficient.
    # TODO: should i add null values if there are no left or right features?

    window = 4

    window_features = {}

    left_start    = max(index - window, 0)
    left_end      = index

    right_start   = index + 1
    right_end     = index + window + 1

    left_features  = {("left_{}_{}".format(i, key), f[key]):True for i, f in enumerate(features_in_sentence[left_start:left_end]) for key in f}
    right_features = {("right_{}_{}".format(i, key), f[key]):True for i, f in enumerate(features_in_sentence[right_start:right_end]) for key in f}

    window_features.update(left_features)
    window_features.update(right_features)

    return window_features


def get_preceding_labels(token, labels):

    """Get the IOB labeling of the previous 4 tokens.
    """

    features = {}
    preceding_labels = []

    window = 4

    start = max(token["token_offset"] - window, 0)
    end   = token["token_offset"]

    if len(labels) > 0:

        # iob label for token
        preceding_labels = [l["entity_label"] for l in labels[token["sentence_num"] - 1][start:end]]

        # normalize the features
        if token["token_offset"] - window < 0:
            preceding_labels = (([None] * abs(token["token_offset"] - window)) + preceding_labels)

    else:
        preceding_labels = [None]*4

    assert len(preceding_labels) == 4, "preceding _labels: {}".format(preceding_labels)

    for i, l in enumerate(preceding_labels):
        features[("preceding_labels_{}".format(i), l)] = True

    return features

def extract_tlink_features(note):
    tlink_features = []

    for tlink_pair in note.get_tlinked_entities():

        pair_features = {}

        # entities. timexes may be composed of multiple tokens
        target_tokens = tlink_pair['target_entity']
        source_tokens = tlink_pair['src_entity']

        tokens = []
        target_pos_tags = set()

        for i, target_token in enumerate(target_tokens):

            text_feature = get_text(target_token,"target_token_{}".format(i))
            tokens.append(text_feature.keys()[0][1])
            pair_features.update(text_feature)
            pair_features.update(get_lemma(target_token,"target_lemma_{}".format(i)))
            target_pos_feature = get_pos_tag(target_token,"target_pos_{}".format(i))
            target_pos_tags.add(target_pos_feature.keys()[0][1])
            pair_features.update(target_pos_feature)

            pass

        chunk = " ".join(tokens)
        pair_features.update({("target_chunk",chunk):1})

        tokens = []
        src_pos_tags = set()

        for i, source_token in enumerate(source_tokens):

            text_feature = get_text(source_token,"src_token_{}".format(i))
            tokens.append(text_feature.keys()[0][1])
            pair_features.update(text_feature)
            pair_features.update(get_lemma(source_token,"src_lemma_{}".format(i)))
            src_pos_feature = get_pos_tag(target_token,"src_pos_{}".format(i))
            src_pos_tags.add(src_pos_feature.keys()[0][1])
            pair_features.update(src_pos_feature)

            pass

        chunk = " ".join(tokens)
        pair_features.update({("src_chunk",chunk):1})
        pair_features.update({("same_pos", None):(src_pos_tags == target_pos_tags)})
        pair_features.update(get_sentence_distance(source_tokens, target_tokens))
        pair_features.update(get_num_inbetween_entities(source_tokens,target_tokens, note))
        pair_features.update(doc_creation_time_in_pair(source_tokens,target_tokens))
        pair_features.update(get_discourse_connectives_features(source_tokens,target_tokens, note))
        pair_features.update(get_temporal_signal_features(source_tokens,target_tokens,note))

        tlink_features.append(pair_features)

    return tlink_features

def _entity_type_entity(entity, note):
    # doc time
    if 'functionInDocument' in entity[0]:
        return "TIMEX"
    return note.get_labels()[entity[0]["sentence_num"]-1][entity[0]["token_offset"]]["entity_type"]

def get_discourse_connectives_features(src_entity, target_entity, note):
    """
    return tokens of temporal discourse connectives and their distance from each entity, if connective exist and entities are on the same line.
    """

    # if both entities are not events, return
    if _entity_type_entity(src_entity, note) != "EVENT" or _entity_type_entity(target_entity,note) != "EVENT":
        return {}

    # extract relevent attributes from entities
    src_line_no      = src_entity[0]["sentence_num"] - 1
    src_start_offset = src_entity[0]["token_offset"]
    src_end_offset   = src_entity[-1]["token_offset"]

    target_line_no      = target_entity[0]["sentence_num"] - 1
    target_start_offset = target_entity[0]["token_offset"]
    target_end_offset   = target_entity[-1]["token_offset"]

    # connectives are only obtained for single sentences, and connot be processed for pairs that cross sentence boundaries
    if src_line_no != target_line_no or src_line_no is None or target_line_no is None:
        return {}

    # get discourse connectives
    connectives = get_discourse_connectives(src_line_no, note)

    connective_id = None
    connective_tokens = ''
    connective_is_between_entities = False
    connective_before_entities = False
    connective_after_entities = False
    src_before_target = False

    for connective_token in connectives:

        # find connective position relative to entities
        if src_start_offset < connective_token["token_offset"] and connective_token["token_offset"] < target_end_offset:
            connective_is_between_entities = True
            src_before_target = True
        elif target_start_offset < connective_token["token_offset"] and connective_token["token_offset"] < src_end_offset:
            connective_is_between_entities = True

        elif src_start_offset < target_start_offset and target_start_offset < connective_token["token_offset"]:
            connective_after_entities = True
            src_before_target = True
        elif target_start_offset < src_start_offset and src_start_offset < connective_token["token_offset"]:
            connective_after_entities = True

        elif connective_token["token_offset"] < src_end_offset and src_start_offset < target_start_offset:
            connective_before_entities = True
            src_before_target = True
        elif connective_token["token_offset"] < target_end_offset and target_start_offset < src_start_offset:
            connective_before_entities = True

        # assuming every sentence will only have one temporal discourse connective. If this isn't the case, it would be nice to know
        if connective_id is not None:
            assert connective_id == connective_token["discourse_id"]

        connective_id = connective_token["discourse_id"]

        # add token to connective
        connective_tokens += connective_token["token"]

    # if no connective was found
    if connective_id is None:
        return {}

    # # sanity check
    # if connective_id is not None:
     # assert connective_before_entities or connective_after_entities or connective_is_between_entities

    # return feature dict
    retval = {("connective_tokens", connective_tokens):1}
    if connective_before_entities:
        retval["connective_before_src"] = 1
        retval["connective_before_target"] = 1
    elif connective_after_entities:
        retval["connective_after_src"] = 1
        retval["connective_after_target"] = 1
    elif connective_is_between_entities and src_before_target:
        retval["connective_after_src"] = 1
        retval["connective_before_target"] = 1
    elif connective_is_between_entities and not src_before_target:
        retval["connective_before_src"] = 1
        retval["connective_after_target"] = 1

    return retval


def get_discourse_connectives(line_no,note):

    constituency_tree = note.get_sentence_features()[line_no+1]['constituency_tree']
    connectives = get_temporal_discourse_connectives(constituency_tree)

    return connectives


def doc_creation_time_in_pair(src_entity, target_entity):

    feature = {("doctimeinpairm",None):0}

    if 'functionInDocument' in src_entity[0]:
        if src_entity[0]['functionInDocument'] == 'CREATION_TIME':
            feature = {("doctimeinpair",None):1}
    if 'functionInDocument' in target_entity[0]:
        if target_entity[0]['functionInDocument'] == 'CREATION_TIME':
            feature = {("doctimeinpair",None):1}

    return feature

def get_num_inbetween_entities(src_entity, target_entity, note):

    """
    possible situations:

        EVENT -> all following EVENTs in same sentence

        EVENT -> all following TIMEX  in same sentence

        TIMEX -> all following EVENTS in same sentence

        main verb EVENTS in sentence -> main verb EVENTS in following sentence, if there is one.
    """

    # start of entity?
    start_of_entity = lambda label: "I_" not in label and label != "O"

    entity_count = 0

    iob_labels = note.get_labels()

    # doctime does not have a position within the text.
    if "sentence_num" not in src_entity[0] or "sentence_num" not in target_entity[0]:
        return {("entity_distance",None):-1}

    # this proj has poorly managed indexing. bad coding practice. SUE ME!
    # get the sentence index of entities
    src_sentence    = src_entity[0]["sentence_num"] - 1
    target_sentence = target_entity[0]["sentence_num"] - 1

    # entities are in adjacent sentences
    if src_sentence != target_sentence:

        # want to get distance between end and start of tokens
        end_src_token      = src_entity[-1]["token_offset"]
        start_target_token = target_entity[0]["token_offset"]

        # get iob labels. concatenate and then find all labels that are not I_ or O
        chunk1 = iob_labels[src_sentence][end_src_token+1:]
        chunk2 = iob_labels[target_sentence][:start_target_token]

        labels = chunk1 + chunk2

        # count all labels with B_ (TIMEX) or not O (EVENT)
        for label in labels:
            if start_of_entity(label):
                entity_count += 1

    # tokens must be in same sentence
    else:

        # we need to check if src or target entity is a EVENT or TIMEX
        # because of the way TEA pairs stuff, I (kevin) always made EVENTS come first within
        # a pairing event if the EVENT came after a TIMEX. I did this because I just did a literal
        # translation from the paper we were following.

        # same sentence.
        sentence_num = src_sentence

        # if end of src comes before start of target just find the number of entities between
        # otherwise if end comes after just take distance between last index of target and first index of src

        end_src_token       = src_entity[-1]["token_offset"]
        start_target_token  = target_entity[0]["token_offset"]

        start_src_token     = src_entity[0]["token_offset"]
        end_target_token    = target_entity[-1]["token_offset"]

        sentence_labels = iob_labels[sentence_num]
        labels          = None

        if end_src_token < start_target_token:
            labels = sentence_labels[end_src_token+1:start_target_token]
        else:
            labels = sentence_labels[end_target_token+1:start_src_token]

        for label in labels:
            if start_of_entity(label):
                entity_count += 1

    return {("entity_distance",None): entity_count}

def get_sentence_distance(src_entity, target_entity):
    """
    Sentence distance (e.g. 0 if e1 and e2 are in the same sentence)
    Since we only consider pairs of entities within same sentence or adjacent
    it must be 0 or 1
    """

    # assuming each entity's tokens are all in the same sentence.

    sentence_dist_feat = {("sent_distance",None):-1}

    # if doctime occurs then there is no distance since it is not in a sentence.
    if 'sentence_num' in src_entity[0] and 'sentence_num' in target_entity[0]:
        src_line_no    = src_entity[0]["sentence_num"]
        target_line_no = target_entity[0]["sentence_num"]

        sentence_dist_feat = {("sent_distance",None):abs(src_line_no - target_line_no)}

    return sentence_dist_feat


def extract_event_feature_set(note, labels, predict=False):
    return extract_iob_features(note, labels, "EVENT", predicting=predict)


def extract_timex_feature_set(note, labels, predict=False):
    return extract_iob_features(note, labels, "TIMEX3", predicting=predict)


def extract_event_class_feature_set(note, labels, eventLabels, predict=False):
    return extract_iob_features(note, labels, "EVENT_CLASS", predicting=predict, eventLabels=eventLabels)


def update_features(token, token_features, labels):
    """ needed when predicting """
    token_features.update(get_preceding_labels(token, labels))


def extract_iob_features(note, labels, feature_set, predicting=False, eventLabels=None):

    """ returns featurized representation of events and timexes """

    features = []

    tokenized_text = note.get_tokenized_text()

    preceding_features = []

    for line in tokenized_text:

        for token in tokenized_text[line]:

            token_features = {}

            # get features specific to a specific label type
            if feature_set == "TIMEX3":
                token_features.update(get_lemma(token))
                token_features.update(get_text(token))
                token_features.update(get_pos_tag(token))
                token_features.update(get_ner_features(token))
                token_features.update(timex_regex_feats(token))
            elif feature_set == "EVENT":
                token_features.update(get_lemma(token))
                token_features.update(get_text(token))
                token_features.update(get_pos_tag(token))
                token_features.update(get_ner_features(token))
                token_features.update(is_nominalization(token))
            elif feature_set == "EVENT_CLASS":
                token_features.update(get_lemma(token))
                token_features.update(get_text(token))
                token_features.update(get_pos_tag(token))
                token_features.update(get_ner_features(token))
                token_features.update(is_main_verb(token))
                token_features.update(is_event(token, eventLabels))
                token_features.update(semantic_roles(token))
                token_features.update(is_nominalization(token))
            else:
                raise Exception("ERROR: invalid feature set")

            feature_copy = copy.deepcopy(token_features)

            # labels are meaningless when this function is called when predicting, don't know the labels yet.
            if not predicting:
                token_features.update(get_preceding_labels(token, labels))

            # get the features of the 4 previous tokens.
            # TODO: might be a problem later on, in terms of performance?
            for i, f in enumerate(preceding_features):
                for key in f:
                    token_features[("preceding_feats_{}_{}".format(i, key[0]), key[1])] = f[key]

            if len(preceding_features) < 4:
                preceding_features.append(feature_copy)
            else:
                preceding_features.pop(0)

            features.append(token_features)

    # get features of following 4 tokens:
    for i, token_features in enumerate(features):
        following = features[i + 1:i + 5]
        for j, f in enumerate(following):
            for key in f:
                if ("preceding_feats_" in key[0]) or ("preceding_labels_" in key[0]):
                    continue

                token_features[("following_{}_{}".format(j, key[0]), key[1])] = f[key]

    return features

def semantic_roles(token):
    feats = {}
    if "semantic_roles" in token:
        for i, role in enumerate(token["semantic_roles"]):
            feats.update({("semantic_role_{}".format(i),role):1})
    return feats

def is_main_verb(token):
    feat = {("main_verb",None):0}
    if "is_main_verb" in token:
        feat = {("main_verb",None):token["is_main_verb"]}
    return feat


def is_event(token, eventLabels):
    return {("is_event", None):(eventLabels[token["sentence_num"]-1][token["token_offset"]]["entity_label"] == "EVENT")}


def get_grammar_categories(token):

    features = {}

    if "grammar_categories" not in token:

        return {("category", "DATE"):1}

    else:

        for key in token["grammar_categories"]:

            features.update({("category", token["grammar_categories"][key]):1})

    return features


def get_wordshapes(self, token):

    return wordshapes.getWordShapes(token["token"])


def get_ner_features(token):

    if "ner_tag" in token:
        return {("ner_tag", token["ner_tag"]):1,
                "in_ne":1,
                ("ne_chunk", token["ne_chunk"]):1}
    # TODO: what problems might arise from labeling tokens as none if no tagging?, we'll find out!
    else:
        return {("ner_tag", 'None'):1,
                "in_ne":0,
                ("ne_chunk", "NULL"):1}

def get_tense(self, token):
    if "tense" in token:

        return {("tense", token["tense"]):1}

    else:

        return {("tense", "PRESENT"):1}


def get_text(token,feat_name="text"):

    if "token" in token:
        return {(feat_name,token["token"]):1}
    else:
        return {(feat_name, token["value"]):1}


def get_pos_tag(token,feat_name="pos_tag"):

    if "pos_tag" in token:
        return {(feat_name, token["pos_tag"]):1}
    else:
        # creation time.
        return {(feat_name, "DATE"):1}

def get_lemma(token,feat_name="lemma"):

    if "pos_tag" in token:

        return {(feat_name, token["lemma"]):True}

    else:

        # creation time
        # TODO: make better?
        return {(feat_name, "DATE"):True}


def get_features_for_entity_pair(self, src_entity, target_entity):

     pair_features = {}

     src_features = {}
     target_features = {}

     src_features.update(self.get_text_features(src_entity))
     target_features.update(self.get_text_features(target_entity))

     src_features.update(self.get_entity_type_features(src_entity))
     target_features.update(self.get_entity_type_features(target_entity))

     src_features.update(self.get_label_features(src_entity))
     target_features.update(self.get_label_features(target_entity))

     src_features.update(self.get_entity_attributes(src_entity))
     target_features.update(self.get_entity_attributes(target_entity))

     pair_features.update(self.get_same_pos_tag_feature(src_entity, target_entity))
     pair_features.update(self.get_sentence_distance_feature(src_entity, target_entity))

     pair_features.update(self.get_num_of_entities_between_tokens(src_entity, target_entity))
     pair_features.update(self.doc_creation_time_in_pair(src_entity, target_entity))

     pair_features.update(self.get_same_attributes(self.get_entity_attributes(src_entity), self.get_entity_attributes(target_entity)))

     pair_features.update(self.get_discourse_connectives_features(src_entity, target_entity))
     pair_features.update(self.get_temporal_signal_features(src_entity, target_entity))

     for key in src_features:

         pair_features[(key[0] + "_src", key[1])] = src_features[key]

     for key in target_features:

         pair_features[(key[0] + "_target", key[1])] = target_features[key]

     return pair_features

def get_same_attributes(self, src_features, target_features):

    # TODO: what happened here???/
    return {"same_attributes":1}

def get_entity_attributes(self, entity):

    features = {}

    for tok in entity:

        if "tense" in tok:

            features.update({("tense", tok["tense"]):1})

        else:

            features.update({("tense", "PRESENT"):1})

        iob_label = None

        if "token_offset" in tok:

            token_offset = tok["token_offset"]
            label = self.get_iob_labels()[tok["sentence_num"] - 1][token_offset]

            iob_label = {("class", label["entity_label"]):1}

        else:

            iob_label = {("class", "O"):1}

        features.update(iob_label)


    return features

def get_entity_position(self, entity):

    line_no = None
    start_offset = None
    end_offset = None

    for token in entity:

        if line_no is None:
            # creation time does not have a sentence number
            if "sentence_num" in token:
                line_no = token["sentence_num"]

        else:
            assert token["sentence_num"] == line_no


        if start_offset is None and end_offset is None:
            if "token_offset" in token:
                start_offset = token["token_offset"]
                end_offset = token["token_offset"]

        else:
            if start_offset > token["token_offset"]:
                start_offset = token["token_offset"]

            if end_offset < token["token_offset"]:
                end_offset = token["token_offset"]

    return {"line_no": line_no, "start_offset": start_offset, "end_offset": end_offset}

def get_preposition_features(self, token):

    features = {}

    if "preposition_tokens" not in token or "semantic_role" not in token:

        return {("preposition_token", "NULL"):1,
                ("semantic_role", "NULL"):1}

    for prep_tok in token["preposition_tokens"]:

        features.update({("preposition_tokens", prep_tok):1})

    for semantic_role in token["semantic_role"]:

        features.update({("semantic_role", semantic_role):1})

    return features


def get_temporal_signal_features(src_entity, target_entity, note):

    # doc time in pair. not same sentence.
    if "sentence_num" not in src_entity[0] or "sentence_num" not in target_entity[0]:
        return {}

    # extract relevent attributes from entities
    src_line_no      = src_entity[0]["sentence_num"]
    src_start_offset = src_entity[0]["token_offset"]
    src_end_offset   = src_entity[-1]["token_offset"]

    target_line_no       = target_entity[0]["sentence_num"]
    target_start_offset = target_entity[0]["token_offset"]
    target_end_offset   = target_entity[-1]["token_offset"]

    # signals are currently only examined for pairs in the same sentence
    if src_line_no != target_line_no or src_line_no is None or target_line_no is None:
        return {}

    # get signals in sentence
    signals = get_temporal_signals_in_sentence(src_line_no, note)
    retval = {}

    # extract positional features for each signal
    for signal in signals:
        signal_text = signal['tokens']
        retval.update({(signal_text + '_signal'):1})
        if signal['end'] < src_start_offset:
            retval.update({(signal_text + "_signal_before_src"): 1})
        if src_end_offset < signal['start']:
            retval.update({(signal_text + "_signal_after_src"): 1})
        if signal['end'] < target_start_offset:
            retval.update({(signal_text + "_signal_before_target"): 1})
        if target_end_offset < signal['start']:
            retval.update({(signal_text + "_signal_after_target"): 1})

    return retval

def get_temporal_signals_in_sentence(line_no, note):

    # get sentence in question
    sentence = note.get_tokenized_text()[line_no]
    signals = []

    # for every token, see if it is in every signal
    for i, token in enumerate(sentence):
        for signal in temporal_signals:
            token_is_signal = True
            signal_text = ""

            # check if the whole signal is present
            signal_end = i
            for j in range(len(signal)):
                if sentence[i+j]['token'] != signal[j]:
                    token_is_signal = False
                    break
                signal_text += signal[j] + ' '
                signal_end = i + j

            # if signal is present, do shit
            if token_is_signal:
                signals.append({"start": i, "end": signal_end, "tokens": signal_text})
                break

    return signals

def get_label_features(self, entity):

    features = {}

    for i, token in enumerate(entity):

        features.update({("label_type", self.token_label_feature(token)["entity_label"]):1})

    return features

def get_entity_type_features(self, entity):

    features = {}

    for i, token in enumerate(entity):

        features.update({("entity_type", self.token_entity_type_feature(token)["entity_type"]):1})

    return features

def token_label_feature(self, token):

    feature = {}
    feature = {}

    line_num = None
    token_offset = None

    label = None

    # TODO: correct this, hacky
    if "functionInDocument" in token:

        if token["functionInDocument"] == 'CREATION_TIME':

            # this is the creation time...
            label = "B_DATE"

            return {"entity_label":label}

    line_num     = token["sentence_num"] - 1
    token_offset = token["token_offset"]

    iob_labels  =  self.get_iob_labels()
    label = iob_labels[line_num][token_offset]["entity_label"]

    assert label not in ['O', None]

    return {"entity_label":label}

def token_entity_type_feature(self, token):

    feature = {}

    line_num = None
    token_offset = None

    entity_type = None

    # TODO: correct this, hacky
    if "functionInDocument" in token:

        if token["functionInDocument"] == 'CREATION_TIME':
            # this is the creation time...
            entity_type = "TIMEX3"

            return {"entity_type":entity_type}

    line_num     = token["sentence_num"] - 1
    token_offset = token["token_offset"]

    iob_labels  =  self.get_iob_labels()

    entity_type = iob_labels[line_num][token_offset]["entity_type"]

    if entity_type not in ["EVENT", "TIMEX3"]:

        print "\n\n"
        print "iob_labels: ", iob_labels
        print "token: ", token
        print "iob_labels[line_num]", iob_labels[line_num]
        print "iob_labels[line_num][token_offset]", iob_labels[line_num][token_offset]

        exit("error...")

    return {"entity_type":entity_type}


def timex_regex_feats(token):

    timex = None

    if "token" in token:
        timex = token["token"]
    else:
        timex = token["value"]

    feats =  {("_YY_",None): 0,
              ("_TIME_", None): 0,
              ("_DURATION_", None): 0,
              ("_NUMBER_", None): 0,
              ("_OTHER_", None): 0,
              ("_UNIT_", None): 0,
              ("_DAY_", None): 0,
              ("_MONTH_", None): 0,
              ("_SEASON_", None): 0,
              ("_ON_", None): 0,
              ("_PD_", None): 0,
              ("_CN_", None): 0,
              ("_AVT_", None): 0,
              ("_NAMES_", None): 0,
              ("_SET_", None): 0,
              ("_", None): 0}

    # contains digits
    if re.search("[0-9]", timex):

        patterns = [re_timex_pattern.yy, re_timex_pattern.time, re_timex_pattern.duration, re_timex_pattern.patternnumber]
        keys = [("_YY_", None), ( "_TIME_", None), ("_DURATION_", None),  ("_NUMBER_", None)]

        for key, pattern in zip(keys, patterns):
            if re.search(pattern, timex):
                feats[key] = 1
                break
        else:
            feats[("_OTHER_", None)] = 1

    # doesn't dontain digits.
    else:

        keys =  [("_UNIT_", None), ("_DAY_", None), ("_MONTH_", None), ("_SEASON_", None), ("_ON_", None),
                 ("_PD_", None), ("_CN_", None), ("_AVT_", None), ("_NAMES_", None), ("_SET_", None)]

        patterns = [re_timex_pattern.unit, re_timex_pattern.day, re_timex_pattern.month, re_timex_pattern.season, re_timex_pattern.ordinal_number,
                    re_timex_pattern.parts_of_the_day, re_timex_pattern.cardinal_number, re_timex_pattern.dverbs, re_timex_pattern.names, re_timex_pattern.set_pattern]

        for key, pattern in zip(keys, patterns):
            if re.search(pattern, timex):
                feats[key] = 1
                break
        else:
            feats[("_", None)] = 1

    return feats

def is_nominalization(token):
    feat = {("is_nominalization",None):0}

    if "token" in token:
        if token["token"] in nominalization.nominalization_list:
            feat = {("is_nominalization",None):1}

    return feat


