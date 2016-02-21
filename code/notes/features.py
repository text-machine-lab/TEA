""" TODO: fix features below, they don't work. i simply just moved them into a different file.
"""

# these were extracted from the TimeBank corpus, and have been hardcoded here for convenience
temporal_signals = [['in'],      ['on'],                  ['after'],       ['since'],
                    ['until'],   ['in', 'advance', 'of'], ['before'],      ['to'],
                    ['at'],      ['during'],              ['ahead', 'of'], ['of'], ['by'],
                    ['between'], ['as', 'of'],            ['from'],        ['as', 'early', 'as'],
                    ['for'],     ['around'],              ['over'],        ['prior', 'to'],
                    ['when'],    ['should'],              ['within'],      ['while']]


def get_event_features(self):

    return self.get_iob_features("EVENT")

def get_timex_features(self):

    return self.get_iob_features("TIMEX3")


def get_iob_features(self, token_type):

    """ returns featurized representation of events and timexes """

    vectors = []

    for line in self.pre_processed_text:

        for token in self.pre_processed_text[line]:

            token_features = self.get_features_for_token(token, token_type)
            vectors.append(token_features)

    return vectors

def get_features_for_token(self, token, token_type):
     """ get the features for given token

         token: a dictionary with various information
     """

     features = {}

     # TODO: need to change feature set for each of these.
     if token_type == "TIMEX3":

         features.update(self.get_text(token))
         features.update(self.get_lemma(token))
         features.update(self.get_pos_tag(token))
         features.update(self.get_ner_features(token))

         features.update(self.get_wordshapes(token))

         # 4-gram
         features.update(self.get_ngram_features(token))

         features.update(self.get_grammar_categories(token))

     elif token_type == "EVENT":

         features.update(self.get_lemma(token))
         features.update(self.get_ner_features(token))
         features.update(self.get_pos_tag(token))

         features.update(self.get_tense(token))
         features.update(self.is_main_verb(token))

         # 4-gram
         features.update(self.get_ngram_features(token))

         features.update(self.get_ngram_label_features(token))

         features.update(self.get_preposition_features(token))

         features.update(self.get_grammar_categories(token))


     else:
         exit("invalid token type")

     return features

def get_grammar_categories(self, token):

    features = {}

    if "grammar_categories" not in token:

        return {("category", "DATE"):1}

    else:

        for key in token["grammar_categories"]:

            features.update({("category", token["grammar_categories"][key]):1})

    return features


def is_main_verb(self, token):

    if "is_main_verb" in token:

        return {"is_main_verb":token["is_main_verb"]}

    else:

        return {"is_main_verb":False}

def get_wordshapes(self, token):

    return wordshapes.getWordShapes(token["token"])


def get_ner_features(self, token):

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


def get_text(self, token):

    if "token" in token:

        return {("text",token["token"]):1}

    else:

        print token

        return {("text", token["value"]):1}


def get_pos_tag(self, token):

    if "pos_tag" in token:

        return {("pos_tag", token["pos_tag"]):1}

    else:

        # creation time.
        return {("pos_tag", "DATE"):1}

def get_lemma(self, token):

    if "pos_tag" in token:

        return {("lemma", token["lemma"]):1}

    else:

        # creation time
        # TODO: make better?
        return {("lemma", "DATE"):1}

def get_ngram_features(self, token):

    features = {}

    features.update(self.get_tokens_to_right(token, span=4))
    features.update(self.get_tokens_to_left(token, span=4))

    return features

def get_ngram_label_features(self, token):

    features=  {}

    features.update(self.get_labels_to_right(token, span=4))
    features.update(self.get_labels_to_left(token, span=4))

    return features


def get_labels_to_right(self, token, span):
    """ get the labeled entities to the right of token

        obtains tokens relative to the right of the current position of token
    """

    # TODO: set none if there are no tokens?

    token_offset = token["token_offset"]
    line = self.get_iob_labels()[token["sentence_num"] - 1]

    assert span > 0, "set to one or more, otherwise it will just get the token itself"

    start = token_offset
    end   = start + 1 + span

    right_labels = line[start:end][1:]

    labels = dict([(("right_label", label["entity_label"]), 1) for i, label in enumerate(right_labels)])

    return labels

def get_tokens_to_right(self, token, span):
    """ get the tokens to the right of token

        obtains labels relative to the right of the current position of token
    """

    # TODO: set none if there are no tokens?

    token_offset = token["token_offset"]
    line = self.pre_processed_text[token["sentence_num"]]

    # make sure we got the right token
    assert  line[token_offset] == token
    assert span > 0, "set to one or more, otherwise it will just get the token itself"

    start = token_offset
    end   = start + 1 + span

    right_tokens = line[start:end][1:]

    tokens = dict([(("right_token", token["token"]), 1) for i, token in enumerate(right_tokens)])

    return tokens

def get_labels_to_left(self, token, span):
    """ get the labels to the left of token

        obtains labels relative to the left of the current position of token
    """

    # TODO: set none if there are no tokens?
    token_offset = token["token_offset"]
    line = self.get_iob_labels()[token["sentence_num"] - 1]

    assert span > 0, "set to one or more, otherwise it will just get the token itself"

    start   = token_offset - span
    end     = token_offset

    left_labels = line[start:end]
    labels = dict([(("left_label", label["entity_label"]), 1) for i, label in enumerate(left_labels)])

    return labels

def get_tokens_to_left(self, token, span):
    """ get the tokens to the left of token

        obtains tokens relative to the left of the current position of token
    """

    # TODO: set none if there are no tokens?
    token_offset = token["token_offset"]
    line = self.pre_processed_text[token["sentence_num"]]

    # make sure we got the right token
    assert  line[token_offset] == token
    assert span > 0, "set to one or more, otherwise it will just get the token itself"

    start   = token_offset - span
    end     = token_offset

    left_tokens = line[start:end]
    tokens = dict([(("left_token", token["token"]), 1) for i, token in enumerate(left_tokens)])

    return tokens

 def get_tlink_features(self):

     """
     TODO: add more substantial features
     """

     """ returns featurized representation of tlinks """

     print "called get_tlink_features"

     vectors = []

     for relation in self.tlinks:

         vector = {}

         target_entity = relation["target_entity"]
         src_entity = relation["src_entity"]

         vector = self.get_features_for_entity_pair(src_entity, target_entity)

         vectors.append(vector)

     return vectors

def get_features_for_entity_pair(self, src_entity, target_entity):

     """ get the features for an entity pair
     """

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
    ''' extract line number, start token offset, and end token offset from a given entity '''

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


def get_sentence_distance_feature(self, src_entity, target_entity):

    src_line_no = None
    target_line_no = None

    for token in src_entity:

        if src_line_no is None:

            if "sentence_num" in token:
                src_line_no = token["sentence_num"]
            else:
                # creation time is not in a sentence.
                return {"sent_distance":'None'}

        else:

            assert token["sentence_num"] == src_line_no

    for token in target_entity:

        if target_line_no is None:

            if "sentence_num" in token:
                target_line_no = token["sentence_num"]
            else:
                # creation time is not in a sentence.
                return {"sent_distance":'None'}

        else:

            assert token["sentence_num"] == target_line_no

    return {"sent_distance":src_line_no - target_line_no}


def get_discourse_connectives_features(self, src_entity, target_entity):
     ''' return tokens of temporal discourse connectives and their distance from each entity, if connective exist and entities are on the same line.'''

     # if both entities are not events, return
     if self.token_entity_type_feature(src_entity[0])["entity_type"] != "EVENT" or self.token_entity_type_feature(target_entity[0])["entity_type"] != "EVENT":
        return {}

     # extract relevent attributes from entities
     src_position = self.get_entity_position(src_entity)
     src_line_no = src_position["line_no"]
     src_start_offset = src_position["start_offset"]
     src_end_offset = src_position["end_offset"]

     target_position = self.get_entity_position(target_entity)
     target_line_no = target_position["line_no"]
     target_start_offset = target_position["start_offset"]
     target_end_offset = target_position["end_offset"]

     # connectives are only obtained for single sentences, and connot be processed for pairs that cross sentence boundaries
     if src_line_no != target_line_no or src_line_no is None or target_line_no is None:
        return {}

     # get discourse connectives
     connectives = self.get_discourse_connectives(src_line_no)

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


def get_discourse_connectives(self, line_no):

    constituency_tree = self.sentence_features[line_no]['constituency_tree']

    connectives = get_temporal_discourse_connectives(constituency_tree)

    return connectives

def get_temporal_signal_features(self, src_entity, target_entity):

    # get position information for both entities
    src_position = self.get_entity_position(src_entity)
    src_line_no = src_position["line_no"]
    src_start_offset = src_position["start_offset"]
    src_end_offset = src_position["end_offset"]

    target_position = self.get_entity_position(target_entity)
    target_line_no = target_position["line_no"]
    target_start_offset = target_position["start_offset"]
    target_end_offset = target_position["end_offset"]

    # signals are currently only examined for pairs in the same sentence
    if src_line_no != target_line_no or src_line_no is None or target_line_no is None:
        return {}

    # get signals in sentence
    signals = self.get_temporal_signals_in_sentence(src_line_no)
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

def get_temporal_signals_in_sentence(self, line_no):

    # get sentence in question
    sentence = self.pre_processed_text[line_no]
    signals = []

    # for every token, see if it is in every signal
    for i, token in enumerate(sentence):
        for signal in self.temporal_signals:
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


def get_text_features(self, entity):

    """
        gets the features for entity:

            pos
            text of tokens
            lemma of tokens
            text chunk of entity

    """

    features = {}
    tokens = []

    for i, token in enumerate(entity):

        if "lemma" in token:
            features.update({("lemma", token["lemma"]):1})
        else:
            features.update({("lemma", "DATE"):1})

        if "token" in token:
            tokens.append(token["token"])
            features.update({("text", token["token"]):1})
        else:
            tokens.append(token["value"])
            features.update({("text", token["value"]):1})

        if "pos_tag" in token:
            features.update({("pos", token["pos_tag"]):1})
        else:
            features.update({("pos", "DATE"):1})

    features.update({("chunk"," ".join(tokens)):1})

    return features


def get_same_pos_tag_feature(self, src_entity, target_entity):

    src_pos_tags = []
    target_pos_tags = []

    for token in src_entity:

        if "pos_tag" in token:

            src_pos_tags.append(token["pos_tag"])

        else:

            src_pos_tags.append("DATE")

    for token in target_entity:

        if "pos_tag" in token:

            target_pos_tags.append(token["pos_tag"])

        else:

            target_pos_tags.append("DATE")

    return {"same_pos_tags":src_pos_tags == target_pos_tags}


def get_label_features(self, entity):

    features = {}

    for i, token in enumerate(entity):

        features.update({("label_type", self.token_label_feature(token)["entity_label"]):1})

    return features


def get_entity_type_features(self, entity):

    """ for some entity, get the entity type labeling for tokens in that entity
    """

    features = {}

    for i, token in enumerate(entity):

        features.update({("entity_type", self.token_entity_type_feature(token)["entity_type"]):1})

    return features

def doc_creation_time_in_pair(self, src_entity, target_entity):

    if 'functionInDocument' in src_entity[0]:

        if src_entity[0]['functionInDocument'] == 'CREATION_TIME':

            return {"doctimeinpair":1}

    if 'functionInDocument' in target_entity[0]:

        if target_entity[0]['functionInDocument'] == 'CREATION_TIME':

            return {"doctimeinpair":1}


    return {"doctimeinpair":0}


def get_num_of_entities_between_tokens(self, src_entity, target_entity  ):

    """ the two tokens either occur on the same sentence or token2 occurs on the next sentence, this is because
        of the way we filter our tlink pairs
    """

    iob_labels = self.get_iob_labels()

    count = 0

    # doctime does not have a position within the text.
    if "sentence_num" not in src_entity[0] or "sentence_num" not in target_entity[0]:
        return {"entity_distance":-1}

    if src_entity[-1]["sentence_num"] != target_entity[-1]["sentence_num"]:

        src_sentence = src_entity[-1]["sentence_num"] - 1
        src_token_offset = src_entity[-1]["token_offset"]

        chunk1 = iob_labels[src_sentence][src_token_offset:]

        target_sentence = target_entity[-1]["sentence_num"] - 1
        target_token_offset = target_entity[-1]["token_offset"]

        chunk2 = iob_labels[target_sentence][:target_token_offset+1]

        for label in chunk1 + chunk2:

            if label["entity_label"] != 'O':
                count += 1

    else:

        sentence_num = src_entity[-1]["sentence_num"] - 1

        src_token_offset = src_entity[-1]["token_offset"]
        target_token_offset = target_entity[-1]["token_offset"]

        sentence_labels = iob_labels[sentence_num]

        start = src_token_offset if src_token_offset < target_token_offset else target_token_offset

        end   = target_token_offset if target_token_offset > src_token_offset else src_token_offset

        for label in sentence_labels[start:end+1]:

            if label["entity_label"] != 'O':
                count += 1


    return {"entity_distance": count}


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

    """ for some token, get the entity type (EVENT or TIMEX3) for it
    """

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




