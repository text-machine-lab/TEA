
import xml_utilities

def _get_entities_element(naf_tagged_doc):

   xml_root = xml_utilities.get_root_from_str(naf_tagged_doc)

   terms_element = None

   for e in xml_root:
       if e.tag == "entities":
           terms_element = e
           break

   return terms_element

def _get_srl_element(naf_tagged_doc):

    xml_root = xml_utilities.get_root_from_str(naf_tagged_doc)

    srl_element = []

    for e in xml_root:

        if e.tag == "srl":

            srl_element = e

            break

    return list(srl_element)


def _get_predicate_tokens(naf_tagged_doc):

    srl_element = _get_srl_element(naf_tagged_doc)

    tokens = []

    for predicate in srl_element:

        span = []

        # i'm assuming all elements within srl_element are predicates
        assert predicate.tag == "predicate"

        for element in predicate:

            if element.tag == "span":

                span = list(element)

                break

        # im assuming one token per span
        assert len(span) == 1

        token = span[0].attrib["id"]
        token = list(token)
        token[0] = 'w'
        token = "".join(token)

        tokens.append(token)

    return tokens


def _get_ner_labels(naf_tagged_doc):

    entities_element =_get_entities_element(naf_tagged_doc)

    if entities_element is None:

        return {}

    entity_elements = list(entities_element)

    def get_ref(entity_element):
        assert len(entity_element) == 1
        return list(entity_element)[0]

    references_elements =  map(get_ref, entity_elements)

    def get_span(references_element):
        assert len(references_element) == 1
        return list(references_element)[0]

    ner_span_elements =  map(get_span, references_elements)

    assert len(entity_elements) == len(ner_span_elements)

    def span_to_ids(span):

        ids = []

        for target in span:

            target_id = target.attrib["id"]
            target_id = list(target_id)
            target_id[0] = 'w'
            target_id = "".join(target_id)

            ids.append(target_id)

        return ids

    def extract_entity_attributes(ner_entity):

        return {"ne_id":ner_entity.attrib["id"], "ner_tag":ner_entity.attrib["type"]}

    clustered_target_ids = map(span_to_ids,
                               ner_span_elements)

    mappings = []

    for entity, ids in zip(entity_elements, clustered_target_ids):

        labels = extract_entity_attributes(entity)
        labels.update({"target_ids":ids})

        mappings.append(labels)

    return mappings

def _get_terms_element(naf_tagged_doc):

   xml_root = xml_utilities.get_root_from_str(naf_tagged_doc)

   terms_element = None

   for e in xml_root:
       if e.tag == "terms":
           terms_element = e
           break

   return terms_element

def _get_naf_terms(naf_tagged_doc):

    terms_element = _get_terms_element(naf_tagged_doc)

    return list(terms_element)

