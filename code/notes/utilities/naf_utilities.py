
import xml_utilities

def _get_entities_element(ixa_tok_output):

   xml_root = xml_utilities.get_root_from_str(ixa_tok_output)

   terms_element = None

   for e in xml_root:
       if e.tag == "entities":
           terms_element = e
           break

   return terms_element

def _get_entity_elements(ixa_tok_output):

    elements = _get_entities_element(ixa_tok_output)

    return list(elements)

def _get_references_elements(ixa_tok_output):

    entity_elements = _get_entity_elements(ixa_tok_output)

    def get_ref(entity_element):

        assert len(entity_element) == 1

        return list(entity_element)[0]

    return map(get_ref, entity_elements)

def _get_span_elements(ixa_tok_output):

    references_elements = _get_references_elements(ixa_tok_output)

    def get_span(references_element):

        assert len(references_element) == 1

        return list(references_element)[0]

    return map(get_span, references_elements)

def _get_ner_labels(ixa_tok_output):

    entities = _get_entity_elements(ixa_tok_output)

    ner_span_elements = _get_span_elements(ixa_tok_output)

    assert len(entities) == len(ner_span_elements)

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

    for entity, ids in zip(entities, clustered_target_ids):

        labels = extract_entity_attributes(entity)
        labels.update({"target_ids":ids})

        mappings.append(labels)

    return mappings

def _get_terms_element(ixa_tok_output):

   xml_root = xml_utilities.get_root_from_str(ixa_tok_output)

   terms_element = None

   for e in xml_root:
       if e.tag == "terms":
           terms_element = e
           break

   return terms_element

def _get_naf_terms(ixa_tok_output):

    terms_element = _get_terms_element(ixa_tok_output)

    return list(terms_element)

