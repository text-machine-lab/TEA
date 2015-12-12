

import xml_utilities


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

    naf_terms = []

    for e in terms_element:
        if e.tag == "term":
            naf_terms.append(e)

    return naf_terms

