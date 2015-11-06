
import subprocess
import os
import re
import sys

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities
import news_reader

def _get_constituency_element(naf_tagged_doc):

   xml_root = xml_utilities.get_root_from_str(naf_tagged_doc)

   constituency_element = None

   for e in xml_root:
       if e.tag == "constituency":
           constituency_element = e
           break

   return constituency_element


def _get_constituency_tree_element(naf_tagged_doc):

    constituency_element = _get_constituency_element(naf_tagged_doc)

    tree_element = None

    for e in constituency_element:

        if e.tag == "tree":
            tree_element = e
            break

    return tree_element


def _create_edge(constituency_parent, constituency_child):

    """ will update respective fields in each node given as input """
    constituency_parent.set_child(constituency_child)
    constituency_child.set_parent(constituency_parent)

class ConstituencyNode(object):

     # TODO: refactor this.

    def __init__(self, xml_node):
        print "constructor"

        self.parent_node = None
        self.child_node = None

        self.node_id = xml_node.attrib["id"]
        self.target_ids = None

        if 'label' in xml_node.attrib:
            self.terminal = False

            # non terminal nodes are given labels
            self.label = xml_node.attrib['label']

        else:

            # node is a terminal node
            self.terminal = True
            self.label = None

            # TODO: not really sure the structure of the span element.
            self.target_ids = []

            # terminal node
            for span in xml_node:
                for target in span:
                    self.target_ids.append(target.attrib["id"].replace('t', 'w'))


    def is_terminal_node(self):

        return self.terminal

    def get_target_ids(self):

        return self.target_ids

    def get_id(self):

        return self.node_id

    def get_label(self):

        return self.label

    def __repr__(self):

        # for debugging purposes
        return "terminal?: {}, id: {}, label: {}, target id: {}\n".format(self.is_terminal_node(), self.get_id(), self.get_label(), self.get_target_ids())

    def set_child(self, node):
        # can have multiple parents
        if self.child_node is None:
            self.child_node = [node]
        else:
            self.child_node.insert(0, node)

    def set_parent(self, node):
        # only one parent
        if self.parent_node is None:
            self.parent_node = node
        else:
            exit( "more than one parent? something bad has happened" )

class ConstituencyTree(object):


    def __init__(self, naf_tagged_doc):

        xml_constituency_tree_element = _get_constituency_tree_element(naf_tagged_doc)
        self.tree = self.process_constituency_tree_element(xml_constituency_tree_element)


    def process_constituency_tree_element(self, xml_constituency_tree_element):
        """ generates and connects proper nodes within ConstituencyTree """

        print "called create_constituency_nodes"
        constituency_nodes = {}

        root = None

        """ the elements within the xml tree element is sequential, i think. just proces them in order """
        for element in xml_constituency_tree_element:

            if element.tag == 'nt' or element.tag == 't':
                node = ConstituencyNode( element )
                constituency_nodes[node.get_id()] = node

                if node.label == "TOP":
                    root = node

            elif element.tag == 'edge':
                #print "TODO: handle edge"

                parent_id = element.attrib["to"]
                child_id  = element.attrib["from"]

                parent_node = constituency_nodes[parent_id]
                child_node  = constituency_nodes[child_id]

                # update fields of each node to create edge.
                _create_edge(parent_node, child_node)

        return root

if __name__ == "__main__":
    pass
# EOF

