
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


def get_constituency_trees(naf_tagged_doc):

    trees = {}

    sentence_num = 1

    for tree in _get_constituency_tree_elements(naf_tagged_doc):

        trees[sentence_num]= ConstituencyTree(tree)
        sentence_num += 1

    return trees

def _get_constituency_tree_elements(naf_tagged_doc):

    constituency_element = _get_constituency_element(naf_tagged_doc)

    tree_elements = []

    for e in constituency_element:

        if e.tag == "tree":
            tree_elements.append(e)

    return tree_elements


def _create_edge(constituency_parent, constituency_child):

    """ will update respective fields in each node given as input """
    constituency_parent.set_child(constituency_child)
    constituency_child.set_parent(constituency_parent)

class ConstituencyNode(object):

     # TODO: refactor this.

    def __init__(self, xml_node):

        self.parent_node = None
        self.child_node = None

        self.node_id = xml_node.attrib["id"]
        self.target_id = None

        if 'label' in xml_node.attrib:
            self.terminal = False

            # non terminal nodes are given labels
            self.label = xml_node.attrib['label']

        else:

            # node is a terminal node
            self.terminal = True
            self.label = None

            i = 1

            # terminal node
            for span in xml_node:
                for target in span:
                    # TODO: assumption
                    # I am assuming there is only one target. if i'm wrong all the code
                    # i have writen is wrong and it will be good to know this in the future.
                    assert( i < 2 )
                    self.target_id = target.attrib["id"].replace('t', 'w')
                    i += 1

    def is_terminal_node(self):

        return self.terminal

    def get_target_id(self):

        return self.target_id

    def get_id(self):

        return self.node_id

    def get_label(self):

        return self.label

    """
    def __repr__(self):

        # for debugging purposes
        return "terminal?: {}, id: {}, label: {}, target id: {}\n".format(self.is_terminal_node(), self.get_id(), self.get_label(), self.get_target_ids())
    """

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

    def is_child(self):

        return self.parent_node is not None

    def is_root(self):

        return self.parent_node is None

    def get_parent(self):

        return self.parent_node

class ConstituencyTree(object):


    def __init__(self, xml_constituency_tree_element):

        self.terminal_nodes = self.process_constituency_tree_element(xml_constituency_tree_element)


    def process_constituency_tree_element(self, xml_constituency_tree_element):
        """ generates a tree structure to determine the categories each token belongs in. """

        #print "called create_constituency_nodes"
        constituency_nodes = {}
        terminal_nodes = {}

        # used to assert if there can be same target id in different nodes
        target_ids_seen = set()

        """ the elements within the xml tree element is sequential, i think. just proces them in order """
        for element in xml_constituency_tree_element:

            if element.tag == 'nt' or element.tag == 't':
                node = ConstituencyNode( element )
                constituency_nodes[node.get_id()] = node

                if node.is_terminal_node():
                    terminal_nodes[node.get_target_id()] = node
                    if node.get_target_id() in target_ids_seen:
                        # TODO: im making an assumption. if this comes back to bite me then i need to redo this.
                        exit("error: target already seen...")
                    else:
                        target_ids_seen.add(node.get_target_id())

            elif element.tag == 'edge':
                #print "TODO: handle edge"

                parent_id = element.attrib["to"]
                child_id  = element.attrib["from"]

                parent_node = constituency_nodes[parent_id]
                child_node  = constituency_nodes[child_id]

                # update fields of each node to create edge.
                _create_edge(parent_node, child_node)

        return terminal_nodes


    def get_phrase_memberships(self, node_id):

        assert( node_id in self.terminal_nodes )

        terminal_node = self.terminal_nodes[node_id]

        level = 0

        grammar_category = {}

        # skip first node, terminal node, no labels available
        node = terminal_node.get_parent()

        # a terminal node should always have a parent.
        assert( node is not None )

        # want to get labels of all non terminal nodes.
        while node.is_root() is False:
            grammar_category[level] = node.get_label()
            level += 1
            node = node.get_parent()

        return grammar_category

if __name__ == "__main__":
    pass
# EOF

