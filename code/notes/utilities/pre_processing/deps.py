
import copy
import subprocess
import os
import re
import sys

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities
import timeml_utilities

class DependencyPath(object):

    def __init__(self, ixa_tok_output):

        self.first_last_to_path = DependencyPath.get_deps_path(ixa_tok_output)

    def get_path(self, start, end):

        if (start, end) in self.first_last_to_path:

            return self.first_last_to_path[(start,end)]

        else:

            return ([],[])

    @staticmethod
    def get_deps_path(ixa_tok_output):

        xml_root = xml_utilities.get_root_from_str(ixa_tok_output)

        deps_element = None

        # get deps xml element
        for e in xml_root:
            if e.tag == "deps":
                deps_element = e
                break

        tmp_paths = []

        # get deps
        for d in deps_element:

            # d = ( [ list of tokens representing path ], [ list of rfuncs ] )

            tmp_paths.append(([d.attrib["from"],
                        d.attrib["to"]],
                        [d.attrib["rfunc"]]))

        paths = []

        i = 0

        for tmp_path in tmp_paths:

            """
                iterate over all the dependencies
                concatenate and merge them as necessary
            """

            # going to potentially modify it and add to list of deps.
            path = copy.deepcopy(tmp_path)

            # by default add to paths
            paths.append(path)

            paths_to_add = []

            for p in paths:

                # want to add to a path
                # example:
                #       20 -> 21
                #       21 - > 23
                #       want to add 20 -> 21 -> 31

                # refer to same path add
                if p == path:
                    continue

                # append to path
                elif p[0][-1] == path[0][0]:
                    paths_to_add.append((p[0] + path[0][1:], p[1] + path[1]))
                    pass

                # prepend to path
                elif p[0][0] == path[0][-1]:
                    paths_to_add.append((path[0] + p[0][1:], path[1] + p[1]))
                    pass

                # do nothing
                else:
                    pass

            paths += paths_to_add
            paths_to_add = []

        first_last_to_path = {}

        for p in paths:

            first_last = (p[0][0], p[0][-1])

            if first_last in first_last_to_path:

                exit("error line 91 in deps.py")

            first_last_to_path[first_last] = p

        return first_last_to_path

if __name__ == "__main__":

    print DependencyPath(open("deps.txt", "rb").read()).get_path("t25", "t5")

    pass
# EOF

