
import os

def valid_path(n_path):

    if os.path.isfile(n_path) is False:
        # TODO: throw an error
        exit("ERROR: Invalid file path.")

#EOF

