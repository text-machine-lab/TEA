
import os
import cPickle

def load_sentenizer():
    print "called load_sentenizer"

    sentenizer = None

    if 'PUNKT_PATH' in os.environ:

        valid_path(os.environ['PUNKT_PATH'])

        # TODO: make an install script or something
        sentenizer = cPickle.load(open(os.environ['PUNKT_PATH']))

    else:
        print "ERROR: PUNKT_PATH environment variable not set. please set to path of english.pickle file"

    return sentenizer

def valid_path(n_path):

    if os.path.isfile(n_path) is False:
        # TODO: throw an error, will be better so I can make custom messages based on the function.
        exit("ERROR: Invalid file path.")

if __name__ == "__main__":
    load_sentenizer()

#EOF

