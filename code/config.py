import os
import sys

TEA_HOME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# sys.path.append(os.path.join(TEA_HOME_DIR, '..', 'ntm_keras'))

# inspired by cliner...
def env_paths():

    paths = open(os.path.join(TEA_HOME_DIR, "config.txt"), "rb").read()
    paths = paths.strip('\n')
    paths = [line.split() for line in paths.split('\n')]

    env_paths = {}
    env_paths['ntm_dir'] = os.path.join(TEA_HOME_DIR, '..', 'ntm_keras')

    for line in paths:
        path = None if line[1] == 'None' else line[1]

        if line[0] == "PY4J_DIR_PATH":
            if path is not None and os.path.isdir(path) is False:
                sys.exit("ERROR: PY4J_DIR_PATH needs to be directory containing proper .jar file")

        if line[0] == "MORPHO_DIR_PATH":
            if path is not None and os.path.isdir(path) is False:
                sys.exit("ERROR: MORPHO_DIR_PATH needs to be root directory of morphopro tool")

        env_paths[line[0]] = path

    return env_paths

if __name__ == "__main__":

    print env_paths()

