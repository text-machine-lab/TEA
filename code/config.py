import os

TEA_HOME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# inspired by cliner...
def env_paths():

    paths = open(os.path.join(TEA_HOME_DIR, "config.txt"), "rb").read()
    paths = paths.strip(b'\n')
    paths = [line.split() for line in paths.split(b'\n')]

    env_paths = {}

    for line in paths:
        path = None if line[1] == 'None' else line[1]

        if line[0] == "PY4J_DIR_PATH":
            if path is not None and os.path.isdir(path) is False:
                raise Exception("ERROR: PY4J_DIR_PATH directory is invalid")

        env_paths[line[0]] = path

    return env_paths

if __name__ == "__main__":

    print(env_paths())

