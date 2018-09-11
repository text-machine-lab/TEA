import os

def valid_path(n_path):

    if os.path.isfile(n_path) is False:
        # TODO: throw an error, will be better so I can make custom messages based on the function.
        exit("ERROR: Invalid file path: " + n_path)

if __name__ == "__main__":
    load_sentenizer()

#EOF

