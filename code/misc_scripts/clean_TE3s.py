import xml.etree.ElementTree as ET
import glob
import argparse

from code.notes.utilities.xml_utilities import write_root_to_file
from code.notes.utilities.timeml_utilities import get_stripped_root

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("TE3_dir",
                        type=str,
                        nargs=1)

    args = parser.parse_args()

    files = glob.glob(args.TE3_dir[0] + '/*')

    files_to_clean = []
    good_files = []
    for f in files:
        if "E3input" in f:
            files_to_clean.append(f)

    print "\nThere are ", len(files_to_clean), " files to be cleaned.\n"

    for f in files_to_clean:
        print "\nProcessing: ", f
        root = get_stripped_root(f)
        write_root_to_file(root, f)

if __name__ == "__main__":
    main()
