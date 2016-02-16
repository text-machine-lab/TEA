import os
import argparse
import glob
from code.notes.utilities.timeml_utilities import get_text_element
import cPickle as cp

def main():
    '''script for extracting temporal signals from TimeML documents with signal annotations'''


    parser = argparse.ArgumentParser()

    parser.add_argument("signal_dir",
                        type=str,
                        nargs=1,
                        help="Directory containing documents annotated with signal information")

    parser.add_argument("output_destination",
                        help="Where to store the list of signals")

    args = parser.parse_args()

    if os.path.isdir(args.signal_dir[0]) is False:
        exit("invalid path to directory containing training data")

    signal_dir = None

    if '/*' != args.signal_dir[0][-2:]:
        signal_dir = args.signal_dir[0] + '/*'

    else:
        signal_dir = args.signal_dir[0]

    files = glob.glob(signal_dir)

    signals = []

    for file in files:
        text_element = get_text_element(file)
        for element in text_element:
            if element.tag == "SIGNAL":
                text = element.text.lower()
                if text in signals:
                    continue
                else:
                    signals += [text]

    print signals

    with open(args.output_destination, "wb") as output:
        cp.dump(signals, output)

if __name__ == "__main__":
    main()