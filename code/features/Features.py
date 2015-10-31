
class Features(object):

    def __init__(self):
        print "called Features constructor"


    def get_features_vect(self):
        # TODO: redo this. my changes broke it.

        vectors = []

        # dummy features for now.
        for line in self.tokenized_data:

            for token in line:

                vectors.append([{"dummy":1}])

        return vectors

if __name__ == "__main__":
    print "nothing to do"

