
class Features(object):

    def __init__(self):
        print "called Features constructor"

    def get_features_vect(self):

        vectors = []

        # dummy features for now.
        for line in self.tokenized_data:

            for token in line:

                feats = self.extract_features(token)

                vectors.append([feats])

        return vectors

    def extract_features(self, token):

        # Just token text for now: pending preprocessed data
        features = {}
        features["word"] = token
        features["length"] = len(token)

        return features

if __name__ == "__main__":
    print "nothing to do"

