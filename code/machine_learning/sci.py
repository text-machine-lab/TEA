from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score

from multiprocessing import cpu_count

import numpy as np

import math

def train( featsDict, Y, do_grid=False ):

    #vectorize dictionary data
    vec = DictVectorizer()
    X = vec.fit_transform(featsDict).toarray()

    # Search space
    C_range     = 10.0 ** np.arange( -5, 9 )
    gamma_range = 10.0 ** np.arange( -5, 9 )

    clf = None

    # Grid search?
    if do_grid:

        print "training model [GRID SEARCH ON]"

        estimates = SVC(kernel='linear')
        parameters = [ {'C':C_range } ]

        # Find best classifier
        clf = GridSearchCV(estimates,
                           parameters,
                           score_func = f1_score,
                           # one job per cpu available.
                           n_jobs = int(math.ceil(cpu_count() / 2.0)),
                           pre_dispatch=1)
        clf.fit(X, Y)

    else:

        print "training model [GRID SEARCH OFF]"

        clf = SVC(kernel='linear')
        clf.fit(X, Y)

    return clf, vec

