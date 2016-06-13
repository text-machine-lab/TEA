from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold

import numpy as np
import math

def train( featsDict, Y, do_grid=False, ovo=False, t="not_set"):
    '''
    train()
        train a single classifier for a given data and label set

    @param featsDict: List of dictionary representations of every data entry
    @param Y: labels for each data entry
    @param do_grid: use grid search? (default False)
    @param ovo: use one-vs-one classification? (default False results in one-vs-rest classification)
    '''

    #vectorize dictionary data
    vec = DictVectorizer()
    X = vec.fit_transform(featsDict)

    print
    print "\ttype(X): ", type(X)
    print "\tX.shape: ", X.shape
    print "\t# labels: ",  len(Y)
    print

    # Search space
#    C_range     = [1]
#    gamma_range = 10.0 ** np.arange( -3, 5 )

    C = 1
    gamma = None

    # NOTE: was found training on all training data for each type (t) respectively.
    if t == "EVENT_CLASS":
        gamma = 0.10000000000000001
    elif t == "TLINK":
        gamma = 100
    else:
        print
        sys.exit("DEV ERROR: specify type we are training.")
        print

    # clf = None

    # one-vs-one?
    func_shape = 'ovr'
    if ovo:
        func_shape = 'ovo'

    # Grid search?
    #if do_grid:

    #    print "training model [GRID SEARCH ON]"

    #    estimates = SVC(kernel='poly', degree=2, max_iter=1000, decision_function_shape=func_shape, class_weight='balanced', cache_size=500)
    #    parameters = [{'C':C_range, 'gamma':gamma_range}]

        # Find best classifier
    #    clf = GridSearchCV(estimates,
    #                       parameters,
    #                       scoring='f1_weighted',
    #                       n_jobs=30,
    #                       cv=10,
    #                       verbose=10)

    #    clf.fit(X, Y)

    #else:

    print "training model [GRID SEARCH OFF]"
    clf = SVC(kernel='poly', C=C, gamma=gamma, degree=2, max_iter=1000, decision_function_shape=func_shape, class_weight='balanced', cache_size=500, verbose=True)
    clf.fit(X, Y)

    return clf, vec

