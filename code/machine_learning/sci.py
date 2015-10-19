from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

def train( featsDict, Y ):

	print "Called sci.train"

	#vectorize dictionary data
	vec = DictVectorizer()
	X = vec.fit_transform(featsDict).toarray()

	clf = LinearSVC()
	clf.fit(X, Y)

	# Validate that training data is classified properly
	testval = list(clf.predict(X))

	if(testval == Y):
		print 'yes'

	return clf
