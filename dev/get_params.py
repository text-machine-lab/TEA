import cPickle as pickle
import glob

print glob.glob("models/*")

tlink_model_path = "models/all_feats_MAY_24th.model_EVENT_CLASS_MODEL_PREDICATE_AS_EVENT"
tlink_model = pickle.load(open(tlink_model_path,"rb"))

print tlink_model.best_params_

tlink_model_path = "models/all_feats_MAY_24th.model_TLINK_MODEL"
tlink_model = pickle.load(open(tlink_model_path,"rb"))

print tlink_model.best_params_

