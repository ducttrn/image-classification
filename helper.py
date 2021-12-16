import pickle


def _save_model(filepath, clf):
    pickle.dump(clf, open(filepath, "wb"))
