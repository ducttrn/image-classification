import pickle


def save_model(filepath, clf):
    pickle.dump(clf, open(filepath, "wb"))


def load_model(model_path: str = "models/model.pkl",):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    return model
