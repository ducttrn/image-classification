import pickle


def save_model(filepath, clf):
    pickle.dump(clf, open(filepath, "wb"))


def load_models(
    model_path: str = "models/model.pkl",
    orb_path: str = "models/orb.pkl",
    voc_path: str = "models/voc.pkl"
):
    with open(model_path, "rb") as estimator_file:
        estimator = pickle.load(estimator_file)

    with open(orb_path, "rb") as orb_file:
        orb = pickle.load(orb_file)

    with open(voc_path, "rb") as voc_file:
        voc = pickle.load(voc_file)

    return estimator, orb, voc
