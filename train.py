import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from helper import extract_feature, save_model
from preprocess import get_training_data


def _get_training_feature():
    training_images, training_labels = get_training_data()
    img_features, codebook = extract_feature(training_images)
    return img_features, codebook, training_labels


def train_svm(model_directory: str = "models"):
    img_features, codebook, training_labels = _get_training_feature()

    # train a SVM classifier
    model = SVC(max_iter=10000)
    estimator = model.fit(img_features, np.array(training_labels))

    # For testing and reuse
    save_model(model_directory + "/svm_estimator.pkl", estimator)
    save_model(model_directory + "/svm_codebook.pkl", codebook)


def train_nb(model_directory: str = "models"):
    img_features, codebook, training_labels = _get_training_feature()

    # train a Naive Bayes classifier
    model = GaussianNB()
    estimator = model.fit(img_features, np.array(training_labels))

    # For testing and reuse
    save_model(model_directory + "/nb_estimator.pkl", estimator)
    save_model(model_directory + "/nb_codebook.pkl", codebook)


if __name__ == "__main__":
    train_svm()
    train_nb()
