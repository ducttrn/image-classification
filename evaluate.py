import cv2
import numpy as np
from scipy.cluster.vq import vq

from config import config
from helper import load_model
from preprocess import get_test_data


def evaluate_model(estimator_path: str, codebook_path: str):
    estimator = load_model(estimator_path)
    codebook = load_model(codebook_path)

    orb = cv2.ORB_create()
    test_images, test_labels = get_test_data()
    n = len(test_images)

    test_descriptors = []
    # extract features
    for image_path in test_images:
        img = cv2.imread(image_path)
        features = orb.detect(img, None)
        _, img_descriptor = orb.compute(img, features)
        test_descriptors.append((image_path, img_descriptor))

    img_features = np.zeros((n, config.CLUSTER_SIZE), "float32")
    # create histogram of test images
    for i in range(n):
        words, distance = vq(test_descriptors[i][1], codebook)
        for word in words:
            img_features[i][word] += 1

    predictions = estimator.predict(img_features)
    correct = 0
    for i in range(n):
        if predictions[i] == test_labels[i]:
            correct += 1

    accuracy = correct / n
    return round(accuracy, 2)


if __name__ == "__main__":
    svm_accuracy = evaluate_model("models/svm_estimator.pkl", "models/svm_codebook.pkl")
    print(f"SVM Classification Accuracy: {svm_accuracy * 100}%")

    nb_accuracy = evaluate_model("models/nb_estimator.pkl", "models/nb_codebook.pkl")
    print(f"Naive Bayes Classification Accuracy: {nb_accuracy * 100}%")
