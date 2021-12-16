import cv2
import numpy as np
from scipy.cluster.vq import vq

from helper import load_models
from preprocess import get_test_data


def evaluate_model():
    estimator, orb, voc = load_models()
    test_images, test_labels = get_test_data()
    n = len(test_images)

    test_descriptors = []
    # extract features
    for image_path in test_images:
        img = cv2.imread(image_path)
        features = orb.detect(img, None)
        _, img_descriptor = orb.compute(img, features)
        test_descriptors.append((image_path, img_descriptor))

    k = 200
    test_features = np.zeros((n, k), "float32")
    for i in range(n):
        words, distance = vq(test_descriptors[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    predictions = estimator.predict(test_features)
    correct = 0
    for i in range(n):
        if predictions[i] == test_labels[i]:
            correct += 1

    return f'Classification accuracy: {(correct / n) * 100}%'


if __name__ == '__main__':
    evaluate_model()
