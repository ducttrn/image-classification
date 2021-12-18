import pickle
from typing import List

import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq

from config import config


def save_model(filepath, clf):
    pickle.dump(clf, open(filepath, "wb"))


def load_model(model_path: str = "models/model.pkl",):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    return model


def extract_feature(images: List[str],):
    n = len(images)
    descriptors = []
    orb = cv2.ORB_create()

    # extract features
    for image_path in images:
        img = cv2.imread(image_path)
        features = orb.detect(img, None)
        _, img_descriptor = orb.compute(img, features)
        descriptors.append((image_path, img_descriptor))

    # reformat training descriptors
    concat_descriptors = descriptors[0][1]
    for image_path, descriptor in descriptors[1:]:
        concat_descriptors = np.vstack((concat_descriptors, descriptor))

    concat_descriptors = concat_descriptors.astype(float)

    # k-means clustering
    codebook, _ = kmeans(concat_descriptors, config.CLUSTER_SIZE, 1)

    # create histogram of training images
    img_features = np.zeros((n, config.CLUSTER_SIZE), "float32")
    for i in range(n):
        words, distance = vq(descriptors[i][1], codebook)
        for word in words:
            img_features[i][word] += 1

    return img_features, codebook
