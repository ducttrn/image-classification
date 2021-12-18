import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.svm import SVC

from helper import save_model
from preprocess import get_training_data


def train(model_directory: str = "models"):
    training_images, training_labels = get_training_data()
    n = len(training_images)
    training_descriptors = []
    orb = cv2.ORB_create()

    # extract features
    for image_path in training_images:
        img = cv2.imread(image_path)
        features = orb.detect(img, None)
        _, img_descriptor = orb.compute(img, features)
        training_descriptors.append((image_path, img_descriptor))

    # reformat training descriptors
    concat_descriptors = training_descriptors[0][1]
    for image_path, descriptor in training_descriptors[1:]:
        concat_descriptors = np.vstack((concat_descriptors, descriptor))

    concat_descriptors = concat_descriptors.astype(float)

    # k-means clustering
    k = 200
    codebook, _ = kmeans(concat_descriptors, k, 1)

    # create histogram of training images
    im_features = np.zeros((n, k), "float32")
    for i in range(n):
        words, distance = vq(training_descriptors[i][1], codebook)
        for w in words:
            im_features[i][w] += 1

    # train a SVM classifier
    model = SVC(max_iter=10000)
    estimator = model.fit(im_features, np.array(training_labels))

    save_model(model_directory + "/estimator.pkl", estimator)
    save_model(model_directory + "/codebook.pkl", codebook)


if __name__ == "__main__":
    train()
