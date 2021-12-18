import os


def _get_data(path):
    class_names = os.listdir(path)
    image_paths = []
    labels = []
    for name in class_names:
        directory = os.path.join(path, name)
        class_path = [os.path.join(directory, f) for f in os.listdir(directory)]
        image_paths += class_path
        labels.extend([name] * len(class_path))

    return image_paths, labels


def get_training_data():
    return _get_data("data/training")


def get_test_data():
    return _get_data("data/test")


if __name__ == "__main__":
    print(get_training_data())
