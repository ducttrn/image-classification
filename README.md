# Image Classification using Bag-of-Visual-Word
Compare the accuracy of SVM and Naive Bayes in image classification when using Bag-of-Visual-Word model.

## ⬇ Installation

#### 0. Prerequisites
Make sure you have installed these following prerequisites on your computer:
- Python 3.6 or later.
You can download different versions of Python here:
http://www.python.org/getit/
- `pip`
- `virtualenv`

#### 1. Install and activate virtual environment
```
$ virtualenv venv --python=python3.8
$ source venv/bin/activate
```

#### 2. Install requirements
```
pip install -r requirements.txt
```

#### 3. Install development requirements (Optional)
```
pip install -r requirements-dev.txt
```

## 🚀 Features
### 1. Train:
```
$ python train.py
```
This command will train 2 models, one based on SVM and the other on Naive Bayes, to classify images of three classes, Airplane, Face, and Motor. The trained models will be saved in the `models` directory.
- Training data:
    - Airplane: 700 images
    - Face: 300 images
    - Motorbike: 700 images
- Test data:
    - Airplane: 100 images
    - Face: 100 images
    - Motorbike: 100 images
### 2. Evaluate:
```
$ python evaluate.py
```
This command will evaluate the accuracy of the two models trained in the previous step.
```text
SVM Classification Accuracy: 94.0%
Naive Bayes Classification Accuracy: 83.0%
```
## 🤟 Acknowledgement
Data used in training and testing for this project are from [Caltech Vision Datasets](https://drive.google.com/drive/folders/1kLMG1pa3xV_TwK0DnibSbjYrj_hjGttf).

## 📄 References
- https://machinelearningknowledge.ai/image-classification-using-bag-of-visual-words-model/
- https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
- https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
- E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," 2011 International Conference on Computer Vision, 2011, pp. 2564-2571, doi: 10.1109/ICCV.2011.6126544.
