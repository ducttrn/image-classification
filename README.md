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
This command will train 2 models, one based on SVM and the other on Naive Bayes, to classify images of two classes, Airplane and Car. The trained models will be saved in the `models` directory.
- Training data:
    - Airplane: 50 images
    - Car: 50 images
- Test data:
    - Airplane: 50 images
    - Car: 50 images
### 2. Evaluate:
```
$ python evaluate.py
```
This command will evaluate the accuracy of the two models trained in the previous step.
```text
SVM Classification Accuracy: 78.0%
Naive Bayes Classification Accuracy: 74.0%
```
## 🤟 Acknowledgement
Data used in training and testing for this project are from [Caltech Vision Datasets](https://drive.google.com/drive/folders/1kLMG1pa3xV_TwK0DnibSbjYrj_hjGttf).
