import os
from numpy.linalg import norm

import scipy.io
import scipy.signal
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
import numpy as np
from pywt import wavedec


AUDIO_PREFIX = "Audio"
EEG_PREFIX = "EEG"
EXTENSION = ".mat"

DATASET_DIR = "dataset"

NUM_EEG_CHANNELS = 6
EEG_LENGTH = 4096


def shuffle_data(x, y):
    assert x.shape[0] == y.shape[0]
    np.random.seed(10)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def feature_extraction(arr):
    feature_vector = []
    for sample in arr:
        sample = scipy.signal.decimate(sample, 8)
        coeffs = wavedec(sample, 'db4', level=5, axis=1)[:-1]
        ei = []
        for decomp in coeffs:
            ei.append(np.sum(np.square(np.abs(decomp)), axis=-1))
        ei = np.array(ei)
        et = np.sum(ei, axis=0)
        rwe = (ei / et)
        rwe= Normalizer().fit_transform(rwe)
        feature_vector.append(np.ravel(rwe))
    return np.array(feature_vector)


def main():
    X = []
    y = []
    subject_dir = sorted([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))])
    for subject in subject_dir:
        eeg_path = os.path.join(DATASET_DIR, subject, '_'.join([subject, EEG_PREFIX]) + EXTENSION)
        eeg_mat = scipy.io.loadmat(eeg_path)[EEG_PREFIX]
        imaginated = (eeg_mat[:, -3] - 1).astype(bool)
        eeg_data = eeg_mat[:, :-3].astype(np.float32)
        eeg_data = eeg_data.reshape(-1, NUM_EEG_CHANNELS, EEG_LENGTH)
        eeg_data_imagineted = eeg_data[~imaginated]
        y.append(eeg_mat[~imaginated, -2])
        X.append(feature_extraction(eeg_data_imagineted))

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0).astype(np.int32)

    X = X[y >= 6]
    y = y[y >= 6]

    X, y = shuffle_data(X, y)

    cv_results = cross_val_score(RandomForestClassifier(n_estimators=500), X, y, cv=10, scoring='accuracy', n_jobs=8)
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

    cv_results = cross_val_score(LinearSVC(C=2.0, max_iter=10000), X, y, cv=10, scoring='accuracy', n_jobs=8)
    print("SVM Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

    cv_results = cross_val_score(MLPClassifier((128, 64), solver="adam", max_iter=1000), X, y, cv=10, scoring="accuracy", n_jobs=8)
    print("MLP Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

if __name__ == "__main__":
    main()
