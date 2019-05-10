from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os
import pickle

def get_X_train():
	df = pd.read_csv("embeds.csv")
	embeds = np.array(df)
	return embeds


def features_model(X_train, feature, lr=0.0001, max_iter=1600, layers=(128, 64)):
    X_train = X_train
    df = pd.read_csv("features.csv")
    y_train = df[feature]
    y_train = np.array(y_train)

    model = SVC(kernel='linear', gamma = 1e-8, probability=True)
    model.fit(X_train, y_train)

    classifier_filename = "./myclassifier/" + feature + "_svm2_classifier.pkl"
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'wb') as f:
        pickle.dump(model, f)

    return model
