# random forest method for QoE classification

import math
import plotly

import pandas as pd
import zipfile as zp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plotly.offline.init_notebook_mode(connected=True)

from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import logging
import os
import joblib
from matplotlib import cm
import itertools


class QoE_RF():
    '''
    Random forest method for QoE classification
    '''
    def __init__(self, num_classes, label_mapping, main_path):
        self.logger = logging.getLogger('QoE.RF')
        self.num_classes = num_classes
        self.label_mapping = label_mapping
        self.main_path = main_path

    def run(self, df, feature_set, feature_sel=False,
            n_estimators=20, max_depth=30, max_features='log2', criterion='entropy'):
        # determine the features
        x_train, y_train = self.to_x_y(df,target_column='label')

        if feature_sel:
            features, rfecv = self.select_best_features(x_train, y_train, n_jobs=16)
        else:
            features = feature_set

        # drop first type duplicates
        df = df.loc[:, features+['label']]
        self.logger.info(f"Selected features: {df.columns}")
        self.logger.info(f"Data shape before dropping duplicates: {df.shape}")
        #df = df.drop_duplicates(subset=features+['label'])
        self.logger.info(f"Data shape after dropping duplicates: {df.shape}")
        df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
        self.logger.info(f"Training data shape: {df_train.shape}. Testing data shape: {df_test.shape}")
        x_train, y_train = self.to_x_y(df_train, target_column='label')
        x_test, y_test = self.to_x_y(df_test, target_column='label')

        clf = RandomForestClassifier(class_weight=None,
                                     criterion=criterion,
                                     max_features=max_features,
                                     min_samples_leaf=1,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)

        self.logger.info(f"RandomForest Classifier with: criterion={criterion}, max_features={max_features}, n_estimators={n_estimators}, max_depth={max_depth}")

        cv = StratifiedKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(clf, x_train, y_train, cv=cv, verbose=True, n_jobs=16)
        self.logger.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        clf.fit(x_train, y_train)

        # save models
        model_path = os.path.join(self.main_path, 'model_rf.joblib')
        with open(model_path, 'wb') as f:
            joblib.dump(clf, f)

        self.logger.info(f"Model saved in {model_path}")
        y_pred = clf.predict(x_test)

        self.logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        sk_report = classification_report(
            digits=6,
            y_true=y_test,
            y_pred=y_pred)
        self.logger.info(f"training report:\n {sk_report}")
        cmatrix = confusion_matrix(y_test, y_pred)
        self.logger.info(f"Confusion matrix: {cmatrix}")
        self.plot_confusion_matrix(conf_matrix=cmatrix,
                                   classes=self.get_labels(range(self.num_classes)),
                                   save_path=self.main_path)

    def inference(self, path, data, features):
        with open(path, 'rb') as f:
            model = joblib.load(f)
        try:
            df = data.loc[:, features + ['label']]
        except:
            raise RuntimeError(f"Data should contain columns: {features}")
        x, y = df.values[:, :-1], df.values[:, -1]
        y_pred = model.predict(x)
        y_prob = np.max(model.predict_proba(x), axis=1)
        sk_report = classification_report(
            digits=6,
            y_true=y.astype(int),
            y_pred=y_pred.astype(int))
        self.logger.info(f"testing report:\n {sk_report}")
        cmatrix = confusion_matrix(y, y_pred)
        self.logger.info(f"Confusion matrix: {cmatrix}")
        self.plot_confusion_matrix(conf_matrix=cmatrix,
                                   classes=self.get_labels(range(self.num_classes)),
                                   save_path=self.main_path)


    def select_best_features(self, x_train, y_train, n_jobs=1, classifier=None):
        self.logger.info('Begin feature selection!')
        if classifier is None:
            classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10)
        rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(3),
                      scoring='accuracy', n_jobs=n_jobs)
        rfecv.fit(x_train, y_train)

        self.logger.info("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        fig_path = os.path.join(self.main_path, 'feature_selection.png')
        plt.savefig(fig_path)
        plt.close()
        #plt.show()
        self.logger.info(f"feature support: {rfecv.support_}")
        self.logger.info(f"feature ranking: {rfecv.ranking_}")
        #print(rfecv.support_)
        #print(rfecv.ranking_)

        features = []
        u = x_train.columns
        for i in range(rfecv.support_.shape[0]):
            if rfecv.support_[i]:
                features.append(u.values[i])
        self.logger.info(f"Final features: {features}")

        return features, rfecv

    def to_x_y(self, df, target_column, cat_type='encoding'):
        if cat_type == 'encoding':
            #         unique_vals_dct = dict(zip(df.loc[:, target_column].unique(), range(df.loc[:, target_column].nunique())))
            #unique_vals_dct = {'bad': 0, 'good': 2, 'normal': 1}
            #y = df.loc[:, target_column].apply(lambda x: unique_vals_dct[x])
            y = df.loc[:, target_column]
        else:
            y = pd.get_dummies(df[target_column], prefix=target_column)
        x = df.drop(target_column, axis=1)
        return x, y

    def get_labels(self, labels):
        label_mapping = self.label_mapping
        new_dict = {v: k for k, v in label_mapping.items()}

        return [new_dict[int(i)] for i in labels]

    def plot_confusion_matrix(self, conf_matrix, classes, save_path):
        cmap = cm.Blues
        plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.grid(False)

        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        fig_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(fig_path)