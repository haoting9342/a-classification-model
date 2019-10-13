import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pdb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pathlib
import glob
import datetime as dt

from numpy import newaxis
import keras
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, sgd, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
import io
import logging
import itertools
from matplotlib import cm


class QoEModel_R4():

    def __init__(self, input_shape, num_classes, label_mapping):
        self.logger = logging.getLogger('QoE.model')
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_mapping = label_mapping

    def dnn_model(self):

        model = Sequential()
        model.add(Dense(512, input_shape=(self.input_shape,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def run(self, df, split_rate, batch_size, epochs, save_path):

        train_data = df.values
        shuffle_list = list(range(train_data.shape[0]))
        np.random.shuffle(shuffle_list)
        train_data = train_data[shuffle_list, :]
        train_size = int(train_data.shape[0] * split_rate)
        train = train_data[:train_size, :]
        test = train_data[train_size:, :]
        # split into input and outputs
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        self.logger.info(f"{train_x.shape}, {train_y.shape}, {test_x.shape}, {test_y.shape}")

        model = self.dnn_model()

        train_y = keras.utils.to_categorical(train_y, self.num_classes)
        test_y = keras.utils.to_categorical(test_y, self.num_classes)

        #model_path = os.path.join(save_path, 'QoE_best_%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S')))

        model_path = os.path.join(save_path, 'QoE_best.h5')
        lr = LearningRateScheduler(self.lr_scheduler)
        lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=10, verbose=1)

        callbacks = [ModelCheckpoint(filepath=model_path, monitor='val_acc', save_best_only=True)]
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(test_x, test_y),
                            callbacks=callbacks)
        score = model.evaluate(test_x, test_y)
        self.logger.info(f"model evaluation score: {score}")
        model_path = os.path.join(save_path, 'QoE_final.h5')
        model.save(model_path)
        self.logger.info(f"Model saved in {model_path}")

        # training logs
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(history.history['loss'], label='training_loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        train_fig_path = os.path.join(save_path, 'loss_%s.png' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S')))
        plt.savefig(train_fig_path)
        #plt.show()
        plt.close()

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(history.history['acc'], label='training_acc')
        ax.plot(history.history['val_acc'], label='val_acc')
        plt.legend()
        train_fig_path = os.path.join(save_path, 'acc_%s.png' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S')))
        plt.savefig(train_fig_path)
        # plt.show()
        plt.close()

        tpred = model.predict(test_x, verbose=1)
        label_tpred = np.argmax(tpred, axis=1).astype(int)

        label_true = np.argmax(test_y, axis=1).astype(int)
        sk_report = classification_report(
            digits=6,
            y_true=label_true,
            y_pred=label_tpred)
        self.logger.info(f"training report:\n {sk_report}")
        cmatrix = confusion_matrix(label_true, label_tpred)
        self.logger.info(f"confusion matrix: {cmatrix}")
        self.plot_confusion_matrix(conf_matrix=cmatrix,
                                   classes=self.get_labels(range(self.num_classes)),
                                   save_path=save_path)

    def inference(self, model_path, data, log_path):
        model = load_model(model_path)

        x, y = data.values[:, :-1], data.values[:, -1]
        pred = model.predict(x, verbose=1)
        y_pred = np.argmax(pred, axis=1).astype(int)

        y = y.astype(int)
        sk_report = classification_report(
            digits=6,
            y_true=y,
            y_pred=y_pred)
        self.logger.info(f"training report:\n {sk_report}")
        cmatrix = confusion_matrix(y, y_pred)
        self.logger.info(f"Confusion matrix:\n {cmatrix}")
        self.plot_confusion_matrix(conf_matrix=cmatrix,
                                   classes=self.get_labels(range(self.num_classes)),
                                   save_path=log_path)


    def lr_scheduler(self, epoch, max_epoch = 100, mode='progressive_drops'):
        lr_base = 0.001
        epochs = max_epoch
        lr_power = 0.9
        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.00005
            elif epoch > 0.75 * epochs:
                lr = 0.0001
            elif epoch > 0.2 * epochs:
                lr = 0.0005
            else:
                lr = 0.001

        return lr

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
