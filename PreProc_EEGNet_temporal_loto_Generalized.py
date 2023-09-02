#!/usr/bin/env python
# coding: utf-8

# import libraries

import os
from os import listdir
import pandas as pd
import numpy as np
import time
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from Models import EEGNet

from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal

from scipy.signal import butter, lfilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#UNWANTED_COLS = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz', 'CP1', 'CP2'] #kept ['FC5', 'FC6', 'C3', 'C4', 'CP5', 'CP6', 'T7', 'T8']
UNWANTED_COLS = ['F3', 'Fz', 'F4', 'FC1', 'FC5', 'FC2', 'FC6', 'CP1', 'CP5', 'CP2', 'CP6', 'T7', 'T8'] #kept ['C3', 'Cz', 'C4']
# all: F3, Fz, F4, FC1, FC5, FC2, FC6, C3, Cz, C4, CP1, CP5, CP2, CP6, T7, T8


# all subjects csv files folder

DATA_FOLDER = "output_trial"

# subject wise performance result csv

SUBJECT_WISE_PERFORMANCE_METRIC_CSV_GENERALIZED = "PreProc_EEGNet_temporal_loto_Generalized_subject_performance_metric.csv"

# list all the subject wise csv files

def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]


# load dataframe from csv

def load_data(filename):
    # read csv file
    df = pd.read_csv(filename)
    return df

def class_scores(y_true, y_pred, reference):
    
    """Function which takes two lists and a reference indicating which class
    to calculate the TP, FP, and FN for."""
    
    # .................................................................
    Y_true = set([i for (i, v) in enumerate(y_true) if v == reference])
    # print("Y_true:{}".format(Y_true))
    Y_pred = set([i for (i, v) in enumerate(y_pred) if v == reference])
    # print("Y_pred:{}".format(Y_pred))
    TP = len(Y_true.intersection(Y_pred))
    # print(TP)
    FP = len(Y_pred - Y_true)
    FN = len(Y_true - Y_pred)
    return TP, FP, FN


def f_beta_score(precision, recall, beta=1):
    """A function which takes the precision and recall of some model, and a value for beta,
    and returns the f_beta-score"""
    #.......................................................................
    return (1+beta**2) * precision * recall / (beta**2 * precision + recall)


def f_score(precision, recall):
    return f_beta_score(precision, recall, beta=1)

filters = 6
class FilterBank(BaseEstimator, TransformerMixin):

# obtained from https://www.kaggle.com/eilbeigi/visual/data
# author: fornax, alexandre

    """Filterbank TransformerMixin.
    Return signal processed by a bank of butterworth filters.
    """

    def __init__(self, filters='LowpassBank'):
        """init."""
        if filters == 'LowpassBank':
            #self.freqs_pairs = [[0.5], [1], [2], [3], [4], [5], [7], [9], [15], [30]]
            self.freqs_pairs = [[0.5], [4], [4, 7], [8, 12], [12, 30], [0.5, 30]] 
        else:
            self.freqs_pairs = filters
        self.filters = filters

    def fit(self, X, y=None):
        """Fit Method, Not used."""
        return self

    def transform(self, X, y=None):
        """Transform. Apply filters."""
        X_tot = []
        for freqs in self.freqs_pairs:
            if len(freqs) == 1:
                b, a = butter(5, freqs[0] / 250.0, btype='lowpass')
            else:
                if freqs[1] - freqs[0] < 3:
                    b, a = butter(3, np.array(freqs) / 250.0, btype='bandpass')
                else:
                    b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')
            X_filtered = lfilter(b, a, X, axis=0)
            X_tot.append(X_filtered)

        return np.array(X_tot)

def preprocessData(data):
    """Preprocess data with filterbank."""
    fb = FilterBank()
    return fb.transform(data)

def resample_single_electrode(electrode_data):
    downsampling_factor = 2 #original_sampling_rate // new_sampling_rate
    return signal.resample(electrode_data, len(electrode_data) // downsampling_factor)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def multi_data_trial_125Hz(df, channels):
    #Version: 1.0 a
    x = []
    y = []
    
    groups = df.groupby('trial')
    trials = groups.ngroups
    
    for i, group in groups:
        group = group.drop(['trial'], axis=1)
        y_trial = group.pop("class").head(1)

        group = group.apply(resample_single_electrode, axis=0)

        x.append(preprocessData(group.to_numpy()).transpose().reshape(channels*filters, len(group), 1))
        y.append(y_trial)
        
    x = np.array(x)
    return x, y

def train_model(X, y):
    lhs = 'Generalized'
    channels = 3
    samples = 1001

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    groups = np.array(range(X.shape[0]))
    logo = LeaveOneGroupOut()
    print(logo.get_n_splits(X, y, groups))

    merged_subjects_df = pd.DataFrame()

    for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
        print(f"Fold {i}:")
        #print(f"  Train: index={train_index}, group={groups[train_index]}")
        #print(f"  Test:  index={test_index}, group={groups[test_index]}")
        x_train=np.concatenate(np.array([X[ii] for ii in train_index]))
        y_train=np.concatenate(np.array([y[ii] for ii in train_index]))

        x_test=np.concatenate(np.array([X[ii] for ii in test_index]))
        y_test=np.concatenate(np.array([y[ii] for ii in test_index]))

        # some parameters
        batch_size = 16
        num_classes = 2
        epochs = 300

        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        #===============================================================================================================
        #=========================================== NEURAL ARCHITECTURE ===============================================
        #===============================================================================================================

        model = EEGNet(2, channels, samples//2)
        model.summary()

        opt = tf.keras.optimizers.Adam()

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print("==========",lhs,"==========")

        st = time.time()
        # train the model
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=False,
                            )
        
        et = time.time()
        train_time = et-st
        print("Train time:", train_time, "seconds")

        # save the model
        model.save('saved_model/EEGNet_Generalized')

        # history plot for accuracy and loss
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.suptitle('PreProc_EEGNet_temporal_Generalized Model Performance {} loto {}'.format(lhs, test_index[0]))
        plt.tight_layout()
        plt.savefig('graphs/PreProc_EEGNet_temporal_Generalized Model Performance {} loto {}.png'.format(lhs, test_index[0]))
        # clean the current fig for save plotting
        plt.clf()
        # plt.show()

        # convert them to single-digit ones
        predictions = model.predict(x_test)

        report = classification_report(y_test, predictions.round(),  output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"reports_csv/{lhs}_{test_index[0]}.csv")

        print(history.history['accuracy'][-1], history.history['val_accuracy'][-1], history.history['loss'][-1], history.history['val_loss'][-1])

        subject_id, acc, val_acc, loss, val_loss, trial_out = lhs, history.history['accuracy'][-1], history.history['val_accuracy'][-1], history.history['loss'][-1], history.history['val_loss'][-1], test_index[0]

        subject_dic = pd.DataFrame(
            [(subject_id, acc, val_acc, loss, val_loss, trial_out)],
            columns=['subject_id', 'train_acc', 'test_acc', 'train_loss', 'test_loss', 'trial_out'])
        merged_subjects_df = pd.concat([merged_subjects_df, subject_dic], ignore_index=True)

    return merged_subjects_df


if __name__ == '__main__':
    subject_filenames = find_csv_filenames(DATA_FOLDER)

    # new dataframe for performance metrics
    merged_subjects_df = pd.DataFrame()

    # new dataframe with all subject's data merged
    merged_dataset_df = pd.DataFrame()

    X = []
    X1 = []
    X2 = []
    X3 = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    for name in subject_filenames:
        df = load_data(name)
        df.drop(UNWANTED_COLS, axis=1, inplace=True, errors='ignore')
        for i in range(3):
            min_trial = df['trial'].min()
            df['trial'] = df['trial']-min_trial+1

            lower_bound = i * 40 + 1
            upper_bound = (i + 1) * 40
            
            # Split trial
            x = df[(df['trial'] >= lower_bound) & (df['trial'] <= upper_bound)].copy()

            # temporal data
            x, y_tmp = multi_data_trial_125Hz(x, channels=3)

            if i == 0:
                X1.append(x)
                y1.append(y_tmp)
            elif i == 1:
                X2.append(x)
                y2.append(y_tmp)
            else:
                X3.append(x)
                y3.append(y_tmp)

            del x, y_tmp
        del df

    X.append(X1)
    X.append(X2)
    X.append(X3)
    X = np.array(X)
    print(X.shape)
    X = X.reshape(3, 2200, 3, 1001, 6, 1)

    y.append(y1)
    y.append(y2)
    y.append(y3)
    y = np.array(y)
    y = y.reshape(3, 2200, 1)

    # unified dataset/model for generalization
    st = time.time()
    subjects_dic = train_model(X, y)
    et = time.time()
    print("Generalization train time:", et-st, "seconds")
    merged_subjects_df = pd.concat([merged_subjects_df, subjects_dic], ignore_index=True)
    merged_subjects_df.to_csv(SUBJECT_WISE_PERFORMANCE_METRIC_CSV_GENERALIZED, index=True)