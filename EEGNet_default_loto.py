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

from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# all subjects csv files folder

DATA_FOLDER = "output_trial"

# subject wise performance result csv

SUBJECT_WISE_PERFORMANCE_METRIC_CSV = "EEGNet_default_loto_subject_performance_metric.csv"

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

def resample_single_electrode(electrode_data):
    downsampling_factor = 2 #original_sampling_rate // new_sampling_rate
    return signal.resample(electrode_data, len(electrode_data) // downsampling_factor)


def multi_data(df, channels, samples, augment=False):
    #Version: 3.12 a
    x = []
    y = []
    trials = df['trial'].max()
    
    printProgressBar(0, trials, prefix='Splitting trials:', suffix='', length=50)
    
    groups = df.groupby('trial')
    
    for i, group in groups:
        group = group.drop(['trial'], axis=1)
        y_trial = group.pop("class").head(1)

        group = group.apply(resample_single_electrode, axis=0)

        temp_image = np.empty((channels, samples, 1), dtype='float32')

        jump = samples//3
        if augment:
            for j in range(group.shape[0] // jump):
                if j * jump + samples <= group.shape[0]:
                    temp_image = group.iloc[j * jump:(j * jump) + samples].to_numpy()
                    temp_image = temp_image.transpose().reshape(channels, samples, 1)
                    x.append(temp_image)
                    y.append(y_trial)
                else:
                    temp_image = group.iloc[group.shape[0]-samples-1:group.shape[0]-1].to_numpy()
                    temp_image = temp_image.transpose().reshape(channels, samples, 1)
                    x.append(temp_image)
                    y.append(y_trial)
        else:
            for j in range(group.shape[0] // samples):
                temp_image = group.iloc[j * jump:(j * jump) + samples].to_numpy()
                #temp_image = preprocessData(temp_image)
                temp_image = temp_image.transpose().reshape(channels, samples, 1)
                x.append(temp_image)
                y.append(y_trial)
        
        printProgressBar(i, trials, prefix='Splitting trials:', suffix='', length=50)
    
    x = np.array(x)
    return x, y


def train_model(filename):
    extracted_filename = os.path.basename(filename)
    
    # split the filename
    lhs, rhs = extracted_filename.split("_", 1)

    df = load_data(filename)

    samples = 125 # one second of samples
    channels = 16

    X = []
    y = []
    for i in range(3):
        min_trial = df['trial'].min()
        df['trial'] = df['trial']-min_trial+1

        lower_bound = i * 40 + 1
        upper_bound = (i + 1) * 40
        
        # trial split
        x = df[(df['trial'] >= lower_bound) & (df['trial'] <= upper_bound)].copy()

        # temporal data division
        x, y_tmp = multi_data(x, channels=channels, samples=samples)

        X.append(x)
        y.append(y_tmp)

    # convert list in numpy array
    X = np.array(X)
    y = np.array(y)

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

        # get the model
        model = EEGNet(num_classes, channels, samples)
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

        # save last model
        model.save('saved_model/EEGNet')

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

        plt.suptitle('EEGNet Performance {} loto {}'.format(lhs, test_index[0]))
        plt.tight_layout()
        plt.savefig('graphs/EEGNet Performance {} loto {}.png'.format(lhs, test_index[0]))
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

    st = time.time()
    # train a model for each subject
    for name in subject_filenames:
        subjects_dic = train_model(name)

        merged_subjects_df = pd.concat([merged_subjects_df, subjects_dic], ignore_index=True)

    et = time.time()
    print("All train time:", et-st, "seconds")
    # write into csv file
    merged_subjects_df.to_csv(SUBJECT_WISE_PERFORMANCE_METRIC_CSV, index=True)