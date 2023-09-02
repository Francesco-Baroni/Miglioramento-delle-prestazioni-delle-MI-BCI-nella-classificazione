#!/usr/bin/env python
# coding: utf-8


# this class is used to do pre_processing subject wise
# create new output folder in current folder. all the new csv files will be generated there

# importing required libraries

import os
from os import listdir
import pandas as pd


# making List of unwanted columns

UNWANTED_COLS = ["TimeStamp", "Trigger", "t", "t2"]



# list all the subject wise csv files

def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]



# preprocessing code block

if __name__ == '__main__':
    
    # find csv from the folder "dataset"
    subject_filenames = find_csv_filenames("dataset")

    # iterate through all subject files
    for id, name in enumerate(subject_filenames):
        filename = os.path.basename(name)
        # print(filename)

        # split the filename by first underscore
        lhs, rhs = filename.split("_", 1)

        # word to search
        search = lhs
        # search every subject's csv files in a list
        result = [subject for subject in subject_filenames if search in subject]
        
        # merge each subject's csv file
        merged_df = pd.DataFrame()
        for idx, subject_name in enumerate(result):
            
            # read csv
            df = pd.read_csv(subject_name)

            # filter on Timestamp, only having values in the range of 1 to 5
            df = df.query('1 <= TimeStamp <=5')

            # split each trial
            df['trial'] = df['trial']+((id*120)+(idx*40))

            # remove unwanted cols
            df.drop(UNWANTED_COLS, axis=1, inplace=True, errors='ignore')

            # replace class 1, with 1 and  -1 with 0
            df['class'] = df['class'].map({1: 1, -1: 0})

            merged_df = pd.concat([merged_df, df], ignore_index=True)

        # remove unwanted cols
        merged_df.drop(["Unnamed: 0"], axis=1, inplace=True, errors='ignore')
        # write into csv file
        merged_df.to_csv("output_trial/{}_merged.csv".format(search), index=False)
        
        print("===== {} done =====".format(search))

