# Improving the classification performance of MI-BCI

In this repository, you can find the implementation and an example of usage of the convolutional neural network presented in my undergraduate thesis at the University of Trento.

The data accompanying this code can be downloaded from Leeuwis, N., Paas, A., & Alimardani, M. (2021). Psychological and Cognitive Factors in Motor Imagery Brain Computer Interfaces.

https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2FZ7ZVOD&version=&q=&fileTypeGroupFacet=&fileAccess=Public&fileSortField=date

# Guidelines to use the code:

1) Create an empty folder
2) In the empty folder create folders named 'dataset', 'graphs', 'saved_model', 'output_trial' and 'reports_csv'
3) In the 'dataset' folder transfer EEG recordings of all the subjects from the dataset mentioned above
4) First Run Preprocessing_Data_Trial_split in Python
5) Then you can Run PreProc_CNN_3D_temporal_loto.py or PreProc_CNN_3D_temporal_loto_Generalized.py to start the training of the model

# Comparison model EEGNet

In Models.py there is also the implementatin of EEGNet (http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta).

After the steps before you can Run EEGNet_default_loto.py for the comparision test using EEGNet with default parameters or PreProc_EEGNet_temporal_loto_Generalized.py for the comparision test using EGGNet using data from different subjects.

# Utility code

In addition to the code for the models and the examples of usege there is a further code (count_FLOPs.py) that is used to count the number of FLOPs and show the summary of the model with an indication of the number of parameters used. In order to use this code, you must have trained the model about which you want to know this information. The saved model should then be in the 'saved_model' folder. Modify the path of the model you want to load in the code, change the size of the sample input data to fit the model and finally just run the code.
