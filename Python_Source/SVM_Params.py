from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import logging
import sys
import seaborn as sns
import datetime
import os

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''
dataset = 1
if dataset ==1:
    datapath = 'C:/Users/AmishaManga/Documents/FinalData/All/'
    log_directory = 'C:/Users/AmishaManga/Documents/Masters/Results/'
    log_file_name = f"logfile_SVM_Params_{current_time}.log"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/AmishaManga/Documents/FinalData/AnimalsOnly/'
    log_directory = 'C:/Users/AmishaManga/Documents/Masters/Results/'
    log_file_name = f"logfile_SVM_PCA_Unscaled_{current_time}.log"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/AmishaManga/Documents/FinalData/HumanVsAnimal/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/AmishaManga/Documents/Masters/Results/'
    log_file_name = f"logfile_SVM_PCA_Unscaled_{current_time}.log"
    target_name_labels = ['Human', 'Animal']
else:
    logging.info("Error: Choose a dataset")

train_data = np.load(datapath + 'Train_Data.npy')
test_data = np.load(datapath +'Test_Data.npy')
train_label = np.load(datapath +'Train_Label.npy')
test_label = np.load(datapath +'Test_Label.npy')

# Open a file to redirect the output
log_file_path = os.path.join(log_directory, log_file_name)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

logging.info("Dataset: ")
logging.info(target_name_labels)
logging.info("\n")

param_grid = {'C':[0.1,1,10],
'gamma': [1,0.1,0.01],
'kernel': ['rbf', 'linear', 'sigmoid', 'poly']}

grid = GridSearchCV(SVC(),  param_grid, cv =3, scoring='accuracy', verbose= 3)
grid.fit(test_data, test_label)

logging.info("best params:   ")
logging.info(grid.best_params_)
logging.info("\n")
logging.info("best estimator:    ")
logging.info(grid.best_estimator_)