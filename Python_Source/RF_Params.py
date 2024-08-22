from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''
dataset = 1

if dataset ==1:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/All/'
    log_file_name = f"logfile_RFParams_All_{current_time}.log"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/AnimalsOnly/'
    log_file_name = f"logfile_RFParams_AnimalsOnly_{current_time}.log"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/HumanVsAnimal/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/HumanVsAnimal/'
    log_file_name = f"logfile_RFParams_HumanVsAnimal_{current_time}.log"
    target_name_labels = ['Human', 'Animal']
else:
    print("Error: Choose a dataset")

train_data = np.load(datapath + 'Train_Data.npy')
train_label = np.load(datapath +'Train_Label.npy')


# Open a file to redirect the output
log_file_path = os.path.join(log_directory, log_file_name)
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

print("Dataset: ")
print(target_name_labels)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', 0.5],  # Number of features to consider at every split
}

# Initialize the classifier
rf = RandomForestClassifier()

# Initialize the GridSearchCV object, not sure why it only does 3 fold and not 10
grid_search = GridSearchCV(rf, param_grid, cv=10,  scoring='accuracy',verbose=3)

# Fit the grid search to the data
grid_search.fit(train_data, train_label)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__



