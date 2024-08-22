from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
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
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/PCA/Unscaled/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/Params/'
    log_file_name = f"logfile_kNNParams_All_{current_time}.log"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/PCA/Unscaled/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/Params/'
    log_file_name = f"logfile_kNNParams_AnimalsOnly_{current_time}.log"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/HumanVsAnimal/PCA/Unscaled/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/Params/'
    log_file_name = f"logfile_kNNParams_HumanVsAnimal_{current_time}.log"
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

print("Dataset: PCA: ")
print(target_name_labels)

# Define the parameter values that should be searched
k_range = list(range(1, 31))

# Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)

# Initialize the KNN model
knn = KNeighborsClassifier()

# Instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', verbose =3)

# Fit the grid with data
grid.fit(train_data, train_label)

# View the complete results
grid_results = grid.cv_results_

# Examine the best model
print(grid.best_score_)
print(grid.best_params_)

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

